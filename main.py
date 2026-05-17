import os
import uuid
import json
import shutil
import logging
import secrets
import tempfile
from collections import deque
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from typing import Dict, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
import jwt
from supabase import create_client, Client

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ADMIN_SECRET       = os.environ.get("ADMIN_SECRET", "change-me-in-production")
JWT_SECRET         = os.environ.get("JWT_SECRET")          # strong random secret
JWT_ALGORITHM      = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", 7))

SUPABASE_URL       = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")   # service-role key (server only)
SUPABASE_BUCKET    = os.environ.get("SUPABASE_BUCKET", "rag-documents")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is required")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

# ─────────────────────────────────────────────
# Supabase client (service role — never expose to frontend)
# ─────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
log_buffer: deque = deque(maxlen=500)

class RingBufferHandler(logging.Handler):
    def emit(self, record):
        log_buffer.append({
            "time":    datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "message": self.format(record),
        })

logger = logging.getLogger("rag_app")
logger.setLevel(logging.DEBUG)

_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_sh)

_rh = RingBufferHandler()
_rh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_rh)

try:
    _fh = logging.FileHandler("app.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
except Exception:
    pass

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API")

FRONTEND_ORIGINS = os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ─────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────
def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    payload = decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    return {"id": payload["sub"], "email": payload["email"]}

# ─────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# ─────────────────────────────────────────────
# RAG component singleton — lazy init
# ─────────────────────────────────────────────
_embedding_manager = None
_vector_store      = None
_retriever         = None
_chunker           = None
_llm               = None

def get_rag_components():
    global _embedding_manager, _vector_store, _retriever, _chunker, _llm

    if all([_embedding_manager, _vector_store, _retriever, _chunker, _llm]):
        return _embedding_manager, _vector_store, _retriever, _chunker, _llm

    try:
        from rag_engine import (
            EmbeddingManager, VectorStore,
            RagRetriever, ChunkingManager, get_llm,
        )
        logger.info("Initialising RAG components ...")
        _embedding_manager = EmbeddingManager()
        _vector_store      = VectorStore()
        _retriever         = RagRetriever(_vector_store, _embedding_manager)
        _chunker           = ChunkingManager()
        _llm               = get_llm()
        logger.info("RAG components ready.")
    except Exception as e:
        _embedding_manager = _vector_store = _retriever = _chunker = _llm = None
        logger.error(f"RAG init failed: {type(e).__name__} — {e}")
        raise

    return _embedding_manager, _vector_store, _retriever, _chunker, _llm

# ─────────────────────────────────────────────
# Admin auth
# ─────────────────────────────────────────────
def verify_admin(x_admin_key: str = Header(None)):
    if not x_admin_key or not secrets.compare_digest(x_admin_key, ADMIN_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")

# ─────────────────────────────────────────────
# Supabase DB helpers
# ─────────────────────────────────────────────
def db_get_user_by_email(email: str) -> Optional[dict]:
    res = supabase.table("users").select("*").eq("email", email).limit(1).execute()
    return res.data[0] if res.data else None

def db_get_user_by_id(user_id: str) -> Optional[dict]:
    res = supabase.table("users").select("*").eq("id", user_id).limit(1).execute()
    return res.data[0] if res.data else None

def db_create_user(user_id: str, email: str, name: str, password_hash: str) -> dict:
    data = {
        "id": user_id,
        "email": email,
        "name": name,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = supabase.table("users").insert(data).execute()
    return res.data[0]

def db_get_chats(user_id: str) -> list:
    res = supabase.table("conversations")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()
    return res.data or []

def db_get_chat(chat_id: str, user_id: str) -> Optional[dict]:
    res = supabase.table("conversations")\
        .select("*")\
        .eq("id", chat_id)\
        .eq("user_id", user_id)\
        .limit(1)\
        .execute()
    return res.data[0] if res.data else None

def db_get_messages(chat_id: str) -> list:
    res = supabase.table("messages")\
        .select("*")\
        .eq("conversation_id", chat_id)\
        .order("created_at", desc=False)\
        .execute()
    return res.data or []

def db_create_chat(user_id: str, title: str) -> dict:
    chat_id = str(uuid.uuid4())
    data = {
        "id": chat_id,
        "user_id": user_id,
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = supabase.table("conversations").insert(data).execute()
    return {**res.data[0], "messages": []}

def db_add_message(chat_id: str, role: str, content: str) -> dict:
    msg_id = str(uuid.uuid4())
    data = {
        "id": msg_id,
        "conversation_id": chat_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = supabase.table("messages").insert(data).execute()
    return res.data[0]

def db_update_chat_title(chat_id: str, title: str):
    supabase.table("conversations").update({"title": title}).eq("id", chat_id).execute()

def db_delete_messages_after(chat_id: str, keep_count: int):
    messages = db_get_messages(chat_id)
    if len(messages) > keep_count:
        ids_to_delete = [m["id"] for m in messages[keep_count:]]
        supabase.table("messages").delete().in_("id", ids_to_delete).execute()

def db_get_documents(user_id: str, session_id: str) -> list:
    res = supabase.table("uploaded_documents")\
        .select("*")\
        .eq("user_id", user_id)\
        .eq("session_id", session_id)\
        .execute()
    return res.data or []

def db_create_document(user_id: str, session_id: str, name: str,
                        chunk_count: int, vector_ids: list,
                        storage_path: str) -> dict:
    doc_id = str(uuid.uuid4())
    data = {
        "id": doc_id,
        "user_id": user_id,
        "session_id": session_id,
        "name": name,
        "chunk_count": chunk_count,
        "vector_ids": vector_ids,
        "storage_path": storage_path,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    res = supabase.table("uploaded_documents").insert(data).execute()
    return res.data[0]

def db_delete_document(doc_id: str, user_id: str) -> Optional[dict]:
    res = supabase.table("uploaded_documents")\
        .select("*")\
        .eq("id", doc_id)\
        .eq("user_id", user_id)\
        .limit(1)\
        .execute()
    if not res.data:
        return None
    supabase.table("uploaded_documents").delete().eq("id", doc_id).execute()
    return res.data[0]

def db_delete_session_documents(user_id: str, session_id: str) -> list:
    res = supabase.table("uploaded_documents")\
        .select("*")\
        .eq("user_id", user_id)\
        .eq("session_id", session_id)\
        .execute()
    docs = res.data or []
    if docs:
        supabase.table("uploaded_documents")\
            .delete()\
            .eq("user_id", user_id)\
            .eq("session_id", session_id)\
            .execute()
    return docs

def db_store_refresh_token(user_id: str, token_hash: str, expires_at: datetime):
    supabase.table("refresh_tokens").insert({
        "user_id": user_id,
        "token_hash": token_hash,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

def db_verify_refresh_token(user_id: str, token_hash: str) -> bool:
    res = supabase.table("refresh_tokens")\
        .select("*")\
        .eq("user_id", user_id)\
        .eq("token_hash", token_hash)\
        .gt("expires_at", datetime.now(timezone.utc).isoformat())\
        .execute()
    return bool(res.data)

def db_revoke_refresh_token(user_id: str, token_hash: str):
    supabase.table("refresh_tokens")\
        .delete()\
        .eq("user_id", user_id)\
        .eq("token_hash", token_hash)\
        .execute()

def db_revoke_all_refresh_tokens(user_id: str):
    supabase.table("refresh_tokens").delete().eq("user_id", user_id).execute()

# ─────────────────────────────────────────────
# Supabase Storage helpers
# ─────────────────────────────────────────────
def storage_upload(path: str, file_bytes: bytes, content_type: str = "application/pdf") -> str:
    supabase.storage.from_(SUPABASE_BUCKET).upload(
        path, file_bytes,
        {"content-type": content_type, "upsert": "false"}
    )
    return path

def storage_download(path: str) -> bytes:
    return supabase.storage.from_(SUPABASE_BUCKET).download(path)

def storage_delete(path: str):
    supabase.storage.from_(SUPABASE_BUCKET).remove([path])

# ─────────────────────────────────────────────
# Cleanup helpers
# ─────────────────────────────────────────────
def _cleanup_session_documents(user_id: str, session_id: str):
    """Delete Pinecone vectors, Supabase storage files, and DB records for a session."""
    docs = db_delete_session_documents(user_id, session_id)
    if not docs:
        return

    all_vector_ids = []
    for doc in docs:
        all_vector_ids.extend(doc.get("vector_ids") or [])
        try:
            storage_delete(doc["storage_path"])
        except Exception as e:
            logger.warning(f"Storage delete failed for {doc['storage_path']}: {e}")

    if all_vector_ids:
        try:
            _, vs, _, _, _ = get_rag_components()
            vs.delete_vectors(all_vector_ids)
            logger.info(f"Deleted {len(all_vector_ids)} Pinecone vectors for session {session_id}")
        except Exception as e:
            logger.error(f"Pinecone cleanup failed for session {session_id}: {e}")

def _cleanup_user_data(user_id: str):
    """Delete all data for a user (account deletion)."""
    # 1. Get all documents and clean storage + Pinecone
    res = supabase.table("uploaded_documents")\
        .select("*").eq("user_id", user_id).execute()
    docs = res.data or []

    all_vector_ids = []
    for doc in docs:
        all_vector_ids.extend(doc.get("vector_ids") or [])
        try:
            storage_delete(doc["storage_path"])
        except Exception:
            pass

    if all_vector_ids:
        try:
            _, vs, _, _, _ = get_rag_components()
            vs.delete_vectors(all_vector_ids)
        except Exception as e:
            logger.error(f"Pinecone cleanup on delete-account failed: {e}")

    # 2. Delete from DB (cascade handles conversations → messages)
    supabase.table("uploaded_documents").delete().eq("user_id", user_id).execute()
    supabase.table("refresh_tokens").delete().eq("user_id", user_id).execute()
    # Get all conversation IDs then delete messages
    conv_res = supabase.table("conversations").select("id").eq("user_id", user_id).execute()
    conv_ids = [c["id"] for c in (conv_res.data or [])]
    if conv_ids:
        supabase.table("messages").delete().in_("conversation_id", conv_ids).execute()
    supabase.table("conversations").delete().eq("user_id", user_id).execute()
    supabase.table("users").delete().eq("id", user_id).execute()
    logger.info(f"Deleted all data for user {user_id}")

# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class LogoutRequest(BaseModel):
    refresh_token: str
    session_id: Optional[str] = None

class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"

class QueryRequest(BaseModel):
    chat_id: str
    query: str
    session_id: str

class ChatRename(BaseModel):
    title: str

class TruncateRequest(BaseModel):
    keep_count: int

class DeleteSessionRequest(BaseModel):
    session_id: str

class UpdateProfileRequest(BaseModel):
    name: str

# ─────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
def register(body: RegisterRequest):
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    if db_get_user_by_email(body.email):
        raise HTTPException(409, "Email already registered")

    user_id       = str(uuid.uuid4())
    password_hash = hash_password(body.password)
    user          = db_create_user(user_id, body.email, body.name, password_hash)

    access_token  = create_access_token(user_id, body.email)
    refresh_token = create_refresh_token(user_id)
    token_hash    = secrets.token_hex(32)   # opaque hash stored in DB

    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    db_store_refresh_token(user_id, token_hash, expires_at)

    logger.info(f"New user registered: {body.email}")
    return {
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_hash":    token_hash,
        "user": {
            "id":    user["id"],
            "email": user["email"],
            "name":  user["name"],
            "created_at": user["created_at"],
        },
    }


@app.post("/auth/login")
def login(body: LoginRequest):
    user = db_get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    access_token  = create_access_token(user["id"], user["email"])
    refresh_token = create_refresh_token(user["id"])
    token_hash    = secrets.token_hex(32)

    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    db_store_refresh_token(user["id"], token_hash, expires_at)

    logger.info(f"User logged in: {body.email}")
    return {
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_hash":    token_hash,
        "user": {
            "id":    user["id"],
            "email": user["email"],
            "name":  user["name"],
            "created_at": user["created_at"],
        },
    }


@app.post("/auth/refresh")
def refresh_token(body: RefreshRequest):
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid refresh token")

    user_id = payload["sub"]
    # We verify the opaque token_hash stored in DB is valid
    # (client must send token_hash alongside refresh_token)
    raise HTTPException(400, "Send token_hash with this request")


@app.post("/auth/refresh-with-hash")
def refresh_with_hash(body: dict):
    refresh_token_str = body.get("refresh_token")
    token_hash        = body.get("token_hash")

    if not refresh_token_str or not token_hash:
        raise HTTPException(400, "refresh_token and token_hash required")

    payload = decode_token(refresh_token_str)
    if payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid refresh token")

    user_id = payload["sub"]
    if not db_verify_refresh_token(user_id, token_hash):
        raise HTTPException(401, "Refresh token revoked or expired")

    user = db_get_user_by_id(user_id)
    if not user:
        raise HTTPException(401, "User not found")

    # Rotate tokens
    db_revoke_refresh_token(user_id, token_hash)
    new_access  = create_access_token(user_id, user["email"])
    new_refresh = create_refresh_token(user_id)
    new_hash    = secrets.token_hex(32)

    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    db_store_refresh_token(user_id, new_hash, expires_at)

    return {
        "access_token":  new_access,
        "refresh_token": new_refresh,
        "token_hash":    new_hash,
    }


@app.post("/auth/logout")
def logout(body: LogoutRequest, current_user: dict = Depends(get_current_user)):
    db_revoke_refresh_token(current_user["id"], body.refresh_token)

    # Cleanup session documents on logout
    if body.session_id:
        try:
            _cleanup_session_documents(current_user["id"], body.session_id)
            logger.info(f"Session {body.session_id} cleaned up on logout")
        except Exception as e:
            logger.error(f"Session cleanup on logout failed: {e}")

    return {"detail": "Logged out"}


@app.get("/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    user = db_get_user_by_id(current_user["id"])
    if not user:
        raise HTTPException(404, "User not found")
    return {
        "id":         user["id"],
        "email":      user["email"],
        "name":       user["name"],
        "created_at": user["created_at"],
    }


@app.patch("/auth/me")
def update_profile(body: UpdateProfileRequest, current_user: dict = Depends(get_current_user)):
    if not body.name.strip():
        raise HTTPException(400, "Name cannot be empty")
    supabase.table("users")\
        .update({"name": body.name.strip()})\
        .eq("id", current_user["id"])\
        .execute()
    return {"detail": "Profile updated"}


@app.delete("/auth/me")
def delete_account(current_user: dict = Depends(get_current_user)):
    _cleanup_user_data(current_user["id"])
    logger.info(f"Account deleted: {current_user['email']}")
    return {"detail": "Account deleted"}

# ─────────────────────────────────────────────
# Chat endpoints (auth-gated)
# ─────────────────────────────────────────────
@app.get("/chats")
def list_chats(current_user: dict = Depends(get_current_user)):
    chats = db_get_chats(current_user["id"])
    result = []
    for chat in chats:
        messages = db_get_messages(chat["id"])
        result.append({**chat, "messages": messages})
    return result


@app.post("/chats")
def create_chat(body: ChatCreate, current_user: dict = Depends(get_current_user)):
    return db_create_chat(current_user["id"], body.title or "New Chat")


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    chat = db_get_chat(chat_id, current_user["id"])
    if not chat:
        raise HTTPException(404, "Chat not found")
    messages = db_get_messages(chat_id)
    return {**chat, "messages": messages}


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    chat = db_get_chat(chat_id, current_user["id"])
    if not chat:
        raise HTTPException(404, "Chat not found")
    supabase.table("messages").delete().eq("conversation_id", chat_id).execute()
    supabase.table("conversations").delete().eq("id", chat_id).execute()
    return {"deleted": True}


@app.patch("/chats/{chat_id}")
def rename_chat(chat_id: str, body: ChatRename, current_user: dict = Depends(get_current_user)):
    chat = db_get_chat(chat_id, current_user["id"])
    if not chat:
        raise HTTPException(404, "Chat not found")
    db_update_chat_title(chat_id, body.title)
    return {**chat, "title": body.title}


@app.patch("/chats/{chat_id}/truncate")
def truncate_chat(chat_id: str, body: TruncateRequest, current_user: dict = Depends(get_current_user)):
    chat = db_get_chat(chat_id, current_user["id"])
    if not chat:
        raise HTTPException(404, "Chat not found")
    if body.keep_count < 0:
        raise HTTPException(400, "keep_count must be >= 0")
    db_delete_messages_after(chat_id, body.keep_count)
    messages = db_get_messages(chat_id)
    return {**chat, "messages": messages}

# ─────────────────────────────────────────────
# RAG prompt builders
# ─────────────────────────────────────────────
def _build_doc_prompt(query: str, history_text: str, context: str) -> str:
    return f"""You are a helpful AI assistant with access to uploaded documents.

Previous conversation:
{history_text}

Relevant document context:
{context}

Instructions:
- Answer based primarily on the document context above.
- Be specific and reference relevant parts of the context.
- If the context only partially answers the question, say so and supplement with general knowledge.
- Never invent facts not present in the context.
- Be concise and clear.

User question: {query}

Answer:"""


def _build_general_prompt(query: str, history_text: str) -> str:
    return f"""You are a helpful AI assistant.

Previous conversation:
{history_text}

Instructions:
- Answer the user's question using your general knowledge.
- Be concise and clear.

User question: {query}

Answer:"""

# ─────────────────────────────────────────────
# Query endpoint
# ─────────────────────────────────────────────
@app.post("/query")
def query_rag(body: QueryRequest, current_user: dict = Depends(get_current_user)):
    chat = db_get_chat(body.chat_id, current_user["id"])
    if not chat:
        raise HTTPException(404, "Chat not found")

    request_id = str(uuid.uuid4())[:8]

    # Persist user message
    db_add_message(body.chat_id, "user", body.query)

    messages = db_get_messages(body.chat_id)
    if len(messages) == 1:
        title = body.query[:50] + ("..." if len(body.query) > 50 else "")
        db_update_chat_title(body.chat_id, title)

    response_text = "I'm having trouble answering that right now. Please try again in a moment."

    try:
        _, _, retriever, _, llm_instance = get_rag_components()

        history_msgs = messages[-7:-1]
        history_text = ""
        for m in history_msgs:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"

        # Check if user has documents in this session
        session_docs = db_get_documents(current_user["id"], body.session_id)

        relevant_docs = retriever.get_documents(body.query)
        logger.info(f"[{request_id}] retrieved {len(relevant_docs)} chunks (first pass)")

        if not relevant_docs and session_docs:
            relevant_docs = retriever.get_documents(body.query, initial_k=40, top_k=3)
            logger.info(f"[{request_id}] retrieved {len(relevant_docs)} chunks (retry pass)")

        if not relevant_docs:
            if session_docs:
                note = "\n\n*Note: No matching content found in uploaded documents — answer uses general knowledge.*"
            else:
                note = "\n\n*No documents uploaded yet. Upload PDFs to get document-aware answers.*"

            prompt = _build_general_prompt(body.query, history_text)
            raw = llm_instance.invoke(prompt)
            response_text = raw.content.strip() + (note if not session_docs else "")
        else:
            context = "\n\n---\n\n".join(d["text"] for d in relevant_docs)

            seen, source_details = set(), []
            for d in relevant_docs:
                meta = d["metadata"]
                key  = f"{meta.get('source', '?')}:p{meta.get('page', '?')}"
                if key not in seen:
                    seen.add(key)
                    source_details.append(
                        f"{meta.get('source', '?')} (Page {meta.get('page', '?')})"
                    )

            prompt = _build_doc_prompt(body.query, history_text, context)
            raw = llm_instance.invoke(prompt)
            response_text = raw.content.strip()

            if source_details:
                sources = "\n".join(f"• {s}" for s in source_details)
                response_text += f"\n\n**Sources:**\n{sources}"

    except Exception as e:
        logger.error(
            f"[{request_id}] query_rag failed | chat={body.chat_id} | "
            f"type={type(e).__name__} | detail={e}"
        )
        response_text = "I'm having trouble answering that right now. Please try again in a moment."

    assistant_msg = db_add_message(body.chat_id, "assistant", response_text)
    updated_messages = db_get_messages(body.chat_id)
    updated_chat = db_get_chat(body.chat_id, current_user["id"])

    return {
        "message": assistant_msg,
        "chat": {**updated_chat, "messages": updated_messages},
    }

# ─────────────────────────────────────────────
# Document endpoints (session-scoped)
# ─────────────────────────────────────────────
@app.get("/documents")
def list_documents(session_id: str, current_user: dict = Depends(get_current_user)):
    return db_get_documents(current_user["id"], session_id)


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-Id"),
    current_user: dict = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id   = str(uuid.uuid4())
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        content = await file.read()

        # Upload to Supabase Storage
        storage_path = f"{current_user['id']}/{session_id}/{doc_id}/{file.filename}"
        storage_upload(storage_path, content)
        logger.info(f"Uploaded {file.filename} to Supabase Storage ({len(content):,} bytes)")

        # Write to temp file for chunking
        with open(tmp_path, "wb") as f:
            f.write(content)

        try:
            emb_mgr, vs, _, chunker, _ = get_rag_components()

            chunks = chunker.chunk_documents([tmp_path])
            logger.info(f"upload_document: {len(chunks)} chunks created")

            if not chunks:
                raise ValueError("No text could be extracted from the PDF.")

            texts      = [c["page_content"] for c in chunks]
            embeddings = emb_mgr.get_embeddings(texts)

            vector_ids  = vs.put_documents(doc_id, chunks, embeddings)
            chunk_count = len(chunks)
            logger.info(f"upload_document: upserted {chunk_count} vectors (doc={doc_id})")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"upload_document pipeline failed | {type(e).__name__}: {e}")
            storage_delete(storage_path)
            raise HTTPException(500, "Document processing failed. Please try uploading again.")

        doc = db_create_document(
            user_id=current_user["id"],
            session_id=session_id,
            name=file.filename,
            chunk_count=chunk_count,
            vector_ids=vector_ids,
            storage_path=storage_path,
        )
        return doc

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"upload_document outer failed | {type(e).__name__}: {e}")
        raise HTTPException(500, "Document processing failed. Please try uploading again.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    doc = db_delete_document(doc_id, current_user["id"])
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        storage_delete(doc["storage_path"])
    except Exception as e:
        logger.warning(f"Storage delete failed: {e}")

    vector_ids = doc.get("vector_ids") or []
    if vector_ids:
        try:
            _, vs, _, _, _ = get_rag_components()
            vs.delete_vectors(vector_ids)
            logger.info(f"Deleted {len(vector_ids)} Pinecone vectors for doc {doc_id}")
        except Exception as e:
            logger.error(f"Pinecone delete failed for doc {doc_id}: {e}")

    return {"deleted": True}


@app.post("/documents/cleanup-session")
def cleanup_session(body: DeleteSessionRequest, current_user: dict = Depends(get_current_user)):
    """Call this on logout/tab-close to clean up session documents."""
    try:
        _cleanup_session_documents(current_user["id"], body.session_id)
        return {"cleaned": True}
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(500, "Cleanup failed")

# ─────────────────────────────────────────────
# Admin endpoints
# ─────────────────────────────────────────────
@app.get("/admin/logs", dependencies=[Depends(verify_admin)])
def admin_logs(level: Optional[str] = None, limit: int = 100):
    logs = list(log_buffer)
    if level:
        logs = [l for l in logs if l["level"] == level.upper()]
    return {"total": len(logs), "logs": logs[-limit:]}


@app.get("/admin/stats", dependencies=[Depends(verify_admin)])
def admin_stats():
    user_count = len(supabase.table("users").select("id").execute().data or [])
    conv_count = len(supabase.table("conversations").select("id").execute().data or [])
    doc_count  = len(supabase.table("uploaded_documents").select("id").execute().data or [])
    return {
        "users":           user_count,
        "conversations":   conv_count,
        "documents":       doc_count,
        "log_buffer_size": len(log_buffer),
        "rag_initialized": all([_embedding_manager, _vector_store, _retriever, _chunker, _llm]),
    }


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "rag_ready": all([_embedding_manager, _vector_store, _retriever, _chunker, _llm]),
    }