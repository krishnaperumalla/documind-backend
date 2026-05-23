import os
import uuid
import cohere
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY=os.getenv('COHERE_API_KEY')

class EmbeddingManager:
    def __init__(self):
        # Initialize Gemini client
        self.embeddings=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key =GOOGLE_API_KEY
        )

    def get_embeddings(self, texts: List[str]) -> np.ndarray:

        vectors = self.embeddings.embed_documents(texts)

        return np.array(vectors, dtype=np.float32)



class VectorStore:
    def __init__(self, index_name: str = "dense-rag-index"):
        self.api_key = os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set.")
        self.index_name = index_name
        self._init_db()

    def _init_db(self):
        self.pc = Pinecone(api_key=self.api_key)
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def put_documents(self, doc_id: str, documents: List[dict], embeddings: np.ndarray, user_id:str, session_id:str) -> List[str]:
        index = self.pc.Index(self.index_name)
        vectors = []
        vector_ids = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            vid = f"{doc_id}_{i}_{uuid.uuid4().hex[:8]}"
            vector_ids.append(vid)
            metadata = dict(doc["metadata"])
            metadata["text"] = doc["page_content"]
            metadata["document_id"] = doc_id
            metadata["user_id"] = user_id        
            metadata["session_id"] = session_id  
            vectors.append({
                "id": vid,
                "values": emb.tolist(),
                "metadata": metadata,
            })

        for i in range(0, len(vectors), 200):
            index.upsert(vectors=vectors[i:i + 200])

        return vector_ids

    def delete_vectors(self, vector_ids: List[str]):
        if not vector_ids:
            return
        index = self.pc.Index(self.index_name)
        for i in range(0, len(vector_ids), 1000):
            index.delete(ids=vector_ids[i:i + 1000])



class RagRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vs = vector_store
        self.emb = embedding_manager
        self.reranker = None

    def _get_reranker(self):
        if self.reranker is None:
            self.reranker = cohere.Client(COHERE_API_KEY)
        return self.reranker

    def get_documents(
        self,
        query: str,
        user_id: str,           
        session_id: str = None, 
        initial_k: int = 20,
        rerank_k: int = 12,
        top_k: int = 5
    ) -> List[Dict]:

        query_embedding = self.emb.get_embeddings([query])[0]

        index = self.vs.pc.Index(self.vs.index_name)

        pinecone_filter = {"user_id": {"$eq": user_id}}
        if session_id:
            pinecone_filter["session_id"] = {"$eq": session_id}

        try:
            results = index.query(
                vector=query_embedding.tolist(),
                top_k=initial_k,
                include_metadata=True,
                filter=pinecone_filter,
            )

        except Exception as e:
            print(f"Pinecone query error: {e}")
            return []

        if not results.get("matches"):
            return []

        docs = []

        for match in results["matches"]:

            if "metadata" not in match:
                continue

            if "text" not in match["metadata"]:
                continue

            docs.append({
                "id": match["id"],
                "score": match["score"],
                "text": match["metadata"]["text"],
                "metadata": match["metadata"],
            })

        if not docs:
            return []

        docs = docs[:rerank_k]

        try:
            reranker = self._get_reranker()

            response = reranker.rerank(
                model="rerank-v3.5",
                query=query,
                documents=[d["text"] for d in docs],
                top_n=top_k
            )

            reranked_docs = []

            for result in response.results:

                doc = docs[result.index]

                doc["rerank_score"] = float(result.relevance_score)

                reranked_docs.append(doc)

            docs = sorted(
                reranked_docs,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

        except Exception as e:
            print(f"Reranker error (falling back to vector scores): {e}")

        return docs[:top_k]


class ChunkingManager:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _process_file(self, fp: str) -> List[dict]:
        loader = PyPDFLoader(fp)
        pages = loader.load()
        file_name = os.path.basename(fp)
        chunks = []
        for page_num, page in enumerate(pages):
            if not page.page_content.strip():
                continue
            for i, chunk in enumerate(self.splitter.split_text(page.page_content)):
                chunks.append({
                    "page_content": chunk,
                    "metadata": {
                        "source": file_name,
                        "page": page_num + 1,
                        "chunk_index": i,
                    },
                })
        return chunks

    def chunk_documents(self, file_paths: List[str]) -> List[dict]:
        all_chunks: List[dict] = []
        with ThreadPoolExecutor(max_workers=min(4, len(file_paths))) as pool:
            futures = {pool.submit(self._process_file, fp): fp for fp in file_paths}
            for future in as_completed(futures):
                try:
                    all_chunks.extend(future.result())
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
        return all_chunks


class StubLLM:
    def invoke(self, prompt: str) -> str:
        return (
            "⚠️ Gemini LLM is not reachable."
        )


def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=2000
        )
        return llm
    except Exception as e:
        print(f"Gemini LLM not available ({type(e).__name__}: {e}) — using StubLLM")
        return StubLLM()