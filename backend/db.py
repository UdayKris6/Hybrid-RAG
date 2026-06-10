import os
import re
import hashlib
import math
import uuid
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables from the .env file
load_dotenv()

# Singleton instance of QdrantClient
_client = None

def get_client() -> QdrantClient:
    """
    Lazy-loads and returns a singleton QdrantClient instance.
    This avoids acquiring file locks at import-time, preventing conflicts
    in multiprocessing/reloader environments like uvicorn reload.
    """
    global _client
    if _client is None:
        QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "./qdrant_storage")
        _client = QdrantClient(path=QDRANT_STORAGE_PATH)
    return _client

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "GEMINI").upper()
# Set vector dimensions dynamically (Gemini embedding-2 is 3072, OpenAI is 1536)
VECTOR_SIZE = 3072 if EMBEDDING_PROVIDER == "GEMINI" else 1536

COLLECTION_NAME = "ideas"
CHUNKS_COLLECTION_NAME = "idea_chunks"

def chunk_text(text: str, max_chars: int = 800, overlap: int = 150) -> List[str]:
    """
    Chunks text into passages of max_chars with overlap, avoiding splitting words.
    """
    if len(text) <= max_chars:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Find the nearest space to avoid cutting a word
        space_idx = text.rfind(' ', start, end)
        if space_idx > start + (max_chars // 2):
            end = space_idx
            
        chunks.append(text[start:end].strip())
        start = end - overlap
        
    return chunks

class BM25Vectorizer:
    """
    Stateful BM25 TF-IDF vectorizer that computes deterministic hashed sparse vectors
    using corpus statistics (DF, doc length) for high-contrast lexical retrieval in Qdrant.
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.df = {}
        self.initialized = False
        
    def fit(self, documents: List[Dict[str, Any]]):
        self.doc_count = len(documents)
        if self.doc_count == 0:
            self.avg_doc_len = 0.0
            self.df = {}
            self.initialized = True
            return
            
        total_len = 0
        self.df = {}
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('description', '')}"
            words = re.findall(r'\b\w+\b', text.lower())
            total_len += len(words)
            
            unique_terms = set(words)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1
                
        self.avg_doc_len = total_len / self.doc_count
        self.initialized = True

    def get_sparse_vector(self, text: str, is_query: bool = False) -> models.SparseVector:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return models.SparseVector(indices=[], values=[])
            
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0.0) + 1.0
            
        indices = []
        values = []
        doc_len = len(words)
        
        for term, count in tf.items():
            # Deterministic 32-bit hash index
            hash_val = int(hashlib.md5(term.encode('utf-8')).hexdigest()[:8], 16) % (2**32 - 1)
            indices.append(hash_val)
            
            if is_query or not self.initialized or self.doc_count == 0:
                # Queries get plain Term Frequency weights
                values.append(count)
            else:
                # Indexed documents get BM25 tf-idf normalized weights
                term_df = self.df.get(term, 0)
                # Smoothed IDF
                idf = math.log(1.0 + (self.doc_count - term_df + 0.5) / (term_df + 0.5))
                # TF Normalization
                tf_norm = (count * (self.k1 + 1)) / (count + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))))
                bm25_weight = idf * tf_norm
                values.append(max(bm25_weight, 0.0))
                
        sorted_pairs = sorted(zip(indices, values))
        sorted_indices = [p[0] for p in sorted_pairs]
        sorted_values = [p[1] for p in sorted_pairs]
        
        return models.SparseVector(indices=sorted_indices, values=sorted_values)

def get_bm25_vectorizer() -> BM25Vectorizer:
    """
    Fits and returns a BM25Vectorizer based on all currently indexed ideas.
    """
    vectorizer = BM25Vectorizer()
    ideas = get_all_ideas()
    vectorizer.fit(ideas)
    return vectorizer

def init_db():
    """
    Initializes the Qdrant database. Creates the collection with named dense
    vectors for Title and Description, plus a sparse vector for lexical search.
    """
    client = get_client()
    
    # 1. Initialize Parent Collection
    collections = client.get_collections().collections
    exists_parent = any(c.name == COLLECTION_NAME for c in collections)
    
    if exists_parent:
        try:
            info = client.get_collection(COLLECTION_NAME)
            if "title" not in info.config.params.vectors or "description" in info.config.params.vectors:
                print(f"Parent collection configuration mismatch (needs title & lexical only). Recreating...")
                client.delete_collection(COLLECTION_NAME)
                exists_parent = False
        except Exception as e:
            print(f"Could not verify parent collection config: {e}. Recreating...")
            client.delete_collection(COLLECTION_NAME)
            exists_parent = False
            
    if not exists_parent:
        print(f"Creating parent collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "title": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "lexical": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False
                    )
                )
            }
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tags",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="topics",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("Parent collection and indexes created successfully.")
        
    # 2. Initialize Chunks Collection
    exists_chunks = any(c.name == CHUNKS_COLLECTION_NAME for c in collections)
    if exists_chunks:
        try:
            info = client.get_collection(CHUNKS_COLLECTION_NAME)
            existing_size = info.config.params.vectors["description"].size
            if existing_size != VECTOR_SIZE:
                print(f"Chunks collection size mismatch (existing: {existing_size}, required: {VECTOR_SIZE}). Recreating...")
                client.delete_collection(CHUNKS_COLLECTION_NAME)
                exists_chunks = False
        except Exception as e:
            print(f"Could not verify chunks collection size: {e}. Recreating...")
            client.delete_collection(CHUNKS_COLLECTION_NAME)
            exists_chunks = False
            
    if not exists_chunks:
        print(f"Creating chunks collection '{CHUNKS_COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=CHUNKS_COLLECTION_NAME,
            vectors_config={
                "description": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            }
        )
        print("Chunks collection created successfully.")

def insert_idea(
    idea_id: int,
    title: str,
    description: str,
    summary: str,
    topics: List[str],
    tags: List[str],
    title_vector: List[float],
    description_vector: List[float],
    vertical_domains: List[str] = None,
    horizontal_technologies: List[str] = None
):
    """
    Stores an idea: parent payload in 'ideas' collection, and description chunks in 'idea_chunks'.
    """
    client = get_client()
    from services import get_embeddings
    
    # Ensure they are lists
    v_domains = vertical_domains or []
    h_techs = horizontal_technologies or []
    
    # 1. Combine title and description for lexical token indexing in parent collection
    lexical_text = f"{title} {description}"
    vectorizer = get_bm25_vectorizer()
    sparse_vector = vectorizer.get_sparse_vector(lexical_text, is_query=False)
    
    # 2. Insert parent document in COLLECTION_NAME
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=idea_id,
                vector={
                    "title": title_vector,
                    "lexical": sparse_vector
                },
                payload={
                    "title": title,
                    "description": description,
                    "summary": summary,
                    "topics": topics,
                    "tags": tags,
                    "vertical_domains": v_domains,
                    "horizontal_technologies": h_techs
                }
            )
        ]
    )
    
    # 3. Clean up any previous chunks for this parent_id to avoid stale data
    client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="parent_id",
                        match=models.MatchValue(value=idea_id)
                    )
                ]
            )
        )
    )
    
    # 4. Chunk the description and insert chunks
    THRESHOLD = 1000
    if len(description) > THRESHOLD:
        # Split into multiple semantic chunks
        chunks = chunk_text(description, max_chars=800, overlap=150)
    else:
        # Keep as single chunk
        chunks = [description]
        
    chunk_points = []
    for idx, chunk_str in enumerate(chunks):
        # Generate chunk embedding vector dynamically (gemini-embedding-2)
        # If it's a single chunk and description <= 1000, we reuse the pre-computed description_vector
        if len(chunks) == 1:
            chunk_vec = description_vector
        else:
            chunk_vec = get_embeddings(chunk_str)
            
        # Generate deterministic UUID for child point
        chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"idea_{idea_id}_chunk_{idx}"))
        
        chunk_points.append(
            models.PointStruct(
                id=chunk_uuid,
                vector={
                    "description": chunk_vec
                },
                payload={
                    "parent_id": idea_id,
                    "parent_title": title,
                    "parent_description": description,
                    "parent_summary": summary,
                    "parent_topics": topics,
                    "parent_tags": tags,
                    "parent_vertical_domains": v_domains,
                    "parent_horizontal_technologies": h_techs,
                    "chunk_text": chunk_str,
                    "chunk_index": idx
                }
            )
        )
        
    client.upsert(
        collection_name=CHUNKS_COLLECTION_NAME,
        points=chunk_points
    )

def search_ideas(
    title_vector: List[float],
    description_vector: List[float],
    query_text: str,
    top_k: int = 15
) -> tuple:
    """
    Performs hybrid search by executing:
      1. Dense search on Title vector (against COLLECTION_NAME)
      2. Dense search on Description vector (against CHUNKS_COLLECTION_NAME with Parent grouping)
      3. Sparse search on Query text (against COLLECTION_NAME)
    Returns lists of results for each search type.
    """
    client = get_client()
    
    # 1. Dense Title Search
    title_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=title_vector,
        using="title",
        limit=top_k,
        with_payload=True
    ).points
    
    # 2. Dense Description Search (against child chunks collection, query_limit expanded to allow grouping)
    chunk_results = client.query_points(
        collection_name=CHUNKS_COLLECTION_NAME,
        query=description_vector,
        using="description",
        limit=top_k * 3,
        with_payload=True
    ).points
    
    # Group chunk results by parent_id, keeping the maximum score (best match passage)
    grouped_hits = {}
    for hit in chunk_results:
        parent_id = hit.payload.get("parent_id")
        if parent_id is None:
            continue
        if parent_id not in grouped_hits or hit.score > grouped_hits[parent_id].score:
            grouped_hits[parent_id] = hit
            
    # Sort grouped parent hits by score descending and truncate to top_k
    sorted_grouped_hits = sorted(grouped_hits.values(), key=lambda h: h.score, reverse=True)[:top_k]
    
    # 3. Sparse Lexical Search
    vectorizer = get_bm25_vectorizer()
    sparse_vector = vectorizer.get_sparse_vector(query_text, is_query=True)
    sparse_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vector,
        using="lexical",
        limit=top_k,
        with_payload=True
    ).points
    
    # Map results to unified dictionaries
    def format_parent_hit(hit):
        return {
            "id": hit.id,
            "score": hit.score,
            "title": hit.payload.get("title"),
            "description": hit.payload.get("description"),
            "summary": hit.payload.get("summary"),
            "topics": hit.payload.get("topics", []),
            "tags": hit.payload.get("tags", []),
            "vertical_domains": hit.payload.get("vertical_domains", []),
            "horizontal_technologies": hit.payload.get("horizontal_technologies", [])
        }
        
    def format_chunk_hit(hit):
        return {
            "id": hit.payload.get("parent_id"),
            "score": hit.score,
            "title": hit.payload.get("parent_title"),
            "description": hit.payload.get("parent_description"),
            "summary": hit.payload.get("parent_summary"),
            "topics": hit.payload.get("parent_topics", []),
            "tags": hit.payload.get("parent_tags", []),
            "vertical_domains": hit.payload.get("parent_vertical_domains", []),
            "horizontal_technologies": hit.payload.get("parent_horizontal_technologies", [])
        }
        
    return (
        [format_parent_hit(h) for h in title_results],
        [format_chunk_hit(h) for h in sorted_grouped_hits],
        [format_parent_hit(h) for h in sparse_results]
    )

def get_all_ideas() -> List[Dict[str, Any]]:
    """
    Retrieves all ideas stored in Qdrant.
    """
    client = get_client()
    
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=False
    )[0]
    
    return [
        {
            "id": hit.id,
            "title": hit.payload.get("title"),
            "description": hit.payload.get("description"),
            "summary": hit.payload.get("summary"),
            "topics": hit.payload.get("topics", []),
            "tags": hit.payload.get("tags", []),
            "vertical_domains": hit.payload.get("vertical_domains", []),
            "horizontal_technologies": hit.payload.get("horizontal_technologies", [])
        }
        for hit in results
    ]
