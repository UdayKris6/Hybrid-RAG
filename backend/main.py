import random
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from db import init_db, insert_idea, search_ideas, get_all_ideas
from services import (
    get_embeddings,
    extract_metadata,
    reciprocal_rank_fusion,
    reranker_service,
    request_warnings
)

# Pydantic schemas for request validation
class IdeaCreate(BaseModel):
    title: str = Field(..., max_length=100, description="Title of the idea, max 100 characters")
    description: str = Field(..., max_length=10000, description="Detailed description of the idea, max 10,000 characters")

class IdeaCheck(BaseModel):
    title: str = Field(..., max_length=100)
    description: str = Field(..., max_length=10000)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler that initializes the Qdrant local collection on server startup.
    """
    init_db()
    # Pre-load the Cross-Encoder model in the background so the first request doesn't lag
    try:
        reranker_service.load_model()
    except Exception as e:
        print(f"Non-critical: Could not pre-load CrossEncoder reranker model: {e}")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid RAG Duplicate Idea Search Engine",
    description="Backend microservice using Qdrant vector database, OpenAI embeddings, and local Cross-Encoder reranker",
    lifespan=lifespan
)

# Enable CORS for Angular frontend integration (running on http://localhost:4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/ideas")
def list_ideas():
    """
    Retrieves all ideas currently stored in the Qdrant local collection.
    """
    try:
        return get_all_ideas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ideas")
def create_idea(idea: IdeaCreate):
    """
    Saves a new idea:
    1. Extracts summary, categories, and tags using OpenAI LLM.
    2. Computes text embeddings for Title and Description.
    3. Saves all data, vectors, and sparse index in Qdrant.
    """
    warnings_list = []
    token = request_warnings.set(warnings_list)
    try:
        # Step 1: Call LLM to extract metadata tags and summary
        metadata = extract_metadata(idea.title, idea.description)
        
        # Step 2: Compute dense vector embeddings
        title_vector = get_embeddings(idea.title)
        description_vector = get_embeddings(idea.description)
        
        # Generate a unique positive integer ID
        idea_id = random.randint(1, 2**31 - 1)
        
        # Step 3: Upsert into Qdrant Local database
        insert_idea(
            idea_id=idea_id,
            title=idea.title,
            description=idea.description,
            summary=metadata.get("summary", ""),
            topics=metadata.get("topics", []),
            tags=metadata.get("tags", []),
            title_vector=title_vector,
            description_vector=description_vector
        )
        
        return {
            "id": idea_id,
            "title": idea.title,
            "description": idea.description,
            "summary": metadata.get("summary", ""),
            "topics": metadata.get("topics", []),
            "tags": metadata.get("tags", []),
            "status": "success",
            "warnings": list(warnings_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        request_warnings.reset(token)

@app.post("/api/ideas/check")
def check_duplicate_idea(query: IdeaCheck):
    """
    Checks if a draft idea title and description matches any existing ideas:
    1. Generates title and description embeddings for the query.
    2. Performs multi-vector dense search & sparse search in Qdrant.
    3. Merges rankings using Reciprocal Rank Fusion (RRF).
    4. Reranks the top candidates using a local Cross-Encoder model.
    5. Returns ranked matches with similarity certainty levels.
    """
    warnings_list = []
    token = request_warnings.set(warnings_list)
    try:
        # 1. Embed query title and description
        query_title_vector = get_embeddings(query.title)
        query_desc_vector = get_embeddings(query.description)
        
        # 2. Retrieve initial candidates using dense (vector) and sparse (lexical) search
        title_hits, desc_hits, sparse_hits = search_ideas(
            title_vector=query_title_vector,
            description_vector=query_desc_vector,
            query_text=f"{query.title} {query.description}",
            top_k=15
        )
        
        # If no ideas exist yet, return empty list
        if not title_hits and not desc_hits and not sparse_hits:
            return {
                "is_duplicate": False,
                "matches": [],
                "warnings": list(warnings_list)
            }
            
        # 3. Reciprocal Rank Fusion (RRF) to merge and balance rankings
        rrf_merged = reciprocal_rank_fusion(title_hits, desc_hits, sparse_hits, k=60)
        
        # Take the top 15 results from RRF to send to the Cross-Encoder Reranker
        top_candidates = rrf_merged[:15]
        
        # 4. Rerank candidates using our local Cross-Encoder model
        query_text = f"Title: {query.title}. Description: {query.description}"
        final_matches = reranker_service.rerank(query_text, top_candidates, top_n=5)
        
        # 5. Determine warning flags
        # If the highest score is > 0.82, we flag it as a highly probable duplicate
        highest_score = final_matches[0]["similarity_score"] if final_matches else 0.0
        is_duplicate = highest_score >= 0.82
        
        return {
            "is_duplicate": is_duplicate,
            "max_similarity_score": highest_score,
            "matches": final_matches,
            "warnings": list(warnings_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        request_warnings.reset(token)

if __name__ == "__main__":
    import uvicorn
    # Start the local development server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
