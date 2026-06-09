import os
import json
import contextvars
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

# Contextvar to store warnings for the current request context
request_warnings = contextvars.ContextVar('request_warnings', default=None)

# Try to import Google GenAI SDK gracefully
try:
    from google import genai
    from google.genai import types
    is_gemini_sdk_available = True
except ImportError:
    is_gemini_sdk_available = False

load_dotenv()

# Active AI provider (GEMINI or OPENAI)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "GEMINI").upper()

# API Keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configuration logs
print(f"Active AI Provider: {EMBEDDING_PROVIDER}")

# 1. Initialize Gemini Client
is_gemini_configured = False
gemini_client = None
if EMBEDDING_PROVIDER == "GEMINI":
    if not is_gemini_sdk_available:
        print("\n" + "!"*80)
        print("WARNING: google-genai SDK is not installed.")
        print("To use Gemini, please run: pip install google-genai")
        print("Falling back to local MOCK embeddings and tag extraction.")
        print("!"*80 + "\n")
    elif not gemini_api_key or gemini_api_key.startswith("your_gemini"):
        print("\n" + "!"*80)
        print("WARNING: GEMINI_API_KEY is not set or is using the placeholder.")
        print("Please configure GEMINI_API_KEY in the backend/.env file.")
        print("Falling back to local MOCK embeddings and tag extraction.")
        print("!"*80 + "\n")
    else:
        try:
            gemini_client = genai.Client(api_key=gemini_api_key)
            is_gemini_configured = True
            print("Gemini API Client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")

# 2. Initialize OpenAI Client
is_openai_configured = False
openai_client = None
if EMBEDDING_PROVIDER == "OPENAI":
    if not openai_api_key or openai_api_key.startswith("your_openai"):
        print("\n" + "!"*80)
        print("WARNING: OPENAI_API_KEY is not set or is using the placeholder.")
        print("Please configure OPENAI_API_KEY in the backend/.env file.")
        print("Falling back to local MOCK embeddings and tag extraction.")
        print("!"*80 + "\n")
    else:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            is_openai_configured = True
            print("OpenAI API Client initialized successfully.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")

class RerankerService:
    """
    Service for reranking candidate documents using a local Cross-Encoder model.
    Loads the model lazily on the first request to speed up server startup time.
    """
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            # We use 'ms-marco-MiniLM-L-6-v2', a highly optimized and lightweight reranker (~80MB).
            # It evaluates query-document pairs on CPU in milliseconds.
            print("Loading local CrossEncoder reranking model ('ms-marco-MiniLM-L-6-v2')...")
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("CrossEncoder model loaded successfully.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        self.load_model()
        
        # Format the query and candidate idea for Cross-Encoder comparison
        pairs = []
        for doc in candidates:
            doc_text = f"Title: {doc['title']}. Description: {doc['description']}"
            pairs.append([query, doc_text])
            
        # Predict the similarity scores (raw logits)
        scores = self.model.predict(pairs)
        
        # Add normal probability scores to each candidate using a sigmoid activation
        for doc, score in zip(candidates, scores):
            sigmoid_score = 1.0 / (1.0 + np.exp(-score))
            doc["similarity_score"] = round(float(sigmoid_score), 4)

        # Sort candidates by similarity score in descending order
        reranked = sorted(candidates, key=lambda d: d["similarity_score"], reverse=True)
        return reranked[:top_n]

# Global instance of RerankerService
reranker_service = RerankerService()

def get_embeddings(text: str) -> List[float]:
    """
    Generates a dense vector embedding using the active provider:
    - Gemini: gemini-embedding-2 (3072 dimensions)
    - OpenAI: text-embedding-3-small (1536 dimensions)
    Falls back to a deterministic normalized mock vector if active provider is unconfigured.
    """
    if EMBEDDING_PROVIDER == "GEMINI" and is_gemini_configured and gemini_client is not None:
        try:
            response = gemini_client.models.embed_content(
                model="gemini-embedding-2",
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            err_str = str(e)
            print(f"Error generating Gemini embeddings: {err_str}. Falling back to mock...")
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                warns = request_warnings.get()
                if warns is not None:
                    warns.append({
                        "model": "gemini-embedding-2",
                        "type": "rate_limit",
                        "message": "Gemini Embedding API rate limit exceeded. Falling back to local mock embeddings."
                    })

    elif EMBEDDING_PROVIDER == "OPENAI" and is_openai_configured and openai_client is not None:
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embeddings: {e}. Falling back to mock...")

    # Fallback Mock Vector Generator
    vector_size = 3072 if EMBEDDING_PROVIDER == "GEMINI" else 1536
    rng = np.random.default_rng(hash(text) & 0xffffffff)
    vec = rng.standard_normal(vector_size)
    vec /= np.linalg.norm(vec)
    return vec.tolist()

def extract_metadata(title: str, description: str) -> Dict[str, Any]:
    """
    Extracts summary, categories, and tags using the active LLM provider (Gemini or OpenAI).
    Falls back to heuristic rules if no credentials are configured.
    """
    prompt = f"""
    You are an AI assistant analyzing a new project/product idea submission.
    Your task is to extract a summary, relevant topics/categories, and key tags.
    
    Idea Title: {title}
    Idea Description: {description}
    
    Respond STRICTLY in JSON format with the following keys:
    {{
        "summary": "a short 2-sentence summary of the core value proposition of the idea",
        "topics": ["1 to 3 general categories/domains, e.g., SaaS, HealthTech, AI, Blockchain, E-commerce"],
        "tags": ["3 to 6 specific keywords/tags related to the idea features or technology"]
    }}
    """

    if EMBEDDING_PROVIDER == "GEMINI" and is_gemini_configured and gemini_client is not None:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )
            return json.loads(response.text)
        except Exception as e:
            err_str = str(e)
            print(f"Error extracting Gemini metadata: {err_str}. Falling back to mock...")
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                warns = request_warnings.get()
                if warns is not None:
                    warns.append({
                        "model": "gemini-3.1-flash-lite",
                        "type": "rate_limit",
                        "message": "Gemini Generation API rate limit exceeded for gemini-3.1-flash-lite. Using local heuristic metadata fallback."
                    })

    elif EMBEDDING_PROVIDER == "OPENAI" and is_openai_configured and openai_client is not None:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error extracting OpenAI metadata: {e}. Falling back to mock...")

    # Heuristic Mock Metadata Fallback
    words = [w.capitalize() for w in title.split() if len(w) > 4][:4]
    summary_text = (description[:120] + "...") if len(description) > 120 else description
    return {
        "summary": f"A project exploring: {summary_text}",
        "topics": ["General Idea"] if not words else [words[0]],
        "tags": ["Idea"] + words
    }

def reciprocal_rank_fusion(
    title_matches: List[Dict[str, Any]], 
    desc_matches: List[Dict[str, Any]], 
    lexical_matches: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) combines multiple search rankings into a single sorted list.
    RRF score is calculated as: RRF(d) = sum(1 / (k + rank_in_list))
    k = 60 is standard and prevents high-ranked items from completely dominating the score.
    """
    scores = {}
    doc_lookup = {}
    
    rank_lists = [title_matches, desc_matches, lexical_matches]
    
    for r_list in rank_lists:
        for rank, doc in enumerate(r_list):
            doc_id = doc["id"]
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc
            
            # Add to RRF score
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
            
    # Sort documents by total RRF score in descending order
    sorted_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    merged_results = []
    for doc_id, rrf_score in sorted_ids:
        doc_copy = dict(doc_lookup[doc_id])
        doc_copy["rrf_score"] = round(rrf_score, 6)
        merged_results.append(doc_copy)
        
    return merged_results
