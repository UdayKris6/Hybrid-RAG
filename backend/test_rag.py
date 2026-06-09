import os
from dotenv import load_dotenv

# Ensure mock/local environments can resolve imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import init_db, insert_idea, get_all_ideas
from main import check_duplicate_idea, IdeaCheck
from services import get_embeddings, extract_metadata

# Sample seed data representing existing project submissions
BASELINE_IDEAS = [
    {
        "title": "Smart AI-powered Grocery List Planner",
        "description": "An intelligent mobile application that analyzes your cooking habits and automatically generates a grocery list. It suggests healthy recipes, predicts when you are running low on pantry essentials, and tracks nutritional intake over time."
    },
    {
        "title": "Decentralized Voting System on Ethereum",
        "description": "A secure, transparent voting platform built on blockchain technology. Using Ethereum smart contracts, it guarantees anonymous voting, immediate auditability, and absolute immunity against voter fraud or database tampering."
    },
    {
        "title": "IoT Home Energy Monitor and Optimizer",
        "description": "A smart home hardware and software solution that attaches to your electrical panel. It uses machine learning to identify individual appliances by their electrical signature, monitors real-time usage, and suggests energy-saving schedules."
    }
]

def run_test():
    load_dotenv()
    print("Initializing test database...")
    init_db()
    
    # Check if we already have data in Qdrant, if not insert base ideas
    existing = get_all_ideas()
    if len(existing) == 0:
        print("Seeding baseline ideas into Qdrant...")
        import time
        for i, item in enumerate(BASELINE_IDEAS):
            # Generate vectors
            title_vec = get_embeddings(item["title"])
            desc_vec = get_embeddings(item["description"])
            
            # Extract metadata tags
            meta = extract_metadata(item["title"], item["description"])
            
            insert_idea(
                idea_id=i + 1,
                title=item["title"],
                description=item["description"],
                summary=meta.get("summary", ""),
                topics=meta.get("topics", []),
                tags=meta.get("tags", []),
                title_vector=title_vec,
                description_vector=desc_vec
            )
            print(f"Seeded idea #{i+1}: '{item['title']}'")
            # Sleep to prevent hitting free-tier concurrent rate limits (503)
            time.sleep(2)
        print("Database seeded successfully!")
    else:
        print(f"Database already contains {len(existing)} ideas. Skipping seeding.")

    # Define test queries to evaluate similarity check
    test_queries = [
        # Query 1: Conceptually identical to Grocery Planner, different words
        {
            "title": "Intelligent Shopping Helper for Food",
            "description": "A cell phone app that uses artificial intelligence to help people buy their groceries. It learns what meals you cook, makes grocery lists for you automatically, and suggests recipes to stay healthy."
        },
        # Query 2: Conceptually identical to Ethereum Voting, different words
        {
            "title": "Secure Blockchain Ballot Box",
            "description": "An online system where citizens can vote using cryptocurrency networks. It runs on decentralized smart contracts to keep votes secure, anonymous, and impossible to fake."
        },
        # Query 3: Random irrelevant idea (should return low similarity scores)
        {
            "title": "AI Coding Assistant for Angular developers",
            "description": "A VSCode extension that generates boilerplate code for Angular components, writes tests, and cleans up CSS stylesheets."
        }
    ]

    print("\n" + "="*80)
    print("RUNNING HYBRID RAG SIMILARITY SEARCH TESTS")
    print("="*80)

    for idx, query_data in enumerate(test_queries, 1):
        print(f"\n[Test Query #{idx}]")
        print(f"Draft Title: {query_data['title']}")
        print(f"Draft Description: {query_data['description'][:100]}...")
        
        # Run the backend checking logic directly
        query_model = IdeaCheck(title=query_data["title"], description=query_data["description"])
        result = check_duplicate_idea(query_model)
        
        print(f"Is Duplicate? -> {result['is_duplicate']} (Max Score: {result['max_similarity_score']})")
        print("Matches returned:")
        for rank, match in enumerate(result["matches"], 1):
            print(f"  {rank}. Title: '{match['title']}'")
            print(f"     -> Similarity Score (Cross-Encoder): {match['similarity_score']}")
            print(f"     -> RRF Score: {match['rrf_score']}")
            print(f"     -> Extracted Tags: {match['tags']}")
            print(f"     -> AI Summary: {match['summary']}")

if __name__ == "__main__":
    run_test()
