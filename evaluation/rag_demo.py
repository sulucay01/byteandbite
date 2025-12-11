#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Demo for TripAdvisor Restaurant Data

This script demonstrates Retrieval-Augmented Generation (RAG) using a CSV file
containing restaurant data scraped from TripAdvisor.

Requirements:
  pip install pandas requests sentence-transformers

Usage:
  python rag_demo.py --csv path/to/restaurants.csv --question "Find Italian restaurants"
"""

import argparse
import pathlib
import time
from typing import List, Dict, Any

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import numpy as np


# Ollama configuration
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

# System prompt for the LLM
SYSTEM_PROMPT = (
    "You are a helpful assistant for food & restaurants.\n"
    "Answer in English in 1–3 concise sentences unless asked otherwise.\n"
    "Start answer with Answer:.\n"
    "Use the provided restaurant information to give specific, accurate recommendations.\n"
    "If the provided information doesn't match the query, say you don't have relevant information.\n"
)


class RestaurantRAG:
    """Simple RAG system for restaurant recommendations."""
    
    def __init__(self, csv_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system with restaurant data.
        
        Args:
            csv_path: Path to CSV file with restaurant data
            embedding_model: Name of sentence-transformer model for embeddings
        """
        print(f"Loading restaurant data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} restaurants")
        
        # Display available columns
        print(f"Available columns: {', '.join(self.df.columns.tolist())}")
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
        # Create text representations for each restaurant
        print("Creating embeddings...")
        self.restaurant_texts = self._create_restaurant_texts()
        self.embeddings = self.encoder.encode(self.restaurant_texts, show_progress_bar=True)
        print("Embeddings created!")
    
    def _create_restaurant_texts(self) -> List[str]:
        """Convert restaurant data into searchable text strings."""
        texts = []
        for _, row in self.df.iterrows():
            # Combine relevant columns into a searchable text
            parts = []
            
            # Common column names to look for
            if 'name' in self.df.columns:
                parts.append(f"Restaurant: {row['name']}")
            if 'restaurant_name' in self.df.columns:
                parts.append(f"Restaurant: {row['restaurant_name']}")
            if 'cuisine' in self.df.columns:
                parts.append(f"Cuisine: {row['cuisine']}")
            if 'cuisine_type' in self.df.columns:
                parts.append(f"Cuisine: {row['cuisine_type']}")
            if 'price_range' in self.df.columns:
                parts.append(f"Price: {row['price_range']}")
            if 'rating' in self.df.columns:
                parts.append(f"Rating: {row['rating']}")
            if 'review' in self.df.columns:
                parts.append(f"Review: {row['review']}")
            if 'description' in self.df.columns:
                parts.append(f"Description: {row['description']}")
            if 'location' in self.df.columns:
                parts.append(f"Location: {row['location']}")
            
            # Add all other columns as fallback
            for col in self.df.columns:
                if col not in ['name', 'restaurant_name', 'cuisine', 'cuisine_type', 
                              'price_range', 'rating', 'review', 'description', 'location']:
                    if pd.notna(row[col]):
                        parts.append(f"{col}: {row[col]}")
            
            text = ". ".join(parts)
            texts.append(text)
        
        return texts
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant restaurants for a query.
        
        Args:
            query: User's question/query
            top_k: Number of restaurants to retrieve
            
        Returns:
            List of dictionaries containing restaurant info and relevance score
        """
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Retrieve restaurants
        results = []
        for idx in top_indices:
            restaurant_data = self.df.iloc[idx].to_dict()
            restaurant_data['relevance_score'] = float(similarities[idx])
            restaurant_data['retrieved_text'] = self.restaurant_texts[idx]
            results.append(restaurant_data)
        
        return results
    
    def format_context(self, retrieved: List[Dict[str, Any]]) -> str:
        """Format retrieved restaurants into context for the LLM."""
        context_parts = ["Retrieved restaurant information:"]
        for i, restaurant in enumerate(retrieved, 1):
            context_parts.append(f"\n{i}. {restaurant['retrieved_text']}")
        return "\n".join(context_parts)


def chat_ollama(model: str, system: str, user: str, host: str = OLLAMA_HOST, port: int = OLLAMA_PORT) -> str:
    """Send a chat request to Ollama."""
    url = f"http://{host}:{port}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "options": {
            "num_ctx": 4096,
            "temperature": 0.3,
            "repeat_penalty": 1.1
        },
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="RAG Demo for TripAdvisor Restaurant Data")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with restaurant data")
    parser.add_argument("--question", type=str, default=None, help="Question to ask (interactive if not provided)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--top-k", type=int, default=3, help="Number of restaurants to retrieve")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", 
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--host", type=str, default=OLLAMA_HOST, help="Ollama host")
    parser.add_argument("--port", type=int, default=OLLAMA_PORT, help="Ollama port")
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not pathlib.Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    # Initialize RAG system
    rag = RestaurantRAG(args.csv, embedding_model=args.embedding_model)
    
    # Interactive or single question mode
    if args.question:
        questions = [args.question]
    else:
        print("\n" + "="*60)
        print("RAG Demo - Restaurant Recommendations")
        print("="*60)
        print("Enter questions about restaurants. Type 'quit' to exit.\n")
        questions = []
        while True:
            q = input("Question: ").strip()
            if q.lower() in ['quit', 'exit', 'q']:
                break
            if q:
                questions.append(q)
    
    # Process questions
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Retrieve relevant restaurants
        print(f"\nRetrieving top {args.top_k} restaurants...")
        retrieved = rag.retrieve(question, top_k=args.top_k)
        
        # Display retrieved restaurants
        print("\nRetrieved restaurants:")
        for i, restaurant in enumerate(retrieved, 1):
            print(f"\n{i}. Relevance: {restaurant['relevance_score']:.3f}")
            print(f"   {restaurant['retrieved_text'][:200]}...")
        
        # Format context
        context = rag.format_context(retrieved)
        
        # Build user message with context
        user_message = f"{context}\n\nQuestion: {question}\n\nAnswer based on the restaurant information provided above."
        
        # Get response from LLM
        print(f"\nGenerating response with {args.model}...")
        t0 = time.time()
        try:
            response = chat_ollama(args.model, SYSTEM_PROMPT, user_message, args.host, args.port)
            dt = time.time() - t0
            print(f"\nAnswer (took {dt:.2f}s):")
            print(f"{response}\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()


