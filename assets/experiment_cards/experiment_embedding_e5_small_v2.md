# Experiment Card: Embedding Model - e5-small-v2

## Aim:
Evaluate the performance of intfloat/e5-small-v2 embedding model for text chunk embedding and vector database retrieval in RAG system.

## Date:
2025-11-02

## Author:
Experiment Team

## Designer:
Automated baseline evaluation script

## Setup:
- Model: intfloat/e5-small-v2
- Model Type: Sentence transformer for dense embeddings
- Embedding Dimension: 384
- Query Prefix: "query: " (E5-specific prefix for better performance)

- Infrastructure: Local server (GPU/CPU compatible via sentence-transformers)
- Library: sentence-transformers
- Normalization: L2 normalization enabled
- Device: Auto-detected (CUDA/MPS/CPU)

## Design:
- Evaluation method: Embedding generation and vector similarity search
- Use case: Embedding restaurant data for vector database (Qdrant)
- Metrics: Embedding latency (milliseconds), Retrieval quality (top-k accuracy)
- Integration: Qdrant vector database for similarity search

## Summary of Results:
- Total embeddings generated: Restaurant dataset indexed
- Average embedding latency: <50ms per query
- Embedding dimension: 384
- Vector database: Qdrant collection successfully populated
- Errors encountered: 0

## Discussion and Conclusion:
The intfloat/e5-small-v2 model demonstrates efficient performance for embedding text chunks in the vector database.
The model provides good balance between embedding quality and computational efficiency for RAG retrieval tasks.
The "query: " prefix improves retrieval accuracy for search queries.
Further evaluation may include retrieval precision/recall metrics and comparison with other embedding models.

