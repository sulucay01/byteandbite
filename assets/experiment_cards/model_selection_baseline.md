# Model Selection Card: Baseline LLM Selection

## Decision Summary
**Selected Model:** llama3.1:8b-instruct-q4_K_M (llama31_8b)  
**Alternative Considered:** mistral:7b-instruct-q4_K_M (mistral7b)  
**Decision Date:** 2025-11-02  
**Decision Status:** ✅ Selected for baseline implementation

## Evaluation Context

### Models Evaluated
1. **llama3.1:8b-instruct-q4_K_M** (llama31_8b)
   - Quantization: q4_K_M (4-bit)
   - Provider: Meta AI
   - Access: Local Ollama instance

2. **mistral:7b-instruct-q4_K_M** (mistral7b)
   - Quantization: q4_K_M (4-bit)
   - Provider: Mistral AI
   - Access: Local Ollama instance

### Evaluation Setup
- **Infrastructure:** Local Ollama instance (RTX 2070 Max-Q compatible)
- **Test Set:** 10 food & restaurant questions
- **Metrics:** Latency (seconds), Answer quality (word count)
- **Context Window:** 2048 tokens
- **Temperature:** 0.3
- **Repeat Penalty:** 1.1
- **Evaluation Method:** Question-answering on test set

## Performance Comparison

### Key Metrics

| Metric | llama31_8b | mistral7b | Winner |
|--------|------------|-----------|--------|
| **Average Latency** | 1.852s | 3.502s | llama31_8b |
| **Median Latency** | 1.673s | 2.745s | llama31_8b |
| **Average Answer Length** | 43.8 words | 83.7 words | llama31_8b |
| **Latency Range** | 1.246s - 4.067s | 1.235s - 7.072s | llama31_8b |
| **Latency Std Dev** | 0.813s | 2.061s | llama31_8b |

### Performance Improvements (llama31_8b vs mistral7b)
- **47.1% faster** average latency
- **39.0% faster** median latency
- **47.7% more concise** answers (shorter response length)
- **1.89x overall speedup** factor
- **Better consistency** (lower standard deviation: 0.813s vs 2.061s)

## Decision Rationale

### Primary Reasons for Selection

1. **Superior Latency Performance**
   - llama31_8b is 47.1% faster on average (1.852s vs 3.502s)
   - Critical for interactive applications where user experience depends on response time
   - Median latency of 1.673s is well within acceptable range for real-time chat applications

2. **More Concise Responses**
   - Average response length of 43.8 words vs 83.7 words
   - Better alignment with system prompt requirements (1-3 concise sentences)
   - Reduces token usage and improves readability for users

3. **Better Consistency**
   - Lower standard deviation (0.813s vs 2.061s) indicates more predictable performance
   - Narrower latency range (1.246s - 4.067s vs 1.235s - 7.072s) provides better user experience
   - More reliable for production deployment

4. **Resource Efficiency**
   - Faster responses reduce server load and improve throughput
   - Lower latency enables better scalability for concurrent users

### Trade-offs Considered

**Advantages of mistral7b:**
- Slightly longer, more detailed responses (may be preferred for some use cases)
- Similar minimum latency (1.235s vs 1.246s)

**Why these trade-offs favor llama31_8b:**
- Our use case prioritizes quick, concise answers over detailed explanations
- The system prompt explicitly requests 1-3 concise sentences
- Interactive applications benefit more from speed than verbosity
- The consistency advantage outweighs the minor minimum latency difference

## Conclusion

The llama31_8b model was selected as the baseline LLM for the following reasons:
- **47.1% faster response times** improve user experience in interactive applications
- **47.7% more concise answers** better match the system requirements
- **Better consistency** provides more predictable performance for production use
- **Overall 1.89x speedup** enables better scalability

This selection aligns with the project's goals of providing fast, concise, and reliable responses for food & restaurant question answering in an interactive chatbot environment.

## Next Steps
- Proceed with llama31_8b as the default generation model
- Continue monitoring performance in production
- Consider re-evaluation if requirements change (e.g., need for more detailed responses)

