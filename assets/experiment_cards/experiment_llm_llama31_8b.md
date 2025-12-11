# Experiment Card: LLM Baseline - llama31_8b

## Aim:
Evaluate the performance of llama31_8b language model for food & restaurant question answering.

## Date:
2025-11-02

## Author:
Experiment Team

## Designer:
Automated baseline evaluation script

## Setup:
- Model: llama31_8b
- System Prompt: Food & restaurant assistant with concise answer guidelines

- Infrastructure: Local Ollama instance (RTX 2070 Max-Q compatible)
- Context Window: 2048 tokens
- Temperature: 0.3
- Repeat Penalty: 1.1

## Design:
- Evaluation method: Question-answering on test set
- Number of questions: 10
- Metrics: Latency (seconds), Answer quality (word count)
- Intent-aware prompting: No

## Summary of Results:
- Total questions evaluated: 10
- Average latency: 1.852 seconds
- Median latency: 1.673 seconds
- Average answer length: 43.8 words
- Errors encountered: 0

## Discussion and Conclusion:
The llama31_8b model demonstrates acceptable performance for food & restaurant question answering.
Average response time of 1.852s is suitable for interactive applications.
Further evaluation may include human assessment of answer quality and relevance.
