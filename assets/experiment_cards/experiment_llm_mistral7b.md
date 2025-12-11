# Experiment Card: LLM Baseline - mistral7b

## Aim:
Evaluate the performance of mistral7b language model for food & restaurant question answering.

## Date:
2025-11-02

## Author:
Experiment Team

## Designer:
Automated baseline evaluation script

## Setup:
- Model: mistral7b
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
- Average latency: 3.502 seconds
- Median latency: 2.745 seconds
- Average answer length: 83.7 words
- Errors encountered: 0

## Discussion and Conclusion:
The mistral7b model demonstrates acceptable performance for food & restaurant question answering.
Average response time of 3.502s is suitable for interactive applications.
Further evaluation may include human assessment of answer quality and relevance.
