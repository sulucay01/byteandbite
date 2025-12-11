# Experiment Card: BERT Intent Classification Training

## Aim:
Train a BERT-based intent classification model to categorize user queries for food & restaurant chatbot.

## Date:
2025-11-02

## Author:
Experiment Team

## Designer:
Automated BERT training script

## Setup:
- Base Model: distilbert-base-uncased
- Task: Multi-class intent classification
- Number of intent classes: 12
- Training Framework: Hugging Face Transformers
- Training Strategy: Stratified train-test split (80/20)
- Early Stopping: Enabled (patience=3)
- Optimizer: AdamW with learning rate 5e-5
- Weight Decay: 0.01

## Design:
- Dataset: Intent patterns from intents.json
- Training Strategy: Supervised learning with early stopping
- Evaluation Strategy: Stratified split, epoch-based evaluation
- Best Model Selection: Based on F1-macro score
- Max Sequence Length: 64 tokens

## Summary of Results:
Metrics not available (model may not be trained yet)

## Discussion and Conclusion:
BERT-based intent classification provides a lightweight alternative to LLM-based approaches.
The model can be used for intent detection before routing queries to specialized LLM prompts.
Further evaluation should include real-world deployment testing and comparison with LLM-only approaches.
