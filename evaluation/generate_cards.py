#!/usr/bin/env python3
"""
Generate Experiment Cards and Model Cards from existing experiments and models.
"""

import json
import pandas as pd
import pathlib
from datetime import datetime
from pathlib import Path


def load_json_safe(path):
    """Safely load JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def analyze_llm_results(csv_path):
    """Analyze LLM baseline results from CSV."""
    try:
        df = pd.read_csv(csv_path)
        return {
            'num_questions': len(df),
            'avg_latency': df['latency_sec'].mean(),
            'median_latency': df['latency_sec'].median(),
            'avg_answer_words': df['answer_words'].mean(),
            'total_errors': len(df[df['answer'].str.contains('__ERROR__', na=False)]),
        }
    except Exception as e:
        return {'error': str(e)}


def generate_experiment_card_llm(output_dir, model_name, intent=None):
    """Generate experiment card for LLM baseline experiments."""
    csv_path = output_dir / f"{model_name}_llm_only_run.csv"
    if not csv_path.exists():
        return None
    
    results = analyze_llm_results(csv_path)
    df = pd.read_csv(csv_path)
    
    card = f"""# Experiment Card: LLM Baseline - {model_name}

## Aim:
Evaluate the performance of {model_name} language model for food & restaurant question answering{f' with {intent} intent' if intent else ''}.

## Date:
{datetime.now().strftime('%Y-%m-%d')}

## Author:
Experiment Team

## Designer:
Automated baseline evaluation script

## Setup:
- Model: {model_name}
- System Prompt: Food & restaurant assistant with concise answer guidelines
{f'- Intent: {intent}' if intent else ''}
- Infrastructure: Local Ollama instance (RTX 2070 Max-Q compatible)
- Context Window: 2048 tokens
- Temperature: 0.3
- Repeat Penalty: 1.1

## Design:
- Evaluation method: Question-answering on test set
- Number of questions: {results.get('num_questions', 'N/A')}
- Metrics: Latency (seconds), Answer quality (word count)
- Intent-aware prompting: {'Yes' if intent else 'No'}

## Summary of Results:
- Total questions evaluated: {results.get('num_questions', 'N/A')}
- Average latency: {results.get('avg_latency', 0):.3f} seconds
- Median latency: {results.get('median_latency', 0):.3f} seconds
- Average answer length: {results.get('avg_answer_words', 0):.1f} words
- Errors encountered: {results.get('total_errors', 0)}

## Discussion and Conclusion:
The {model_name} model demonstrates {'acceptable' if results.get('total_errors', 1) == 0 else 'some issues with'} performance for food & restaurant question answering.
Average response time of {results.get('avg_latency', 0):.3f}s is suitable for interactive applications.
Further evaluation may include human assessment of answer quality and relevance.
"""
    return card


def generate_experiment_card_bert(output_dir):
    """Generate experiment card for BERT intent classification training."""
    metrics_path = output_dir / "intent_bert" / "eval_metrics.csv"
    
    metrics_info = "Metrics not available (model may not be trained yet)"
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            metrics_info = f"""
- Test Accuracy: {metrics_df.get('eval_accuracy', [0])[0]:.4f}
- Test F1-Macro: {metrics_df.get('eval_f1_macro', [0])[0]:.4f}
- Training Loss: {metrics_df.get('train_loss', ['N/A'])[0]}
"""
        except:
            pass
    
    intents_path = Path("intents.json")
    num_intents = "Unknown"
    if intents_path.exists():
        try:
            data = load_json_safe(intents_path)
            if data and 'intents' in data:
                num_intents = len(data['intents'])
        except:
            pass
    
    card = f"""# Experiment Card: BERT Intent Classification Training

## Aim:
Train a BERT-based intent classification model to categorize user queries for food & restaurant chatbot.

## Date:
{datetime.now().strftime('%Y-%m-%d')}

## Author:
Experiment Team

## Designer:
Automated BERT training script

## Setup:
- Base Model: distilbert-base-uncased
- Task: Multi-class intent classification
- Number of intent classes: {num_intents}
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
{metrics_info}

## Discussion and Conclusion:
BERT-based intent classification provides a lightweight alternative to LLM-based approaches.
The model can be used for intent detection before routing queries to specialized LLM prompts.
Further evaluation should include real-world deployment testing and comparison with LLM-only approaches.
"""
    return card


def generate_model_card_bert(output_dir):
    """Generate model card for BERT intent classification model."""
    intents_path = Path("intents.json")
    intents_data = load_json_safe(intents_path)
    
    intent_classes = "Unknown"
    if intents_data and 'intents' in intents_data:
        intent_classes = ", ".join([it['tag'] for it in intents_data['intents'][:10]])
        if len(intents_data['intents']) > 10:
            intent_classes += f", ... (total: {len(intents_data['intents'])} classes)"
    
    metrics_path = output_dir / "intent_bert" / "eval_metrics.csv"
    performance_metrics = "N/A - Model metrics not available"
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            perf = {}
            if 'eval_accuracy' in metrics_df.columns:
                perf['accuracy'] = metrics_df['eval_accuracy'].iloc[0]
            if 'eval_f1_macro' in metrics_df.columns:
                perf['f1_macro'] = metrics_df['eval_f1_macro'].iloc[0]
            if perf:
                performance_metrics = f"""
- Accuracy: {perf.get('accuracy', 'N/A'):.4f}
- F1-Macro: {perf.get('f1_macro', 'N/A'):.4f}
"""
        except:
            pass
    
    card = f"""# Model Card: BERT Intent Classifier

## Model Details

### Person or organization developing model
Experiment Team

### Model date
{datetime.now().strftime('%Y-%m-%d')}

### Model version
1.0

### Model type
Sequence Classification (DistilBERT-based)

### Information about training algorithms, parameters, fairness constraints or other applied approaches, and features
- Architecture: DistilBERT (distilbert-base-uncased)
- Training Algorithm: Fine-tuning with AdamW optimizer
- Learning Rate: 5e-5
- Batch Size: 8
- Weight Decay: 0.01
- Max Sequence Length: 64 tokens
- Early Stopping: Enabled (patience=3)
- Features: Raw text input, tokenized with DistilBERT tokenizer

### Paper or other resource for more information
- DistilBERT: https://arxiv.org/abs/1910.01108
- Hugging Face Transformers: https://huggingface.co/docs/transformers

### License
Check base model license (DistilBERT)

### Where to send questions or comments about the model
Contact experiment team

## Intended Use

### Primary intended uses
- Intent classification for food & restaurant chatbot
- Preprocessing step to route queries to appropriate response handlers
- Integration with LLM-based question answering systems

### Primary intended users
- Chatbot developers
- Food & restaurant service providers
- NLP researchers

### Out-of-scope use cases
- General purpose intent classification beyond restaurant/food domain
- Direct question answering (requires downstream LLM)
- Multi-turn conversation understanding

## Factors

### Relevant factors
- Query language (English)
- Food & restaurant domain
- Intent type (e.g., menu inquiry, reservation, dietary restrictions)

### Evaluation factors
- Intent class distribution
- Query length and complexity
- Similar intent classes (potential confusion)

## Metrics

### Model performance measures
{performance_metrics}

### Decision thresholds
- Classification: Argmax over logits
- Confidence threshold: Not explicitly set (can be added for filtering low-confidence predictions)

### Variation approaches
- Stratified train-test split (80/20)
- Evaluation on held-out test set
- Early stopping to prevent overfitting

## Evaluation Data

### Datasets
- Test set: 20% stratified split from intents.json patterns
- Number of intent classes: {intents_data['intents'].__len__() if intents_data and 'intents' in intents_data else 'Unknown'}

### Motivation
Stratified split ensures balanced representation of all intent classes in evaluation.

### Preprocessing
- Tokenization with DistilBERT tokenizer
- Max sequence length: 64 tokens
- Padding and truncation applied

## Training Data

### Details
- Training set: 80% stratified split from intents.json patterns
- Intent classes: {intent_classes}
- Label encoding: Standard label encoder mapping classes to integers
- Preprocessing: Same as evaluation data (tokenization, padding, truncation)

## Quantitative Analyses

### Unitary results
See performance metrics above.

### Intersectional results
Not applicable for this intent classification task.

## Ethical Considerations
- Model is trained on domain-specific patterns for food & restaurant queries
- No sensitive demographic information is used
- Model should not be used to discriminate or bias responses based on user characteristics
- Intent classification is a preprocessing step; final responses are generated by LLM components

## Caveats and Recommendations
- Model accuracy depends on quality and coverage of training patterns in intents.json
- May struggle with out-of-vocabulary queries or novel phrasings
- Should be combined with LLM fallback for robust handling
- Regular retraining recommended as new intents or patterns are added
- Consider confidence threshold for filtering uncertain predictions
"""
    return card


def generate_model_card_llm(model_name, model_tag):
    """Generate model card for LLM models."""
    card = f"""# Model Card: {model_name}

## Model Details

### Person or organization developing model
- llama3.1:8b: Meta AI
- mistral:7b: Mistral AI

### Model date
- Model release dates vary (check original model documentation)

### Model version
- Quantization: q4_K_M (4-bit quantization, K-quant method)

### Model type
Large Language Model (Instruction-tuned)

### Information about training algorithms, parameters, fairness constraints or other applied approaches, and features
- Architecture: Transformer-based decoder
- Context Window: 2048 tokens (optimized for RTX 2070 Max-Q)
- Temperature: 0.3 (low temperature for deterministic responses)
- Repeat Penalty: 1.1
- Quantization: 4-bit quantization for memory efficiency
- Instruction Format: System message + user message

### Paper or other resource for more information
- Check original model documentation from Meta AI (Llama) or Mistral AI

### License
Check original model licenses

### Where to send questions or comments about the model
Contact original model developers or experiment team

## Intended Use

### Primary intended uses
- Food & restaurant question answering
- Providing general information about food, dining, and restaurant-related topics
- Supporting chatbot applications in food service domain

### Primary intended users
- Food service businesses
- Restaurant information systems
- General consumers seeking food-related information

### Out-of-scope use cases
- Providing specific venue addresses, phone numbers, or URLs
- Making actual reservations or orders
- Medical or dietary advice requiring professional consultation
- Real-time menu information (requires integration with restaurant systems)

## Factors

### Relevant factors
- Domain: Food & restaurants
- Query complexity: Varies from simple to complex
- Intent type: Menu inquiries, dietary restrictions, general questions

### Evaluation factors
- Response latency
- Answer quality (length, relevance)
- Error rate

## Metrics

### Model performance measures
- Latency: Measured per-query response time
- Answer length: Word count as proxy for completeness
- Error rate: Frequency of API/connection errors

### Decision thresholds
- Response time targets: < 5 seconds per query (interactive use)

### Variation approaches
- Multiple model evaluation (llama3.1:8b vs mistral:7b)
- Intent-aware prompting variations
- Different system prompts

## Evaluation Data

### Datasets
- Test set: questions.txt (food & restaurant related questions)
- Questions cover various topics: dietary restrictions, cuisine types, restaurant practices, etc.

### Motivation
Evaluate model performance on diverse food & restaurant questions to assess practical usability.

### Preprocessing
- Direct question input (no preprocessing)
- System prompt injection
- Intent-specific guidelines (when applicable)

## Training Data
Training data details are specific to the base models (Llama, Mistral) - not retrained in this project.

## Quantitative Analyses

### Unitary results
See experiment card results for latency and answer quality metrics.

### Intersectional results
Not applicable for this evaluation setup.

## Ethical Considerations
- Models are instructed to avoid providing specific venue information to prevent misinformation
- System prompts emphasize general, helpful responses
- Models should not generate offensive or inappropriate content
- Responses should respect dietary restrictions and preferences
- No personal data collection or storage in model usage

## Caveats and Recommendations
- Models run locally via Ollama - performance depends on hardware (tested on RTX 2070 Max-Q)
- 4-bit quantization may slightly reduce output quality compared to full precision
- Responses are general and may not reflect current restaurant offerings
- Should not be used as sole source for dietary or medical advice
- Integration with live restaurant data recommended for accurate menu/hours information
- Response quality should be validated by human reviewers for production use
"""
    return card


def main():
    """Generate all experiment and model cards."""
    output_dir = Path("outputs")
    cards_dir = Path("experiment_cards")
    cards_dir.mkdir(exist_ok=True)
    
    print("Generating experiment and model cards...")
    
    # Generate LLM experiment cards
    models = [
        ("llama31_8b", "llama3.1:8b-instruct-q4_K_M"),
        ("mistral7b", "mistral:7b-instruct-q4_K_M")
    ]
    
    for model_short, model_tag in models:
        # Experiment card
        exp_card = generate_experiment_card_llm(output_dir, model_short)
        if exp_card:
            exp_path = cards_dir / f"experiment_llm_{model_short}.md"
            with open(exp_path, 'w', encoding='utf-8') as f:
                f.write(exp_card)
            print(f"Generated: {exp_path}")
        
        # Model card
        model_card = generate_model_card_llm(model_short, model_tag)
        model_path = cards_dir / f"model_card_{model_short}.md"
        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        print(f"Generated: {model_path}")
    
    # Generate BERT experiment card
    bert_exp_card = generate_experiment_card_bert(output_dir)
    if bert_exp_card:
        bert_exp_path = cards_dir / "experiment_bert.md"
        with open(bert_exp_path, 'w', encoding='utf-8') as f:
            f.write(bert_exp_card)
        print(f"Generated: {bert_exp_path}")
    
    # Generate BERT model card
    bert_model_card = generate_model_card_bert(output_dir)
    if bert_model_card:
        bert_model_path = cards_dir / "model_card_bert.md"
        with open(bert_model_path, 'w', encoding='utf-8') as f:
            f.write(bert_model_card)
        print(f"Generated: {bert_model_path}")
    
    print("\nAll cards generated successfully!")


if __name__ == "__main__":
    main()

