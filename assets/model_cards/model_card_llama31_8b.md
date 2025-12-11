# Model Card: llama31_8b

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
