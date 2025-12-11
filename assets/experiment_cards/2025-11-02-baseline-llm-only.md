# Experiment Card — LLM-only Baseline

**Date:** 2025-11-02  
**Hardware:** Local GPU (RTX 2070 Max-Q, 8 GB), Ollama ≥ 0.12.9  
**Models:** llama3.1:8b-instruct-q4_K_M (llama31_8b), mistral:7b-instruct-q4_K_M (mistral7b)  
**Prompt (system):** Short, generic, safety-constrained; avoid specific venues/countries. Secondary system message carries the intent guideline.  
**Data:** Questions file: `questions.txt`

## Results (Automatic Quick Checks)
| Model | Latency(s) | Len OK | I-don’t-know | Halluc% | TR-mention% |
|------|------------:|-------:|-------------:|--------:|------------:|
| llama31_8b | 1.852 | 1.000 | 0.100 | 0.000 | 0.000 |
| mistral7b | 3.502 | 0.500 | 0.000 | 0.600 | 0.000 |


## Notes
- No RAG data used in this baseline.
- Answers are constrained to be generic and short; country/venue specifics are avoided by prompt design.
- Intent is injected as a secondary system message (static in this baseline).

## Next Steps
- Integrate a lightweight BERT intent classifier to select the intent dynamically.
- Add RAG layer with Yelp corpus and a vector DB; inject retrieved context as an additional system message.
- Expand evaluation with human rubrics (Accuracy/Coverage/Clarity/Safety).
