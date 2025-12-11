# RAG Demo for TripAdvisor Restaurant Data

This demo implements Retrieval-Augmented Generation (RAG) using a CSV file containing restaurant data from TripAdvisor.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running with the model:
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Usage

### Basic Usage
```bash
python rag_demo.py --csv path/to/restaurants.csv --question "Find Italian restaurants"
```

### Interactive Mode
```bash
python rag_demo.py --csv path/to/restaurants.csv
```

### Options
- `--csv`: Path to your TripAdvisor restaurant CSV file (required)
- `--question`: Single question to ask (optional, defaults to interactive mode)
- `--model`: Ollama model to use (default: `llama3.1:8b-instruct-q4_K_M`)
- `--top-k`: Number of restaurants to retrieve (default: 3)
- `--embedding-model`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--host`: Ollama host (default: `127.0.0.1`)
- `--port`: Ollama port (default: `11434`)

## CSV Format

The script expects a CSV file with restaurant data. It will automatically detect common column names:

**Preferred columns:**
- `name` or `restaurant_name`: Restaurant name
- `cuisine` or `cuisine_type`: Type of cuisine
- `price_range`: Price range
- `rating`: Rating
- `review`: Review text
- `description`: Description
- `location`: Location

**Example CSV structure:**
```csv
name,cuisine,price_range,rating,review,location
Mario's Italian,Italian,$$,4.5,"Great pasta and pizza",Downtown
Sushi Paradise,Japanese,$$$,4.8,"Fresh fish, excellent service",City Center
```

The script will use **all available columns** in the CSV to create searchable text, so you can include any additional columns you have.

## How It Works

1. **Loading**: The script loads the CSV file and displays available columns
2. **Embedding**: Creates vector embeddings for each restaurant using sentence transformers
3. **Retrieval**: When you ask a question, it finds the most relevant restaurants using cosine similarity
4. **Generation**: The retrieved restaurant information is passed to the LLM along with your question
5. **Response**: The LLM generates an answer based on the retrieved context

## Example

```bash
python rag_demo.py --csv restaurants.csv --question "Find affordable Italian restaurants with vegetarian options"
```

The script will:
1. Retrieve the top 3 most relevant restaurants from your CSV
2. Show the retrieved restaurants with relevance scores
3. Generate an answer using the LLM with the retrieved context


