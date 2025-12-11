#!/usr/bin/env python3
"""
Client script to interact with the Food & Restaurant Assistant web API
"""
import requests
import json
import sys
import argparse

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:5000"

def chat(question, model="llama3.1:8b-instruct-q4_K_M", server_url=DEFAULT_SERVER_URL):
    """
    Send a question to the web API and get response
    
    Args:
        question: The question to ask
        model: Model to use (default: llama3.1:8b-instruct-q4_K_M)
        server_url: Base URL of the web server
    
    Returns:
        dict with answer, model_name, latency_sec, answer_words
    """
    url = f"{server_url}/api/chat"
    payload = {
        "model": model,
        "question": question
    }
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}", file=sys.stderr)
        return {"error": str(e)}

def list_models(server_url=DEFAULT_SERVER_URL):
    """Get list of available models from the server"""
    url = f"{server_url}/api/models"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}", file=sys.stderr)
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(
        description="Client for Food & Restaurant Assistant API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a question with default model
  python client_script.py "What are common gluten-free dishes?"
  
  # Ask with specific model
  python client_script.py "What is ramen?" --model mistral:7b-instruct-q4_K_M
  
  # List available models
  python client_script.py --list-models
  
  # Use custom server URL
  python client_script.py "Your question" --server http://192.168.1.100:5000
        """
    )
    
    parser.add_argument(
        'question',
        nargs='?',
        help='Question to ask the assistant'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='llama3.1:8b-instruct-q4_K_M',
        help='Model to use (default: llama3.1:8b-instruct-q4_K_M)'
    )
    
    parser.add_argument(
        '--server', '-s',
        default=DEFAULT_SERVER_URL,
        help=f'Server URL (default: {DEFAULT_SERVER_URL})'
    )
    
    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output result as JSON'
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        result = list_models(args.server)
        if "error" in result:
            sys.exit(1)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Available models:")
            for model in result["models"]:
                print(f"  - {model['tag']} ({model['name']})")
        return
    
    # Ask question
    if not args.question:
        parser.error("Question is required (or use --list-models)")
    
    result = chat(args.question, args.model, args.server)
    
    if "error" in result:
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nModel: {result['model_name']}")
        print(f"Latency: {result['latency_sec']}s")
        print(f"Answer words: {result['answer_words']}")
        print(f"\nAnswer:\n{result['answer']}\n")

if __name__ == "__main__":
    main()

