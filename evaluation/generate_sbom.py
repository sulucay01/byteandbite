#!/usr/bin/env python3
"""
Generate Software Bill of Materials (SBOM) requirements
for both Python packages and AI models used in the project.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def get_package_metadata() -> Dict[str, Dict[str, str]]:
    """Return metadata (description and license) for known packages."""
    return {
        "flask": {
            "description": "Lightweight WSGI web framework.",
            "license": "BSD-3-Clause"
        },
        "flask-cors": {
            "description": "CORS support for Flask apps.",
            "license": "MIT"
        },
        "requests": {
            "description": "Human-friendly HTTP client for Python.",
            "license": "Apache-2.0"
        },
        "pandas": {
            "description": "Data analysis library with DataFrame.",
            "license": "BSD-3-Clause"
        },
        "sentence-transformers": {
            "description": "Transformer-based sentence embeddings (SBERT).",
            "license": "Apache-2.0"
        },
        "numpy": {
            "description": "Fundamental package for n-dimensional arrays.",
            "license": "BSD-3-Clause"
        },
        "fastapi": {
            "description": "High-performance API framework on ASGI/type hints.",
            "license": "MIT"
        },
        "uvicorn": {
            "description": "ASGI server implementation.",
            "license": "BSD-3-Clause"
        },
        "qdrant-client": {
            "description": "Python client for Qdrant vector DB (HTTP/gRPC).",
            "license": "Apache-2.0"
        },
        "pydantic": {
            "description": "Data validation via type hints.",
            "license": "MIT"
        }
    }


def parse_requirements(requirements_file: str = "requirements.txt") -> List[Dict[str, str]]:
    """Parse requirements.txt and extract package information."""
    packages = []
    req_path = Path(requirements_file)
    metadata = get_package_metadata()
    
    if not req_path.exists():
        print(f"Warning: {requirements_file} not found")
        return packages
    
    with open(req_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse package name and version constraint
            # Format: package>=version or package==version or package
            match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)([>=<!=]+)?([\d.]+)?', line)
            if match:
                pkg_name = match.group(1).split('[')[0]  # Remove extras like [dev]
                version_constraint = match.group(2) or ""
                version = match.group(3) or ""
                
                # Get metadata if available, otherwise use defaults
                pkg_meta = metadata.get(pkg_name, {})
                
                packages.append({
                    "name": pkg_name,
                    "version_constraint": version_constraint + version,
                    "description": pkg_meta.get("description", f"Python package: {pkg_name}"),
                    "license": pkg_meta.get("license", "To be determined"),
                    "supplier": "PyPI",
                    "checksum": "To be determined"
                })
    
    return packages


def get_ai_models() -> List[Dict[str, str]]:
    """Extract AI models used in the codebase."""
    models = [
        {
            "model_name": "intfloat/e5-small-v2",
            "purpose": "Embedding text chunks for use in the vector database",
            "accesser": "System",
            "date_of_access": datetime.now().strftime("%b. %d"),
            "method_of_interaction": "Run via sentence-transformers library; interact via SentenceTransformer API",
            "experiment_and_results": "Default embedding model for RAG system"
        },
        {
            "model_name": "llama3.1:8b-instruct-q4_K_M",
            "purpose": "Generate answers given a context (Llama 3.1 8B quantized)",
            "accesser": "System",
            "date_of_access": datetime.now().strftime("%b. %d"),
            "method_of_interaction": "Run on GPU on server; interact via API calls to Ollama host",
            "experiment_and_results": "Default generation model for RAG chat"
        },
        {
            "model_name": "mistral:7b-instruct-q4_K_M",
            "purpose": "Generate answers given a context (Mistral 7B quantized)",
            "accesser": "System",
            "date_of_access": datetime.now().strftime("%b. %d"),
            "method_of_interaction": "Run on GPU on server; interact via API calls to Ollama host",
            "experiment_and_results": "Alternative generation model option"
        }
    ]
    return models


def format_software_sbom(packages: List[Dict[str, str]]) -> str:
    """Format Software Bill of Materials table as string."""
    output = []
    output.append("\n" + "="*80)
    output.append("SOFTWARE BILL OF MATERIALS")
    output.append("="*80)
    output.append("\nThis register tracks all third-party libraries, tools and models used during development.")
    output.append("\nTo generate a complete SBOM, you can use:")
    output.append("- requirements.txt file with syft tool")
    output.append("- Docker image with syft tool")
    output.append("\nPackage Details:")
    output.append("-" * 80)
    output.append(f"{'Package Name':<25} {'Version':<20} {'Description':<30} {'License':<15} {'Supplier':<15}")
    output.append("-" * 80)
    
    for pkg in packages:
        name = pkg['name'][:24]
        version = pkg['version_constraint'][:19] if pkg['version_constraint'] else "N/A"
        desc = pkg['description'][:29]
        license_info = pkg['license'][:14]
        supplier = pkg['supplier'][:14]
        output.append(f"{name:<25} {version:<20} {desc:<30} {license_info:<15} {supplier:<15}")
    
    output.append("-" * 80)
    output.append(f"\nTotal packages: {len(packages)}")
    return "\n".join(output)


def format_ai_sbom(models: List[Dict[str, str]]) -> str:
    """Format AI Software Bill of Materials table as string."""
    output = []
    output.append("\n" + "="*80)
    output.append("AI SOFTWARE BILL OF MATERIALS")
    output.append("="*80)
    output.append("\nThis SBOM section documents third-party AI models.")
    output.append("\nFor Hugging Face models, you can use AI SBOM Generator to create")
    output.append("an AI SBOM in CycloneDX 1.6 JSON format with helpful metadata.")
    output.append("\nWhen naming a model, specify all characteristics including:")
    output.append("- Model version")
    output.append("- Size")
    output.append("- Quantization")
    output.append("\nInclude links to where the model's weights and more details were accessed.")
    output.append("Specify what purpose each model will satisfy.")
    output.append("Document who accessed the model and when.")
    output.append("Mention the specific libraries used to interact with the models.")
    output.append("Include an associated experiment and its result, indicating whether the model")
    output.append("will be used or rejected, and why.")
    output.append("\n" + "-" * 80)
    output.append(f"{'Model Name':<40} {'Purpose':<50} {'Accesser':<15} {'Date of Access':<15} {'Method of Interaction':<50} {'Experiment and Results':<30}")
    output.append("-" * 80)
    
    for model in models:
        name = model['model_name'][:39]
        purpose = model['purpose'][:49]
        accesser = model['accesser'][:14]
        date = model['date_of_access'][:14]
        method = model['method_of_interaction'][:49]
        experiment = model['experiment_and_results'][:29]
        output.append(f"{name:<40} {purpose:<50} {accesser:<15} {date:<15} {method:<50} {experiment:<30}")
    
    output.append("-" * 80)
    output.append(f"\nTotal models: {len(models)}")
    return "\n".join(output)


def main():
    """Main function to generate SBOM requirements."""
    output_lines = []
    
    output_lines.append("Generating Software Bill of Materials...")
    output_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate Software SBOM
    packages = parse_requirements()
    output_lines.append(format_software_sbom(packages))
    
    # Generate AI SBOM
    models = get_ai_models()
    output_lines.append(format_ai_sbom(models))
    
    output_lines.append("\n" + "="*80)
    output_lines.append("SBOM Generation Complete")
    output_lines.append("="*80)
    
    # Combine all output
    full_output = "\n".join(output_lines)
    
    # Print to console
    print(full_output)
    
    # Save to file
    output_file = "sbom_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_output)
    
    print(f"\n\nSBOM report saved to: {output_file}")


if __name__ == "__main__":
    main()

