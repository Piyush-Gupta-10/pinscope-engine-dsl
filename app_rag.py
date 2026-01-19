#!/usr/bin/env python3
"""
RAG-Enhanced PDF + README to YAML Generator using Vertex AI Gemini

This script:
1. Uses RAG to retrieve relevant PDF chunks
2. Combines with README.md format rules
3. Sends focused context to Gemini for YAML generation
4. Saves the cleaned YAML output to output.yaml

Usage: python app_rag.py <rules.pdf> <README.md>
"""

import sys
import os
import re
import yaml
import warnings
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Set credentials from environment if available
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    credentials_path = os.path.join(script_dir, 'service-account.json')
    if os.path.exists(credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

try:
    from vertexai.generative_models import GenerativeModel, Part
    import vertexai
except ImportError:
    print("Error: Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")
    sys.exit(1)

# Import RAG components
from rag_ingestion import RAGIngestionPipeline
from rag_retrieval import RAGRetriever


class RAGYAMLGenerator:
    """RAG-enhanced YAML generator."""
    
    def __init__(self, readme_path: str = None):
        self.ingestion_pipeline = RAGIngestionPipeline(readme_path)
        self.retriever = RAGRetriever()
        
    def ensure_pdf_processed(self, pdf_path: str) -> None:
        """Ensure PDF is processed and stored in vector database."""
        print("Checking if PDF is already processed...")
        
        # Check if we have chunks for this PDF
        pdf_filename = os.path.basename(pdf_path)
        stats = self.retriever.get_collection_stats()
        
        if stats.get('total_chunks', 0) == 0:
            print("No chunks found. Processing PDF...")
            self.ingestion_pipeline.process_pdf(pdf_path)
        else:
            print(f"Found {stats['total_chunks']} chunks in database")
    
    def generate_yaml_with_rag(self, pdf_path: str, readme_path: str) -> str:
        """Generate YAML using RAG-enhanced approach."""
        print("Starting RAG-enhanced YAML generation...")
        
        # Step 1: Ensure PDF is processed
        self.ensure_pdf_processed(pdf_path)
        
        # Step 2: Retrieve relevant context
        print("Retrieving relevant context from PDF...")
        pdf_filename = os.path.basename(pdf_path)
        retrieval_result = self.retriever.retrieve_context_for_yaml_generation(readme_path, source_file=pdf_filename)
        
        # Step 3: Read README for format rules
        print("Reading README format rules...")
        try:
            with open(readme_path, 'r', encoding='utf-8') as file:
                readme_content = file.read()
        except Exception as e:
            raise Exception(f"Error reading README {readme_path}: {str(e)}")
        
        # Step 4: Build focused prompt for Gemini
        prompt = self._build_rag_prompt(readme_content, retrieval_result)
        
        # Step 5: Call Gemini
        print("Calling Gemini with RAG context...")
        response_text = self._call_gemini(prompt)
        
        return response_text
    
    def _build_rag_prompt(self, readme_content: str, retrieval_result) -> str:
        """Build focused prompt using RAG context."""
        prompt = f"""You are given:
1) A README that defines how YAML should be structured
2) Relevant chunks from a technical PDF document containing component specifications

TASK:
- Read the README carefully to understand the exact YAML format required
- Use ONLY the provided PDF chunks to extract component information
- Generate YAML that strictly follows the README format
- If information is missing in the chunks, do not invent it - leave fields empty or omit them
- Return ONLY valid YAML, no explanations or code fences

README FORMAT RULES:
{readme_content}

RELEVANT PDF CHUNKS:
{retrieval_result.context}

Retrieval Statistics: {retrieval_result.metadata}

Generate the YAML output based on the above information:"""
        
        return prompt
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Vertex AI Gemini API."""
        try:
            # Get project ID from environment or use default
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            if not project_id:
                print("Warning: GOOGLE_CLOUD_PROJECT not set, using default project")
            
            vertexai.init(project=project_id, location="us-central1")
            model = GenerativeModel("gemini-2.5-flash-lite")
            
            # Set temperature to 0 for consistent results
            generation_config = {
                "temperature": 0.0,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            raise Exception(f"Error calling Gemini API: {str(e)}")


def clean_yaml_response(response_text: str) -> str:
    """Clean YAML response by removing code fences and extra text."""
    yaml_text = re.sub(r'```yaml\s*', '', response_text)
    yaml_text = re.sub(r'```\s*', '', yaml_text)
    
    yaml_text = yaml_text.strip()
    
    yaml_text = yaml_text.replace('!signal_level(low)', 'signal_level: low')
    yaml_text = yaml_text.replace('!signal_level(high)', 'signal_level: high')
    yaml_text = yaml_text.replace('!signal_level(medium)', 'signal_level: medium')

    try:
        yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        print(f"Warning: Generated text may not be valid YAML: {e}")
    
    return yaml_text


def save_yaml(yaml_content: str, pdf_path: str, output_dir: str = "output", custom_filename: str = None) -> None:
    """Save YAML content to file with component name."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if custom_filename:
            output_filename = f"{custom_filename}.yaml"
        else:
            pdf_filename = os.path.basename(pdf_path)
            component_name = os.path.splitext(pdf_filename)[0]
            output_filename = f"output-{component_name}-RAG.yaml"
        
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(yaml_content)
        
        print(f"YAML saved to {output_path}")
    except Exception as e:
        raise Exception(f"Error saving YAML to {output_path}: {str(e)}")


def validate_inputs(pdf_path: str, readme_path: str) -> None:
    """Validate input files exist and are accessible."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README file not found: {readme_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    if not readme_path.lower().endswith('.md'):
        raise ValueError(f"README file must be a markdown file: {readme_path}")


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python app_rag.py <rules.pdf> <README.md>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    readme_path = sys.argv[2]
    
    try:
        validate_inputs(pdf_path, readme_path)
        
        print("RAG-Enhanced PDF + README to YAML Generator")
        print("=" * 50)
        
        # Initialize RAG generator
        generator = RAGYAMLGenerator()
        
        print("Step 1: Validating files...")
        
        print("Step 2: Processing PDF with RAG...")
        
        print("Step 3: Retrieving relevant context...")
        response_text = generator.generate_yaml_with_rag(pdf_path, readme_path)
        
        print("Step 4: Cleaning YAML...")
        yaml_content = clean_yaml_response(response_text)
        
        print("Step 5: Saving output...")
        save_yaml(yaml_content, pdf_path)
        
        print("RAG-enhanced YAML generated successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
