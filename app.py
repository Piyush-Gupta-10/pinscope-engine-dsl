#!/usr/bin/env python3
"""
PDF + README to YAML Generator using Vertex AI Gemini

This script:
1. Reads a PDF file and README.md
2. Builds a prompt combining both contents
3. Sends to Vertex AI (Gemini) for YAML generation
4. Saves the cleaned YAML output to output.yaml

Usage: python app.py <rules.pdf> <README.md>
"""

import sys
import os
import re
import yaml
import warnings
from pathlib import Path
from typing import Optional, Tuple

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set service account path (service-account.json is in the same folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(script_dir, 'service-account.json')

# PDF processing
try:
    import PyPDF2
except ImportError:
    print("Error: PyPDF2 not installed. Run: pip install PyPDF2")
    sys.exit(1)

# Vertex AI
try:
    from vertexai.generative_models import GenerativeModel, Part
    import vertexai
except ImportError:
    print("Error: Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")
    sys.exit(1)


def call_gemini_with_files(pdf_path: str, readme_path: str) -> str:
    """Call Vertex AI Gemini API with direct file upload."""
    try:
        # Initialize Vertex AI (uses GOOGLE_APPLICATION_CREDENTIALS)
        vertexai.init(location="us-central1")
        
        # Use Gemini model
        model = GenerativeModel("gemini-2.5-flash-lite")
        
        # Read files and create parts
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        
        with open(readme_path, 'r', encoding='utf-8') as readme_file:
            readme_data = readme_file.read()
        
        # Create parts with correct method
        pdf_part = Part.from_data(pdf_data, mime_type="application/pdf")
        readme_part = Part.from_text(readme_data)
        
        # Generate content with files
        prompt = """Extract rules from the PDF and generate YAML strictly following the README format. 
        Read the README carefully to understand the YAML structure, then extract component information from the PDF.
        Return only valid YAML, no explanations or code fences."""
        
        response = model.generate_content([
            pdf_part,
            readme_part,
            prompt
        ])
        
        return response.text
    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")


def read_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")


def read_readme(readme_path: str) -> str:
    """Read README.md file."""
    try:
        with open(readme_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error reading README {readme_path}: {str(e)}")


def build_prompt(pdf_text: str, readme_text: str) -> str:
    """Build the prompt for Gemini."""
    prompt = f"""You are given:
1) A PDF that contains rules
2) A README that defines how YAML should be structured

Task:
- Read the README carefully to understand YAML format
- Extract rules from the PDF
- Generate YAML strictly following README format
- Return only YAML, no explanation

README:
{readme_text}

PDF CONTENT:
{pdf_text}

Generate the YAML output:"""
    return prompt


def call_gemini(prompt: str) -> str:
    """Call Vertex AI Gemini API."""
    try:
        # Initialize Vertex AI (uses GOOGLE_APPLICATION_CREDENTIALS)
        vertexai.init(location="us-central1")
        
        # Use Gemini model
        model = GenerativeModel("gemini-2.5-flash-lite")
        
        # Generate content
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")


def clean_yaml_response(response_text: str) -> str:
    """Clean YAML response by removing code fences and extra text."""
    # Remove YAML code fences
    yaml_text = re.sub(r'```yaml\s*', '', response_text)
    yaml_text = re.sub(r'```\s*', '', yaml_text)
    
    # Remove any leading/trailing whitespace
    yaml_text = yaml_text.strip()
    
    # Fix common YAML issues
    # Replace problematic custom tags with standard ones
    yaml_text = yaml_text.replace('!signal_level(low)', 'signal_level: low')
    yaml_text = yaml_text.replace('!signal_level(high)', 'signal_level: high')
    yaml_text = yaml_text.replace('!signal_level(medium)', 'signal_level: medium')
    
    # Try to validate it's valid YAML
    try:
        yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        print(f"Warning: Generated text may not be valid YAML: {e}")
    
    return yaml_text


def save_yaml(yaml_content: str, pdf_path: str, output_dir: str = "output", custom_filename: str = None) -> None:
    """Save YAML content to file with component name."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if custom_filename:
            output_filename = f"{custom_filename}.yaml"
        else:
            # Extract component name from PDF path
            pdf_filename = os.path.basename(pdf_path)
            component_name = os.path.splitext(pdf_filename)[0]
            output_filename = f"output-{component_name}.yaml"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Save YAML content
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
        print("Usage: python app.py <rules.pdf> <README.md>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    readme_path = sys.argv[2]
    
    try:
        # Validate inputs
        validate_inputs(pdf_path, readme_path)
        
        print("Step A: Validating files...")
        # Step A: Validate files (no text extraction needed)
        
        print("Step B: Preparing files for upload...")
        # Step B: Files are ready for direct upload
        
        print("Step C: Calling Gemini with files...")
        # Step C: Call Gemini with direct file upload
        response_text = call_gemini_with_files(pdf_path, readme_path)
        
        print("Step D: Cleaning YAML...")
        # Step D: Clean YAML
        yaml_content = clean_yaml_response(response_text)
        
        print("Step E: Saving output...")
        # Step E: Save YAML
        save_yaml(yaml_content, pdf_path)
        
        print("✅ YAML generated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
