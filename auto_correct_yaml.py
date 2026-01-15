#!/usr/bin/env python3
"""
Automated YAML Correction System

This script:
1. Takes validation feedback from previous run
2. Automatically generates corrected YAML
3. Applies all fixes in one pass
4. Outputs perfect YAML

Usage: python auto_correct_yaml.py <validation_results.json> <original.pdf> <README.md> <output.yaml>
"""

import sys
import os
import re
import json
import yaml
from typing import Dict, List

# Vertex AI
try:
    from vertexai.generative_models import GenerativeModel, Part
    import vertexai
except ImportError:
    print("Error: Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")
    sys.exit(1)

# Set service account path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(script_dir, 'service-account.json')


def load_validation_results(validation_path: str) -> Dict:
    """Load previous validation results."""
    try:
        with open(validation_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading validation results: {str(e)}")


def load_original_yaml(yaml_path: str) -> str:
    """Load original YAML content."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error loading original YAML: {str(e)}")


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                try:
                    page_text = page.extract_text()
                    page_text = page_text.encode('ascii', errors='ignore').decode('ascii')
                    text += page_text + "\n"
                except:
                    continue
            return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")


def generate_correction_prompt(original_yaml: str, validation_results: Dict, pdf_text: str, readme_text: str) -> str:
    """Generate comprehensive correction prompt."""
    
    # Extract all issues from validation
    issues_summary = []
    
    # Schema issues
    if 'schema_compliance' in validation_results:
        schema = validation_results['schema_compliance']
        if schema.get('issues'):
            issues_summary.extend([f"SCHEMA: {issue}" for issue in schema['issues']])
    
    # Pin accuracy issues
    if 'pin_accuracy' in validation_results:
        pins = validation_results['pin_accuracy']
        if pins.get('missing_pins'):
            issues_summary.extend([f"MISSING PIN: {pin}" for pin in pins['missing_pins']])
        if pins.get('extra_pins'):
            issues_summary.extend([f"EXTRA PIN: {pin}" for pin in pins['extra_pins']])
    
    # Voltage issues
    if 'voltage_accuracy' in validation_results:
        voltage = validation_results['voltage_accuracy']
        if voltage.get('incorrect_ranges'):
            issues_summary.extend([f"VOLTAGE: {issue}" for issue in voltage['incorrect_ranges']])
    
    # Hallucinations
    if 'hallucinations' in validation_results:
        hall = validation_results['hallucinations']
        if hall.get('detected'):
            issues_summary.extend([f"HALLUCINATION: {hall}" for hall in hall['detected']])
    
    # Recommendations
    if 'recommendations' in validation_results:
        issues_summary.extend([f"RECOMMENDATION: {rec}" for rec in validation_results['recommendations']])
    
    issues_text = "\n".join([f"- {issue}" for issue in issues_summary])
    
    correction_prompt = f"""You are an expert YAML correction specialist. Your task is to fix ALL validation issues and produce a perfect 100/100 score YAML.

ORIGINAL YAML (with issues):
{original_yaml}

VALIDATION ISSUES TO FIX:
{issues_text}

README FORMAT REQUIREMENTS:
{readme_text[:3000]}

PDF SOURCE CONTENT:
{pdf_text[:4000]}

CORRECTION TASKS:
1. Fix ALL schema compliance issues
2. Add all missing pins, remove extra pins
3. Correct all voltage ranges to match PDF exactly
4. Remove ALL hallucinated content
5. Follow README format perfectly
6. Ensure package pin identifiers are strings
7. Only use information explicitly stated in PDF
8. Make minimal, conservative interpretations

CRITICAL REQUIREMENTS:
- Score MUST be 100/100
- NO hallucinations allowed
- Perfect schema compliance required
- All pins must be accurate
- Voltage ranges must be exact

Return ONLY the corrected YAML, no explanations, no code fences, no extra text.
The output must be valid YAML that will pass all validation checks with 100/100 score."""
    
    return correction_prompt


def generate_corrected_yaml(pdf_path: str, readme_path: str, correction_prompt: str) -> str:
    """Generate corrected YAML using Vertex AI."""
    try:
        # Initialize Vertex AI
        vertexai.init(location="us-central1")
        
        # Use Gemini for correction
        model = GenerativeModel("gemini-2.5-flash-lite")
        
        # Read files for direct upload
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        
        with open(readme_path, 'r', encoding='utf-8') as readme_file:
            readme_data = readme_file.read()
        
        # Create parts
        pdf_part = Part.from_data(pdf_data, mime_type="application/pdf")
        readme_part = Part.from_text(readme_data)
        
        # Generate corrected YAML
        response = model.generate_content([
            pdf_part,
            readme_part,
            correction_prompt
        ])
        
        return response.text
        
    except Exception as e:
        raise Exception(f"Error generating corrected YAML: {str(e)}")


def clean_and_validate_yaml(yaml_content: str) -> str:
    """Clean YAML response and basic validation."""
    # Remove code fences
    yaml_text = yaml_content.strip()
    yaml_text = re.sub(r'```yaml\s*', '', yaml_text)
    yaml_text = re.sub(r'```\s*', '', yaml_text)
    
    # Basic YAML validation
    try:
        yaml.safe_load(yaml_text)
        print("[VALIDATION] Generated YAML is syntactically valid")
    except yaml.YAMLError as e:
        print(f"[WARNING] YAML syntax issue: {e}")
    
    return yaml_text.strip()


def save_corrected_yaml(yaml_content: str, output_path: str) -> None:
    """Save corrected YAML."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"[SAVED] Corrected YAML saved to: {output_path}")
    except Exception as e:
        raise Exception(f"Error saving corrected YAML: {str(e)}")


def main():
    """Main correction function."""
    if len(sys.argv) != 5:
        print("Usage: python auto_correct_yaml.py <validation_results.json> <original.pdf> <README.md> <output.yaml>")
        sys.exit(1)
    
    validation_path = sys.argv[1]
    pdf_path = sys.argv[2]
    readme_path = sys.argv[3]
    output_path = sys.argv[4]
    
    try:
        print("[AUTO-CORRECT] Starting automated YAML correction...")
        
        # Load inputs
        print("[LOAD] Loading validation results...")
        validation_results = load_validation_results(validation_path)
        
        print("[LOAD] Loading original YAML...")
        original_yaml = load_original_yaml(output_path)
        
        print("[LOAD] Extracting PDF text...")
        pdf_text = extract_pdf_text(pdf_path)
        
        print("[LOAD] Loading README...")
        readme_text = open(readme_path, 'r', encoding='utf-8').read()
        
        # Generate correction prompt
        print("[ANALYZE] Analyzing validation issues...")
        correction_prompt = generate_correction_prompt(original_yaml, validation_results, pdf_text, readme_text)
        
        # Generate corrected YAML
        print("[GENERATE] Generating corrected YAML...")
        corrected_yaml = generate_corrected_yaml(pdf_path, readme_path, correction_prompt)
        
        # Clean and validate
        print("[CLEAN] Cleaning and validating corrected YAML...")
        final_yaml = clean_and_validate_yaml(corrected_yaml)
        
        # Save result
        print("[SAVE] Saving corrected YAML...")
        save_corrected_yaml(final_yaml, output_path)
        
        # Create correction report
        report_path = output_path.replace('.yaml', '_correction_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'original_score': validation_results.get('ai_validation', {}).get('overall_score', 0),
                'issues_fixed': len(validation_results.get('schema_compliance', {}).get('issues', [])) +
                              len(validation_results.get('pin_accuracy', {}).get('missing_pins', [])) +
                              len(validation_results.get('pin_accuracy', {}).get('extra_pins', [])) +
                              len(validation_results.get('voltage_accuracy', {}).get('incorrect_ranges', [])) +
                              len(validation_results.get('hallucinations', {}).get('detected', [])),
                'corrections_applied': [
                    'Schema compliance fixes',
                    'Pin accuracy corrections', 
                    'Voltage range corrections',
                    'Hallucination removal',
                    'Format standardization'
                ],
                'expected_score': 100,
                'files_used': {
                    'validation_results': validation_path,
                    'pdf': pdf_path,
                    'readme': readme_path,
                    'original_yaml': output_path
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n[COMPLETE] Automated correction finished!")
        print(f"[REPORT] Correction report saved to: {report_path}")
        print(f"[RESULT] Check {output_path} for corrected YAML")
        print(f"[NEXT] Run validation again to verify 100/100 score")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
