#!/usr/bin/env python3
"""
YAML Validation System for Generated Component Definitions

This script validates AI-generated YAML files for:
- Correctness against source PDF and README
- Hallucination detection
- Schema compliance
- Consistency checking

Usage: python validate_yaml.py <output.yaml> <original.pdf> <README.md>
"""

import sys
import os
import re
import yaml
import json
from typing import Dict, List, Tuple

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


def load_yaml_file(yaml_path: str) -> Dict:
    """Load and parse YAML file."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f"Error loading YAML {yaml_path}: {str(e)}")


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF for fact-checking."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                try:
                    page_text = page.extract_text()
                    # Clean non-ASCII characters
                    page_text = page_text.encode('ascii', errors='ignore').decode('ascii')
                    text += page_text + "\n"
                except:
                    # Skip problematic pages
                    continue
            return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")


def validate_with_ai(yaml_content: str, pdf_text: str, readme_text: str) -> Dict:
    """Use AI to validate YAML for correctness and hallucinations."""
    try:
        # Initialize Vertex AI
        vertexai.init(location="us-central1")
        
        # Use different model for validation (same as generation but different prompt strategy)
        model = GenerativeModel("gemini-2.5-flash-lite")
        
        validation_prompt = f"""You are a YAML validation expert specializing in component definitions. 

TASK: Analyze the generated YAML for correctness and hallucinations.

Generated YAML:
{yaml_content}

README Format Reference:
{readme_text[:2000]}  # First 2k chars for context

PDF Content (for fact-checking):
{pdf_text[:3000]}  # First 3k chars for fact-checking

VALIDATION CRITERIA:
1. Schema Compliance: Does YAML follow README format exactly?
2. Pin Accuracy: Do all pins exist in the PDF?
3. Voltage Values: Are voltage ranges correct per PDF?
4. Rule Validity: Are DSL rules syntactically correct?
5. Hallucination Detection: Any content not supported by PDF?

Return JSON response:
{{
    "overall_score": 85,
    "schema_compliance": {{"score": 90, "issues": []}},
    "pin_accuracy": {{"score": 80, "missing_pins": [], "extra_pins": []}},
    "voltage_accuracy": {{"score": 85, "incorrect_ranges": []}},
    "rule_validity": {{"score": 95, "invalid_rules": []}},
    "hallucinations": {{"detected": [], "confidence": 95}},
    "recommendations": ["Fix voltage ranges", "Add missing pins"]
}}

Be thorough and specific in your analysis."""
        
        response = model.generate_content(validation_prompt)
        
        # Try to parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse JSON response", "raw_response": response.text}
        except:
            return {"error": "JSON parsing failed", "raw_response": response.text}
            
    except Exception as e:
        raise Exception(f"Error in AI validation: {str(e)}")


def schema_validation(yaml_data: Dict) -> Dict:
    """Perform rule-based schema validation."""
    issues = []
    
    # Check required top-level fields
    required_fields = ['component', 'category', 'packages', 'pins']
    for field in required_fields:
        if field not in yaml_data:
            issues.append(f"Missing required field: {field}")
    
    # Validate packages structure
    if 'packages' in yaml_data:
        for pkg_name, pkg_data in yaml_data['packages'].items():
            if 'pins' not in pkg_data:
                issues.append(f"Package {pkg_name} missing pins definition")
    
    # Validate pins structure
    if 'pins' in yaml_data:
        valid_roles = ['power', 'ground', 'io', 'analog', 'clock', 'config', 'protection']
        valid_directions = ['input', 'output', 'bidirectional', 'passive']
        
        for pin_uid, pin_data in yaml_data['pins'].items():
            # Check required pin fields
            required_pin_fields = ['names', 'role', 'direction']
            for field in required_pin_fields:
                if field not in pin_data:
                    issues.append(f"Pin {pin_uid} missing field: {field}")
            
            # Validate role
            if 'role' in pin_data and pin_data['role'] not in valid_roles:
                issues.append(f"Pin {pin_uid} has invalid role: {pin_data['role']}")
            
            # Validate direction
            if 'direction' in pin_data and pin_data['direction'] not in valid_directions:
                issues.append(f"Pin {pin_uid} has invalid direction: {pin_data['direction']}")
            
            # Validate voltage ranges
            if 'voltage' in pin_data:
                voltage = pin_data['voltage']
                if isinstance(voltage, dict):
                    if 'abs_min' in voltage and 'abs_max' in voltage:
                        if voltage['abs_min'] >= voltage['abs_max']:
                            issues.append(f"Pin {pin_uid} has invalid voltage range: min >= max")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "score": max(0, 100 - len(issues) * 10)
    }


def cross_reference_pins(yaml_data: Dict, pdf_text: str) -> Dict:
    """Cross-reference YAML pins with PDF content."""
    yaml_pins = set()
    pdf_pins = set()
    
    # Extract pins from YAML
    if 'pins' in yaml_data:
        for pin_uid in yaml_data['pins'].keys():
            yaml_pins.add(pin_uid.lower())
    
    # Extract pin names from PDF (simple pattern matching)
    pin_patterns = [
        r'\b(PIN\s*[0-9]+)\b',
        r'\b([A-Z][0-9]+)\b',
        r'\b(VIN|VOUT|GND|EN|FB|PG)\b',
        r'\b(SDA|SCL|SCK|MISO|MOSI)\b'
    ]
    
    for pattern in pin_patterns:
        matches = re.findall(pattern, pdf_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                pdf_pins.add(match[0].lower())
            else:
                pdf_pins.add(match.lower())
    
    # Also extract pin names from YAML names field
    if 'pins' in yaml_data:
        for pin_data in yaml_data['pins'].values():
            if 'names' in pin_data:
                for name in pin_data['names']:
                    pdf_pins.add(name.lower())
    
    return {
        "yaml_pins": list(yaml_pins),
        "pdf_pins": list(pdf_pins),
        "missing_pins": list(pdf_pins - yaml_pins),
        "extra_pins": list(yaml_pins - pdf_pins),
        "coverage": len(yaml_pins & pdf_pins) / max(len(pdf_pins), 1) * 100
    }


def generate_validation_report(yaml_path: str, validation_results: Dict) -> None:
    """Generate a detailed validation report."""
    print(f"\n[VALIDATION] YAML Validation Report for: {os.path.basename(yaml_path)}")
    print("=" * 60)
    
    if "error" in validation_results:
        print(f"[ERROR] Validation Error: {validation_results['error']}")
        return
    
    # Overall score
    overall_score = validation_results.get('overall_score', 0)
    print(f"[SCORE] Overall Score: {overall_score}/100")
    
    # Schema compliance
    schema = validation_results.get('schema_compliance', {})
    print(f"\n[SCHEMA] Schema Compliance: {schema.get('score', 0)}/100")
    if schema.get('issues'):
        for issue in schema['issues']:
            print(f"   [WARNING] {issue}")
    
    # Pin accuracy
    pins = validation_results.get('pin_accuracy', {})
    print(f"\n[PINS] Pin Accuracy: {pins.get('score', 0)}/100")
    if pins.get('missing_pins'):
        print(f"   [MISSING] Missing pins: {pins['missing_pins']}")
    if pins.get('extra_pins'):
        print(f"   [EXTRA] Extra pins: {pins['extra_pins']}")
    
    # Voltage accuracy
    voltage = validation_results.get('voltage_accuracy', {})
    print(f"\n[VOLTAGE] Voltage Accuracy: {voltage.get('score', 0)}/100")
    if voltage.get('incorrect_ranges'):
        for range_issue in voltage['incorrect_ranges']:
            print(f"   [WARNING] {range_issue}")
    
    # Rule validity
    rules = validation_results.get('rule_validity', {})
    print(f"\n[RULES] Rule Validity: {rules.get('score', 0)}/100")
    if rules.get('invalid_rules'):
        for rule in rules['invalid_rules']:
            print(f"   [INVALID] Invalid rule: {rule}")
    
    # Hallucination detection
    hallucinations = validation_results.get('hallucinations', {})
    print(f"\n[AI] Hallucination Detection: {hallucinations.get('confidence', 0)}% confidence")
    if hallucinations.get('detected'):
        for halluc in hallucinations['detected']:
            print(f"   [HALLUCINATION] Potential hallucination: {halluc}")
    else:
        print("   [OK] No hallucinations detected")
    
    # Recommendations
    recommendations = validation_results.get('recommendations', [])
    if recommendations:
        print(f"\n[RECOMMENDATIONS]")
        for rec in recommendations:
            print(f"   - {rec}")
    
    print("\n" + "=" * 60)


def main():
    """Main validation function."""
    if len(sys.argv) != 4:
        print("Usage: python validate_yaml.py <output.yaml> <original.pdf> <README.md>")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    pdf_path = sys.argv[2]
    readme_path = sys.argv[3]
    
    try:
        print("[LOADING] Loading files...")
        yaml_data = load_yaml_file(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8', errors='ignore') as f:
            yaml_content = f.read()
        pdf_text = extract_pdf_text(pdf_path)
        with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
            readme_text = f.read()
        
        print("[AI] Running AI validation...")
        ai_results = validate_with_ai(yaml_content, pdf_text, readme_text)
        
        print("[SCHEMA] Running schema validation...")
        schema_results = schema_validation(yaml_data)
        
        print("[CROSS-REF] Cross-referencing pins...")
        pin_results = cross_reference_pins(yaml_data, pdf_text)
        
        # Generate comprehensive report
        generate_validation_report(yaml_path, ai_results)
        
        # Save detailed results
        results_file = yaml_path.replace('.yaml', '_validation_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'ai_validation': ai_results,
                'schema_validation': schema_results,
                'pin_cross_reference': pin_results
            }, f, indent=2)
        
        print(f"\n[SAVED] Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
