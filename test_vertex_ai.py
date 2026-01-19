#!/usr/bin/env python3
"""
Test Vertex AI setup
"""

import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

def test_vertex_ai():
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize Vertex AI
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'faradworks-volt-dev')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        print(f"Testing Vertex AI with project: {project_id}")
        print(f"Location: {location}")
        print(f"Service account: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
        
        aiplatform.init(project=project_id, location=location)
        
        # Test model
        model = GenerativeModel("gemini-2.0-flash-exp")
        
        # Simple test
        response = model.generate_content("Hello, can you see this message?")
        
        print("SUCCESS: Vertex AI is working!")
        print(f"Response: {response.text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Vertex AI setup failed: {e}")
        return False

if __name__ == "__main__":
    test_vertex_ai()
