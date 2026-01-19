#!/usr/bin/env python3
"""
Test script for RAG pipeline
"""

import os
import sys
from rag_ingestion import RAGIngestionPipeline
from rag_retrieval import RAGRetriever


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("🧪 Testing RAG Pipeline")
    print("=" * 40)
    
    # Test 1: Check if we can import everything
    print("Test 1: Importing modules...")
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Check if embedding model loads
    print("\nTest 2: Loading embedding model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False
    
    # Test 3: Test embedding generation
    print("\nTest 3: Testing embedding generation...")
    try:
        test_text = "MPL3115A2 pressure sensor specifications"
        embedding = model.encode(test_text)
        print(f"✅ Embedding generated: {embedding.shape}")
    except Exception as e:
        print(f"❌ Embedding generation error: {e}")
        return False
    
    # Test 4: Test vector database connection
    print("\nTest 4: Testing vector database...")
    try:
        retriever = RAGRetriever()
        stats = retriever.get_collection_stats()
        print(f"✅ Vector database connected: {stats}")
    except Exception as e:
        print(f"❌ Vector database error: {e}")
        return False
    
    print("\n🎉 All tests passed! RAG pipeline is ready.")
    return True


def show_usage_examples():
    """Show usage examples for the RAG pipeline."""
    print("\n📚 Usage Examples:")
    print("=" * 40)
    
    print("\n1. Process a PDF (one-time setup):")
    print("   python rag_ingestion.py your_rules.pdf")
    
    print("\n2. Test retrieval with README:")
    print("   python rag_retrieval.py your_README.md")
    
    print("\n3. Generate YAML with RAG:")
    print("   python app_rag.py your_rules.pdf your_README.md")
    
    print("\n📁 Files Created:")
    print("   - rag_ingestion.py    (PDF processing)")
    print("   - rag_retrieval.py    (Context retrieval)")
    print("   - app_rag.py          (RAG YAML generation)")
    print("   - chroma_db/          (Vector database)")
    
    print("\n🔧 Requirements:")
    print("   - sentence-transformers")
    print("   - chromadb")
    print("   - numpy")
    print("   - (existing: PyPDF2, vertexai, PyYAML)")


if __name__ == "__main__":
    success = test_rag_pipeline()
    show_usage_examples()
    
    if success:
        print("\n✅ RAG pipeline is ready to use!")
    else:
        print("\n❌ Please fix the errors before using the pipeline")
        sys.exit(1)
