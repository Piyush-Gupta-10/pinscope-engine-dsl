#!/usr/bin/env python3
"""
Force Reset Pinecone Database Utility

This script completely clears the Pinecone vector database
without requiring interactive confirmation.
"""

import os
import sys
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def force_reset_pinecone_database(collection_name: str = "pdf-chunks"):
    """Force reset Pinecone database by deleting and recreating the index."""
    
    # Get API key from environment variable
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("Error: PINECONE_API_KEY not found in environment variables or .env file")
        sys.exit(1)
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        index_name = collection_name.replace("_", "-")
        
        print(f"Force resetting Pinecone index: {index_name}")
        
        # Check if index exists
        if index_name in [index.name for index in pc.list_indexes()]:
            print(f"Deleting existing index: {index_name}")
            pc.delete_index(index_name)
            print("Index deleted successfully")
        else:
            print(f"Index {index_name} does not exist - nothing to delete")
        
        print("Database reset completed!")
        print("Next run of app_rag.py will recreate the index with Docling-processed chunks")
        
        return True
        
    except Exception as e:
        print(f"Error resetting database: {str(e)}")
        return False

def main():
    """Main function."""
    collection_name = "pdf-chunks"
    
    print("=" * 50)
    print("Force Pinecone Database Reset")
    print("=" * 50)
    print("This will permanently delete all stored chunks!")
    print()
    
    success = force_reset_pinecone_database(collection_name)
    
    if success:
        print("Reset completed successfully!")
        print("You can now run: python app_rag.py <pdf> <readme>")
    else:
        print("Reset failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
