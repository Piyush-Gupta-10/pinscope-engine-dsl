#!/usr/bin/env python3
"""
RAG Retrieval System

This module handles:
1. Query generation from README
2. Vector database search
3. Relevant chunk retrieval
4. Context preparation for Gemini
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import google.genai as genai
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RetrievalResult:
    """Represents a retrieval result with context."""
    query: str
    relevant_chunks: List[Dict[str, Any]]
    context: str
    metadata: Dict[str, Any]


class QueryGenerator:
    """Generates semantic queries from README content."""
    
    def __init__(self):
        self.query_patterns = [
            "component specifications and technical details",
            "electrical characteristics and voltage requirements", 
            "pin configuration and interface protocols",
            "operating parameters and performance specifications",
            "communication protocols and data formats",
            "physical dimensions and mechanical properties",
            "environmental specifications and operating conditions"
        ]
    
    def generate_queries_from_readme(self, readme_path: str) -> List[str]:
        """Generate intent-driven queries based on README analysis."""
        try:
            with open(readme_path, 'r', encoding='utf-8') as file:
                readme_content = file.read()
            
            # Extract README's main intent
            intent = self._extract_readme_intent(readme_content)
            print(f"README Intent: {intent}")
            
            # Generate 3-5 strong, intent-driven queries
            queries = self._generate_intent_queries(intent, readme_content)
            
            return queries
            
        except Exception as e:
            print(f"Warning: Could not read README for query generation: {e}")
            return ["component specifications and technical details"]
    
    def _extract_readme_intent(self, readme_content: str) -> str:
        """Extract the main intent from README content."""
        
        # Look for key sections that indicate what's needed
        if "Component YAML" in readme_content and "DSL rules" in readme_content:
            return "component_definition_with_requirements"
        
        if "packages" in readme_content.lower() and "pins" in readme_content.lower():
            return "package_pin_mapping"
        
        if "requirements" in readme_content.lower() and "rules" in readme_content.lower():
            return "pin_requirements_dsl"
        
        if "interface" in readme_content.lower() or "communication" in readme_content.lower():
            return "interface_communication"
        
        # Default to general component definition
        return "general_component_specification"
    
    def _generate_intent_queries(self, intent: str, readme_content: str) -> List[str]:
        """Generate strong queries based on README intent."""
        
        if intent == "component_definition_with_requirements":
            return [
                "component pin definitions and electrical characteristics",
                "power supply requirements and voltage specifications", 
                "interface protocols and communication pin configurations",
                "package pin mapping and physical layout",
                "component requirements and design rules"
            ]
        
        elif intent == "package_pin_mapping":
            return [
                "package pin definitions and mapping",
                "physical pin locations and assignments",
                "pin numbering and package specifications"
            ]
        
        elif intent == "pin_requirements_dsl":
            return [
                "pin electrical requirements and constraints",
                "power supply specifications and voltage levels",
                "interface pin configurations and protocols",
                "component design rules and requirements"
            ]
        
        elif intent == "interface_communication":
            return [
                "communication interface specifications",
                "protocol pin configurations and requirements",
                "signal interface definitions and characteristics"
            ]
        
        else:  # general_component_specification
            return [
                "component specifications and electrical characteristics",
                "power requirements and voltage specifications",
                "interface configurations and communication protocols",
                "pin definitions and package mapping"
            ]


class RAGRetriever:
    """Handles retrieval of relevant chunks from Pinecone vector database."""
    
    def __init__(self, collection_name: str = "pdf-chunks-vision"):
        self.collection_name = collection_name
        
        # Get API keys from environment variables
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables or .env file")
        
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Get index
        self.index_name = collection_name.replace("_", "-")
        self.index = self.pc.Index(self.index_name)
        
        # Initialize Gemini client
        print(f"Loading Gemini embedding model for retrieval...")
        self.client = genai.Client(api_key=google_api_key)
        self.model_name = "text-embedding-004"
        print("Gemini embedding model loaded for retrieval")
        
        self.query_generator = QueryGenerator()
        print(f"Connected to vector database for retrieval: {collection_name}")
    
    def search_relevant_chunks(self, query: str, n_results: int = 5, min_similarity: float = 0.3, source_file: str = None) -> List[Dict[str, Any]]:
        """Search for relevant chunks with similarity filtering using Pinecone."""
        try:
            # Generate embedding for query using Gemini
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[query]
            )
            query_embedding = result.embeddings[0].values
            
            # Build filter for source file if specified
            filter_dict = {}
            if source_file:
                filter_dict["source_file"] = source_file
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=n_results * 2,  # Get more to filter
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            filtered_results = []
            for match in results['matches']:
                similarity = match['score']
                if similarity >= min_similarity:
                    filtered_results.append({
                        'text': match['metadata'].get('text', ''),
                        'metadata': match['metadata'],
                        'similarity': similarity
                    })
            
            # Sort by similarity and return top n_results
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
            return filtered_results[:n_results]
            
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []
    
    def retrieve_context_for_yaml_generation(self, readme_path: str, n_chunks_per_query: int = 3, source_file: str = None) -> RetrievalResult:
        """Retrieve relevant context for YAML generation."""
        print("Generating queries from README...")
        queries = self.query_generator.generate_queries_from_readme(readme_path)
        
        all_relevant_chunks = []
        query_results = {}
        
        print(f"Searching with {len(queries)} queries...")
        for i, query in enumerate(queries):
            print(f"   Query {i+1}/{len(queries)}: {query[:50]}...")
            chunks = self.search_relevant_chunks(query, n_chunks_per_query, source_file=source_file)
            query_results[query] = chunks
            all_relevant_chunks.extend(chunks)
        
        # Remove duplicates based on text content
        unique_chunks = []
        seen_texts = set()
        for chunk in all_relevant_chunks:
            if chunk['text'] not in seen_texts:
                unique_chunks.append(chunk)
                seen_texts.add(chunk['text'])
        
        # Sort by similarity and take top chunks
        unique_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        final_chunks = unique_chunks[:10]  # Top 10 most relevant chunks
        
        # Create context string
        context = self._create_context_string(final_chunks)
        
        metadata = {
            'total_queries': len(queries),
            'total_chunks_found': len(all_relevant_chunks),
            'unique_chunks': len(unique_chunks),
            'final_chunks': len(final_chunks),
            'average_similarity': sum(c['similarity'] for c in final_chunks) / len(final_chunks) if final_chunks else 0
        }
        
        print(f"Retrieved {len(final_chunks)} relevant chunks (avg similarity: {metadata['average_similarity']:.2f})")
        
        return RetrievalResult(
            query=f"YAML generation based on {readme_path}",
            relevant_chunks=final_chunks,
            context=context,
            metadata=metadata
        )
    
    def _create_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        """Create a formatted context string from chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Chunk {i} - Page {chunk['metadata'].get('page_number', 'Unknown')} - "
                f"Similarity: {chunk['similarity']:.2f}]\n"
                f"{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            return {
                'total_chunks': stats.get('total_vector_count', 0),
                'index_name': self.index_name,
                'dimension': stats.get('dimension', 'unknown')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def has_chunks_for_pdf(self, pdf_filename: str) -> bool:
        """Check if chunks exist for a specific PDF file."""
        try:
            # Search for chunks with this specific source_file
            result = self.index.query(
                vector=[0.0] * 768,  # Dummy vector for metadata filtering
                top_k=1,
                include_metadata=True,
                filter={"source_file": pdf_filename}
            )
            return len(result['matches']) > 0
        except Exception as e:
            print(f"Error checking PDF chunks: {e}")
            return False


def main():
    """Test the retrieval system."""
    if len(sys.argv) != 2:
        print("Usage: python rag_retrieval.py <readme_path>")
        sys.exit(1)
    
    readme_path = sys.argv[1]
    
    if not os.path.exists(readme_path):
        print(f"README file not found: {readme_path}")
        sys.exit(1)
    
    try:
        retriever = RAGRetriever()
        
        # Show collection stats
        stats = retriever.get_collection_stats()
        print(f"Collection Stats: {stats}")
        
        # Retrieve context
        result = retriever.retrieve_context_for_yaml_generation(readme_path)
        
        print("\nContext Preview:")
        print("=" * 50)
        print(result.context[:500] + "..." if len(result.context) > 500 else result.context)
        print("=" * 50)
        
        print(f"\nRetrieval Metadata:")
        for key, value in result.metadata.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
