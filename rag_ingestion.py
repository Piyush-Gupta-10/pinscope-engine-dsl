#!/usr/bin/env python3
"""
RAG Ingestion System for PDF Processing

This module handles:
1. PDF text extraction
2. Intelligent chunking with overlap
3. Metadata preservation
4. Embedding generation
5. Vector database storage
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import google.genai as genai
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import json

# Vertex AI imports
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Image as VertexImage

# Load environment variables
load_dotenv()


@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata."""
    text: str
    page_number: int
    chunk_id: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class StructuredDocumentChunk:
    """Chunk with both text and structured data from vision extraction."""
    text: str  # Original OCR text
    structured_data: Dict  # JSON from Gemini Vision
    page_number: int
    chunk_id: str
    source_file: str
    confidence: float
    embedding: Optional[List[float]] = None


class VertexVisionExtractor:
    """Extracts structured data from PDF page images using Vertex AI Vision."""
    
    def __init__(self):
        # Initialize Vertex AI
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        try:
            aiplatform.init(project=project_id, location=location)
            self.model = GenerativeModel("gemini-2.0-flash-exp")
            print("Vertex AI Vision initialized successfully")
        except Exception as e:
            print(f"Vertex AI initialization failed: {e}")
            self.model = None
    
    def extract_structured_from_image(self, image: Image.Image, prompt: str) -> Dict:
        """Extract structured data from PDF page image using Vertex AI Vision."""
        try:
            # Rate limiting to avoid 429 errors (2 requests per minute)
            import time
            time.sleep(5)  # Wait 30 seconds between API calls
            # Convert PIL to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Debug: Check image properties
            print(f"  Image: {image.size}, Mode: {image.mode}, Size: {len(img_bytes)} bytes")
            
            if not self.model:
                raise ValueError("Vertex AI model not initialized")
            
            # Create Vertex AI Image object
            vertex_image = VertexImage.from_bytes(img_bytes)
            
            # Make API call with Vertex AI Vision
            response = self.model.generate_content([
                prompt,
                vertex_image
            ])
            
            # Debug: Log response
            print(f"  Response type: {type(response)}")
            print(f"  Response text preview: {response.text[:200]}...")
            
            # Parse JSON response
            if response and hasattr(response, 'text'):
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    # Clean response and retry
                    cleaned = self._clean_json_response(response.text)
                    return json.loads(cleaned)
            else:
                raise ValueError("No valid response received")
                
        except Exception as e:
            print(f"Error in Vertex AI vision extraction: {str(e)}")
            # Return fallback data with basic info
            return {
                "error": str(e), 
                "confidence": 0.0,
                "component_name": "extraction_failed",
                "component_type": "unknown",
                "extraction_method": "vertex_vision_failed"
            }
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response from Gemini."""
        # Remove code fences and extra text
        json_text = response_text.strip()
        json_text = json_text.replace('```json', '').replace('```', '')
        json_text = json_text.strip()
        return json_text
    
    def _calculate_confidence(self, structured_data: Dict) -> float:
        """Calculate confidence score based on extracted data completeness."""
        if "error" in structured_data:
            return 0.0
        
        # Count non-null fields
        total_fields = 0
        filled_fields = 0
        
        def count_fields(obj):
            nonlocal total_fields, filled_fields
            if isinstance(obj, dict):
                for key, value in obj.items():
                    total_fields += 1
                    if value and value != "" and value != {} and value != []:
                        filled_fields += 1
                    count_fields(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_fields(item)
        
        count_fields(structured_data)
        return filled_fields / total_fields if total_fields > 0 else 0.0



 
class VisionPDFProcessor:
    """Processes PDF pages with vision-based structured data extraction."""
    
    def __init__(self, readme_path: str = None):
        self.readme_content = self._load_readme(readme_path) if readme_path else ""
        self.vision_extractor = VertexVisionExtractor()
        self.extraction_prompt = self._build_vision_extraction_prompt(self.readme_content)
    
    def _load_readme(self, readme_path: str) -> str:
        """Load README content for context."""
        try:
            with open(readme_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Warning: Could not load README {readme_path}: {e}")
            return ""
    
    def _build_vision_extraction_prompt(self, readme_content: str) -> str:
        """Build prompt with README context for structured extraction."""
        return f"""
        You are extracting structured component data from a technical PDF page.
        
        README CONTEXT (defines what to extract):
        {readme_content}
        
        TASK:
        1. Analyze the PDF page image carefully
        2. Extract ONLY fields mentioned in README context
        3. Return as valid JSON object
        4. If data is not found, use null or omit the field
        5. Focus on technical specifications, pin configurations, and electrical characteristics
        
        FIELDS TO EXTRACT (based on README):
        - Component name/type
        - Electrical specifications (voltage, current, power consumption)
        - Physical characteristics (package type, pin count, dimensions)
        - Interface protocols (I2C, SPI, etc.)
        - Performance specifications (ranges, accuracy)
        - Pin configurations and functions
        - Operating conditions (temperature, pressure ranges)
        
        Return JSON object with nested structure:
        {{
            "component_name": "string",
            "component_type": "string",
            "electrical": {{
                "supply_voltage": {{"min": number, "max": number, "unit": "string"}},
                "current_consumption": {{"typical": number, "max": number, "unit": "string"}},
                "power_consumption": {{"typical": number, "unit": "string"}}
            }},
            "physical": {{
                "package": "string",
                "pins": number,
                "dimensions": {{"length": number, "width": number, "height": number, "unit": "string"}}
            }},
            "interfaces": ["string"],
            "performance": {{
                "operating_temperature": {{"min": number, "max": number, "unit": "string"}},
                "pressure_range": {{"min": number, "max": number, "unit": "string"}}
            }},
            "pin_configuration": {{"pin_number": "function"}},
            "additional_specs": {{"key": "value"}}
        }}
        """
    
    def pdf_page_to_image(self, pdf_path: str, page_num: int) -> Image.Image:
        """Convert PDF page to high-res image."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # High DPI for better OCR
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            print(f"Error converting page {page_num} to image: {e}")
            return None
    
    def extract_structured_from_pdf(self, pdf_path: str) -> List[StructuredDocumentChunk]:
        """Extract structured data from all PDF pages."""
        structured_chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                print(f"Processing page {page_num + 1}/{total_pages} with vision...")
                
                # Convert page to image
                image = self.pdf_page_to_image(pdf_path, page_num)
                if not image:
                    continue
                
                # Extract structured data
                page_data = self.vision_extractor.extract_structured_from_image(
                    image, self.extraction_prompt
                )
                
                # Calculate confidence
                confidence = self.vision_extractor._calculate_confidence(page_data)
                
                # Create searchable text from structured data
                searchable_text = self._create_searchable_text(page_data)
                
                # Generate chunk ID
                chunk_id = self._generate_chunk_id(
                    os.path.basename(pdf_path), page_num + 1, 0
                )
                
                # Create structured chunk
                chunk = StructuredDocumentChunk(
                    text=searchable_text,
                    structured_data=page_data,
                    page_number=page_num + 1,
                    chunk_id=chunk_id,
                    source_file=os.path.basename(pdf_path),
                    confidence=confidence
                )
                
                structured_chunks.append(chunk)
                print(f"Page {page_num + 1} extracted with confidence: {confidence:.2f}")
            
            doc.close()
            print(f"Successfully extracted structured data from {len(structured_chunks)} pages")
            
        except Exception as e:
            print(f"Error in vision-based PDF processing: {e}")
        
        return structured_chunks
    
    def _create_searchable_text(self, structured_data: Dict) -> str:
        """Create searchable text from structured data."""
        if "error" in structured_data:
            return ""
        
        text_parts = []
        
        def extract_text(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix} {key}" if prefix else key
                    extract_text(value, new_prefix)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item, prefix)
            elif obj:
                text_parts.append(f"{prefix}: {obj}")
        
        extract_text(structured_data)
        return " ".join(text_parts)
    
    def _generate_chunk_id(self, source_file: str, page_number: int, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{source_file}_{page_number}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class EmbeddingGenerator:
    """Handles embedding generation using Gemini model."""
    
    def __init__(self, model_name: str = "text-embedding-004"):
        print(f"Loading Gemini embedding model: {model_name}")
        
        # Get API key from environment
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print("Gemini embedding model loaded successfully")
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for all chunks using Gemini."""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Process in batches to respect rate limits
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            texts = [chunk.text for chunk in batch_chunks]
            
            # Generate embeddings
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            
            # Store embeddings
            for chunk, embedding in zip(batch_chunks, result.embeddings):
                chunk.embedding = embedding.values
                chunk.metadata['embedding_length'] = len(embedding.values)
        
        print("Embeddings generated successfully")
        return chunks


class VectorDatabase:
    """Handles vector database operations using Pinecone."""
    
    def __init__(self, collection_name: str = "pdf-chunks", embedding_generator=None, api_key=None):
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator
        
        # Get API key from parameter, environment variable, or raise error
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('PINECONE_API_KEY')
            if not self.api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables or .env file")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create or get index
        self.index_name = collection_name.replace("_", "-")
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        print(f"Connected to Pinecone index: {self.index_name}")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Gemini text-embedding-004 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created new Pinecone index: {self.index_name}")
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Store chunks in Pinecone vector database."""
        if not chunks:
            return
        
        # Clear existing vectors for this source file (only if index has data)
        source_file = chunks[0].source_file
        try:
            # Check if index has any data first
            stats = self.index.describe_index_stats()
            if stats.get('total_vector_count', 0) > 0:
                # Delete existing vectors for this source file
                self.index.delete(filter={"source_file": source_file})
                print(f"Cleared existing chunks for {source_file}")
        except Exception as e:
            # Ignore deletion errors for new indexes
            print(f"Info: No existing chunks to clear")
        
        # Prepare vectors for upsert
        vectors = []
        for chunk in chunks:
            # Use pre-computed embedding if available
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding. Call generate_embeddings first.")
            
            # Prepare metadata
            metadata = chunk.metadata.copy()
            metadata.update({
                'text': chunk.text,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'source_file': chunk.source_file
            })
            
            vectors.append({
                'id': chunk.chunk_id,
                'values': chunk.embedding,
                'metadata': metadata
            })
        
        # Upsert vectors in batches (Pinecone limit is 1000 per request)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Stored {len(chunks)} chunks in Pinecone")
    
    def store_structured_chunks(self, chunks: List[StructuredDocumentChunk]) -> None:
        """Store structured chunks in Pinecone vector database."""
        if not chunks:
            return
        
        # Clear existing vectors for this source file
        source_file = chunks[0].source_file
        try:
            stats = self.index.describe_index_stats()
            if stats.get('total_vector_count', 0) > 0:
                self.index.delete(filter={"source_file": source_file})
                print(f"Cleared existing structured chunks for {source_file}")
        except Exception as e:
            print(f"Info: No existing chunks to clear")
        
        # Prepare vectors for upsert
        vectors = []
        for chunk in chunks:
            # Use pre-computed embedding if available
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding. Call generate_embeddings first.")
            
            # Enhanced metadata for structured data
            metadata = {
                'original_text': chunk.text,
                'structured_data': json.dumps(chunk.structured_data),
                'page_number': chunk.page_number,
                'source_file': chunk.source_file,
                'confidence': chunk.confidence,
                'extraction_method': 'gemini_vision'
            }
            
            # Add extracted fields for filtering
            structured = chunk.structured_data
            if structured and 'error' not in structured:
                metadata.update({
                    'component_name': structured.get('component_name'),
                    'component_type': structured.get('component_type'),
                    'package': structured.get('physical', {}).get('package') if structured.get('physical') else None,
                    'pins': structured.get('physical', {}).get('pins') if structured.get('physical') else None,
                    'interfaces': structured.get('interfaces', []),
                    'supply_voltage_min': structured.get('electrical', {}).get('supply_voltage', {}).get('min') if structured.get('electrical') and structured.get('electrical', {}).get('supply_voltage') else None,
                    'supply_voltage_max': structured.get('electrical', {}).get('supply_voltage', {}).get('max') if structured.get('electrical') and structured.get('electrical', {}).get('supply_voltage') else None
                })
            
            vectors.append({
                'id': chunk.chunk_id,
                'values': chunk.embedding,
                'metadata': metadata
            })
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Stored {len(chunks)} structured chunks in Pinecone")
    
    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using Pinecone."""
        # Generate embedding for query using Gemini
        result = self.embedding_generator.client.models.embed_content(
            model=self.embedding_generator.model_name,
            contents=[query]
        )
        query_embedding = result.embeddings[0].values
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'text': match['metadata'].get('text', ''),
                'metadata': match['metadata'],
                'similarity': match['score']
            })
        
        return formatted_results


class RAGIngestionPipeline:
    """Vision-based RAG ingestion pipeline with structured data extraction."""
    
    def __init__(self, readme_path: str = None):
        self.vision_processor = VisionPDFProcessor(readme_path)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase(collection_name="pdf-chunks-vision", embedding_generator=self.embedding_generator)
    
    def process_pdf(self, pdf_path: str) -> List[StructuredDocumentChunk]:
        """Complete vision-based PDF processing pipeline."""
        print(f"Processing PDF with Vision: {pdf_path}")
        
        # Extract structured data using vision
        structured_chunks = self.vision_processor.extract_structured_from_pdf(pdf_path)
        print(f"Extracted structured data from {len(structured_chunks)} pages")
        
        # Filter out low-confidence chunks
        high_confidence_chunks = [chunk for chunk in structured_chunks if chunk.confidence > 0.3]
        print(f"Kept {len(high_confidence_chunks)} high-confidence chunks")
        
        if not high_confidence_chunks:
            print("Warning: No high-confidence chunks found")
            return []
        
        # Generate embeddings for structured chunks
        # Convert to list of DocumentChunk-like objects for embedding generation
        chunks_for_embedding = []
        for chunk in high_confidence_chunks:
            # Create temporary DocumentChunk for embedding generation
            temp_chunk = DocumentChunk(
                text=chunk.text,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                chunk_index=0,
                metadata={'confidence': chunk.confidence}
            )
            chunks_for_embedding.append(temp_chunk)
        
        # Generate embeddings
        chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks_for_embedding)
        
        # Copy embeddings back to structured chunks
        for i, structured_chunk in enumerate(high_confidence_chunks):
            structured_chunk.embedding = chunks_with_embeddings[i].embedding
        
        # Store in vector database
        self.vector_db.store_structured_chunks(high_confidence_chunks)
        
        return high_confidence_chunks
    
    def search_by_specifications(self, **filters) -> List[Dict[str, Any]]:
        """Search by structured specifications."""
        query_filter = {}
        
        if 'min_voltage' in filters:
            query_filter['supply_voltage_min'] = {'$gte': filters['min_voltage']}
        
        if 'max_voltage' in filters:
            query_filter['supply_voltage_max'] = {'$lte': filters['max_voltage']}
        
        if 'package_type' in filters:
            query_filter['package'] = filters['package_type']
        
        if 'component_type' in filters:
            query_filter['component_type'] = filters['component_type']
        
        if 'min_confidence' in filters:
            query_filter['confidence'] = {'$gte': filters['min_confidence']}
        
        # Generate a generic embedding for the search
        query_text = "component specifications technical details"
        result = self.embedding_generator.client.models.embed_content(
            model=self.embedding_generator.model_name,
            contents=[query_text]
        )
        query_embedding = result.embeddings[0].values
        
        # Search with filters
        results = self.vector_db.index.query(
            vector=query_embedding,
            top_k=filters.get('limit', 10),
            include_metadata=True,
            filter=query_filter if query_filter else None
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'text': match['metadata'].get('original_text', ''),
                'structured_data': json.loads(match['metadata'].get('structured_data', '{}')),
                'metadata': match['metadata'],
                'similarity': match['score']
            })
        
        return formatted_results


def main():
    """Test the vision-based ingestion pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python rag_ingestion.py <pdf_path> [readme_path]")
        print("Example: python rag_ingestion.py component_database/MPL3115A2/MPL3115A2.pdf component_database/README.md")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    readme_path = sys.argv[2] if len(sys.argv) > 2 else "component_database/README.md"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        # Use vision-based pipeline
        print("=== Vision-Based RAG Pipeline ===")
        pipeline = RAGIngestionPipeline(readme_path)
        structured_chunks = pipeline.process_pdf(pdf_path)
        
        print(f"\nSuccessfully processed {len(structured_chunks)} structured chunks")
        
        if structured_chunks:
            avg_confidence = sum(chunk.confidence for chunk in structured_chunks) / len(structured_chunks)
            print(f"Average confidence: {avg_confidence:.2f}")
            
            # Show sample structured data
            print(f"\nSample structured data from page 1:")
            sample_chunk = structured_chunks[0]
            print(json.dumps(sample_chunk.structured_data, indent=2)[:500] + "...")
            
            # Test structured search
            print("\n=== Testing Structured Search ===")
            results = pipeline.search_by_specifications(
                min_voltage=1.0,
                max_voltage=5.0,
                package_type="TSSOP",
                limit=3
            )
            
            print(f"\nFound {len(results)} components matching criteria:")
            for i, result in enumerate(results, 1):
                structured = result['structured_data']
                print(f"\n{i}. Similarity: {result['similarity']:.3f}")
                print(f"   Confidence: {result['metadata']['confidence']:.2f}")
                print(f"   Component: {structured.get('component_name', 'Unknown')}")
                print(f"   Package: {structured.get('physical', {}).get('package', 'Unknown')}")
                print(f"   Voltage: {structured.get('electrical', {}).get('supply_voltage', {})}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
