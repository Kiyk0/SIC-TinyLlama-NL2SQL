"""
Embedding and Vector Store Pipeline for NL-to-SQL System
Creates embeddings from schema metadata and builds searchable vector index
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
import pickle


@dataclass
class SchemaDocument:
    """A searchable document representing schema element"""
    doc_id: str
    doc_type: str  # 'table', 'column', 'relationship'
    table_name: str
    content: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None


class EmbeddingGenerator:
    """Generates embeddings for schema documents"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Options:
        - sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast, free)
        - sentence-transformers/all-mpnet-base-v2 (768 dims, better quality)
        - OpenAI text-embedding-3-small (via API)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


class SchemaDocumentBuilder:
    """Builds searchable documents from schema metadata"""
    
    def __init__(self, schema_metadata: Dict[str, Any]):
        self.schema_metadata = schema_metadata
        self.documents: List[SchemaDocument] = []
    
    def build_all_documents(self) -> List[SchemaDocument]:
        """Build all document types"""
        print("Building schema documents...")
        
        for table in self.schema_metadata['tables']:
            # Table-level documents
            self.documents.extend(self._build_table_documents(table))
            
            # Column-level documents
            self.documents.extend(self._build_column_documents(table))
            
            # Relationship documents
            self.documents.extend(self._build_relationship_documents(table))
        
        print(f"✓ Built {len(self.documents)} documents")
        return self.documents
    
    def _build_table_documents(self, table: Dict) -> List[SchemaDocument]:
        """Create searchable documents for table"""
        docs = []
        table_name = table['table_name']
        
        # Main table document
        content = self._format_table_content(table)
        docs.append(SchemaDocument(
            doc_id=f"table_{table_name}",
            doc_type="table",
            table_name=table_name,
            content=content,
            metadata={
                'row_count': table['row_count'],
                'column_count': len(table['columns']),
                'common_queries': table.get('common_queries', [])
            }
        ))
        
        # Business context document (if available)
        if table.get('business_context'):
            docs.append(SchemaDocument(
                doc_id=f"table_{table_name}_context",
                doc_type="table_context",
                table_name=table_name,
                content=f"Business Context: {table['business_context']}",
                metadata={'parent_table': table_name}
            ))
        
        return docs
    
    def _format_table_content(self, table: Dict) -> str:
        """Format table information into searchable text"""
        parts = [
            f"Table: {table['table_name']}",
            f"Description: {table['description']}",
            f"Contains {table['row_count']} rows",
        ]
        
        # Add column names for better matching
        col_names = [col['name'] for col in table['columns']]
        parts.append(f"Columns: {', '.join(col_names)}")
        
        # Add common queries
        if table.get('common_queries'):
            parts.append("Common use cases:")
            parts.extend([f"  - {q}" for q in table['common_queries']])
        
        return "\n".join(parts)
    
    def _build_column_documents(self, table: Dict) -> List[SchemaDocument]:
        """Create documents for each column"""
        docs = []
        table_name = table['table_name']
        
        for col in table['columns']:
            content = self._format_column_content(table_name, col)
            
            docs.append(SchemaDocument(
                doc_id=f"column_{table_name}_{col['name']}",
                doc_type="column",
                table_name=table_name,
                content=content,
                metadata={
                    'column_name': col['name'],
                    'data_type': col['data_type'],
                    'is_primary_key': col['is_primary_key'],
                    'is_foreign_key': col['is_foreign_key'],
                    'foreign_key_ref': col.get('foreign_key_ref'),
                    'sample_values': col.get('sample_values', [])
                }
            ))
        
        return docs
    
    def _format_column_content(self, table_name: str, col: Dict) -> str:
        """Format column information into searchable text"""
        parts = [
            f"Column: {table_name}.{col['name']}",
            f"Type: {col['data_type']}",
        ]
        
        if col.get('description'):
            parts.append(f"Description: {col['description']}")
        
        if col['is_primary_key']:
            parts.append("Primary Key")
        
        if col['is_foreign_key'] and col.get('foreign_key_ref'):
            parts.append(f"Foreign Key → {col['foreign_key_ref']}")
        
        if col.get('sample_values'):
            sample_str = ', '.join(str(v) for v in col['sample_values'][:3])
            parts.append(f"Sample values: {sample_str}")
        
        return "\n".join(parts)
    
    def _build_relationship_documents(self, table: Dict) -> List[SchemaDocument]:
        """Create documents for table relationships"""
        docs = []
        table_name = table['table_name']
        
        for idx, rel in enumerate(table.get('relationships', [])):
            content = self._format_relationship_content(table_name, rel)
            
            docs.append(SchemaDocument(
                doc_id=f"relationship_{table_name}_{idx}",
                doc_type="relationship",
                table_name=table_name,
                content=content,
                metadata={
                    'relationship_type': rel['type'],
                    'to_table': rel['to_table'],
                    'join_condition': rel['join_condition']
                }
            ))
        
        return docs
    
    def _format_relationship_content(self, table_name: str, rel: Dict) -> str:
        """Format relationship into searchable text"""
        return (
            f"Relationship: {table_name} to {rel['to_table']}\n"
            f"Type: {rel['type']}\n"
            f"Join: {rel['join_condition']}"
        )


class VectorStore:
    """FAISS-based vector store for schema retrieval"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.documents: List[SchemaDocument] = []
        self.doc_id_to_idx: Dict[str, int] = {}
    
    def add_documents(self, documents: List[SchemaDocument]):
        """Add documents with embeddings to index"""
        embeddings = np.array([doc.embedding for doc in documents]).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and mapping
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.documents.append(doc)
            self.doc_id_to_idx[doc.doc_id] = idx
        
        print(f"✓ Added {len(documents)} documents to vector store")
        print(f"  Total documents: {len(self.documents)}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        filter_type: str = None,
        filter_table: str = None
    ) -> List[Tuple[SchemaDocument, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_type: Filter by doc_type ('table', 'column', 'relationship')
            filter_table: Filter by table_name
        """
        # Search in FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k * 3, len(self.documents)))
        
        # Get results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            doc = self.documents[idx]
            
            # Apply filters
            if filter_type and doc.doc_type != filter_type:
                continue
            if filter_table and doc.table_name != filter_table:
                continue
            
            # Convert L2 distance to similarity score (higher is better)
            similarity = 1 / (1 + dist)
            results.append((doc, similarity))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents and metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_id_to_idx': self.doc_id_to_idx,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"✓ Vector store saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VectorStore':
        """Load vector store from disk"""
        # Load FAISS index
        index = faiss.read_index(f"{filepath}.faiss")
        
        # Load documents and metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct vector store
        store = cls(data['embedding_dim'])
        store.index = index
        store.documents = data['documents']
        store.doc_id_to_idx = data['doc_id_to_idx']
        
        print(f"✓ Vector store loaded from {filepath}")
        print(f"  Total documents: {len(store.documents)}")
        return store


class EmbeddingPipeline:
    """End-to-end pipeline for building vector store"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = EmbeddingGenerator(model_name)
        self.vector_store = VectorStore(self.embedder.embedding_dim)
    
    def build_from_schema(self, schema_metadata_file: str):
        """Build vector store from schema metadata JSON"""
        print(f"\n{'='*60}")
        print("Building Vector Store from Schema Metadata")
        print(f"{'='*60}\n")
        
        # Load schema metadata
        with open(schema_metadata_file, 'r') as f:
            schema_metadata = json.load(f)
        
        # Build documents
        doc_builder = SchemaDocumentBuilder(schema_metadata)
        documents = doc_builder.build_all_documents()
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed_batch(texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        # Build vector store
        print("\nBuilding vector index...")
        self.vector_store.add_documents(documents)
        
        print(f"\n{'='*60}")
        print("✓ Vector store built successfully!")
        print(f"{'='*60}\n")
        
        return self.vector_store
    
    def save_vector_store(self, filepath: str = "schema_vector_store"):
        """Save vector store to disk"""
        self.vector_store.save(filepath)
    
    def test_retrieval(self, query: str, top_k: int = 5):
        """Test retrieval with a sample query"""
        print(f"\n{'='*60}")
        print(f"Test Query: '{query}'")
        print(f"{'='*60}\n")
        
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Search
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Display results
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. [{doc.doc_type}] {doc.table_name} (score: {score:.3f})")
            print(f"   {doc.content[:150]}...")
            print()


# Example usage
if __name__ == "__main__":
    # Build pipeline
    pipeline = EmbeddingPipeline(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Build vector store from extracted schema
    vector_store = pipeline.build_from_schema("elearning_schema_metadata.json")
    
    # Save for later use
    pipeline.save_vector_store("elearning_vector_store")
    
    # Test retrieval
    print("\n" + "="*60)
    print("Testing Retrieval")
    print("="*60)
    
    test_queries = [
        "Find students who completed courses",
        "Show course ratings and reviews",
        "Get instructor information and experience",
        "Display quiz scores and performance",
        "Find courses by category"
    ]
    
    for query in test_queries:
        pipeline.test_retrieval(query, top_k=3)
        print()
    
    print("\n✓ Pipeline complete! Vector store ready for use.")