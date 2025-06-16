from typing import List, Dict, Any, Optional
import faiss
import numpy as np
import pickle
import os
from app.utils.logger import logger
from app.config.llm_config import llm_config
from datetime import datetime
import tiktoken

class VectorStoreService:
    def __init__(self):
        self.index = None
        self.documents = []
        self.dimension = None  # Will be determined dynamically
        self.metadata_index = {}  # For quick metadata lookups
        self._embedding_dimension_determined = False
        self.max_embedding_tokens = 8192  # Your model's limit
        self.tokenizer = None
        
    def _get_tokenizer(self):
        """Get tokenizer for counting tokens"""
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
            except:
                # Fallback to a generic tokenizer
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        return self.tokenizer
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))
    
    def _chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """Chunk text to fit within token limits"""
        if max_tokens is None:
            max_tokens = self.max_embedding_tokens - 100  # Leave some buffer
            
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
        
    def _determine_embedding_dimension(self):
        """Dynamically determine the embedding dimension from the model"""
        if self._embedding_dimension_determined:
            return
            
        try:
            # Test embedding to get actual dimension
            test_embedding = llm_config._embed_model.get_text_embedding("test")
            self.dimension = len(test_embedding)
            self._embedding_dimension_determined = True
            logger.info(f"Determined embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Could not determine embedding dimension: {e}")
            # Fallback to common Azure OpenAI dimension
            self.dimension = 1536
            self._embedding_dimension_determined = True
            logger.warning(f"Using fallback dimension: {self.dimension}")
        
    def create_index(self):
        """Create a new FAISS index"""
        if not self._embedding_dimension_determined:
            self._determine_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created FAISS index with dimension: {self.dimension}")
        
    def _validate_embedding_dimension(self, embedding: np.ndarray, context: str = ""):
        """Validate that embedding has the correct dimension"""
        if not self._embedding_dimension_determined:
            self._determine_embedding_dimension()
            
        if embedding.shape[0] != self.dimension:
            error_msg = (f"Embedding dimension mismatch {context}: "
                        f"expected {self.dimension}, got {embedding.shape[0]}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using LlamaIndex Azure OpenAI embedding model"""
        try:
            # Check token count and chunk if necessary
            token_count = self._count_tokens(text)
            if token_count > self.max_embedding_tokens:
                logger.warning(f"Text too long ({token_count} tokens), chunking...")
                chunks = self._chunk_text(text)
                # Use the first chunk for embedding (you might want to handle this differently)
                text = chunks[0]
                logger.warning(f"Using first chunk with {self._count_tokens(text)} tokens")
            
            # Use LlamaIndex's embedding model
            embedding = await llm_config._embed_model.aget_text_embedding(text)
            embedding_array = np.array(embedding).astype('float32')
            
            # Validate dimension
            self._validate_embedding_dimension(embedding_array, "in get_embedding")
            
            return embedding_array
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
        
    def get_embedding_sync(self, text: str) -> np.ndarray:
        """Get embedding synchronously using LlamaIndex"""
        try:
            # Check token count and chunk if necessary
            token_count = self._count_tokens(text)
            if token_count > self.max_embedding_tokens:
                logger.warning(f"Text too long ({token_count} tokens), chunking...")
                chunks = self._chunk_text(text)
                # Use the first chunk for embedding (you might want to handle this differently)
                text = chunks[0]
                logger.warning(f"Using first chunk with {self._count_tokens(text)} tokens")
            
            # Use LlamaIndex's embedding model synchronously
            embedding = llm_config._embed_model.get_text_embedding(text)
            embedding_array = np.array(embedding).astype('float32')
            
            # Validate dimension
            self._validate_embedding_dimension(embedding_array, "in get_embedding_sync")
            
            return embedding_array
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
        
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store with enhanced metadata and chunking"""
        if not documents:
            logger.warning("No documents provided to add_documents")
            return
            
        if not self.index:
            self.create_index()
            
        # Process documents and create chunks
        chunks_to_add = []
        
        for doc in documents:
            # Ensure metadata exists
            if "metadata" not in doc:
                doc["metadata"] = {}
                
            content = doc["content"]
            base_metadata = doc["metadata"].copy()
            
            # Check if content needs chunking
            token_count = self._count_tokens(content)
            
            if token_count <= self.max_embedding_tokens:
                # Content fits in one chunk
                chunk_metadata = base_metadata.copy()
                chunk_metadata["timestamp"] = datetime.now().isoformat()
                chunk_metadata["doc_id"] = len(self.documents) + len(chunks_to_add)
                chunk_metadata["chunk_index"] = 0
                chunk_metadata["total_chunks"] = 1
                chunk_metadata["token_count"] = token_count
                
                chunks_to_add.append({
                    "content": content,
                    "metadata": chunk_metadata
                })
            else:
                # Content needs chunking
                logger.info(f"Chunking document with {token_count} tokens into smaller pieces")
                text_chunks = self._chunk_text(content)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["timestamp"] = datetime.now().isoformat()
                    chunk_metadata["doc_id"] = len(self.documents) + len(chunks_to_add)
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(text_chunks)
                    chunk_metadata["token_count"] = self._count_tokens(chunk_text)
                    chunk_metadata["is_chunk"] = True
                    
                    chunks_to_add.append({
                        "content": chunk_text,
                        "metadata": chunk_metadata
                    })
        
        # Generate embeddings for chunks
        embeddings = []
        successful_chunks = []
        
        for i, chunk in enumerate(chunks_to_add):
            try:
                embedding = await self.get_embedding(chunk["content"])
                embeddings.append(embedding)
                successful_chunks.append(chunk)
                
                # Store in metadata index for quick lookups
                doc_id = chunk["metadata"]["doc_id"]
                self.metadata_index[doc_id] = chunk["metadata"]
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {i}: {e}")
                continue
        
        if embeddings and successful_chunks:
            # Add to FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            # Add successful chunks to documents list
            self.documents.extend(successful_chunks)
            
            logger.info(f"Added {len(successful_chunks)} document chunks to vector store")
        else:
            logger.warning("No embeddings generated, no documents added")
            
    def add_documents_sync(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store synchronously with chunking"""
        if not documents:
            logger.warning("No documents provided to add_documents_sync")
            return
            
        if not self.index:
            self.create_index()
            
        # Process documents and create chunks
        chunks_to_add = []
        
        for doc in documents:
            # Ensure metadata exists
            if "metadata" not in doc:
                doc["metadata"] = {}
                
            content = doc["content"]
            base_metadata = doc["metadata"].copy()
            
            # Check if content needs chunking
            token_count = self._count_tokens(content)
            
            if token_count <= self.max_embedding_tokens:
                # Content fits in one chunk
                chunk_metadata = base_metadata.copy()
                chunk_metadata["timestamp"] = datetime.now().isoformat()
                chunk_metadata["doc_id"] = len(self.documents) + len(chunks_to_add)
                chunk_metadata["chunk_index"] = 0
                chunk_metadata["total_chunks"] = 1
                chunk_metadata["token_count"] = token_count
                
                chunks_to_add.append({
                    "content": content,
                    "metadata": chunk_metadata
                })
            else:
                # Content needs chunking
                logger.info(f"Chunking document with {token_count} tokens into smaller pieces")
                text_chunks = self._chunk_text(content)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["timestamp"] = datetime.now().isoformat()
                    chunk_metadata["doc_id"] = len(self.documents) + len(chunks_to_add)
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(text_chunks)
                    chunk_metadata["token_count"] = self._count_tokens(chunk_text)
                    chunk_metadata["is_chunk"] = True
                    
                    chunks_to_add.append({
                        "content": chunk_text,
                        "metadata": chunk_metadata
                    })
        
        # Generate embeddings for chunks
        embeddings = []
        successful_chunks = []
        
        for i, chunk in enumerate(chunks_to_add):
            try:
                embedding = self.get_embedding_sync(chunk["content"])
                embeddings.append(embedding)
                successful_chunks.append(chunk)
                
                # Store in metadata index for quick lookups
                doc_id = chunk["metadata"]["doc_id"]
                self.metadata_index[doc_id] = chunk["metadata"]
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {i}: {e}")
                continue
        
        if embeddings and successful_chunks:
            # Add to FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            # Add successful chunks to documents list
            self.documents.extend(successful_chunks)
            
            logger.info(f"Added {len(successful_chunks)} document chunks to vector store")
        else:
            logger.warning("No embeddings generated, no documents added")
        
    async def search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents with optional filtering"""
        if not self.index:
            logger.warning("Index not initialized for search")
            return []
            
        # Check if index is empty
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no documents to search")
            return []
            
        try:
            # Generate query embedding
            query_embedding = await self.get_embedding(query)
            
            # Debug logging
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            logger.debug(f"Index dimension: {self.index.d}")
            logger.debug(f"Index total vectors: {self.index.ntotal}")
            
            # Prepare query vector for FAISS
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Ensure we don't search for more results than available
            search_k = min(k * 2, self.index.ntotal)
            
            # Search in FAISS
            distances, indices = self.index.search(query_vector, search_k)
            
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return []
        
        # Filter and process results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            metadata = doc["metadata"]
            
            # Apply filters if provided
            if filters:
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            result = {
                "content": doc["content"],
                "metadata": metadata,
                "score": float(1 / (1 + distance))  # Convert distance to similarity score
            }
            results.append(result)
            
            # Break if we have enough results after filtering
            if len(results) >= k:
                break
                
        return results
    
    def search_sync(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents synchronously"""
        if not self.index:
            logger.warning("Index not initialized for search")
            return []
            
        # Check if index is empty
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no documents to search")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.get_embedding_sync(query)
            
            # Debug logging
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            logger.debug(f"Index dimension: {self.index.d}")
            logger.debug(f"Index total vectors: {self.index.ntotal}")
            
            # Prepare query vector for FAISS
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Ensure we don't search for more results than available
            search_k = min(k * 2, self.index.ntotal)
            
            # Search in FAISS
            distances, indices = self.index.search(query_vector, search_k)
            
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return []
        
        # Filter and process results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            metadata = doc["metadata"]
            
            # Apply filters if provided
            if filters:
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            result = {
                "content": doc["content"],
                "metadata": metadata,
                "score": float(1 / (1 + distance))  # Convert distance to similarity score
            }
            results.append(result)
            
            # Break if we have enough results after filtering
            if len(results) >= k:
                break
                
        return results
    
    async def get_related_context(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Get related context with enhanced metadata and organization"""
        try:
            results = await self.search(query, k, filters)
        except Exception as e:
            logger.error(f"Error in get_related_context search: {e}")
            return {
                "context": "",
                "metadata": {},
                "related_files": []
            }
            
        if not results:
            return {
                "context": "",
                "metadata": {},
                "related_files": []
            }
        
        # Organize results by file
        file_contexts = {}
        for result in results:
            file_path = result["metadata"].get("file_path", "unknown")
            if file_path not in file_contexts:
                file_contexts[file_path] = {
                    "content": [],
                    "metadata": result["metadata"],
                    "score": result["score"]
                }
            file_contexts[file_path]["content"].append(result["content"])
        
        # Sort files by relevance score
        sorted_files = sorted(
            file_contexts.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Build context string
        context_parts = []
        for file_path, data in sorted_files:
            context_parts.append(f"File: {file_path}")
            
            metadata = data["metadata"]
            if "language" in metadata:
                context_parts.append(f"Language: {metadata['language']}")
            if "start_line" in metadata and "end_line" in metadata:
                context_parts.append(f"Lines: {metadata['start_line']}-{metadata['end_line']}")
            if metadata.get("is_chunk"):
                context_parts.append(f"Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}")
                
            context_parts.append("Content:")
            context_parts.extend(data["content"])
            context_parts.append("---")
        
        return {
            "context": "\n".join(context_parts),
            "metadata": {
                "total_files": len(file_contexts),
                "total_chunks": len(results),
                "avg_score": sum(r["score"] for r in results) / len(results)
            },
            "related_files": [
                {
                    "file_path": file_path,
                    "language": data["metadata"].get("language", "unknown"),
                    "score": data["score"]
                }
                for file_path, data in sorted_files
            ]
        }
    
    def save(self, directory: str):
        """Save the vector store to disk"""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
            
            # Save documents and metadata index
            with open(os.path.join(directory, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            with open(os.path.join(directory, "metadata_index.pkl"), "wb") as f:
                pickle.dump(self.metadata_index, f)
                
            # Save dimension info
            with open(os.path.join(directory, "dimension.txt"), "w") as f:
                f.write(str(self.dimension))
                
            logger.info(f"Saved vector store to {directory}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
            
    def load(self, directory: str):
        """Load the vector store from disk"""
        index_path = os.path.join(directory, "index.faiss")
        documents_path = os.path.join(directory, "documents.pkl")
        metadata_path = os.path.join(directory, "metadata_index.pkl")
        dimension_path = os.path.join(directory, "dimension.txt")
        
        try:
            if all(os.path.exists(p) for p in [index_path, documents_path, metadata_path]):
                # Load dimension first
                if os.path.exists(dimension_path):
                    with open(dimension_path, "r") as f:
                        self.dimension = int(f.read().strip())
                        self._embedding_dimension_determined = True
                else:
                    # Fallback: determine dimension from model
                    self._determine_embedding_dimension()
                
                self.index = faiss.read_index(index_path)
                
                # Validate loaded index dimension matches expected dimension
                if self.index.d != self.dimension:
                    logger.warning(f"Loaded index dimension ({self.index.d}) doesn't match "
                                 f"expected dimension ({self.dimension}). Recreating index.")
                    self.create_index()
                    return
                
                with open(documents_path, "rb") as f:
                    self.documents = pickle.load(f)
                with open(metadata_path, "rb") as f:
                    self.metadata_index = pickle.load(f)
                    
                logger.info(f"Loaded vector store from {directory} with {len(self.documents)} documents")
            else:
                logger.warning(f"Vector store files not found in {directory}")
                self.create_index()
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.create_index()