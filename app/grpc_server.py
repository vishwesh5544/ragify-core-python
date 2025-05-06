"""
gRPC server implementation for the RAGify service.
Handles document processing and question-answering requests.
"""

import asyncio
import grpc
from concurrent import futures
from typing import List, Dict
from loguru import logger
import json
from .services.rag_engine import RAGEngine
from .config import settings

# Note: In production, you would import the generated protobuf classes
# from your shared proto package. For this example, we'll define stub classes.
class RagServiceServicer:
    """
    gRPC service implementation for RAG operations.
    """
    
    def __init__(self):
        """Initialize the gRPC servicer with RAG engine."""
        self.rag_engine = RAGEngine()
        
    async def ProcessDocument(
        self,
        request,
        context: grpc.aio.ServicerContext
    ):
        """
        Process a document and store its chunks in the vector database.
        """
        try:
            # Extract request parameters
            file_path = request.file_path
            metadata = json.loads(request.metadata) if request.metadata else None
            
            # Process document
            chunk_ids = await self.rag_engine.process_document(
                file_path=file_path,
                metadata=metadata
            )
            
            # Return response
            return {
                'chunk_ids': chunk_ids,
                'success': True,
                'message': f'Successfully processed document with {len(chunk_ids)} chunks'
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {
                'success': False,
                'message': f'Document processing failed: {str(e)}'
            }
            
    async def AnswerQuestion(
        self,
        request,
        context: grpc.aio.ServicerContext
    ):
        """
        Answer a question using RAG.
        """
        try:
            # Extract request parameters
            question = request.question
            num_chunks = request.num_chunks or 3
            temperature = request.temperature or 0.7
            
            # Generate answer
            result = await self.rag_engine.answer_question(
                question=question,
                num_chunks=num_chunks,
                temperature=temperature
            )
            
            # Return response
            return {
                'answer': result['answer'],
                'sources': json.dumps(result['sources']),
                'model': result['model'],
                'chunks_used': result['chunks_used'],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {
                'success': False,
                'message': f'Question answering failed: {str(e)}'
            }
            
    async def SearchDocuments(
        self,
        request,
        context: grpc.aio.ServicerContext
    ):
        """
        Search for relevant documents.
        """
        try:
            # Extract request parameters
            query = request.query
            limit = request.limit or 5
            filters = json.loads(request.filters) if request.filters else None
            
            # Search documents
            results = await self.rag_engine.search_documents(
                query=query,
                limit=limit,
                filter_conditions=filters
            )
            
            # Return response
            return {
                'results': json.dumps(results),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {
                'success': False,
                'message': f'Document search failed: {str(e)}'
            }

async def serve():
    """
    Start the gRPC server.
    """
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    
    # Register servicer
    # In production, use your generated proto:
    # rag_pb2_grpc.add_RagServiceServicer_to_server(
    #     RagServiceServicer(), server
    # )
    
    # Add insecure port for development
    # In production, use SSL/TLS:
    server.add_insecure_port(f'{settings.SERVICE_HOST}:{settings.SERVICE_PORT}')
    
    logger.info(f'Starting gRPC server on port {settings.SERVICE_PORT}')
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info('Shutting down gRPC server')
        await server.stop(0)

if __name__ == '__main__':
    # Configure logging
    logger.add(
        'logs/grpc_server.log',
        rotation='500 MB',
        retention='10 days',
        level=settings.LOG_LEVEL
    )
    
    # Start server
    asyncio.run(serve())