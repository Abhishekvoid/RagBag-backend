import asyncio
from asgiref.sync import async_to_sync
from qdrant_client.http.exceptions import UnexpectedResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics, permissions
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializers, ChatMessageSerializer, ChatSessionSerializer, DocumentSerializer, SubjectWriteSerializer, SubjectReadSerializer, ChapterReadSerializer, ChapterWriteSerializer,  RAGChatMessageSerializer
import logging
from django.core.exceptions import ValidationError
from rest_framework.throttling import UserRateThrottle
from rest_framework.permissions import IsAuthenticated
from .models import ChatMessage, ChatSession, Document, Subject, Chapter
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models, AsyncQdrantClient
import google.generativeai as genai
from groq import Groq, AsyncGroq
from .tasks import process_document_ingestion, create_chapter_from_document
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.pagination import PageNumberPagination


logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "studywise_documents"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "llama3-70b-8192"



try: 
    if not GOOGLE_API_KEY:
        raise ValueError("google api key is not set")
    genai.configure(api_key=GOOGLE_API_KEY)

    if not GROQ_API_KEY:
        raise ValueError("groq api key is not set")
    groq_client = Groq(api_key=GROQ_API_KEY)

    qdrant_client = QdrantClient(QDRANT_URL)
except Exception as e:
    logger.critical(f"Failed to initialize AI clients in views.py: {e}")
    groq_client = None
    qdrant_client = None

async_qdrant_client = AsyncQdrantClient(QDRANT_URL)
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)


async def expand_queries_async(query: str, num: int = 4) -> list[str]:
    expansion_prompt = f"Generate {num} alternative phrasings of the following query for retrieval:\n\n{query}"
    # You must `await` the async client call.
    completion = await async_groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": expansion_prompt}]
    )
    expanded = completion.choices[0].message.content.strip().split("\n")
    return [q.strip("-• ") for q in expanded if q.strip()]

async def generate_rag_response(query: str, user_id: str, chapter_id: str):
    """
    Performs the full RAG pipeline, now scoped to a specific chapter.
    """


    try:
        count_result = await async_qdrant_client.count(
            collection_name=QDRANT_COLLECTION_NAME,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="chapter_id", match=models.MatchValue(value=str(chapter_id)))]
            ),
            exact=False
        )
        if count_result.count == 0:
            logger.warning(f"SELF-HEALING: No vectors found for COMPLETED chapter {chapter_id}. Triggering re-ingestion.")
            try:
                doc_to_reingest = await asyncio.to_thread(Document.objects.get, chapter__id=chapter_id)
                process_document_ingestion.delay(str(doc_to_reingest.id))
                return "The data for this chapter is being refreshed. Please try your question again in a minute."
            except Document.DoesNotExist:
                return "Sorry, the source document for this chapter could not be found. Please re-upload it."
    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.warning(f"SELF-HEALING: Collection does not exist. Triggering re-ingestion for chapter {chapter_id}.")
            try:
                doc_to_reingest = await asyncio.to_thread(Document.objects.get, chapter__id=chapter_id)
                process_document_ingestion.delay(str(doc_to_reingest.id))
                return "The workspace is being initialized. Please try your question again in a minute."
            except Document.DoesNotExist:
                return "Sorry, the source document for this chapter could not be found. Please re-upload it."
        else:
            raise e
    expanded_queries = await expand_queries_async(query, num=4)
    all_queries = [query] + expanded_queries

    logger.info(f"Batch Embedding{len(all_queries)} queries...")
    embedding_response = await asyncio.to_thread(
        genai.embed_content,
        model=f"models/{EMBEDDING_MODEL}",
        content=all_queries,
        task_type="RETRIEVAL_QUERY"
    )
    all_embeddings = embedding_response['embedding']

    search_filter = models.Filter (
        must = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id))),
            models.FieldCondition(key="chapter_id", match=models.MatchValue(value=str(chapter_id)))
        ]
    )

    search_requests = [
        models.SearchRequest(vector=vector, filter=search_filter, limit=5)
        for vector in all_embeddings
    ]


    logger.info(f"Batch searching Qdrant with {len(search_requests)} requests...")
    try :
        all_search_results = await async_qdrant_client.search_batch(
            collection_name= QDRANT_COLLECTION_NAME,
            requests=search_requests
        )
    except UnexpectedResponse as e:
        if e.status_code == 404:
            # Handle the specific 404 error gracefully
            return "Sorry, the data for this chapter is missing. Please re-upload the source document to enable chat."
        else:
            # Re-raise any other unexpected error
            raise e

    flat_results = [result for sublist in all_search_results for result in sublist]
   
    seen = set()
    unique_results = []
   
    for r in flat_results:
        if r.payload['text'] not in seen:
            seen.add(r.payload['text'])
            unique_results.append(r)

    sorted_results = sorted(unique_results, key=lambda r: r.score, reverse=True)
    context = "\n\n---\n\n".join([r.payload['text'] for r in sorted_results[:10]])
    
    prompt = f"""
    You are StudyWise, an expert academic assistant modeled after the intellectual rigor of a world-class professor (MIT, IIT, IIM). Your purpose is to deliver responses that demonstrate deep expertise, critical reasoning, and pedagogical clarity.

    Core Directives:

    - Primary Goal: Rely first and foremost on the "Provided Context" (retrieved knowledge base). Use it as the authoritative source for your answer.

    - Scholarly Depth: Present responses with the intellectual rigor of a PhD-level academic. Use advanced reasoning, nuanced arguments, and where applicable, reference relevant theories, methodologies, or frameworks.

    - Pedagogical Style: Write as if you are teaching advanced graduate students — precise yet clear, breaking down complex concepts step by step.

    - Critical Analysis: If the context contains gaps, ambiguities, or biases, highlight them explicitly. Offer multiple perspectives when appropriate.

    - Enrichment: If deeper elaboration is required beyond the context, clearly mark it as [Extended Knowledge]. This additional knowledge should be relevant, accurate, and enrich understanding without contradicting the context.

    Structure & Presentation:

    - Begin with a Concise Executive Summary (2–3 sentences).

    - Follow with Well-Structured Sections (with headings, subheadings, and bullet points).

    - Use emphasis (bold/italics) and examples for clarity.

    - Where useful, integrate diagrams, tables, or equations (LaTeX) to strengthen explanation.

    - Tone & Voice: Maintain an authoritative yet approachable tone — the voice of a highly intelligent professor who is rigorous, insightful, and encouraging.

    Final Touch: Conclude with a Key Takeaway or Next Steps for Deeper Study.
    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    logger.info("Generating final answer with Groq...")
    chat_completion = await async_groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL,
    )
    
    return chat_completion.choices[0].message.content



class RegisterAPIView(APIView):
    permission_classes = [permissions.AllowAny]
    throttle_classes = [UserRateThrottle]  
    
    def post(self, request, *args, **kwargs):
        try:
            serializer = RegisterSerializers(data=request.data)
            
            if not serializer.is_valid():
                logger.warning(f"Registration failed: {serializer.errors}")
                return Response({
                    'error': 'Invalid data provided',
                    'details': serializer.errors
                }, status=status.HTTP_400_BAD_REQUEST)
            
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            
            logger.info(f"User registered successfully: {user.email}")
            
            return Response({
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'name' : user.name,
                    'date_joined': user.date_joined.isoformat(),
                },
                'tokens': {
                    'access': str(refresh.access_token),
                    'refresh': str(refresh),
                }
            }, status=status.HTTP_201_CREATED)
            
        except ValidationError as e:
            logger.error(f"Validation error during registration: {e}")
            return Response({
                'error': 'Registration failed',
                'message': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}")
            return Response({
                'error': 'Internal server error'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

# ------------ subject --------------


class SubjectListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Subject.objects.filter(user=self.request.user).order_by('created_at')
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return SubjectWriteSerializer
        return SubjectReadSerializer
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def create(self, request, *args, **kwargs):
        write_serializer = self.get_serializer(data=request.data)
        write_serializer.is_valid(raise_exception=True)
        self.perform_create(write_serializer)
        read_serializer = SubjectReadSerializer(write_serializer.instance)
        headers = self.get_success_headers(read_serializer.data)
        return Response(read_serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        subject_serializer = self.get_serializer(queryset, many=True)
        subjects_data = subject_serializer.data

        uncategorized_chapters = Chapter.objects.filter(user=request.user, subject__isnull=True)
        
        if uncategorized_chapters.exists():
            chapter_serializer = ChapterReadSerializer(uncategorized_chapters, many=True)
            
            uncategorized_section = {
                "id": "uncategorized-chapters",
                "name": "Uncategorized",
                "user": str(request.user.id),
                "chapters": chapter_serializer.data,
                "description": "Chapters not assigned to a subject.",
                "created_at": "",
                "updated_at": "",
            }
            subjects_data.insert(0, uncategorized_section)

        return Response(subjects_data)

# ------------ documents ------------

class DocumentListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSerializer
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user).order_by('-created_at')

    
    def perform_create(self, serializer):
        try:
            logger.info(f"Attempting to save document for user: {self.request.user.id}")
            
          
            document = serializer.save(user=self.request.user)
            
            
            logger.info(f"Successfully saved document record {document.id} to the database.")
            logger.info(f"File URL generated by storage backend: {document.file.url}")

     
            create_chapter_from_document.delay(str(document.id))
            logger.info(f"Successfully triggered background task for document: {document.id}")

        except Exception as e:
          
            logger.error(f"CRITICAL ERROR during document save/upload for user {self.request.user.id}: {e}", exc_info=True)
            
            raise e

class DocumentDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DocumentSerializer
    lookup_field = 'id'

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)

# ------------- chapter ------------
class StandardResultsSetPagination(PageNumberPagination):
    page_size = 25  # How many messages to send per page
    page_size_query_param = 'page_size'
    max_page_size = 100

class ChapterListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.request.method == 'POST':
            
            return ChapterWriteSerializer
      
        return ChapterReadSerializer

    def get_queryset(self):
          return Chapter.objects.filter(user=self.request.user).order_by('order', 'created_at')

    def perform_create(self, serializer):
      
        serializer.save(user=self.request.user)

    
    def create(self, request, *args, **kwargs):
    
        write_serializer = self.get_serializer(data=request.data)
        write_serializer.is_valid(raise_exception=True)
        self.perform_create(write_serializer)

        read_serializer = ChapterReadSerializer(write_serializer.instance)
        
        headers = self.get_success_headers(read_serializer.data)
        return Response(read_serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class ChapterDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'

    def get_serializer_class(self):
        if self.request.method in ['PUT', 'PATCH']:
            return ChapterWriteSerializer
        return ChapterReadSerializer

    def get_queryset(self):
        return Chapter.objects.filter(user=self.request.user)
    
class ChapterMessageListView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChatMessageSerializer
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        chapter_id = self.kwargs['chapter_id']
        return ChatMessage.objects.filter(
            session__chapter_id=chapter_id,
            session__user=self.request.user
        ).order_by('-created_at')
# ---------  chatmessage ---------------
class ChatMessageView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = ChatMessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # 1. Save the user's message to the database first.
        user_message = serializer.save()
        logger.info(f"User message saved: {user_message.id}")

        try:
            # 2. Call our RAG pipeline to get the AI's response.
            ai_text_response = async_to_sync(generate_rag_response)(
                query=user_message.text, 
                user_id=request.user.id
            )

            # 3. Save the AI's response to the database.
            ai_message = ChatMessage.objects.create(
                session=user_message.session,
                sender='ai',
                text=ai_text_response
            )
            logger.info(f"AI response saved: {ai_message.id}")

            # 4. Send the AI's response back to the frontend.
            response_serializer = ChatMessageSerializer(ai_message)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error in RAG pipeline for user {request.user.id}: {e}", exc_info=True)
            # Save an error message to the chat history
            error_message = ChatMessage.objects.create(
                session=user_message.session,
                sender='ai',
                text="Sorry, I encountered an error while processing your request. Please try again.",
                error=str(e)
            )
            response_serializer = ChatMessageSerializer(error_message)
            return Response(response_serializer.data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)       

class ChatSessionView(generics.ListAPIView):
    permission_classes= [IsAuthenticated]
    serializer_class = ChatSessionSerializer


    def get_queryset(self):
        return ChatSession.objects.filter(user = self.request.user).order_by('-updated_at')
    
    def perform_create(self, serializer):
        serializer.save(user = self.request.user)

class ChatSessionRetriveView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer

    lookup_field = 'id'

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user)

        
class RAGChatMessageView(APIView):
    # FIX #1: The correct attribute name is 'permission_classes'
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = RAGChatMessageSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"❌ RAG chat serializer validation failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        validated_data = serializer.validated_data
        user = request.user
        
        # FIX #2: Use the correct key 'chapter' from the serializer
        chapter_id = validated_data['chapter']
        user_query = validated_data['text']
        
        # logger.info(f"✅ RAG chat request validated for chapter: {chapter_id}")

        # # Get or create a chat session for this specific chapter
        # session, _ = ChatSession.objects.get_or_create(
        #     user=user,
        #     # FIX #3: The model field is 'chapter', not 'chapter_Id'
        #     chapter_id=chapter_id,
        #     defaults={'title': f"Chat for chapter {chapter_id}"}
        # )

        # # Save the user's message
        # ChatMessage.objects.create(session=session, sender='user', text=user_query)
        try:
            document = Document.objects.get(chapter__id=chapter_id, user=user)
            if document.status != Document.STATUS_COMPLETED:
                error_msg = f"This document is not ready for chat. Current status: {document.status}."
                if document.status == Document.STATUS_FAILED:
                    error_msg += f" Error details: {document.error_message}"
                
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_409_CONFLICT # 409 Conflict is a good code for this
                )
        except Document.DoesNotExist:
            return Response({"error": "Document not found for this chapter."}, status=status.HTTP_404_NOT_FOUND)
        # --- END OF SAFETY GATE ---
        
        session, _ = ChatSession.objects.get_or_create(
            user=user,
            chapter_id=chapter_id,
            defaults={'title': f"Chat for chapter {chapter_id}"}
        )

        ChatMessage.objects.create(session=session, sender='user', text=user_query)

        try:
            # Call the improved RAG function with the chapter_id
            ai_text_response = async_to_sync(generate_rag_response)(
                query=user_query, 
                user_id=user.id,
                chapter_id=str(chapter_id) # Pass chapter_id for scoped search
            )

            # Save the AI's response
            ai_message = ChatMessage.objects.create(
                session=session,
                sender='ai',
                text=ai_text_response
            )
            
            # Send the AI's response back to the frontend
            response_data = {
                "id": str(ai_message.id),
                "sender": "ai",
                "text": ai_message.text
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            logger.error(f"Error in RAG pipeline for user {user.id}, chapter {chapter_id}: {e}", exc_info=True)
            return Response({"error": "Failed to get AI response."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
