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
from qdrant_client import QdrantClient, models
import google.generativeai as genai
from groq import Groq
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
LLM_MODEL = "mixtral-8x7b-32768" 


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

def generate_rag_response(query: str, user_id: str):
    """
    Performs the full RAG pipeline: embed, retrieve, augment, and generate.
    """
    if not qdrant_client or not groq_client:
        raise Exception("AI clients are not initialized.")

    # 1. EMBED: Convert the user's question into a vector.
    logger.info(f"Embedding query for user {user_id}: '{query[:50]}...'")
    query_embedding = genai.embed_content(
        model=f"models/{EMBEDDING_MODEL}",
        content=query,
        task_type="RETRIEVAL_QUERY" # Use 'RETRIEVAL_QUERY' for search
    )['embedding']

    # 2. RETRIEVE: Search Qdrant for the most relevant text chunks.
    logger.info("Searching Qdrant for relevant context...")
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=3, # Get the top 3 most relevant chunks
        # CRITICAL: Filter results to only include documents owned by the current user.
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=str(user_id))
                )
            ]
        )
    )
    
    context = "\n\n---\n\n".join([result.payload['text'] for result in search_results])
    logger.info(f"Retrieved context: {context[:200]}...")

    # 3. AUGMENT: Create a detailed prompt for the LLM.
    prompt = f"""
    You are an expert academic assistant named StudyWise. Your goal is to help the user understand their documents.
    Based *only* on the context provided below, answer the user's question.
    If the context does not contain the answer, state that you cannot answer based on the provided information.
    Do not use any external knowledge.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    logger.info("Generating final answer with Groq...")
    chat_completion = groq_client.chat.completions.create(
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
            ai_text_response = generate_rag_response(
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
    permissions = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = RAGChatMessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        validated_data = serializer.validated_data
        user = request.user
        chapter_Id = validated_data['chapter_Id']
        user_query = validated_data['text']


        session, _ =  ChatSession.objects.get_or_create(
            user= user,
            chapter_Id =chapter_Id,
            defaults={'title': f"chat for chapter{chapter_Id}"}
        )


        ChatMessage.objects.create(session=session, sender='user', text=user_query)

        try:
            # 3. Call your RAG pipeline
            ai_text_response = generate_rag_response(
                query=user_query, 
                user_id=user.id
            )

            # 4. Save the AI's response
            ai_message = ChatMessage.objects.create(
                session=session,
                sender='ai',
                text=ai_text_response
            )
            
            # 5. Send the AI's response back to the frontend
            response_data = {
                "id": str(ai_message.id),
                "sender": "ai",
                "text": ai_message.text
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            logger.error(f"Error in RAG pipeline for user {user.id}: {e}", exc_info=True)
            return Response({"error": "Failed to get AI response."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)