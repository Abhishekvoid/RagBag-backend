import asyncio
from asgiref.sync import async_to_sync
from qdrant_client.http.exceptions import UnexpectedResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics, permissions
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializers, ChatMessageSerializer, ChatSessionSerializer, DocumentSerializer, SubjectWriteSerializer, SubjectReadSerializer, ChapterReadSerializer, ChapterWriteSerializer,  RAGChatMessageSerializer, GeneratedQuestionsSerializer, GeneratedFlashCardsSerializer
import logging, time
from django.core.exceptions import ValidationError
from rest_framework.throttling import UserRateThrottle
from rest_framework.permissions import IsAuthenticated
from .models import ChatMessage, ChatSession, Document, Subject, Chapter, GenerateQuestion, GenerateFlashCards
import os
from dotenv import load_dotenv

import google.generativeai as genai
from .tasks import process_document_ingestion, create_chapter_from_document, process_document_for_existing_chapter
from rest_framework import parsers
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny 
from utils.formatting import enforce_markdown_spacing
import json
from django.http import Http404
from django.utils.decorators import method_decorator
from utils.timing import time_sync, time_async
from django.views.decorators.csrf import csrf_exempt
logger = logging.getLogger(__name__)

from .rag_pipeline import RagPipeline
from .ai_clients import qdrant_client, GROQ_API_KEY, groq_client

rag_pipeline = RagPipeline(
    groq_api_key=GROQ_API_KEY,
    qdrant_client=qdrant_client,
    embedding_model="text-embedding-004",
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "studywise_documents"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "llama-3.1-8b-instant"


v = os.getenv("GROQ_API_KEY")
logger.info("GROQ_API_KEY repr: %s", repr(v) if v is not None else "None")
if v:
    logger.info("GROQ_API_KEY masked: %s...%s", v[:4], v[-4:])


try:
    if not GOOGLE_API_KEY:
        raise ValueError("google api key is not set")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.critical(f"Failed to configure Google GenAI: {e}")



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
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user).order_by('-created_at')
    
    def perform_create(self, serializer):
        print("DEBUG: perform_create called in DocumentListCreateView")
        try:
            logger.info(f"Attempting to save document for user: {self.request.user.id}")
            
            # Save the document. The serializer's create method will handle
            # associating it with an existing chapter if chapter_id was provided.
            document = serializer.save(user=self.request.user) # Pass user context to serializer

            logger.info(f"Successfully saved document record {document.id} to the database.")
            logger.info(f"  - File: {document.file.name}")
            logger.info(f"  - Chapter ID: {document.chapter.id if document.chapter else 'None'}")
            logger.info(f"  - File URL generated by storage backend: {document.file.url}")

            # Trigger the correct background task based on whether a chapter was assigned
            if document.chapter:
                # Document was associated with an existing chapter
                logger.info(f"Triggering 'process_document_for_existing_chapter' task for document {document.id} and chapter {document.chapter.id}...")
                process_document_for_existing_chapter.delay(str(document.id), str(document.chapter.id))
                logger.info(f"Task 'process_document_for_existing_chapter' triggered successfully.")
            else:
                # Document was uploaded standalone, create a new chapter from it
                logger.info(f"Triggering 'create_chapter_from_document' task for document {document.id}...")
                create_chapter_from_document.delay(str(document.id))
                logger.info(f"Task 'create_chapter_from_document' triggered successfully.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR during document save/upload for user {self.request.user.id}: {e}", exc_info=True)
            raise e

    # def perform_create(self, serializer):
       
    #     chapter_id = self.kwargs.get("chapter_id") or self.request.data.get("chapter")
    #     chapter =None

    #     if chapter_id:
    #         try:
    #             chapter = Chapter.objects.get(id=chapter_id, user=self.request.user)
    #         except Chapter.DoesNotExist:
    #             return Response({"detail": "Chapter not found."}, status=status.HTTP_400_BAD_REQUEST)
    
    #         document = serializer.save( user=self.request.user, chapter=chapter)
    #         logger.info(f"✅ Document {document.id} created in DB. Chapter: {chapter_id}")

    #     # 2. DIRECTLY trigger tasks (Removed transaction.on_commit for reliability)
    #     # This ensures the task is sent to Redis immediately.
    #         if chapter:
    #             logger.info(f"🚀 Sending 'process_document_for_existing_chapter' task...")
    #             process_document_for_existing_chapter.delay(str(document.id), str(chapter.id))
    #         else:
    #             logger.info(f"🚀 Sending 'create_chapter_from_document' task...")
    #             create_chapter_from_document.delay(str(document.id))

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
        ).order_by('created_at')
    





# -------------------- auth ---------------- 

class OAuthSignInView(APIView):
    permissions_classes = [AllowAny]

    def post (self,request, *args, **kwargs):
        email =  request.data.get("email")
        name = request.data.get("name")

        if not email or not name:
            return Response(
                {"error": "Email and name are required."},
                status= status.HTTP_400_BAD_REQUEST
            )
        
        User =  get_user_model()

        user, created = User.objects.get_or_create(
            email=email,
            defaults={'name': name}
        )

        if created:
            
            user.set_unusable_password()
            user.save()

        refresh = RefreshToken.for_user(user)

        return Response({
            'user': {
                'id': user.id,
                'email': user.email,
                'name' : user.name,
            },
            'tokens': {
                'access': str(refresh.access_token),
                'refresh': str(refresh),
            }
        }, status=status.HTTP_200_OK)
        

# ---------  chatmessage -------    --------
# class ChatMessageView(APIView):
#     permission_classes = [IsAuthenticated]
#     throttle_classes = [UserRateThrottle]

#     def post(self, request, *args, **kwargs):
#         serializer = ChatMessageSerializer(data=request.data)
#         if not serializer.is_valid():
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         # 1. Save the user's message to the database first.
#         user_message = serializer.save()
#         logger.info(f"User message saved: {user_message.id}")

#         try:
#             # 2. Call our RAG pipeline to get the AI's response.
#             ai_text_response = async_to_sync(generate_rag_response)(
#                 query=user_message.text, 
#                 user_id=request.user.id
#             )
#             logger.info(f"RAW AI RESPONSE WITH REPR: {repr(ai_text_response)}")

#             # 3. Save the AI's response to the database.
#             ai_message = ChatMessage.objects.create(
#                 session=user_message.session,
#                 sender='ai',
#                 text=ai_text_response
#             )
#             logger.info(f"AI response saved: {ai_message.id}")

#             # 4. Send the AI's response back to the frontend.
#             response_serializer = ChatMessageSerializer(ai_message)
#             return Response(response_serializer.data, status=status.HTTP_201_CREATED)

#         except Exception as e:
#             logger.error(f"Error in RAG pipeline for user {request.user.id}: {e}", exc_info=True)
#             # Save an error message to the chat history
#             error_message = ChatMessage.objects.create(
#                 session=user_message.session,
#                 sender='ai',
#                 text="Sorry, I encountered an error while processing your request. Please try again.",
#                 error=str(e)
#             )
#             response_serializer = ChatMessageSerializer(error_message)
#             return Response(response_serializer.data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)       
class ChatMessageView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [UserRateThrottle]

    def post(self, request, *args, **kwargs):
        return Response(
            {"detail": "Legacy chat endpoint is disabled. Use RAGChatMessageView instead."},
            status=status.HTTP_410_GONE,
        )
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
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = RAGChatMessageSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"❌ RAG chat serializer validation failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        validated_data = serializer.validated_data
        user = request.user
        chapter_id = validated_data['chapter']
        user_query = validated_data['text']
        
        logger.info(f"RAG chat request validated for chapter: {chapter_id}")

        # --- CORRECTED: The Safety Gate is the primary control flow ---
        try:
            document = Document.objects.get(chapter__id=chapter_id, user=user)
            if document.status != Document.STATUS_COMPLETED:
                error_msg = f"This document is not ready for chat. Current status: {document.status}."
                if document.status == Document.STATUS_FAILED:
                    error_msg += f" Error details: {document.error_message}"
                
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_409_CONFLICT
                )

            # Only after the status check passes, we create the session and message
            session, _ = ChatSession.objects.get_or_create(
                user=user,
                chapter_id=chapter_id,
                defaults={'title': f"Chat for chapter {chapter_id}"}
            )
            ChatMessage.objects.create(session=session, sender='user', text=user_query)

            # Call the high-performance RAG function
            ai_text_response = async_to_sync(rag_pipeline.run)(
                user_query,
                chat_history=[],          
                chapter_id=str(chapter_id),
                user_id=user.id,
            )

            # Save the AI's response
            ai_message = ChatMessage.objects.create(
                session=session,
                sender='ai',
                text=ai_text_response
            )
            
            response_data = {
                "id": str(ai_message.id),
                "sender": "ai",
                "text": ai_message.text
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        
        except Document.DoesNotExist:
            return Response({"error": "Document not found for this chapter."}, status=status.HTTP_404_NOT_FOUND)
        
        
        except Exception as e:
            logger.error(f"Error in RAG pipeline for user {user.id}, chapter {chapter_id}: {e}", exc_info=True)
            return Response({"error": "Failed to get AI response."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ------------- generated Questions

class GenerateQuestionsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, chapter_id, *args, **kwargs):
        try:
            # 1. Find the chapter and its documents
            chapter = Chapter.objects.get(id=chapter_id, user=request.user)
            documents = chapter.documents.all()
            if not documents:
                return Response({"error": "This chapter has no documents to generate questions from."}, status=status.HTTP_400_BAD_REQUEST)

            # 2. Consolidate the text from all documents
            full_text = "\n\n---\n\n".join([doc.extracted_text for doc in documents if doc.extracted_text])
            if not full_text.strip():
                 return Response({"error": "Could not find any text in the documents for this chapter."}, status=status.HTTP_400_BAD_REQUEST)

            # 3. Create a powerful prompt for the AI
            prompt = f"""
            Based on the following text, generate 5-7 challenging study questions that a student could use to test their knowledge.
            For each question, provide a concise, accurate answer based only on the text.

            Format your response as a valid JSON array of objects, where each object has a "question" key and an "answer" key.
            Example: [{{"question": "What is the capital of France?", "answer": "Paris."}}]

            TEXT:
            {full_text[:8000]} # Use a generous context window

            JSON:
            """
            
            # 4. Call the AI
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
               
                response_format={"type": "json_object"},
            )
            
          
            generated_data = json.loads(chat_completion.choices[0].message.content)
            
           
            GenerateQuestion.objects.filter(chapter=chapter).delete()

            new_questions = []
            for item in generated_data.get("questions", []): 
                question = GenerateQuestion.objects.create(
                    chapter=chapter,
                    question_text=item.get("question"),
                    answer_text=item.get("answer")
                )
                new_questions.append(question)

            # 6. Send the new questions back to the frontend
            serializer = GeneratedQuestionsSerializer(new_questions, many=True)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Chapter.DoesNotExist:
            return Response({"error": "Chapter not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error generating questions for chapter {chapter_id}: {e}", exc_info=True)
            return Response({"error": "Failed to generate questions."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class GenerateFlashCardView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = GeneratedFlashCardsSerializer

    def post(self, request, chapter_id, *args, **kwargs):
        try:
            # Find chapter and documents
            chapter = Chapter.objects.get(id=chapter_id, user=request.user)
            documents = chapter.documents.all()
            if not documents:
                return Response(
                    {"error": "No document found to generate flashcards."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Consolidate text
            full_text = "\n\n---\n\n".join(
                [doc.extracted_text for doc in documents if doc.extracted_text]
            )
            if not full_text.strip():
                return Response(
                    {"error": "No readable text found in this document."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # AI Prompt
            prompt = f"""
            You are an elite educator with deep interdisciplinary expertise. 
            Your task is to generate *high-quality educational flashcards* from the following study material.

            🎯 OBJECTIVE:
            Create flashcards that help a student actively recall and deeply understand key ideas.

            📚 CONTEXT (from source material):
            {full_text[:8000]}

            ---
            🧩 INSTRUCTIONS:
            1. Extract 9–15 of the most important concepts, definitions, and relationships.
            2. Each flashcard must include:
                - "flashcard_front": a question or prompt
                - "flashcard_back": a short answer or explanation (2–3 sentences max)
            3. Avoid vague, duplicated, or off-topic cards.

            ---
            🎨 OUTPUT FORMAT:
            Return the flashcards as a valid JSON object with a single key "flashcards".
            The value must be an array of 9–15 flashcard objects in this exact structure:
            {{
              "flashcards": [
                {{
                  "flashcard_front": "What is the primary function of mitochondria?",
                  "flashcard_back": "They generate ATP through cellular respiration, providing energy for the cell."
                }}
              ]
            }}
            ⚠️ Do not include commentary or markdown. Output only JSON.
            """

            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                response_format={"type": "json_object"},
            )

            generate_data = json.loads(chat_completion.choices[0].message.content)
            flashcard_list = generate_data.get("flashcards", [])

            if not isinstance(flashcard_list, list):
                return Response({"error": "Unexpected AI response format"}, status=status.HTTP_400_BAD_REQUEST)

            new_flashcards = []
            for item in flashcard_list:
                if isinstance(item, dict) and "flashcard_front" in item and "flashcard_back" in item:
                    flashcard = GenerateFlashCards.objects.create(
                        chapter=chapter,
                        user=request.user,
                        flashcard_front=item["flashcard_front"],
                        flashcard_back=item["flashcard_back"],
                    )
                    new_flashcards.append(flashcard)

            if not new_flashcards:
                return Response(
                    {"error": "AI failed to generate flashcards in correct format."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            serializer = GeneratedFlashCardsSerializer(new_flashcards, many=True)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Chapter.DoesNotExist:
            return Response({"error": "Chapter not found."}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            logger.error(f"Error generating flashcards for chapter {chapter_id}: {e}", exc_info=True)
            return Response(
                {"error": "Failed to generate flashcards."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class FlashCardDetailView(generics.RetrieveUpdateDestroyAPIView):

    permission_classes = [IsAuthenticated]
    serializer_class = GeneratedFlashCardsSerializer
    lookup_field = 'id'

    def get_queryset(self):

        return GenerateFlashCards.objects.filter(user=self.request.user)