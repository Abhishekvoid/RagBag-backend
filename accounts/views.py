import asyncio
from asgiref.sync import async_to_sync
from django.shortcuts import render, get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics, permissions
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializers, ChatMessageSerializer, ChatSessionSerializer, DocumentSerializer, SubjectWriteSerializer, SubjectReadSerializer, ChapterReadSerializer, ChapterWriteSerializer,  RAGChatMessageSerializer, GeneratedQuestionsSerializer,GeneratedFlashCardsSerializer, MeSerializer, NoteSerializer, DocumentPageSerializer
import logging, time
from django.core.exceptions import ValidationError
from rest_framework.throttling import UserRateThrottle
from rest_framework.permissions import IsAuthenticated
from .models import ChatMessage, ChatSession, Document, Subject, Chapter, GenerateQuestion, GenerateFlashCards, Note
from utils.circuit_breaker import llm_circuit_breaker, tei_circuit_breaker
from utils.metrics.latency import latency_tracker
from utils.metrics.cost import cost_tracker
from utils.metrics.retrieval import retrieval_evaluator
import os



from .tasks import  create_chapter_from_document, process_document_for_existing_chapter, process_document_ingestion, cleanup_document_data, rescan_document_with_vision
from rest_framework import parsers
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny 
import json

logger = logging.getLogger(__name__)

from .rag_pipeline import RagPipeline
from .ai_clients import GROQ_API_KEY, groq_client

rag_pipeline = RagPipeline(
    groq_api_key=GROQ_API_KEY,
    embedding_model="gemini-embedding-001",
)

LLM_MODEL = "llama-3.1-8b-instant"


class AIRateThrottle(UserRateThrottle):
    """Tighter, dedicated bucket for the AI note actions (explain / synthesize /
    notes->flashcards). Rate configured under DEFAULT_THROTTLE_RATES['ai']."""
    scope = 'ai'


class LLMUnavailable(Exception):
    """Raised by the sync Groq helpers when the circuit breaker is open or the
    client is unconfigured; views translate this to a 503."""


def _guarded_groq(messages, *, json_mode):
    """Sync Groq call wrapped in the shared circuit breaker (mirrors the async
    llm_gateway used by the RAG pipeline, but for these sync HTTP views)."""
    if groq_client is None:
        raise LLMUnavailable("LLM client is not configured.")
    if not llm_circuit_breaker.allow_request():
        raise LLMUnavailable("LLM is temporarily unavailable. Please retry shortly.")
    try:
        kwargs = {"messages": messages, "model": LLM_MODEL}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = groq_client.chat.completions.create(**kwargs)
        llm_circuit_breaker.record_success()
        return completion.choices[0].message.content
    except Exception:
        llm_circuit_breaker.record_failure()
        raise


def groq_json(prompt):
    return json.loads(_guarded_groq([{"role": "user", "content": prompt}], json_mode=True))


def groq_text(prompt):
    return _guarded_groq([{"role": "user", "content": prompt}], json_mode=False)


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
        
class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = MeSerializer(request.user)
        return Response(serializer.data)


class MetricsView(APIView):
    """Staff-only snapshot of the in-process trackers: per-stage latency,
    per-token spend, retrieval quality, and circuit-breaker state.

    Counters are per worker process and in-memory, so numbers reflect the one
    process that served this request — fine for a read on a single web dyno,
    not a substitute for a real metrics backend across many.
    """
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        # Redis backs the breakers; if it is down we still want the rest of the
        # payload, since that is exactly when someone is reading this endpoint.
        try:
            breakers = {
                "llm": "open" if llm_circuit_breaker.is_open() else "closed",
                "tei_embedding": "open" if tei_circuit_breaker.is_open() else "closed",
            }
        except Exception:
            logger.warning("circuit breaker state unavailable", exc_info=True)
            breakers = {"error": "unavailable"}

        return Response({
            "latency_ms_by_stage": latency_tracker.get_metrics(),
            "cost": cost_tracker.get_summary(),
            "retrieval": retrieval_evaluator.get_summary(),
            "circuit_breakers": breakers,
        })

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

def _purge_documents(documents_qs):
    """Fully remove a set of documents: enqueue vector + file cleanup, then
    delete the DB rows. Used by chapter/subject cascade deletes so documents are
    truly removed rather than left detached by the ``SET_NULL`` default."""
    documents = list(documents_qs)
    if not documents:
        return
    document_ids = [str(doc.id) for doc in documents]
    file_names = [doc.file.name for doc in documents if doc.file]
    cleanup_document_data.delay(document_ids, file_names)
    documents_qs.delete()


class SubjectDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'

    def get_serializer_class(self):
        if self.request.method in ['PUT', 'PATCH']:
            return SubjectWriteSerializer
        return SubjectReadSerializer

    def get_queryset(self):
        return Subject.objects.filter(user=self.request.user)

    def perform_destroy(self, instance):
        # Full cascade: remove all documents (and their vectors/files) belonging
        # to this subject's chapters, then delete the subject. Chapters cascade
        # via the FK; chat sessions detach (SET_NULL) and are intentionally kept.
        _purge_documents(Document.objects.filter(chapter__subject=instance))
        instance.delete()

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


class DocumentRetryView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, id):
        doc = get_object_or_404(Document, id=id, user=request.user)
        doc.status = Document.STATUS_PENDING
        doc.error_message = None
        doc.save(update_fields=["status", "error_message"])
        if doc.chapter:
            process_document_ingestion.delay(str(doc.id))
        else:
            create_chapter_from_document.delay(str(doc.id))
        return Response(
            {"status": "requeued", "document_id": str(doc.id)},
            status=status.HTTP_202_ACCEPTED,
        )
                


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

    def perform_destroy(self, instance):
        # Full cascade: remove this chapter's documents (and their vectors/files),
        # then delete the chapter. Generated questions/flashcards cascade via FK;
        # chat sessions detach (SET_NULL) and are intentionally kept.
        _purge_documents(instance.documents.all())
        instance.delete()

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
    





# -------------------- auth ---------------- 

class OAuthSignInView(APIView):
    permission_classes = [AllowAny]

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
            document = Document.objects.filter(chapter__id=chapter_id, user=user).order_by("-created_at").first()

            if not document:
                return Response(
                    {"error": "Document not found for this chapter."},
                    status=status.HTTP_404_NOT_FOUND
                )
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

            history = ChatMessage.objects.filter(
            session=session
            ).order_by("-created_at")[:10]
            history = list(reversed(history))
            # Call the high-performance RAG function
            result = async_to_sync(rag_pipeline.run)(
                user_query,
                chat_history=list(history),
                chapter_id=str(chapter_id),
                user_id=user.id,
            )

            ai_text = result["answer"]
            sources = result.get("sources", [])
            followups = result.get("followups", [])

            # Enrich source chips with a human title (sync DB is fine here).
            doc_ids = [s["document_id"] for s in sources]
            titles = {
                str(pk): title
                for pk, title in Document.objects.filter(
                    id__in=doc_ids
                ).values_list("id", "title")
            }
            for s in sources:
                s["title"] = titles.get(s["document_id"], "Source")

            # Save the AI's response
            ai_message = ChatMessage.objects.create(
                session=session,
                sender='ai',
                text=ai_text,
                citations=sources,
                suggestions=followups,
            )

            response_data = {
                "id": str(ai_message.id),
                "sender": "ai",
                "text": ai_message.text,
                "sources": sources,
                "followups": followups,
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

            Format your response as a valid JSON object with a single key "questions".
            The value must be an array of objects, where each object has a "question" key and an "answer" key.
            Example: {{"questions": [{{"question": "What is the capital of France?", "answer": "Paris."}}]}}

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

class ChapterFlashCardListView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = GeneratedFlashCardsSerializer

    def get_queryset(self):
        return GenerateFlashCards.objects.filter(
            chapter_id=self.kwargs['chapter_id'],
            user=self.request.user,
        )


class ChapterQuestionListView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = GeneratedQuestionsSerializer

    def get_queryset(self):
        # GenerateQuestion has no user field; scope ownership through the chapter.
        return GenerateQuestion.objects.filter(
            chapter_id=self.kwargs['chapter_id'],
            chapter__user=self.request.user,
        ).order_by('created_at')


# ------------- co-reading: document content -------------

class DocumentContentView(APIView):
    """Serve a document's extracted reader text. The list/detail document
    serializers deliberately omit the (large) body; this is the only place the
    frontend fetches it. Cleanup/reflow happens client-side."""
    permission_classes = [IsAuthenticated]

    def get(self, request, id):
        doc = get_object_or_404(Document, id=id, user=request.user)
        if doc.status != Document.STATUS_COMPLETED:
            return Response(
                {"error": "Document is not ready.", "status": doc.status},
                status=status.HTTP_409_CONFLICT,
            )
        return Response({
            "id": str(doc.id),
            "title": doc.title,
            "text": doc.extracted_text or "",
        })


# ------------- co-reading: notes CRUD -------------

class NoteListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = NoteSerializer

    def get_queryset(self):
        return Note.objects.filter(
            user=self.request.user,
            chapter_id=self.kwargs['chapter_id'],
        )

    def perform_create(self, serializer):
        chapter = get_object_or_404(Chapter, id=self.kwargs['chapter_id'], user=self.request.user)
        serializer.save(user=self.request.user, chapter=chapter)


class NoteDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = NoteSerializer
    lookup_field = 'id'

    def get_queryset(self):
        return Note.objects.filter(user=self.request.user)


class ChapterScratchView(APIView):
    """The single freeform scratch pad per chapter (one kind=scratch Note),
    get-or-created on first access."""
    permission_classes = [IsAuthenticated]

    def _get_or_create(self, request, chapter_id):
        chapter = get_object_or_404(Chapter, id=chapter_id, user=request.user)
        note, _ = Note.objects.get_or_create(
            user=request.user,
            chapter=chapter,
            kind=Note.KIND_SCRATCH,
            defaults={'body': ''},
        )
        return note

    def get(self, request, chapter_id):
        note = self._get_or_create(request, chapter_id)
        return Response(NoteSerializer(note, context={'request': request}).data)

    def put(self, request, chapter_id):
        note = self._get_or_create(request, chapter_id)
        note.body = request.data.get('body', '')
        note.save(update_fields=['body', 'updated_at'])
        return Response(NoteSerializer(note, context={'request': request}).data)


# ------------- co-reading: smart AI actions -------------

class ExplainPassageView(APIView):
    """Explain a selected passage and persist the result as a kind=ai note
    anchored to that passage."""
    permission_classes = [IsAuthenticated]
    throttle_classes = [AIRateThrottle]

    def post(self, request, chapter_id, *args, **kwargs):
        chapter = get_object_or_404(Chapter, id=chapter_id, user=request.user)
        passage = (request.data.get("passage") or "").strip()
        if not passage:
            return Response({"error": "A passage is required."}, status=status.HTTP_400_BAD_REQUEST)

        document = None
        document_id = request.data.get("document")
        if document_id:
            document = get_object_or_404(Document, id=document_id, user=request.user)

        prompt = (
            "You are a patient tutor. Explain the following passage clearly and "
            "concisely for a student, in 2-4 sentences. Use plain language; do not "
            "add information beyond what the passage supports.\n\nPASSAGE:\n"
            f"{passage[:4000]}\n\nEXPLANATION:"
        )
        try:
            explanation = groq_text(prompt).strip()
        except LLMUnavailable as e:
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Explain failed for chapter {chapter_id}: {e}", exc_info=True)
            return Response({"error": "Failed to explain passage."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        note = Note.objects.create(
            user=request.user,
            chapter=chapter,
            document=document,
            kind=Note.KIND_AI,
            anchor_start=request.data.get("anchor_start"),
            anchor_end=request.data.get("anchor_end"),
            quoted_text=passage,
            body=explanation,
        )
        return Response(NoteSerializer(note, context={'request': request}).data, status=status.HTTP_201_CREATED)


class NotesToFlashCardsView(APIView):
    """Turn selected notes/highlights into flashcards in ONE batched LLM call,
    reusing the GenerateFlashCards store."""
    permission_classes = [IsAuthenticated]
    throttle_classes = [AIRateThrottle]

    def post(self, request, chapter_id, *args, **kwargs):
        chapter = get_object_or_404(Chapter, id=chapter_id, user=request.user)
        note_ids = request.data.get("note_ids") or []

        notes = Note.objects.filter(user=request.user, chapter=chapter).exclude(kind=Note.KIND_SCRATCH)
        if note_ids:
            notes = notes.filter(id__in=note_ids)
        notes = list(notes)
        if not notes:
            return Response({"error": "No notes to convert."}, status=status.HTTP_400_BAD_REQUEST)

        source = "\n\n---\n\n".join(
            "\n".join(filter(None, [n.quoted_text.strip(), n.body.strip()])) for n in notes
        )
        prompt = (
            "You are an elite educator. From the student's highlights and notes "
            "below, generate high-quality recall flashcards.\n\n"
            f"NOTES:\n{source[:8000]}\n\n"
            'Return a valid JSON object with a single key "flashcards": an array of '
            'objects each with "flashcard_front" (a question/prompt) and '
            '"flashcard_back" (a 2-3 sentence answer). Output only JSON.'
        )
        try:
            data = groq_json(prompt)
        except LLMUnavailable as e:
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Notes->flashcards failed for chapter {chapter_id}: {e}", exc_info=True)
            return Response({"error": "Failed to generate flashcards."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        cards = data.get("flashcards", [])
        if not isinstance(cards, list):
            return Response({"error": "Unexpected AI response format."}, status=status.HTTP_400_BAD_REQUEST)

        new_flashcards = []
        for item in cards:
            if isinstance(item, dict) and "flashcard_front" in item and "flashcard_back" in item:
                new_flashcards.append(GenerateFlashCards.objects.create(
                    chapter=chapter,
                    user=request.user,
                    flashcard_front=item["flashcard_front"],
                    flashcard_back=item["flashcard_back"],
                ))
        if not new_flashcards:
            return Response({"error": "AI returned no usable flashcards."}, status=status.HTTP_400_BAD_REQUEST)

        return Response(
            GeneratedFlashCardsSerializer(new_flashcards, many=True).data,
            status=status.HTTP_201_CREATED,
        )


class SynthesizeNotesView(APIView):
    """Synthesize all of a chapter's highlights/notes into a markdown study
    sheet in one LLM call. Returns the summary (frontend may save it as a note)."""
    permission_classes = [IsAuthenticated]
    throttle_classes = [AIRateThrottle]

    def post(self, request, chapter_id, *args, **kwargs):
        chapter = get_object_or_404(Chapter, id=chapter_id, user=request.user)
        notes = list(
            Note.objects.filter(user=request.user, chapter=chapter).exclude(kind=Note.KIND_SCRATCH)
        )
        if not notes:
            return Response({"error": "No notes to synthesize yet."}, status=status.HTTP_400_BAD_REQUEST)

        source = "\n\n---\n\n".join(
            "\n".join(filter(None, [n.quoted_text.strip(), n.body.strip()])) for n in notes
        )
        prompt = (
            "You are a study coach. Synthesize the student's highlights and notes "
            "below into a concise, well-structured study sheet in markdown. Use "
            "short headings and bullet points; group related ideas; do not invent "
            "facts beyond the notes.\n\n"
            f"NOTES:\n{source[:8000]}\n\nSTUDY SHEET (markdown):"
        )
        try:
            summary = groq_text(prompt).strip()
        except LLMUnavailable as e:
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Synthesize failed for chapter {chapter_id}: {e}", exc_info=True)
            return Response({"error": "Failed to synthesize notes."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"summary": summary}, status=status.HTTP_200_OK)

# class FlashCardListView(generics.ListAPIView):
#     permission_classes = [IsAuthenticated]
#     serializer_class = GeneratedFlashCardsSerializer
    
#     def get_queryset(self):
#         chapter_id = self.kwargs['chapter_id']
#         return GenerateFlashCards.objects.filter(
#             session__chapter_id = chapter_id,
#             session__user=self.request.user
#         ).order_by('created_at')

class DocumentPagesView(generics.ListAPIView):
    """GET /auth/documents/<id>/pages/ — the reconstructed pages of an owned document."""
    serializer_class = DocumentPageSerializer

    def get_queryset(self):
        doc = get_object_or_404(Document, id=self.kwargs["id"], user=self.request.user)
        return doc.pages.all()


class DocumentRescanView(APIView):
    """POST /auth/documents/<id>/rescan/ — re-run the vision pipeline for an owned document."""
    def post(self, request, id):
        doc = get_object_or_404(Document, id=id, user=request.user)
        rescan_document_with_vision.delay(str(doc.id))
        return Response({"status": "queued"}, status=status.HTTP_202_ACCEPTED)
