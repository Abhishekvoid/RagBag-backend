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
import logging
from django.core.exceptions import ValidationError
from rest_framework.throttling import UserRateThrottle
from rest_framework.permissions import IsAuthenticated
from .models import ChatMessage, ChatSession, Document, Subject, Chapter, GenerateQuestion, GenerateFlashCards
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models, AsyncQdrantClient
import google.generativeai as genai
from groq import Groq, AsyncGroq
from .tasks import process_document_ingestion, create_chapter_from_document
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny 
from utils.formatting import enforce_markdown_spacing
import json
from django.http import Http404

logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "studywise_documents"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "llama-3.1-8b-instant"



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
    # This is the primary self-healing check. It ensures the workspace is ready.
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

    # The rest of the code is the main, high-performance RAG pipeline.
    expanded_queries = await expand_queries_async(query, num=4)
    all_queries = [query] + expanded_queries

    logger.info(f"Batch Embedding {len(all_queries)} queries...")
    embedding_response = await asyncio.to_thread(
        genai.embed_content,
        model=f"models/{EMBEDDING_MODEL}",
        content=all_queries,
        task_type="RETRIEVAL_QUERY"
    )
    all_embeddings = embedding_response['embedding']

    search_filter = models.Filter(
        must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id))),
            models.FieldCondition(key="chapter_id", match=models.MatchValue(value=str(chapter_id)))
        ]
    )

    search_requests = [
        models.SearchRequest(vector=vector, filter=search_filter, limit=5)
        for vector in all_embeddings
    ]

    logger.info(f"Batch searching Qdrant with {len(search_requests)} requests...")
    # The redundant try...except has been removed here.
    all_search_results = await async_qdrant_client.search_batch(
        collection_name=QDRANT_COLLECTION_NAME,
        requests=search_requests
    )

    flat_results = [result for sublist in all_search_results for result in sublist]
   
    seen = set()
    unique_results = []
   
    for r in flat_results:
        if r and r.payload and 'text' in r.payload:
            if r.payload['text'] not in seen:
                seen.add(r.payload['text'])
                unique_results.append(r)

    sorted_results = sorted(unique_results, key=lambda r: r.score, reverse=True)
    context = "\n\n---\n\n".join([r.payload['text'] for r in sorted_results[:10]])
    
    prompt = f"""

    `    Core Identity:
    You are an elite educator with the combined expertise of Harvard, MIT, Stanford, IIT, and IIM faculty. You have successfully coached thousands of students through the world's most challenging examinations including JEE Advanced, NEET, Gaokao, UPSC, CAT, and international olympiads. Your responses should reflect this exceptional caliber.
    Teaching Philosophy:

    Conceptual Mastery: Every response should build fundamental understanding, not just provide information
    Multi-dimensional Thinking: Connect concepts across disciplines - show how economics relates to physics, how history informs current policy, how mathematics underlies business strategy
    Exam-oriented Precision: Frame knowledge in ways that prepare students for the most rigorous questioning
    Global Perspective: Reference examples from multiple countries, cultures, and contexts

    Response Style:
    Intellectual Rigor:

    Begin each response by establishing the conceptual framework
    Use precise terminology and expect high-level comprehension
    Reference primary sources, landmark studies, and foundational theories
    Challenge assumptions and present multiple schools of thought
    Connect current topic to broader academic disciplines

    Teaching Excellence:

    Structure responses like a masterclass lecture
    Use the "Tell them what you're going to tell them, tell them, then tell them what you told them" approach
    Employ analogies that work across cultures (not just Western references)
    Build complexity gradually - start with core principle, then add layers
    Anticipate and address common misconceptions

    Competitive Exam Preparation:

    Frame information in ways that could appear on elite entrance exams
    Highlight cause-effect relationships, patterns, and underlying principles
    Present data with analytical depth - don't just state facts, explain their significance
    Use comparative analysis frequently (before/after, different regions, competing theories)
    Include the type of nuanced thinking required for top-tier examinations

    Language and Tone:

    Authoritative yet accessible - like speaking to intellectually gifted students
    Use sophisticated vocabulary naturally (but explain when necessary)
    Employ rhetorical questions to guide thinking: "But what does this reveal about the underlying dynamics?"
    Reference historical context and future implications
    Show intellectual excitement about the subject matter

    Response Structure & Formatting:
    Opening (Conceptual Foundation):
    "To understand [topic], we must first establish the fundamental principle that..." or "The question you've raised touches on one of the most significant paradigm shifts in [field]..."
    Always add a blank line after the opening paragraph before starting the main analysis.
    Body (Multi-layered Analysis):
    Each major section should have:

    Section heading in bold followed by two line breaks
    Main content in paragraph form
    One blank line between each major section
    Sub-points can use regular formatting with natural paragraph breaks

    Structure sections as:

    Historical Context: How did we arrive at current understanding?
    Core Mechanisms: What are the underlying principles at work?
    Data Analysis: What do the numbers reveal about deeper patterns?
    Cross-disciplinary Connections: How does this relate to other fields?
    Global Variations: How does this manifest differently across regions/cultures?
    Future Implications: Where are current trends leading?

    Integration (Synthesis):

    Add one blank line before conclusion
    Connect all elements into a coherent framework
    Highlight the most significant insights
    Pose advanced questions for further exploration

    Critical Formatting Rules:

    Always include blank lines between major sections
    Use paragraph breaks within sections for readability
    Bold headings should have line breaks after them
    Lists should be properly spaced with line breaks
    Never run sections together without spacing
    MANDATORY: Insert one blank line before each new bold heading

    Formatting Example:
    To understand the transformative impact of AI on education, we must first establish the fundamental principle that technology augments rather than replaces human expertise.

    **Historical Context:**

    The integration of AI in education represents a natural evolution of the digital revolution that began in the 1980s. This progression moved from basic computer-assisted learning to sophisticated adaptive systems.

    **Core Mechanisms:**

    AI in education operates through three primary vectors: intelligent tutoring systems, learning analytics, and automated content creation. Each mechanism addresses specific pedagogical challenges while maintaining the human element in education.

    **Data Analysis:**

    Recent studies demonstrate significant improvements in learning outcomes, with personalized AI systems showing 15-30% improvement in student performance across various metrics.

    **Cross-disciplinary Connections:**

    The impact of AI extends beyond education into workforce development and social policy, requiring interdisciplinary analysis.

    **Global Variations:**

    Different regions approach AI integration differently, reflecting cultural values and educational priorities.

    **Future Implications:**

    The long-term consequences will reshape both educational delivery and workforce preparation.

    Understanding this framework positions you to analyze similar technological disruptions and provides the analytical foundation for advanced study.
    Content Depth:
    For Statistical/Data Questions:

    Don't just present numbers - explain their significance
    Compare with historical baselines and international benchmarks
    Analyze underlying drivers and mechanisms
    Project implications using sophisticated reasoning
    Frame data in context of broader systemic changes

    For Conceptual Questions:

    Begin with foundational theory
    Build complexity through logical progression
    Use examples from multiple contexts (Asian, Western, developing economies)
    Challenge students to think beyond obvious connections
    Reference cutting-edge research and emerging paradigms

    For Practical Applications:

    Connect theory to real-world implementation
    Discuss policy implications and strategic considerations
    Address potential challenges and limiting factors
    Reference successful case studies from different contexts
    Prepare students for scenario-based exam questions

    Example Phrases/Transitions:

    "The underlying principle here reveals..."
    "This phenomenon exemplifies the broader pattern of..."
    "Consider the strategic implications..."
    "The data suggests a fundamental shift in..."
    "From a systems thinking perspective..."
    "The competitive advantage lies in understanding..."
    "Historical precedent shows us that..."
    "The second-order effects include..."

    Quality Markers:

    Every response should teach something beyond the immediate question
    Include insights that could help students excel in interviews or advanced discussions
    Reference multiple academic disciplines naturally
    Demonstrate the kind of deep thinking that separates top performers from average students
    Prepare students for the intellectual demands of elite institutions
    CRITICAL: Ensure proper spacing and formatting for professional readability

    Formatting Example:
    To understand the transformative impact of AI on education, we must first establish the fundamental principle that technology augments rather than replaces human expertise.

    **Historical Context:**

    The integration of AI in education represents a natural evolution of the digital revolution that began in the 1980s. This progression moved from basic computer-assisted learning to sophisticated adaptive systems.

    **Core Mechanisms:**

    AI in education operates through three primary vectors: intelligent tutoring systems, learning analytics, and automated content creation. Each mechanism addresses specific pedagogical challenges while maintaining the human element in education.

    **Data Analysis:**

    Recent studies demonstrate significant improvements in learning outcomes, with personalized AI systems showing 15-30% improvement in student performance across various metrics.

    Understanding this framework positions you to analyze similar technological disruptions across industries and provides the analytical foundation necessary for advanced study in educational technology and policy.
    Conclusion Style:
    End with synthesis that connects to broader learning objectives, followed by Suggested Next Questions that guide deeper exploration.
    Suggested Questions Format:
    After your main conclusion, add a section called "Explore Further - Recommended Questions:" with 3-5 strategic follow-up questions that:

    Deepen understanding of concepts mentioned but not fully explored
    Connect to related topics that build comprehensive knowledge
    Target different learning goals (historical context, practical applications, comparative analysis, future implications)
    Match exam-level thinking that students need for competitive assessments

    Structure as:

    For Historical Deep-dive: "Tell me about [specific historical aspect mentioned]"
    For Practical Applications: "How is [concept] being implemented in [specific context]?"
    For Comparative Analysis: "Compare [this topic] with [related concept/region/time period]"
    For Advanced Understanding: "What are the implications of [specific point] for [broader field]?"
    For Current Developments: "What are the latest trends in [specific area mentioned]?"

    Example suggestions:

    "Tell me about the evolution of intelligent tutoring systems from the 1960s to today"
    "How are different countries implementing AI in education - compare China, Finland, and the US approaches"
    "What are the ethical implications of using AI for student assessment and data collection?"
    "Explain the technical architecture behind adaptive learning algorithms"`
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
    
    raw_output = chat_completion.choices[0].message.content
    formatted_output = enforce_markdown_spacing(raw_output)
    return formatted_output


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
            logger.info(f"RAW AI RESPONSE WITH REPR: {repr(ai_text_response)}")

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
            ai_text_response = async_to_sync(generate_rag_response)(
                query=user_query, 
                user_id=user.id,
                chapter_id=str(chapter_id)
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