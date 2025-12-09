# backend/rag_pipeline.py

import os
import asyncio
import logging
from django.conf import settings
from dotenv import load_dotenv

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse


from .ai_clients import async_qdrant_client  # we’ll still use self.groq_client for LLM
from .models import Document
from .tasks import process_document_ingestion
from utils.formatting import enforce_markdown_spacing

from .rag_service import (
    embed_texts,
    search_qdrant_vectors,
    make_chapter_user_filter,
)

load_dotenv()

logger = logging.getLogger(__name__)

LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "text-embedding-004"
QDRANT_COLLECTION_NAME = "studywise_documents"


class RagPipeline:
    def __init__(self, groq_api_key, qdrant_client, embedding_model):
        self.api_key = groq_api_key

        if not self.api_key:
            self.api_key = getattr(settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            logger.error("RagPipeline initialized without GROQ_API_KEY")
            raise ValueError("GROQ_API_KEY is required for RagPipeline. Please check your .env file.")

        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
        logger.info(f"RagPipeline initialized with GROQ_API_KEY: {masked_key}")

        from groq import AsyncGroq
        self.groq_client = AsyncGroq(api_key=self.api_key)
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.LLM_model = LLM_MODEL

    async def run(self, user_query, chat_history, chapter_id, user_id):
        # step 1: contextualization
        refined_query = await self.contextualize_query(user_query, chat_history)
        logger.info(f"Refined query: {refined_query}")

        # step 2: Router
        intent = await self.route_query(refined_query)
        logger.info(f"Detected intent: {intent}")

        # step 3: Execute strategy
        if intent == "greeting":
            return await self.handle_greeting(user_query)
        elif intent == "summary":
            return await self.handle_summary(chapter_id, user_id)
        elif intent == "ambiguous":
            return "I'm not sure I understand. Could you clarify your question about this document?"
        else:
            return await self.handle_rag_search(refined_query, chapter_id, user_id)

    async def contextualize_query(self, query, history):
        """
        Turn last user question into a standalone question using chat history.
        """
        if not history:
            return query

        # use last few messages – you can tweak slice later
        history_context = "\n".join([f"{msg.sender}: {msg.text}" for msg in history[-5:]])

        prompt = f""" 
        Given the following chat history and the latest user question, 
        rewrite the question to be a standalone query that can be understood without the history.
        Do NOT answer the question. Just rewrite it.

        Chat History:
        {history_context}

        user Question: {query}

        standalone Question:
        """

        try:
            completion = await self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                temperature=0.1,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Contextualization failed: {e}")
            return query

    async def route_query(self, query):
        """
        Classifies the query intent.
        """
        prompt = f""" 
        Classify the following user query into one of these categories:
        1. "greeting" (Hello, Hi, who are you)
        2. "summary" (Summarize this, what is this doc about, give me an overview)
        3. "ambiguous" (Vague requests like "explain", "more", "tell me")
        4. "question" (Specific questions about content, definitions, concepts)

        Query: {query}

        Return only the category name (lowercase)
        """

        try:
            completion = await self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                temperature=0,
            )
            intent = completion.choices[0].message.content.strip().lower()
            if intent not in ["greeting", "summary", "ambiguous", "question"]:
                return "question"
            return intent
        except Exception as e:
            logger.error(f"Intent routing failed: {e}")
            return "question"

    async def handle_greeting(self, query):
        return (
            "Hello! I'm your study assistant. I'm ready to help you analyze this chapter. "
            "What would you like to know?"
        )

    async def handle_summary(self, chapter_id, user_id):
        # later you can actually summarize chapter documents here
        return "Here is a summary of the chapter... (Implementation pending DB fetch)"

    async def _expand_queries(self, query: str, num: int = 4) -> list[str]:
        """
        Your old expand_queries_async, but now as a method using self.groq_client.
        """
        expansion_prompt = f"Generate {num} alternative phrasings of the following query for retrieval:\n\n{query}"
        completion = await self.groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": expansion_prompt}],
        )
        expanded = completion.choices[0].message.content.strip().split("\n")
        return [q.strip("-• ") for q in expanded if q.strip()]

    async def handle_rag_search(self, query: str, chapter_id: str, user_id: str):
        """
        This is basically your old generate_rag_response function,
        now living inside the class.
        """
        # 1) SELF-HEALING: ensure vectors exist in Qdrant for this chapter
        try:
            count_result = await async_qdrant_client.count(
                collection_name=QDRANT_COLLECTION_NAME,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chapter_id",
                            match=models.MatchValue(value=str(chapter_id)),
                        )
                    ]
                ),
                exact=False,
            )
            if count_result.count == 0:
                logger.warning(
                    f"SELF-HEALING: No vectors found for COMPLETED chapter {chapter_id}. "
                    f"Triggering re-ingestion."
                )
                try:
                    doc_to_reingest = await asyncio.to_thread(
                        Document.objects.get, chapter__id=chapter_id
                    )
                    process_document_ingestion.delay(str(doc_to_reingest.id))
                    return (
                        "The data for this chapter is being refreshed. "
                        "Please try your question again in a minute."
                    )
                except Document.DoesNotExist:
                    return (
                        "Sorry, the source document for this chapter could not be found. "
                        "Please re-upload it."
                    )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.warning(
                    f"SELF-HEALING: Collection does not exist. "
                    f"Triggering re-ingestion for chapter {chapter_id}."
                )
                try:
                    doc_to_reingest = await asyncio.to_thread(
                        Document.objects.get, chapter__id=chapter_id
                    )
                    process_document_ingestion.delay(str(doc_to_reingest.id))
                    return (
                        "The workspace is being initialized. "
                        "Please try your question again in a minute."
                    )
                except Document.DoesNotExist:
                    return (
                        "Sorry, the source document for this chapter could not be found. "
                        "Please re-upload it."
                    )
            else:
                raise e

        # 2) Expand queries
        expanded_queries = await self._expand_queries(query, num=4)
        all_queries = [query] + expanded_queries

        # 3) Embed queries using rag_service
        logger.info(f"Batch Embedding {len(all_queries)} queries via rag_service...")
        all_embeddings = await embed_texts(all_queries)

        # 4) Build search filter via rag_service helper
        search_filter = make_chapter_user_filter(chapter_id=str(chapter_id), user_id=str(user_id))

        # 5) Search Qdrant via rag_service
        logger.info(f"Batch searching Qdrant via rag_service with {len(all_embeddings)} vectors...")
        flat_results = await search_qdrant_vectors(all_embeddings, filter=search_filter, limit_per_vector=5)


        

        seen = set()
        unique_results = []
        for r in flat_results:
            if r and r.payload and "text" in r.payload:
                if r.payload["text"] not in seen:
                    seen.add(r.payload["text"])
                    unique_results.append(r)

        sorted_results = sorted(unique_results, key=lambda r: r.score, reverse=True)
        context = "\n\n---\n\n".join(
            [r.payload["text"] for r in sorted_results[:10]]
        )

        
        prompt = f"""
        Core Identity:
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

        # 7) Call Groq for final answer
        logger.info("Generating final answer with Groq...")
        chat_completion = await self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
        )

        raw_output = chat_completion.choices[0].message.content
        formatted_output = enforce_markdown_spacing(raw_output)
        return formatted_output
