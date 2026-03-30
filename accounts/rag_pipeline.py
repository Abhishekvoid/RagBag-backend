# backend/rag_pipeline.py

import os
import asyncio
import logging
from django.conf import settings
from dotenv import load_dotenv
import numpy as np 
from django.db import transaction

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
import json

from .ai_clients import async_qdrant_client  # we’ll still use self.groq_client for LLM
from .models import Document
from .tasks import process_document_ingestion
from utils.formatting import enforce_markdown_spacing
import time
import uuid

from utils.llm_gateway import ask_llm, LLMUnavailable
from .rag_service import (
    embed_texts,
    search_qdrant_vectors,
    make_chapter_user_filter,
)
from utils.reranking_crossencoder import reranker
from utils.metrics.latency import latency_tracker
from utils.metrics.retrieval import retrieval_evaluator
from utils.metrics.cost import cost_tracker

load_dotenv()

logger = logging.getLogger(__name__)

LLM_MODEL = "llama-3.1-8b-instant"

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

        

    def is_greeting(self, user_query: str) -> bool:
        greetings = {
        "hi","hii","hello","hey","yo","sup",
        "good morning","good afternoon","good evening"
        }

        return user_query.lower().strip() in greetings
        
    async def run(self, user_query, chat_history, chapter_id, user_id):
        # step 1: contextualization

        request_id = str(uuid.uuid4())
        start_time = time.monotonic()
        status = "unknown"

        logger.info(
            "rag_request_stared",
            extra= {
                "event": "Rag_request_started",
                "request_id": request_id,
                "user_id": str(user_id),
                "chapter_id": str(chapter_id),
            }
        )
        try:
            refined_query = await self.contextualize_query(user_query, chat_history, request_id, user_id, chapter_id)
            logger.info(f"Refined query: {refined_query}")

            # step 2: Router
            intent = await self.route_query(refined_query, request_id, user_id, chapter_id)
            logger.info(f"Detected intent: {intent}")

            # step 3: Execute strategy
            if self.is_greeting(user_query):
                result =  await self.handle_greeting(user_query)
            elif intent == "summary":
                result =  await self.handle_summary(chapter_id, user_id)
            elif intent == "ambiguous":
                result =  "I'm not sure I understand. Could you clarify your question about this document?"
            else:
                result = await self.handle_rag_search(refined_query, chapter_id, user_id, request_id)
            
            status = "success"
            return result
        except Exception as e:
            status = "failed"

            logger.exception(
                "rag_request_failed",
                extra={
                    "event": "rag_request_failed",
                    "request_id": request_id,
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                }
            )

            raise

        finally:
            total_latency_ms = (time.monotonic() - start_time) * 1000


            logger.info(
                 "rag_request_completed",
                extra={
                    "event": "rag_request_completed",
                    "request_id": request_id,
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                    "status": status,
                    "total_latency_ms": round(total_latency_ms, 2),
                }
            )

    
    async def contextualize_query(self, query, history, request_id, user_id, chapter_id):
        """
        Turn last user question into a standalone question using chat history.
        """

        start_time = time.monotonic()
        status = "unknown"
        result = None

        logger.info(
            "contextualization_request_stared",
            extra= {
                "event": "contextualization_start",
                "stage": "contexttualization",
                "request_id": request_id,
                "user_id": str(user_id),
                "chapter_id": str(chapter_id),
            }
        )
        try: 
            if not history:
                result = query
                status = "skipped"

            else:
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
                    completion = await ask_llm(
                        self.groq_client,
                        messages=[{"role": "user", "content": prompt}],
                        model=LLM_MODEL,
                        temperature=0.1,
                        timeout=5.0,
                    )
                    result =  completion.choices[0].message.content.strip()

                    status = "success"
                    
                
                except LLMUnavailable:
                    logger.info("Contextualization skipped - LLM unaviavble")
                    result = query
                    status = "degraded"
                except Exception as e:
                    logger.error(f"Contextualization failed: {e}")
                    result =  query
                    status = "degraded"
        except Exception as e:

            status = "failed"
            logger.exception(
                "Contextualization failed",
                extra = {
                    "event": "contextualization_failed",
                    "stage": "contextualization",
                    "request_id": request_id,
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                }
            )
            raise

        finally:
            total_latency_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                 "contextualization completed",
                extra={
                    "event": "contextualization completed",
                    "stage": "contextualization",
                    "request_id": request_id,
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                    "status": status,
                    "total_latency_ms": round(total_latency_ms, 2),
                }
            )
        return result

    async def route_query( self, query, request_id,  user_id, chapter_id):


        start_time = time.monotonic()
        status = "unknown"
        result = None

        try:
            """
            Classifies the query intent.
            """
            logger.info(
                "routing started",
                extra = {

                    "event": "routing_started",
                    "stage": "routing",
                    "request_id": str(request_id),
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                }
            )
            
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
                completion = await ask_llm(
                    self.groq_client,
                    messages=[{"role": "user", "content": prompt}],
                    model=LLM_MODEL,
                    temperature=0,
                    timeout=3.0,
                )
                intent = completion.choices[0].message.content.strip().lower()
                if intent not in ["greeting", "summary", "ambiguous", "question"]:
                    result =  "question"
                else:
                    result =  intent

                status = "success"
            
            except LLMUnavailable:
                logger.info(f"cannot decide the route -> llm is unavailable")
                result = "question"
                status = "degraded"
            except Exception as e:
                logger.error(f"Intent routing failed: {e}")
                result = "question"
                status = "degraded"
        except Exception as e:
            
            status = "failed"

            logger.exception(
                "routing failed",
                extra={
                    "event": "routing_failed",
                    "stage": "routing",
                    "request_id": str(request_id),
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                }
            )

            raise
        finally:
            total_latency_ms = (time.monotonic() - start_time) * 1000
        
            logger.info(
                "routing query completed", extra = {
                 "event": "routing completed",
                    "stage": "routing",
                    "request_id": request_id,
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                    "status": status,
                    "total_latency_ms": round(total_latency_ms, 2), }
            )       
        return result     

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
        completion = await ask_llm(
            self.groq_client,
            model=LLM_MODEL,
            messages=[{"role": "user", "content": expansion_prompt}],
            timeout=5.0,
        )
        expanded = completion.choices[0].message.content.strip().split("\n")
        return [q.strip("-• ") for q in expanded if q.strip()]
    async def handle_rag_search(self, query: str, chapter_id: str, user_id: str, request_id=None):
       

        logger.info(f"starting RAg search for chapter{chapter_id}, user {user_id}")
        logger.info(f"query: {query}")
        
    # ===== STEP 1: SELF-HEALING CHECK =====
        logger.info("Skipping count check — going directly to search")
            

        
    # ===== STEP 2: INTELLIGENT QUERY EXPANSION =====
        logger.info("🔍 Expanding query intelligently...")
        
        expansion_prompt = f"""Analyze this student's question and generate 3 strategic search queries to find the most relevant information.

    Question: {query}

    Generate queries that:
    1. Target the core concept/definition
    2. Look for explanations/mechanisms  
    3. Search for examples/applications

    Return as JSON: {{"queries": ["query1", "query2", "query3"]}}
    """
    
        try:
            async with latency_tracker.track_async("query_expansion"):
                expansion_response = await ask_llm(
                    self.groq_client,
                    messages=[{"role": "user", "content": expansion_prompt}],
                    model=LLM_MODEL,
                    json_mode = True,
                    temperature=0.2,
                )
                expansion_data = json.loads(expansion_response.choices[0].message.content)
                expanded_queries = expansion_data.get("queries", [query])

        except LLMUnavailable:
            logger.info(f"Query Expansion failed -> llm unavialable")
            expanded_queries = [query]
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            expanded_queries = [query]
    
        all_queries = [query] + expanded_queries
        logger.info(f"📝 Search queries: {all_queries}")

    # ===== STEP 3: EMBED & SEARCH =====
        logger.info("🔢 Embedding queries...")
        try:
            async with latency_tracker.track_async("embeddings"):
                all_embeddings = await embed_texts(all_queries)
                logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            return "Failed to process your question. Please try again."
        
        logger.info("🧪 Testing search WITHOUT filter to verify embeddings work...")

        # try:
        #     # Search without any filter to see if we get ANY results
        #     test_results = await search_qdrant_vectors(
        #         [all_embeddings[0]],  # Just test with first embedding
        #         filter=None,  # NO FILTER
        #         limit_per_vector=5
        #     )
            
        #     logger.info(f"Test search (no filter) returned {len(test_results)} results")
            
        #     if test_results and len(test_results) > 0:
        #         logger.info("Embeddings are working! Problem is with the filter.")
        #         logger.info(f"Sample result chapter_id: {test_results[0].payload.get('chapter_id')}")
        #         logger.info(f"Sample result user_id: {test_results[0].payload.get('user_id')}")
        #         logger.info(f"Your filter chapter_id: {chapter_id}")
        #         logger.info(f"Your filter user_id: {user_id}")
        #     else:
        #         logger.error("Even without filter, no results! Embedding model mismatch?")
                
        # except Exception as e:
        #     logger.error(f"Test search failed: {e}", exc_info=True)

        # # Now do the normal filtered search
        # logger.info(f"Searching with filter: chapter_id={chapter_id}, user_id={user_id}")

        # # ===== Also check what's actually stored in Qdrant =====
        # logger.info("Checking what's in Qdrant for this chapter...")

        try:
            # Scroll through some vectors to see what chapter_ids exist
            scroll_result = await async_qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=5,
                with_payload=True,
                with_vectors=False,
            )
            
            logger.info(f" Sample vectors in collection:")
            for point in scroll_result[0]:
                logger.info(f"Stored → chapter_id={point.payload.get('chapter_id')}, user_id={point.payload.get('user_id')}")
                logger.info(f"Text preview: {point.payload.get('text', '')[:100]}")
            
            logger.info(f"FILTER DEBUG → user_id={user_id}, chapter_id={chapter_id}")
        except Exception as e:
            logger.error(f"Scroll check failed: {e}")

        
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=str(user_id))
                ),
                models.FieldCondition(
                        key="chapter_id",
                        match=models.MatchValue(value=str(chapter_id)),
                ),
            ]
        )

        logger.info("🔍 Searching vector database...")
        try: 
            async with latency_tracker.track_async("vector_search"):
                flat_results = await search_qdrant_vectors(
                    all_embeddings, 
                    filter=search_filter, 
                    limit_per_vector=15  # Get more results for reranking
                )
                logger.info(f" Retrieved {len(flat_results)} results from Qdrant")

            if not flat_results:
                logger.warning("Strict filter failed → fallback to user_id only")

                fallback_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=str(user_id))
                        )
                    ]
                )
                flat_results = await search_qdrant_vectors(
                    all_embeddings,
                    filter=fallback_filter,
                    limit_per_vector=15        
                )


            import re
            query_words = set(re.findall(r"\w+", query.lower()))
            KEYWORD_BOOST = 0.05

            for r in flat_results:
                if r.payload and "text" in r.payload:
                    text_words = r.payload.get("text", "").lower()
                    keyword_hits = sum(1 for w in query_words if w in text_words)
                    r.score = float(r.score) + (KEYWORD_BOOST * keyword_hits)

            flat_results.sort(key=lambda x: x.score, reverse=True)


            if flat_results and len(flat_results) > 0:
                first_result = flat_results[0]
                if first_result and first_result.payload:
                    preview = first_result.payload.get('text', '')[:200]
                    logger.info(f"📄 First result preview: {preview}...")
                    logger.info(f"📊 First result score: {first_result.score}")
                else:
                    logger.error("❌ First result has no payload!")
            else:
                logger.error("❌ NO RESULTS returned from Qdrant search!")
                return "I couldn't find relevant information in your document. This might be a technical issue."
        except Exception as e:
            logger.error(f"❌ Qdrant search failed: {e}", exc_info=True)
            return "Search failed. Please try again."
            

        # reranking 

        logger.info("Reranking results by relevance...")
        flat_results = flat_results[:40]

# deduplicate
        seen = set()
        unique_results = []
        for r in flat_results:
            text = r.payload.get("text") if r.payload else None
            if text and text not in seen:
                seen.add(text)
                unique_results.append(r)

        logger.info(f"Deduped: {len(unique_results)} chunks")

        if len(unique_results) > 5:
            RERANK_LIMIT = min(len(unique_results), 20)
            pairs = [[query, r.payload["text"]] for r in unique_results[:RERANK_LIMIT]]
            
            async with latency_tracker.track_async("reranking"):
                if reranker:
                    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
                else:
                    logger.warning("Reranker not loaded, skipping reranking")
                    scores = [0] *len(pairs)
            
            # ✅ FIXED: Safe numpy check + proper score assignment
            if len(scores) == 0:
                logger.warning("Reranker returned no scores")
                final_results = unique_results[:8]
            else:
                # Apply scores ONLY to reranked items
                for i, r in enumerate(unique_results[:RERANK_LIMIT]):
                    r.score = float(scores[i])
                
                # ✅ FIXED: Sort ONLY scored items
                scored_results = unique_results[:RERANK_LIMIT]
                final_results = sorted(scored_results, key=lambda x: x.score, reverse=True)[:8]
                
                logger.info(f"Reranked {len(final_results)} chunks. Top score={final_results[0].score:.3f}")
        else:
            final_results = unique_results[:8]

        retrieval_evaluator.evaluate(
            query=query,
            chunks=[r.payload['text'] for r in final_results]
        )
    # ===== STEP 5: BUILD CONTEXT =====
        context = "\n\n---\n\n".join([
            r.payload["text"] for r in final_results
        ])

        context_length = len(context)
        logger.info(f"📄 Context built: {context_length} characters")
        logger.info(f"📄 Context preview: {context[:300]}...")
        
        if context_length < 100:
            logger.error(f"❌ Context too short: {context_length} chars")
            return "I found very limited information in your document. Please ensure it uploaded correctly."
    
        logger.info(f"📄 Context built: {len(context)} chars from {len(final_results)} chunks")

        # ===== STEP 6: GENERATE ANSWER =====
        logger.info("🤖 Generating answer...")
        
        final_prompt = f"""You are an expert AI tutor helping a student understand their study material. Provide a clear, accurate, and helpful response.

            **MANDATORY FORMATTING RULES (Follow exactly):**
            1. **Always** start with a 1-sentence direct answer in bold.
            2. Use **one blank line** between every paragraph/section.
            3. Every paragraph = MAX 2 sentences (40 words).
            4. Use `-` bullets for ANY list (steps, factors, examples).
            5. **Bold key terms** on first mention only.
            6. End technical explanations with `**In simple terms:** ...`

            **YOUR APPROACH:**
            1. Answer the question directly and concisely
            2. Ground every claim in the context provided below
            3. Use natural, conversational language
            4. Structure your response for easy scanning (bold, bullets, etc.)
            5. If the context doesn't contain the answer, be honest about it

            **FORMATTING:**
            - Use **bold** for key terms and important concepts
            - Use bullet points for lists or multiple items
            - Keep paragraphs short (2-3 sentences)
            - Add one blank line between sections
            - For technical terms, explain them clearly

            **CITATION:**
            When referencing the material, use phrases like:
            - "According to the material..."
            - "The document explains that..."
            - "As covered in [topic]..."

            **IF INFORMATION IS MISSING:**
            If the context doesn't contain enough information:
            "I don't see information about [topic] in your document. The material I have covers [related topics]. Would you like to know about those instead?"

            ---

            **CONTEXT FROM DOCUMENT:**
            {context}

            **STUDENT'S QUESTION:**
            {query}

            **YOUR RESPONSE:**
            """
       
        try:
            async with latency_tracker.track_async("llm_generation"):
                chat_completion = await ask_llm(
                    self.groq_client,
                    messages=[{"role": "user", "content": final_prompt}],
                    model=LLM_MODEL,
                    temperature=0.1,  # Very low temperature for factual accuracy
                    max_tokens=800,
                    timeout=30.0,
                )

            raw_output = chat_completion.choices[0].message.content
            logger.info(f"✅ Generated response ({len(raw_output)} chars)")
            logger.info(f"📄 Response preview: {raw_output[:200]}...")
            
            formatted_output = enforce_markdown_spacing(raw_output)
            return formatted_output
        
        except LLMUnavailable:
            logger.warning("Answer generation skipped — LLM unavailable")
            return "AI is temporarily unavailable. Please try again shortly."
        
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}", exc_info=True)
            return "Failed to generate an answer. Please try again."
        
       

        