
import os
import re
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

import httpx
from openai import APITimeoutError
from tenacity import RetryError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError
from django.conf import settings
from dotenv import load_dotenv
import numpy as np 
from django.db import transaction

import json

from .models import Document
from .tasks import process_document_ingestion
from utils.formatting import enforce_markdown_spacing
import time
import uuid

from asgiref.sync import sync_to_async

from utils.llm_gateway import ask_llm, LLMUnavailable
from .rag_service import (
    embed_texts,
    hybrid_search,
    make_chapter_user_filter,
)
from utils.tei_rerank import rerank_client
from utils.token_budget import is_safe_to_embed
from utils.metrics.latency import latency_tracker
from utils.metrics.retrieval import retrieval_evaluator
from utils.metrics.cost import cost_tracker

load_dotenv()

logger = logging.getLogger(__name__)

from .ai_clients import ANSWER_MODEL, LLM_MODEL  # noqa: F401  (re-exported)

TUTOR_SYSTEM_PROMPT = """You are StudyWise, an expert tutor helping a student understand their own study material. Your job is to make the concept click — not to sound like a textbook.

How you answer:
- Lead with a direct, plain-language answer to exactly what was asked. No preamble.
- Match length to the question: a single line for simple questions, a few short paragraphs for complex ones. Never pad.
- Write in natural prose. Use **bold** for key terms, and short `-` bullet lists, only when they genuinely aid clarity — not by default.
- Define any jargon the first time it appears, in plain words.
- Be warm and encouraging but precise. Sound like a sharp person explaining to a friend.

Grounding rules:
- Base your answer primarily on the STUDENT'S MATERIAL provided.
- You MAY add general knowledge to clarify or complete an explanation, but when you go beyond their material, flag it briefly like: "(Beyond your notes:) ...".
- If the material doesn't cover something and you're not confident, say so plainly instead of guessing.
- Never invent specifics (numbers, definitions, names) that are not in the material or well-established general knowledge.

Do not list multiple follow-up questions yourself — those are handled separately. You may end with at most one short invitation to go deeper."""


def build_answer_messages(context: str, query: str) -> list:
    """Two-message chat payload for the answer step: a fixed tutor system
    prompt plus a user message carrying the retrieved context and question."""
    user_content = (
        f"STUDENT'S MATERIAL:\n{context}\n\n"
        f"QUESTION:\n{query}"
    )
    return [
        {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# How much chapter text a summary may consume. ANSWER_MODEL has a very large
# context window, so this is a cost and latency bound rather than a technical
# one — and when it binds, the user is told, because a summary that silently
# covers only the first third of a chapter is worse than no summary.
SUMMARY_CHAR_BUDGET = 60_000

# The retrieval funnel, named rather than inlined so the shape is readable in
# one place: fuse everything, hand a shortlist to the cross-encoder, keep the
# best few for the prompt. FINAL_CHUNKS * ~200 tokens is the context budget.
RERANK_CANDIDATES = 20
FINAL_CHUNKS = 8

VALID_INTENTS = ("greeting", "summary", "ambiguous", "question")

DEFAULT_INTENT = "question"


class QueryExpansionPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    queries: list[Annotated[str, StringConstraints(
        strict=True, strip_whitespace=True, min_length=3, max_length=150,
    )]] = Field(min_length=1, max_length=3)


def validated_search_queries(raw: str, original: str, contextualized: str = "") -> list[str]:
    """Always retain the validated input; at most three distinct alternatives."""
    try:
        payload = QueryExpansionPayload.model_validate_json(raw)
    except (ValidationError, TypeError, ValueError):
        logger.warning("Invalid query expansion; using the original question")
        return [original]
    queries = [original]
    seen = {original.strip().casefold()}
    for candidate in [contextualized, *payload.queries]:
        candidate = candidate.strip()
        key = candidate.casefold()
        if 3 <= len(candidate) <= 150 and key not in seen and is_safe_to_embed(candidate):
            queries.append(candidate)
            seen.add(key)
        if len(queries) == 4:
            break
    return queries

CONTEXTUALIZE_AND_ROUTE_PROMPT = """You prepare a student's message for a study assistant.

Do TWO things and return only JSON.

1. standalone_question — rewrite the message so it can be understood WITHOUT the
   chat history: resolve pronouns and references like "it", "that", "the second one".
   If it already stands alone, return it unchanged. Do NOT answer it.

2. intent — classify the ORIGINAL message as exactly one of:
   "greeting"  — hello, hi, who are you
   "summary"   — summarize this, what is this document about, give me an overview
   "ambiguous" — too vague to act on even with the history ("explain", "more", "tell me")
   "question"  — anything specific: concepts, definitions, mechanisms, examples

Chat history:
{history}

Message: {query}

Return ONLY: {{"standalone_question": "...", "intent": "..."}}"""


def parse_contextualize_and_route(raw: str, fallback_query: str):
    
    """
    parsing contextualization + intent_routing LLM call...
    
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return fallback_query, DEFAULT_INTENT        

    if not isinstance(data, dict):
        return fallback_query, DEFAULT_INTENT

    question = data.get("standalone_question")
    question = question.strip() if isinstance(question, str) else ""

    intent = data.get("intent")
    intent = intent.strip().lower() if isinstance(intent, str) else ""

    return (
        question or fallback_query,
        intent if intent in VALID_INTENTS else DEFAULT_INTENT,
    )


def parse_followups(raw: str) -> list:
    """Parse a follow-ups JSON string into up to 3 clean questions."""
    try:
        data = json.loads(raw)
        items = data.get("followups", [])
        if not isinstance(items, list):
            return []
        cleaned = [str(x).strip() for x in items if str(x).strip()]
        return cleaned[:3]
    except Exception:
        return []


def build_sources(final_results) -> list:
    """Distinct source chunks (by document_id, top 3) with a short snippet."""
    seen = set()
    sources = []
    for r in final_results:
        payload = getattr(r, "payload", None) or {}
        doc_id = payload.get("document_id")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        sources.append({
            "document_id": str(doc_id),
            "snippet": (payload.get("text", "") or "")[:140],
        })
        if len(sources) >= 3:
            break
    return sources


class PipelineOutcome(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RATE_LIMITED = "rate_limited"


@dataclass
class PipelineResult:
    outcome: PipelineOutcome
    answer: str = ""
    sources: list = field(default_factory=list)
    followups: list = field(default_factory=list)
    error: str = ""


def _result(answer: str, sources=None, followups=None,
            outcome=PipelineOutcome.SUCCESS) -> PipelineResult:
    return PipelineResult(outcome, answer, sources or [], followups or [])


def _failure(error: Exception) -> PipelineResult:
    """Classify provider errors, including exhausted retries and SDK wrappers.

    Provider exception text can contain credentials or request bodies. Only
    fixed, public messages leave the pipeline in an error response.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, "response", None)
        code = (getattr(error, "status_code", None)
                or getattr(error, "status", None)
                or getattr(response, "status_code", None))
        if str(code) == "429":
            return PipelineResult(
                PipelineOutcome.RATE_LIMITED,
                error="AI service rate limit reached. Please try again shortly.",
            )
        if isinstance(error, (TimeoutError, httpx.TimeoutException,
                              APITimeoutError, Urllib3TimeoutError)) or str(code) in ("408", "504"):
            return PipelineResult(
                PipelineOutcome.DEADLINE_EXCEEDED,
                error="The AI service timed out. Please try again.",
            )
        if isinstance(error, RetryError):
            error = error.last_attempt.exception()
        else:
            error = (error.__cause__ or getattr(error, "reason", None)
                     or error.__context__)
        if not isinstance(error, BaseException):
            break
    return PipelineResult(
        PipelineOutcome.DEPENDENCY_UNAVAILABLE,
        error="AI is temporarily unavailable. Please try again shortly.",
    )


class RagPipeline:
    def __init__(self, embedding_model):
        # The shared client encodes the provider choice; constructing a second
        # one here would bypass the retry / empty-completion wiring in
        # utils.llm_wrapper.
        from .ai_clients import async_llm_client

        self.llm_client = async_llm_client

        if self.llm_client is None:
            logger.error("RagPipeline initialized with no LLM provider configured")
            raise ValueError(
                "No LLM provider configured. Set OPENROUTER_API_KEY in your .env file."
            )

        logger.info("RagPipeline initialized (answer=%s, routing=%s)", ANSWER_MODEL, LLM_MODEL)

        self.embedding_model = embedding_model
        self.LLM_model = LLM_MODEL

        

    def is_greeting(self, user_query: str) -> bool:
        greetings = [
            "hi", "hello", "hey", "yo", "sup",
            "good morning", "good afternoon", "good evening"
        ]

        query = user_query.lower().strip()

        # (hiiii → hii)
        query = re.sub(r"(.)\1{2,}", r"\1\1", query)

        return any(re.search(rf"\b{greet}\b", query) for greet in greetings)
        
    async def run(self, user_query, chat_history, chapter_id, user_id) -> PipelineResult:
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

            if self.is_greeting(user_query):
                status = PipelineOutcome.SUCCESS.value
                return _result(await self.handle_greeting(user_query))

            refined_query, intent = await self.contextualize_and_route(
                user_query, chat_history, request_id, user_id, chapter_id
            )
            logger.info(f"Refined query: {refined_query} | intent: {intent}")

            if not is_safe_to_embed(refined_query):
                logger.warning(
                    "contextualized query exceeds the embedding budget; "
                    "falling back to the original question"
                )
                refined_query = user_query

            # step 2: Execute strategy
            if intent == "greeting":
                result = _result(await self.handle_greeting(refined_query))
            elif intent == "summary":
                result = await self.handle_summary(chapter_id, user_id)
            elif intent == "ambiguous":
                result =  _result("I'm not sure I understand. Could you clarify your question about this document?")
            else:
                result = await self.handle_rag_search(
                    refined_query, chapter_id, user_id, request_id, original_query=user_query,
                )
            
            status = result.outcome.value
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

    
    async def contextualize_and_route(self, query, history, request_id, user_id, chapter_id):
        """Rewrite the question to stand alone AND classify its intent, in ONE call."""
        start_time = time.monotonic()
        status = "unknown"
        result = (query, DEFAULT_INTENT)

        logger.info(
            "contextualize_and_route_started",
            extra={
                "event": "contextualize_and_route_started",
                "stage": "contextualize_and_route",
                "request_id": request_id,
                "user_id": str(user_id),
                "chapter_id": str(chapter_id),
            },
        )

        try:
            history_context = "\n".join(
                f"{msg.sender}: {msg.text}" for msg in (history or [])[-5:]
            ) or "(no previous messages)"

            prompt = CONTEXTUALIZE_AND_ROUTE_PROMPT.format(
                history=history_context, query=query
            )

            try:
                async with latency_tracker.track_async("contextualize_and_route"):
                    completion = await ask_llm(
                        self.llm_client,
                        messages=[{"role": "user", "content": prompt}],
                        model=LLM_MODEL,
                        json_mode=True,
                        temperature=0,
                        max_tokens=1200,
                        timeout=20.0,
                    )
                result = parse_contextualize_and_route(
                    completion.choices[0].message.content, query
                )
                status = "success"
            except LLMUnavailable:
                logger.info("contextualize_and_route skipped — LLM unavailable")
                status = "degraded"
            except Exception as e:
                logger.error(f"contextualize_and_route failed: {e}")
                status = "degraded"

        except Exception:
            status = "failed"
            logger.exception(
                "contextualize_and_route_failed",
                extra={
                    "event": "contextualize_and_route_failed",
                    "stage": "contextualize_and_route",
                    "request_id": request_id,
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                },
            )
            raise
        finally:
            logger.info(
                "contextualize_and_route_completed",
                extra={
                    "event": "contextualize_and_route_completed",
                    "stage": "contextualize_and_route",
                    "request_id": request_id,
                    "user_id": str(user_id),
                    "chapter_id": str(chapter_id),
                    "status": status,
                    "intent": result[1],
                    "total_latency_ms": round((time.monotonic() - start_time) * 1000, 2),
                },
            )

        return result


    async def _generate_followups(self, query: str, answer: str) -> list:
        """Cheap 8B call: 2-3 next questions a student might ask"""
        prompt = (
            "You suggest what a student might naturally ask NEXT. "
            "Given their question and the tutor's answer, return 2-3 short, "
            "specific follow-up questions that build on this answer and deepen "
            "understanding. Phrase them in the student's voice, under 12 words "
            'each. Return ONLY JSON: {"followups": ["...", "..."]}\n\n'
            f"QUESTION: {query}\n\nANSWER: {answer}"
        )
        try:
            resp = await ask_llm(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                json_mode=True,
                temperature=0.5,
                max_tokens=1200,
                timeout=30.0,
            )
            return parse_followups(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            return []

    async def handle_greeting(self, query):
        return (
            "Hello! I'm your study assistant. I'm ready to help you analyze this chapter. "
            "What would you like to know?"
        )

    async def handle_summary(self, chapter_id, user_id):
        """Summarise a chapter from its stored txt deliberately not via retrieval."""

        text = await self._chapter_text(chapter_id, user_id)

        if not text:
            return _result(
                "I couldn't find any readable text in this chapter yet. "
                "If you just uploaded it, give it a moment to finish processing.",
                outcome=PipelineOutcome.INSUFFICIENT_EVIDENCE,
            )

        if len(text) > SUMMARY_CHAR_BUDGET:
            # Say so rather than quietly summarising the first N characters and
            # calling it a summary of the chapter.
            logger.warning(
                "summary input truncated: %d -> %d chars (chapter %s)",
                len(text), SUMMARY_CHAR_BUDGET, chapter_id,
            )
            text = text[:SUMMARY_CHAR_BUDGET]
            truncated = True
        else:
            truncated = False

        messages = [
            {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Summarise the student's material below so they can see the shape "
                "of the whole chapter: what it covers, the main ideas in order, and "
                "how they connect. Lead with one sentence on what the chapter is "
                "about, then the key points. Keep it tight.\n\n"
                f"STUDENT'S MATERIAL:\n{text}"
            )},
        ]

        try:
            async with latency_tracker.track_async("summary_generation"):
                completion = await ask_llm(
                    self.llm_client,
                    messages=messages,
                    model=ANSWER_MODEL,
                    temperature=0.4,
                    max_tokens=4000,
                    timeout=45.0,
                )
            summary = enforce_markdown_spacing(completion.choices[0].message.content or "")
        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            return _failure(e)

        if truncated:
            summary += (
                "\n\n*(This chapter is long — the summary above covers the earlier "
                "sections. Ask about a specific topic for the rest.)*"
            )
        return _result(summary)

    @staticmethod
    @sync_to_async
    def _chapter_text(chapter_id, user_id) -> str:
     
        from .models import Document as _Document

        texts = (
            _Document.objects
            .filter(chapter_id=chapter_id, user_id=user_id)
            .exclude(extracted_text="")
            .exclude(extracted_text__isnull=True)
            .order_by("created_at")
            .values_list("extracted_text", flat=True)
        )
        return "\n\n---\n\n".join(t for t in texts if t and t.strip())

    async def _expand_queries(self, query: str, num: int = 4) -> list[str]:
       
        expansion_prompt = f"Generate {num} alternative phrasings of the following query for retrieval:\n\n{query}"
        completion = await ask_llm(
            self.llm_client,
            model=LLM_MODEL,
            messages=[{"role": "user", "content": expansion_prompt}],
            timeout=5.0,
        )
        expanded = completion.choices[0].message.content.strip().split("\n")
        return [q.strip("-• ") for q in expanded if q.strip()]
    
    async def handle_rag_search(self, query: str, chapter_id: str, user_id: str,
                                request_id=None, *, original_query=None):
       

        logger.info(f"starting RAg search for chapter{chapter_id}, user {user_id}")
        logger.info(f"query: {query}")

        logger.info("Skipping count check going directly to search")
        
        logger.info("Expanding query intelligently...")
        
        expansion_prompt = f"""Analyze this student's question and generate 3 strategic search queries to find the most relevant information.

    Question: {query}

    Generate queries that:
    1. Target the core concept/definition
    2. Look for explanations/mechanisms  
    3. Search for examples/applications

    Return as JSON: {{"queries": ["query1", "query2", "query3"]}}
    """
    
        original_query = original_query if original_query is not None else query
        try:
            async with latency_tracker.track_async("query_expansion"):
                expansion_response = await ask_llm(
                    self.llm_client,
                    messages=[{"role": "user", "content": expansion_prompt}],
                    model=LLM_MODEL,
                    json_mode = True,
                    temperature=0.2,
                    max_tokens=800,
                )
                all_queries = validated_search_queries(
                    expansion_response.choices[0].message.content, original_query, query,
                )

        except LLMUnavailable:
            logger.info(f"Query Expansion failed -> llm unavialable")
            all_queries = [original_query]

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            all_queries = [original_query]

        # ------------------------------------------------------------
        logger.info(f" Search queries: {all_queries}")

        logger.info("Embedding queries...")
        try:
            async with latency_tracker.track_async("embeddings"):
                all_embeddings = await embed_texts(all_queries)
                logger.info(f"Generated {len(all_embeddings)} embeddings")
                
        except Exception as e:
            logger.error(f" Embedding failed: {e}")
            return _failure(e)
        
        logger.info(f"Search scope → user_id={user_id}, chapter_id={chapter_id}")

        search_filter = {
            "user_id": {"$eq": str(user_id)},
            "chapter_id": {"$eq": str(chapter_id)},
        }

        logger.info("Searching vector database (hybrid + RRF...")
        try:
    
            async with latency_tracker.track_async("vector_search"):
                flat_results = await hybrid_search(
                    all_embeddings,
                    query_text=query,
                    filter=search_filter,
                    limit_per_vector=15,  # controls how many candidates are retrieved for each embedding
                )
                logger.info(f" Retrieved {len(flat_results)} fused results")

            if not flat_results:
                logger.warning("Strict filter failed → fallback to user_id only")

                fallback_filter = {"user_id": {"$eq": str(user_id)}}
                flat_results = await hybrid_search(
                    all_embeddings,
                    query_text=query,
                    filter=fallback_filter,
                    limit_per_vector=15,
                )

      

            if flat_results and len(flat_results) > 0:
                first_result = flat_results[0]
                
                if first_result and first_result.payload:
                    preview = first_result.payload.get('text', '')[:200]
                    logger.info(f"First result preview: {preview}...")
                    logger.info(f" First result score: {first_result.score}")
                else:
                    logger.error(" First result has no payload!")
            else:
                logger.error(" NO RESULTS returned from vector search!")
                return _result("I couldn't find relevant information in your document.",
                               outcome=PipelineOutcome.INSUFFICIENT_EVIDENCE)
        except Exception as e:
            logger.error(f" Vector search failed: {e}", exc_info=True)
            return _failure(e)
            

        seen = set()
        unique_results = []
        for r in flat_results:
            text = r.payload.get("text") if r.payload else None
            if text and text not in seen:
                seen.add(text)
                unique_results.append(r)

        logger.info(f"Deduped: {len(unique_results)} chunks")

        if len(unique_results) > 5:
            candidates = unique_results[:RERANK_CANDIDATES]
            texts = [r.payload["text"] for r in candidates]

            async with latency_tracker.track_async("reranking"):
                scores = await rerank_client.rerank(query, texts)

            if scores:
         
                for r, s in zip(candidates, scores):
                    r.score = float(s)
                final_results = sorted(candidates, key=lambda x: x.score, reverse=True)[:FINAL_CHUNKS]
                logger.info(f"Reranked {len(final_results)} chunks. Top score={final_results[0].score:.3f}")
            else:
            
                logger.warning("Rerank unavailable; falling back to RRF ordering")
                final_results = unique_results[:FINAL_CHUNKS]
        else:
            final_results = unique_results[:FINAL_CHUNKS]

        retrieval_evaluator.evaluate(
            query=query,
            chunks=[r.payload['text'] for r in final_results]
        )
    # ===== STEP 5: BUILD CONTEXT =====
        context = "\n\n---\n\n".join([
            r.payload["text"] for r in final_results
        ])

        context_length = len(context)
        logger.info(f" Context built: {context_length} characters")
        logger.info(f"Context preview: {context[:300]}...") 
        
        if context_length < 100:
            logger.error(f" Context too short: {context_length} chars")
            return _result("I found very limited information in your document. Please ensure it uploaded correctly.",
                           outcome=PipelineOutcome.INSUFFICIENT_EVIDENCE)
    
        logger.info(f" Context built: {len(context)} chars from {len(final_results)} chunks")

        # ===== STEP 6: GENERATE ANSWER =====
        logger.info(" Generating answer...")
        answer_messages = build_answer_messages(context, query)

        try:
            async with latency_tracker.track_async("llm_generation"):
                chat_completion = await ask_llm(
                    self.llm_client,
                    messages=answer_messages,
                    model=ANSWER_MODEL,
                    temperature=0.4,      # natural prose, not robotic
                    max_tokens=4000,
                    timeout=45.0,
                )

            raw_output = chat_completion.choices[0].message.content
            logger.info(f" Generated response ({len(raw_output)} chars)")
            logger.info(f" Response preview: {raw_output[:200]}...")

            formatted_output = enforce_markdown_spacing(raw_output)

            sources = build_sources(final_results)
            followups = await self._generate_followups(query, formatted_output)

            return _result(formatted_output, sources=sources, followups=followups)

        except Exception as e:
            logger.error(f" Answer generation failed: {e}", exc_info=True)
            return _failure(e)
        
       

        
