import logging
import os
from groq import Groq, AsyncGroq
from django.conf import settings
import json
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)
LLM_MODEL = "llama-3.1-8b-instant"

class RagPipeline:

    def __init__(self, groq_api_key, qdrant_client, embedding_model):
        self.api_key = groq_api_key
        
        if not self.api_key:
            self.api_key = getattr(settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")

        # 2. Validate
        if not self.api_key:
            logger.error("RagPipeline initialized without GROQ_API_KEY")
            raise ValueError("GROQ_API_KEY is required for RagPipeline. Please check your .env file.")
            
        # Debug log (masked) to confirm key is loaded
        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
        logger.info(f"RagPipeline initialized with GROQ_API_KEY: {masked_key}")
    
        self.groq_client = AsyncGroq(api_key=groq_api_key)
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.LLM_model = LLM_MODEL

    async def run(self, user_query, chat_history, chapter_id, user_id):

        #step 1: contextualization
        refined_query = await self.contextualize_query(user_query, chat_history)
        logger.info = (f"Refine Query: {refined_query}")

        #step 2: Router

        intent = await self.route_query(refined_query)
        logger.info(f"Detected intent: {intent}")

        #step 3: Execute strategy

        if intent == "greeting":
            return await self.handle_greeting(user_query)
        
        elif intent == "summary":
            return await self.handle_summary(chapter_id, user_id)
        
        elif intent == "ambiguous":
            return "I'm not sure I understand. Could you clarify your question about this document?"
        
        else:
            return await self.handle_rag_search(refined_query, chapter_id, user_id)

    async def contextualize_query(self, history, query):

        if not history:
            return query
        
        history_context = "\n".join([f"{msg.sender}: {msg.text}" for msg in history[:-5]])

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
                messages=[{"role": "user", "content":prompt}],
                model=LLM_MODEL,
                temperature=0.1
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Contextualization failed:{e}")
            return query
        
    async def route_query(self, query):

        """
        classifies the query intent.
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
                messages = [{"role":"user", "content": prompt}],
                model = LLM_MODEL,
                temperature=0
            )
            intent = completion.choices[0].message.content.strip().lower()

            if intent not in ["greeting", "summary", "ambiguous", "question"]:
                return "question"
            return intent
        except Exception:
            return "question"
        
    async def handle_greeting(self, query):
        return "Hello! I'm your study assistant. I'm ready to help you analyze this chapter. What would you like to know?"
    
    async def handle_summary(self, chapter_id, user_id):
        return "Here is a summary of the chapter... (Implementation pending DB fetch)"
    
    async def handle_rag_search(self, query, chapter_id, user_id):
    
        from accounts.views import generate_rag_response 
        return await generate_rag_response(query, user_id, chapter_id)