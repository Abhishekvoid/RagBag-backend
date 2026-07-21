# Tutor Answer Quality + Follow-ups + Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RAG tutor answer like a top AI — upgrade the answer model to Groq 70B, rewrite the prompt into an adaptive Claude-like tutor voice, and add auto-sending follow-up question chips plus lightweight source chips.

**Architecture:** The answer step of `RagPipeline.handle_rag_search` switches to `llama-3.3-70b-versatile` with a restructured system+user prompt. A second cheap 8B call generates 2–3 follow-up questions. Source chips are built from the retrieved chunks' metadata. `run()` returns a structured `{answer, sources, followups}` dict; the view stores sources in the existing `citations` field and follow-ups in a new `suggestions` field, and returns both. The frontend renders clickable chips that auto-send the follow-up as the next question.

**Tech Stack:** Backend — Django 5, DRF, Groq (async), Celery. Frontend — Next.js, Zustand, axios, zod, TypeScript.

## Global Constraints

- Backend tests: `./venv/Scripts/python.exe manage.py test accounts.tests.<Class> -v 2 --keepdb` (the `--keepdb` avoids a Postgres teardown error in this environment).
- Frontend has no test runner: verify with `npx tsc --noEmit` from `frontend/RAG_tutor_frontend`. Keep pure logic isolated. Do NOT add a test framework.
- Backend paths relative to `C:\Users\Abhishek\RAG tutor\backend`. Frontend paths relative to `C:\Users\Abhishek\RAG tutor\frontend\RAG_tutor_frontend`.
- Answer model = `llama-3.3-70b-versatile` (Groq). Sub-tasks (query expansion, follow-ups) stay on `llama-3.1-8b-instant`.
- `ask_llm(groq_client, messages, *, model, json_mode=False, **kwargs)` is the LLM entry point (`utils/llm_gateway.py:14`); pass `model` per call.
- Follow-up generation must NEVER break the answer: any parse/LLM error yields `[]`.
- Commit after every task.

---

## File Structure

**Backend**
- Modify `accounts/rag_pipeline.py` — answer model constant, `build_answer_messages()`, `parse_followups()`, `build_sources()`, `_generate_followups()`, richer `handle_rag_search`/`run` return shape.
- Modify `accounts/models.py` — add `ChatMessage.suggestions` JSONField.
- Create migration `accounts/migrations/00XX_chatmessage_suggestions.py` (via `makemigrations`).
- Modify `accounts/serializers.py` — add `suggestions` to `ChatMessageSerializer`.
- Modify `accounts/views.py` — unpack the dict, enrich sources with titles, store + return sources/follow-ups.
- Modify `accounts/tests.py` — tests for the three pure helpers + the view response shape.

**Frontend**
- Modify `src/features/notebook/notebook.schema.ts` — extend `ragChatResponseSchema` and `chatMessageDTOSchema`.
- Modify `src/lib/store/useNotebook.ts` — `Message` type gains `sources`/`followups`; `sendMessage` maps them through.
- Modify `src/components/dashboard/ContentPanel/ChatView.tsx` — render source chips + follow-up chips (auto-send on click).

---

## Task 1: Answer model + rewritten adaptive prompt

**Files:**
- Modify: `accounts/rag_pipeline.py` (add constant near line 35; add helper; edit answer call ~559-613)
- Test: `accounts/tests.py`

**Interfaces:**
- Produces: `ANSWER_MODEL = "llama-3.3-70b-versatile"`; `build_answer_messages(context: str, query: str) -> list[dict]` returning exactly two messages `[{"role":"system",...},{"role":"user",...}]` where the user message contains both `context` and `query`.

- [ ] **Step 1: Write the failing test**

Add to `accounts/tests.py`:

```python
class BuildAnswerMessagesTests(TestCase):
    def test_returns_system_then_user_with_context_and_query(self):
        from accounts.rag_pipeline import build_answer_messages
        msgs = build_answer_messages("CTX-TEXT", "What is X?")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("CTX-TEXT", msgs[1]["content"])
        self.assertIn("What is X?", msgs[1]["content"])
        # tutor persona + grounding rule present in system prompt
        self.assertIn("tutor", msgs[0]["content"].lower())
        self.assertIn("Beyond your notes", msgs[0]["content"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.BuildAnswerMessagesTests -v 2 --keepdb`
Expected: FAIL — `ImportError: cannot import name 'build_answer_messages'`.

- [ ] **Step 3: Add the constant and helper**

In `accounts/rag_pipeline.py`, just after `LLM_MODEL = "llama-3.1-8b-instant"` (line 35):

```python
ANSWER_MODEL = "llama-3.3-70b-versatile"

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
```

- [ ] **Step 4: Use the helper + model in the answer call**

Replace the `final_prompt = f"""..."""` block (lines ~559-602) and the `ask_llm(...)` call (lines ~604-613) so the generation step reads:

```python
        # ===== STEP 6: GENERATE ANSWER =====
        logger.info("🤖 Generating answer...")

        answer_messages = build_answer_messages(context, query)

        try:
            async with latency_tracker.track_async("llm_generation"):
                chat_completion = await ask_llm(
                    self.groq_client,
                    messages=answer_messages,
                    model=ANSWER_MODEL,
                    temperature=0.4,      # natural prose, not robotic
                    max_tokens=1500,      # room for adaptive length
                    timeout=45.0,
                )
```

Leave the rest of the `try` (raw_output extraction, `enforce_markdown_spacing`, return) unchanged for now.

- [ ] **Step 5: Run test to verify it passes**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.BuildAnswerMessagesTests -v 2 --keepdb`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add accounts/rag_pipeline.py accounts/tests.py
git commit -m "feat(rag): 70B answer model + rewritten adaptive tutor prompt"
```

---

## Task 2: Follow-up question generation

**Files:**
- Modify: `accounts/rag_pipeline.py` (add helpers + method)
- Test: `accounts/tests.py`

**Interfaces:**
- Produces:
  - `parse_followups(raw: str) -> list[str]` — parses a JSON string `{"followups":[...]}`, returns up to 3 stripped non-empty strings; `[]` on any error.
  - `async _generate_followups(self, query: str, answer: str) -> list[str]` — one 8B `json_mode` call; returns `parse_followups(...)`, or `[]` on LLM error.

- [ ] **Step 1: Write the failing test**

```python
class ParseFollowupsTests(TestCase):
    def test_valid_json(self):
        from accounts.rag_pipeline import parse_followups
        raw = '{"followups": ["What is chaining?", "When resize?", "Load factor?"]}'
        self.assertEqual(
            parse_followups(raw),
            ["What is chaining?", "When resize?", "Load factor?"],
        )

    def test_caps_at_three_and_strips(self):
        from accounts.rag_pipeline import parse_followups
        raw = '{"followups": ["  a ", "b", "c", "d"]}'
        self.assertEqual(parse_followups(raw), ["a", "b", "c"])

    def test_drops_empty_entries(self):
        from accounts.rag_pipeline import parse_followups
        raw = '{"followups": ["", "  ", "real one"]}'
        self.assertEqual(parse_followups(raw), ["real one"])

    def test_garbage_returns_empty(self):
        from accounts.rag_pipeline import parse_followups
        self.assertEqual(parse_followups("not json"), [])
        self.assertEqual(parse_followups('{"nope": 1}'), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.ParseFollowupsTests -v 2 --keepdb`
Expected: FAIL — cannot import `parse_followups`.

- [ ] **Step 3: Add `parse_followups` and `_generate_followups`**

Add `parse_followups` near `build_answer_messages` in `accounts/rag_pipeline.py`:

```python
def parse_followups(raw: str) -> list:
    """Parse a follow-ups JSON string into up to 3 clean questions. Never raises."""
    try:
        data = json.loads(raw)
        items = data.get("followups", [])
        if not isinstance(items, list):
            return []
        cleaned = [str(x).strip() for x in items if str(x).strip()]
        return cleaned[:3]
    except Exception:
        return []
```

Add the method inside `class RagPipeline` (e.g. after `_expand_queries`):

```python
    async def _generate_followups(self, query: str, answer: str) -> list:
        """Cheap 8B call: 2-3 next questions a student might ask. []-safe."""
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
                self.groq_client,
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                json_mode=True,
                temperature=0.5,
                max_tokens=200,
                timeout=15.0,
            )
            return parse_followups(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.ParseFollowupsTests -v 2 --keepdb`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add accounts/rag_pipeline.py accounts/tests.py
git commit -m "feat(rag): 8B follow-up question generation with safe parsing"
```

---

## Task 3: Source chip builder

**Files:**
- Modify: `accounts/rag_pipeline.py` (add helper)
- Test: `accounts/tests.py`

**Interfaces:**
- Produces: `build_sources(final_results) -> list[dict]` — from result objects with `.payload` dicts, dedup by `document_id`, keep first 3, each `{"document_id": str, "snippet": str}` where snippet is the chunk text truncated to 140 chars.

- [ ] **Step 1: Write the failing test**

```python
class BuildSourcesTests(TestCase):
    def test_dedup_and_snippet(self):
        from types import SimpleNamespace
        from accounts.rag_pipeline import build_sources
        results = [
            SimpleNamespace(payload={"document_id": "d1", "text": "A" * 200}),
            SimpleNamespace(payload={"document_id": "d1", "text": "dup"}),
            SimpleNamespace(payload={"document_id": "d2", "text": "second"}),
        ]
        srcs = build_sources(results)
        self.assertEqual(len(srcs), 2)
        self.assertEqual(srcs[0]["document_id"], "d1")
        self.assertEqual(len(srcs[0]["snippet"]), 140)
        self.assertEqual(srcs[1]["document_id"], "d2")

    def test_caps_at_three(self):
        from types import SimpleNamespace
        from accounts.rag_pipeline import build_sources
        results = [
            SimpleNamespace(payload={"document_id": f"d{i}", "text": "t"})
            for i in range(6)
        ]
        self.assertEqual(len(build_sources(results)), 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.BuildSourcesTests -v 2 --keepdb`
Expected: FAIL — cannot import `build_sources`.

- [ ] **Step 3: Add `build_sources`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.BuildSourcesTests -v 2 --keepdb`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add accounts/rag_pipeline.py accounts/tests.py
git commit -m "feat(rag): build lightweight source chips from retrieved chunks"
```

---

## Task 4: Return `{answer, sources, followups}` from the pipeline

**Files:**
- Modify: `accounts/rag_pipeline.py` (`handle_rag_search` return points + `run`)

**Interfaces:**
- Consumes: `build_sources`, `_generate_followups` (Tasks 2-3).
- Produces: `run(...)` and `handle_rag_search(...)` return a dict `{"answer": str, "sources": list, "followups": list}`. All non-RAG branches (greeting/summary/ambiguous/error) return the same shape with empty `sources`/`followups`.

- [ ] **Step 1: Add a wrapper helper**

In `accounts/rag_pipeline.py` (module level, near the other helpers):

```python
def _result(answer: str, sources=None, followups=None) -> dict:
    return {
        "answer": answer,
        "sources": sources or [],
        "followups": followups or [],
    }
```

- [ ] **Step 2: Convert `handle_rag_search` early-exit strings to `_result(...)`**

In `handle_rag_search`, wrap every `return "<string>"` with `_result(...)`. The exact lines to change (strings kept verbatim):

```python
        # ~line 487
                return _result("I couldn't find relevant information in your document. This might be a technical issue.")
        # ~line 490
            return _result("Search failed. Please try again.")
        # ~line 552
            return _result("I found very limited information in your document. Please ensure it uploaded correctly.")
```

And the LLM error returns near the end of the method:

```python
        except LLMUnavailable:
            logger.warning("Answer generation skipped — LLM unavailable")
            return _result("AI is temporarily unavailable. Please try again shortly.")

        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}", exc_info=True)
            return _result("Failed to generate an answer. Please try again.")
```

- [ ] **Step 3: Build sources + follow-ups on the success path**

Replace the success return (`formatted_output = enforce_markdown_spacing(raw_output)` / `return formatted_output`, ~lines 619-620) with:

```python
            formatted_output = enforce_markdown_spacing(raw_output)

            sources = build_sources(final_results)
            followups = await self._generate_followups(query, formatted_output)

            return _result(formatted_output, sources=sources, followups=followups)
```

- [ ] **Step 4: Normalize the non-RAG branches in `run`**

In `run`, wrap the branches that currently assign plain strings (lines ~90, 101, 103):

```python
            if self.is_greeting(user_query):
                return _result(await self.handle_greeting(user_query))
```

```python
            if intent == "summary":
                result = _result(await self.handle_summary(chapter_id, user_id))
            elif intent == "ambiguous":
                result = _result("I'm not sure I understand. Could you clarify your question about this document?")
            else:
                result = await self.handle_rag_search(refined_query, chapter_id, user_id, request_id)
```

(`handle_rag_search` already returns a dict, so the `else` needs no wrapping.)

- [ ] **Step 5: Write an integration test for the shape**

```python
class RunReturnShapeTests(TestCase):
    def test_greeting_returns_result_dict(self):
        import asyncio
        from accounts.rag_pipeline import RagPipeline
        pipe = RagPipeline(groq_api_key="x", embedding_model="m")
        out = asyncio.new_event_loop().run_until_complete(
            pipe.run("hello", chat_history=[], chapter_id="c", user_id="u")
        )
        self.assertIn("answer", out)
        self.assertEqual(out["sources"], [])
        self.assertEqual(out["followups"], [])
        self.assertIsInstance(out["answer"], str)
```

> This relies on `is_greeting("hello")` being True and `handle_greeting` not calling the network. If `handle_greeting` does call the LLM, assert only on `set(out.keys()) == {"answer","sources","followups"}` by mocking `RagPipeline.handle_greeting` with `unittest.mock.patch.object` to return `"Hi!"`.

- [ ] **Step 6: Run test**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.RunReturnShapeTests -v 2 --keepdb`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add accounts/rag_pipeline.py accounts/tests.py
git commit -m "feat(rag): return {answer, sources, followups} from pipeline"
```

---

## Task 5: `ChatMessage.suggestions` field + serializer

**Files:**
- Modify: `accounts/models.py` (ChatMessage, ~line 130-142)
- Create: migration
- Modify: `accounts/serializers.py` (`ChatMessageSerializer`, ~line 187-193)

**Interfaces:**
- Produces: `ChatMessage.suggestions` JSONField (null/blank); serializer exposes `suggestions` (read-only).

- [ ] **Step 1: Add the field**

In `accounts/models.py`, in `ChatMessage` after `citations = models.JSONField(null=True, blank=True)`:

```python
    suggestions = models.JSONField(null=True, blank=True)
```

- [ ] **Step 2: Make the migration**

Run: `./venv/Scripts/python.exe manage.py makemigrations accounts`
Expected: creates `accounts/migrations/00XX_chatmessage_suggestions.py`.

- [ ] **Step 3: Apply it**

Run: `./venv/Scripts/python.exe manage.py migrate accounts`
Expected: `Applying accounts.00XX_chatmessage_suggestions... OK`

- [ ] **Step 4: Expose in serializer**

In `accounts/serializers.py` `ChatMessageSerializer.Meta`:

```python
        fields = ['id', 'session', 'sender', 'text', 'created_at', 'citations', 'tokens', 'error', 'suggestions']
        read_only_fields = ['id', 'created_at', 'citations', 'tokens', 'error', 'suggestions']
```

- [ ] **Step 5: Verify check**

Run: `./venv/Scripts/python.exe manage.py check`
Expected: `System check identified no issues`.

- [ ] **Step 6: Commit**

```bash
git add accounts/models.py accounts/migrations/ accounts/serializers.py
git commit -m "feat(chat): add ChatMessage.suggestions field for follow-ups"
```

---

## Task 6: View — store + return sources & follow-ups

**Files:**
- Modify: `accounts/views.py` (`RAGChatMessageView.post`, ~lines 478-497)

**Interfaces:**
- Consumes: `run()` dict from Task 4; `Document` for title enrichment.
- Produces: response JSON `{id, sender:"ai", text, sources:[{document_id,title,snippet}], followups:[...]}` and persists `citations=sources`, `suggestions=followups`.

- [ ] **Step 1: Replace the run-call + save + response block**

Replace lines ~478-497 with:

```python
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
```

- [ ] **Step 2: Write a view test (pipeline mocked)**

```python
class RagChatResponseShapeTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="c@b.com", password="pw12345", name="C")
        self.client.force_authenticate(self.user)

    def test_response_includes_sources_and_followups(self):
        from unittest.mock import patch
        from accounts.models import Chapter, Document
        chapter = Chapter.objects.create(user=self.user, name="Ch")
        doc = Document.objects.create(
            user=self.user, chapter=chapter, title="My Doc",
            file="uploads/x.txt", file_type="txt",
            status=Document.STATUS_COMPLETED)
        fake = {
            "answer": "The answer.",
            "sources": [{"document_id": str(doc.id), "snippet": "snip"}],
            "followups": ["Next q?"],
        }
        with patch("accounts.views.rag_pipeline.run", return_value=fake):
            res = self.client.post(
                "/auth/rag-chat/",
                {"chapter": str(chapter.id), "text": "hi"},
                format="json")
        self.assertEqual(res.status_code, 201, res.content)
        self.assertEqual(res.data["text"], "The answer.")
        self.assertEqual(res.data["followups"], ["Next q?"])
        self.assertEqual(res.data["sources"][0]["title"], "My Doc")
```

> `async_to_sync(rag_pipeline.run)` wraps the patched sync return fine because `patch` replaces `run` with a `MagicMock` returning `fake`; `async_to_sync` of a non-coroutine value in this path is exercised by the existing code — if it errors, patch with `new=AsyncMock(return_value=fake)` from `unittest.mock`.

- [ ] **Step 3: Run test**

Run: `./venv/Scripts/python.exe manage.py test accounts.tests.RagChatResponseShapeTests -v 2 --keepdb`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add accounts/views.py accounts/tests.py
git commit -m "feat(chat): persist and return source chips + follow-up suggestions"
```

---

## Task 7: Frontend response schema

**Files:**
- Modify: `src/features/notebook/notebook.schema.ts` (`ragChatResponseSchema` ~line 120, `chatMessageDTOSchema` ~line 108)

**Interfaces:**
- Produces: `ragChatResponseSchema` and `chatMessageDTOSchema` include optional `sources` (array of `{document_id, title, snippet}`) and `followups` (array of strings). `RagChatMessageDTO` type updated by inference.

- [ ] **Step 1: Add a shared source schema and extend both message schemas**

Above `ragChatResponseSchema`:

```typescript
export const sourceChipSchema = z.object({
  document_id: z.string(),
  title: z.string(),
  snippet: z.string(),
});
```

Extend `ragChatResponseSchema`:

```typescript
export const ragChatResponseSchema = z.object({
  id: z.uuid(),
  sender: z.enum(["user", "ai"]),
  text: z.string(),
  sources: z.array(sourceChipSchema).optional().default([]),
  followups: z.array(z.string()).optional().default([]),
});
```

Extend `chatMessageDTOSchema` (history load) by adding, before its closing `});`:

```typescript
  suggestions: z.array(z.string()).nullable().optional(),
```

- [ ] **Step 2: Typecheck**

Run (from `frontend/RAG_tutor_frontend`): `npx tsc --noEmit`
Expected: no errors. (`citations` is already `z.any().nullable()`, so source chips loaded from history parse without further change.)

- [ ] **Step 3: Commit**

```bash
git add src/features/notebook/notebook.schema.ts
git commit -m "feat(notebook): schema for source chips and follow-ups"
```

---

## Task 8: Store — carry sources & follow-ups on messages

**Files:**
- Modify: `src/lib/store/useNotebook.ts` (`Message` type ~line 38; `sendMessage` ~line 535-540)

**Interfaces:**
- Consumes: `RagChatMessageDTO` (now with `sources`/`followups`).
- Produces: `Message` gains `sources?: {document_id;title;snippet}[]` and `followups?: string[]`; `sendMessage` copies them onto the AI message.

- [ ] **Step 1: Extend the `Message` type**

```typescript
export type Message = {
  id: string;
  sender: "user" | "ai";
  text: string;
  error?: boolean;
  sources?: { document_id: string; title: string; snippet: string }[];
  followups?: string[];
};
```

- [ ] **Step 2: Map fields through in `sendMessage`**

Replace the mapping line (`const aiMessage: Message = { ...aiResponse, sender: "ai" };`) with:

```typescript
          const aiMessage: Message = {
            id: aiResponse.id,
            sender: "ai",
            text: aiResponse.text,
            sources: aiResponse.sources ?? [],
            followups: aiResponse.followups ?? [],
          };
```

> `MessageDTO` in the store is a loose local type; `aiResponse` is actually `RagChatMessageDTO`. If TypeScript complains that `sources`/`followups` don't exist on `MessageDTO`, change the `sendRagMessage` return annotation usage: the value already conforms to `RagChatMessageDTO`, so update the local `const aiResponse` type to `RagChatMessageDTO` (import it if not already) instead of `MessageDTO`.

- [ ] **Step 3: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/lib/store/useNotebook.ts
git commit -m "feat(store): carry source chips and follow-ups on AI messages"
```

---

## Task 9: ChatView — render source chips + auto-sending follow-ups

**Files:**
- Modify: `src/components/dashboard/ContentPanel/ChatView.tsx` (AI message branch ~lines 130-137)

**Interfaces:**
- Consumes: `Message.sources`, `Message.followups`, store `sendMessage`, `isAiResponding`.
- Produces: under each AI answer, a "Sources" row of chips and a follow-up chip row; clicking a follow-up calls `sendMessage(question)` (auto-send), disabled while `isAiResponding`.

- [ ] **Step 1: Render chips in the AI branch**

Replace the AI message branch (lines ~130-137) with:

```tsx
              ) : (
                <div key={msg.id} className="flex items-start gap-3 duration-300 animate-in fade-in slide-in-from-bottom-2">
                  <BotAvatar />
                  <div className="min-w-0 flex-1 break-words">
                    <AnswerMarkdown>{msg.text}</AnswerMarkdown>

                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3">
                        <p className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                          Sources
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {msg.sources.map((s, i) => (
                            <span
                              key={`${s.document_id}-${i}`}
                              title={s.snippet}
                              className="max-w-[220px] truncate rounded-md border border-border bg-secondary/40 px-2 py-1 text-[11px] text-muted-foreground"
                            >
                              {s.title}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {msg.followups && msg.followups.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {msg.followups.map((q, i) => (
                          <button
                            key={i}
                            type="button"
                            disabled={isAiResponding}
                            onClick={() => sendMessage(q)}
                            className="rounded-full border border-primary/30 bg-primary/5 px-3 py-1.5 text-[12.5px] text-foreground transition-colors hover:bg-primary/10 disabled:opacity-50"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ),
```

- [ ] **Step 2: Ensure `sendMessage` is in scope**

Confirm `ChatView` already pulls `sendMessage` from the store (used in `handleSubmit`). If not, add near the other store selectors:

```tsx
  const sendMessage = useNotebookStore((state) => state.sendMessage);
```

- [ ] **Step 3: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Manual end-to-end verification**

With the full stack running (Redis, TEI, Celery, Django, Next), open a COMPLETED chapter and ask a question. Expected:
1. Answer reads naturally/adaptively (70B), not the old rigid bold-first-line format.
2. A "Sources" chip row shows the document title(s); hovering shows the snippet.
3. 2–3 follow-up chips appear; clicking one immediately sends it and produces a new answer + new chips.
4. If the 8B follow-up call fails, the answer still renders (no chips) — no error.

- [ ] **Step 5: Commit**

```bash
git add src/components/dashboard/ContentPanel/ChatView.tsx
git commit -m "feat(chat): render source chips and auto-sending follow-up chips"
```

---

## Self-Review Notes

- **Spec coverage:** Q1 70B answer model → Task 1. Q2 adaptive voice → Task 1 prompt. Q3 follow-ups via 8B → Tasks 2,4; chips → Task 9. Q4 source chips → Tasks 3,6,9. Q5 auto-send → Task 9 (`onClick={() => sendMessage(q)}`). Q6 grounded-but-teaching → Task 1 system prompt ("Beyond your notes"). Q7 streaming deferred — not in this plan by design.
- **Type consistency:** `build_answer_messages`, `parse_followups`, `build_sources`, `_result`, `_generate_followups` defined in Tasks 1-4 and consumed unchanged. Source object shape `{document_id,title,snippet}` consistent across Task 6 (view adds `title`), Task 7 (`sourceChipSchema`), Task 8 (`Message.sources`), Task 9 (render). `followups: string[]` consistent across Tasks 2/4/6/7/8/9.
- **Safety:** follow-up failures return `[]` (Task 2); source enrichment tolerates missing titles ("Source"); non-RAG branches return the full dict shape (Task 4).
- **No migration risk:** only additive nullable `suggestions` field (Task 5).
```
