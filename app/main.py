"""
FastAPI application -- /health and /chat endpoints.

The /chat handler is the orchestrator. On every call it:
    1. Parses the conversation history (deterministic)
    2. Extracts structured requirements (one LLM call)
    3. Classifies intent (rule-based)
    4. Retrieves candidate assessments (hybrid pipeline)
    5. Generates the response (one LLM call)
    6. Post-processes and validates the output

The service is stateless: all state is reconstructed from the
messages[] array on every request.
"""

import json
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, Recommendation
from app.catalog import CatalogStore
from app.conversation_parser import ConversationParser
from app.requirement_extractor import RequirementExtractor
from app.intent_classifier import IntentClassifier, Intent
from app.retrieval import RetrievalEngine
from app.reasoning import ReasoningEngine
from app.post_processor import RecommendationPostProcessor
from app.validator import RecommendationValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# global singletons, initialized at startup
catalog: CatalogStore = None
parser: ConversationParser = None
extractor: RequirementExtractor = None
classifier: IntentClassifier = None
retrieval: RetrievalEngine = None
reasoning: ReasoningEngine = None
post_processor: RecommendationPostProcessor = None
validator: RecommendationValidator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the catalog and index once at startup."""
    global catalog, parser, extractor, classifier
    global retrieval, reasoning, post_processor, validator

    logger.info("Starting up -- loading catalog and index...")
    start = time.time()

    catalog = CatalogStore()
    parser = ConversationParser()
    extractor = RequirementExtractor()
    classifier = IntentClassifier()
    retrieval = RetrievalEngine(catalog)
    reasoning = ReasoningEngine()
    post_processor = RecommendationPostProcessor(catalog)
    validator = RecommendationValidator(catalog)

    elapsed = time.time() - start
    logger.info("Startup complete in %.1fs (%d catalog entries)", elapsed, len(catalog.entries))

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check. Returns 200 when the service is ready."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversation endpoint.

    Receives the full message history and returns a response with
    an optional recommendation shortlist.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # convert pydantic messages to plain dicts for internal use
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    latest_user_msg = messages[-1]["content"]

    logger.info("[%s] Turn %d: %s", request_id, len(messages), latest_user_msg[:80])

    try:
        # step 1: parse conversation history
        parsed = parser.parse(messages)

        # step 2: extract structured requirements
        structured_state = extractor.extract(messages, parsed)

        # step 3: classify intent
        intent = classifier.classify(latest_user_msg, structured_state, parsed)
        logger.info("[%s] Intent: %s", request_id, intent.value)

        # step 4: retrieve candidates
        # skip retrieval for REFUSE, CONFIRM, and COMPARE (when we already have a shortlist)
        if intent in (Intent.REFUSE, Intent.CONFIRM):
            candidates = []
        elif intent == Intent.COMPARE:
            # for COMPARE, retrieve the named assessments from the catalog
            candidates = retrieval.retrieve(structured_state)
        else:
            candidates = retrieval.retrieve(structured_state)

        retrieval_time = time.time()

        # step 5: generate response via LLM
        raw_response = reasoning.generate(intent, structured_state, candidates, messages)

        llm_time = time.time()

        # step 6a: post-process recommendations
        raw_recs = raw_response.get("recommendations")
        processed_recs = post_processor.process(raw_recs)

        # step 6b: validate recommendations
        validated_recs, validation_failures = validator.validate(processed_recs)

        # build the final response
        response = ChatResponse(
            reply=raw_response.get("reply", ""),
            recommendations=[
                Recommendation(**r) for r in validated_recs
            ] if validated_recs else None,
            end_of_conversation=raw_response.get("end_of_conversation", False),
        )

        # enforce: end_of_conversation only when we have a shortlist
        if response.end_of_conversation and response.recommendations is None:
            response.end_of_conversation = False

        total_time = time.time() - start_time

        # log the request summary
        logger.info(
            "[%s] intent=%s recs=%s validation_failures=%d "
            "retrieval=%.0fms llm=%.0fms total=%.0fms",
            request_id,
            intent.value,
            len(validated_recs) if validated_recs else "null",
            validation_failures,
            (retrieval_time - start_time) * 1000,
            (llm_time - retrieval_time) * 1000,
            total_time * 1000,
        )

        return response

    except Exception as e:
        logger.error("[%s] Unhandled error: %s", request_id, e, exc_info=True)
        # return a safe response rather than a 500
        return ChatResponse(
            reply="Something went wrong processing your request. Could you rephrase?",
            recommendations=None,
            end_of_conversation=False,
        )
