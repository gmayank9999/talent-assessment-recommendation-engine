"""
Pydantic models for the /chat API request and response.

These match the evaluator's expected JSON schema exactly.
recommendations is Optional (nullable) -- the reference traces use null
rather than an empty array when the agent is not committing to a shortlist.
"""

from pydantic import BaseModel, field_validator
from typing import Optional


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def must_have_valid_messages(cls, v):
        if not v:
            raise ValueError("messages list must not be empty")
        if v[-1].role != "user":
            raise ValueError("last message must be from the user")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[list[Recommendation]] = None
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def check_recommendation_count(cls, v):
        if v is not None:
            if len(v) < 1 or len(v) > 10:
                raise ValueError("recommendations must have 1-10 items when not null")
        return v
