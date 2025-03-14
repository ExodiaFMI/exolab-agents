from fastapi import APIRouter, HTTPException, Body
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
from pydantic import BaseModel
from agents import Agent, Runner, ModelSettings

# Define a model for user messages
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

# Define a model for the chat request
class ChatRequest(BaseModel):
    history: List[ChatMessage]  # Conversation history between the user and assistant
    user_message: str  # The latest message from the user

    class Config:
        schema_extra = {
            "example": {
                "history": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I help you today?"}
                ],
                "user_message": "Can you explain Newton's first law?"
            }
        }

# Define a model for the chat response
@dataclass
class ChatResponse:
    role: str
    content: str

# Set up the agent for chat
agent_chat = Agent(
    name="Chat Assistant",
    output_type=ChatResponse,
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.7),
    instructions='''You are a helpful AI assistant. Respond to user messages conversationally while maintaining context from previous interactions.

Consider the provided conversation history when generating responses.

Output a JSON object with "role" as "assistant" and "content" containing the response message.
'''
)

# Initialize FastAPI router
router = APIRouter()

async def generate_chat_response(history: List[ChatMessage], user_message: str) -> ChatResponse:
    """Generate a chat response based on conversation history and the latest user message."""
    
    # Format conversation history as context
    chat_history = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history])
    prompt_context = f"""
Previous Conversation:
{chat_history}

User: {user_message}
Assistant:
"""
    
    result = await Runner.run(agent_chat, prompt_context)
    return result.final_output

@router.post("/chat", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "response": {
                                     "role": "assistant",
                                     "content": "Newton's first law states that an object at rest stays at rest..."
                                 }
                             }
                         }
                     }
                 }
             })
async def chat(request: ChatRequest = Body(...)):
    """
    Chat endpoint that maintains conversation history.

    **Request Example:**
    ```json
    {
        "history": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"}
        ],
        "user_message": "Can you explain Newton's first law?"
    }
    ```

    **Response Example:**
    ```json
    {
        "response": {
            "role": "assistant",
            "content": "Newton's first law states that an object at rest stays at rest..."
        }
    }
    ```
    """
    try:
        response = await generate_chat_response(request.history, request.user_message)
        return {"response": response.__dict__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))