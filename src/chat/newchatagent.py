import os
import uuid
import psycopg
import json
import asyncio
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory

# --- Imports for Agents ---
from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema
from dataclasses import dataclass
from dataclasses_json import dataclass_json

# Import a valid LangChain LLM for summarization
from langchain.chat_models import ChatOpenAI

# Initialize FastAPI router
router = APIRouter()

# --- Database and Chat History Setup ---
DB_NAME = os.getenv("DB_NAME", "langchain")
DB_USER = os.getenv("DB_USER", "langchain")
DB_PASSWORD = os.getenv("DB_PASSWORD", "langchain")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "6024")  # Adjust if needed

conn_info = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
sync_connection = psycopg.connect(conn_info)

table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

# --- Global Conversation Memory Dictionary ---
summary_memories = {}

# Create a separate LLM instance for conversation summarization.
summary_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# --- Helper Functions ---
def get_chat_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )

def get_conversation_context(session_id: str):
    """
    Return a list of message objects representing the conversation context.
    If a summary is available, prepend it as a system message.
    """
    chat_history = get_chat_history(session_id)
    if session_id in summary_memories:
        memory_data = summary_memories[session_id].load_memory_variables({})
        summary = memory_data.get("summary", "")
        recent_messages = chat_history.messages[-5:]
        context = [SystemMessage(content=f"Conversation Summary: {summary}")] + recent_messages
    else:
        context = chat_history.messages
    return context

def get_agent_input(session_id: str) -> str:
    """
    Converts the conversation context into a plain text block to be passed as a prompt.
    """
    conversation = get_conversation_context(session_id)
    context_str = "\n".join(
        f"{msg.__class__.__name__.replace('Message','')}: {msg.content}" for msg in conversation
    )
    return context_str

# --- Pydantic Models for Chat Endpoints ---
class ChatCreateRequest(BaseModel):
    message: str

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

# --- Chat Agent Definition ---
# This agent is used to generate replies in our conversation endpoints.
chat_agent = Agent(
    name="Chat Assistant",
    output_type=str,  # We expect plain text output.
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.7,
    ),
    instructions=(
        "You are a helpful assistant. Given the conversation context provided, "
        "produce a clear and helpful reply to the latest user message."
    )
)

async def run_chat_agent(prompt: str) -> str:
    result = await Runner.run(chat_agent, prompt)
    return result.final_output

# --- Endpoint: Create a New Chat Session ---
@router.post("/newchat/create")
async def create_chat(chat_request: ChatCreateRequest):
    try:
        # Generate a new session id and get chat history.
        session_id = str(uuid.uuid4())
        chat_history = get_chat_history(session_id)
        
        # Initialize conversation summary memory using the valid summary_llm.
        summary_memories[session_id] = ConversationSummaryBufferMemory(
            llm=summary_llm,
            memory_key="summary",
            return_messages=True
        )
        
        # Add a welcome system message if history is empty.
        if not chat_history.messages:
            system_msg = SystemMessage(content="Welcome to the chat!")
            chat_history.add_messages([system_msg])
            summary_memories[session_id].save_context(
                {"input": system_msg.content},
                {"output": system_msg.content}
            )
        
        # Append the user's starting message.
        user_message = HumanMessage(content=chat_request.message)
        chat_history.add_messages([user_message])
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": ""}
        )
        
        # Build conversation context and create the prompt for the agent.
        context_text = get_agent_input(session_id)
        prompt = f"{context_text}\nUser: {chat_request.message}"
        
        # Run the chat agent.
        ai_reply_text = await run_chat_agent(prompt)
        reply_message = AIMessage(content=ai_reply_text)
        chat_history.add_messages([reply_message])
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": ai_reply_text}
        )
        
        return {
            "session_id": session_id,
            "reply": ai_reply_text,
            "history": [msg.content for msg in chat_history.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint: Continue an Existing Chat Session ---
@router.post("/newchat/message")
async def chat_message(chat_request: ChatMessageRequest):
    try:
        session_id = chat_request.session_id
        chat_history = get_chat_history(session_id)
        
        # Ensure conversation memory exists.
        if session_id not in summary_memories:
            summary_memories[session_id] = ConversationSummaryBufferMemory(
                llm=summary_llm,
                memory_key="summary",
                return_messages=True
            )
        
        # Append the new user message.
        user_message = HumanMessage(content=chat_request.message)
        chat_history.add_messages([user_message])
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": ""}
        )
        
        # Build updated conversation context.
        context_text = get_agent_input(session_id)
        prompt = f"{context_text}\nUser: {chat_request.message}"
        
        # Run the chat agent.
        ai_reply_text = await run_chat_agent(prompt)
        reply_message = AIMessage(content=ai_reply_text)
        chat_history.add_messages([reply_message])
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": ai_reply_text}
        )
        
        return {
            "session_id": session_id,
            "reply": ai_reply_text,
            "history": [msg.content for msg in chat_history.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Course Schedule Extractor Agent ---
@dataclass_json
@dataclass
class CourseContentOutput:
    topics: list[str]
    description: str
    reading_materials: list[str]

output_schema = AgentOutputSchema(output_type=CourseContentOutput)

course_agent = Agent(
    name="Course Schedule Extractor", 
    output_type=CourseContentOutput,
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.1,
    ),
    instructions='''Extract the following details from the provided course schedule:
1. Lecture topics as a JSON list of strings (exclude exams, holidays, non-course material, course introductions, discussion events, etc.).
2. A short description of the course summarizing its content.
3. Reading materials as a JSON array of strings containing recommended texts.
Output the result as a JSON object with the keys "topics", "description", and "reading_materials".'''
)

async def run_course_agent(content: str) -> CourseContentOutput:
    result = await Runner.run(course_agent, content)
    return result.final_output