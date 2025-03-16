import os
import uuid
import psycopg
import json
import asyncio
import openai
from typing import Optional, Any, List
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from agents import Agent, Runner, ModelSettings, RunContextWrapper, FunctionTool, WebSearchTool
from agents.agent_output import AgentOutputSchema

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

def get_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def search_similar_subtopics(query_text: str, top_n: int = 3):
    query_vector = get_embedding(query_text)
    with psycopg.connect(conn_info) as conn:
        with conn.cursor() as cursor:
            sql = """
            SELECT id, name, text, "topicId", (embedding::vector) <#> %s::vector AS similarity
            FROM subtopics
            ORDER BY similarity
            LIMIT %s;
            """
            cursor.execute(sql, (query_vector, top_n))
            results = cursor.fetchall()
    return results

# Define the input schema for the query tool
class SubtopicQueryArgs(BaseModel):
    query: str
    top_n: int

# Define the async function that will be invoked by the query tool
async def run_subtopics_query(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = SubtopicQueryArgs.model_validate_json(args)
    results = search_similar_subtopics(parsed.query, parsed.top_n)
    formatted_results = [
        {
            "id": row[0], 
            "name": row[1], 
            "text": row[2], 
            "topicId": row[3], 
            "similarity": row[4]
        }
        for row in results
    ]
    return json.dumps(formatted_results)

params_schema = SubtopicQueryArgs.model_json_schema()
params_schema["additionalProperties"] = False

subtopics_query_tool = FunctionTool(
    name="query_subtopics",
    description=(
        "Queries the subtopics table to find similar subtopics based on a natural language query. "
        "Input should be a JSON with 'query' and 'top_n'."
    ),
    params_json_schema=params_schema,
    on_invoke_tool=run_subtopics_query,
)

# --- Global Conversation Memory Dictionary ---
summary_memories = {}

# Create an LLM instance for conversation summarization
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
    Return a list of messages representing the conversation context.
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
    Converts the conversation context into a plain text block for the prompt.
    """
    conversation = get_conversation_context(session_id)
    context_str = "\n".join(
        f"{msg.__class__.__name__.replace('Message','')}: {msg.content}" 
        for msg in conversation
    )
    return context_str

# --- Dataclasses for Chat Agent Output ---
@dataclass_json
@dataclass
class Source:
    name: str
    link: str
    id: str

@dataclass_json
@dataclass
class ChatReplyOutput:
    message: str
    sources: List[Source]

output_schema = AgentOutputSchema(output_type=ChatReplyOutput)

# --- Chat Agent Definition ---
chat_agent = Agent(
    name="Chat Assistant",
    output_type=ChatReplyOutput,
    model="gpt-4o",
    tools=[subtopics_query_tool, WebSearchTool()],
    model_settings=ModelSettings(
        temperature=0.7,
    ),
    instructions=(
        "You are a helpful assistant. Based on the conversation context provided, produce a clear and helpful reply "
        "to the latest user message. Return your answer as a JSON object with two keys: 'message' and 'sources'. "
        "'message' should be your reply text, and 'sources' should be a list of objects, each containing 'name', 'link', and 'id'. "
        "Include sources only if you have used the query tool to retrieve relevant information; otherwise, return an "
        "When asked for notes and other stuff search the subtopics for sure"
        "empty list for 'sources'. If you cannot find relevant section info, search the internet for credible resources "
        "and return their URLs."
    )
)

async def run_chat_agent(prompt: str) -> ChatReplyOutput:
    result = await Runner.run(chat_agent, prompt)
    return result.final_output

# --- Pydantic Models for Chat Endpoints ---

# ChatCreateRequest now has subtopic_id as optional
class ChatCreateRequest(BaseModel):
    message: str
    subtopic_id: Optional[str] = None

# ChatMessageRequest also has subtopic_id as optional
class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
    subtopic_id: Optional[str] = None

# --- Endpoint: Create a New Chat Session ---
@router.post("/chat/create")
async def create_chat(chat_request: ChatCreateRequest):
    try:
        # Generate a new session id and get chat history
        session_id = str(uuid.uuid4())
        chat_history = get_chat_history(session_id)
        
        # Initialize conversation summary memory
        summary_memories[session_id] = ConversationSummaryBufferMemory(
            llm=summary_llm,
            memory_key="summary",
            return_messages=True
        )
        
        # Add a welcome system message if history is empty
        if not chat_history.messages:
            system_msg = SystemMessage(content="")
            chat_history.add_messages([system_msg])
            summary_memories[session_id].save_context(
                {"input": system_msg.content},
                {"output": system_msg.content}
            )
        
        # Check if the user passed a subtopic_id and fetch subtopic details if present
        subtopic_prompt = ""
        if chat_request.subtopic_id:
            with psycopg.connect(conn_info) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT name, text FROM subtopics WHERE id = %s",
                        (chat_request.subtopic_id,)
                    )
                    subtopic = cursor.fetchone()
                    if subtopic:
                        subtopic_name, subtopic_text = subtopic
                        subtopic_prompt = (
                            f"Subtopic Info:\n"
                            f"Name: {subtopic_name}\n"
                            f"Text: {subtopic_text}\n"
                        )
        
        # Append the user's starting message
        user_message_content = f"{chat_request.message}"
        if subtopic_prompt:
            # Optionally, add subtopic info to the start of user's message or keep it separate
            user_message_content = f"{chat_request.message}"

        user_message = HumanMessage(content=user_message_content)
        chat_history.add_messages([user_message])
        summary_memories[session_id].save_context(
            {"input": user_message_content},
            {"output": ""}
        )
        
        # Build conversation context and create the prompt
        context_text = get_agent_input(session_id)
        prompt = f"{context_text}\nUser: {user_message_content}"
        
        # Run the chat agent
        ai_reply_output = await run_chat_agent(prompt)
        reply_message = AIMessage(content=ai_reply_output.message)
        
        # Append the AI reply to the conversation history
        chat_history.add_messages([reply_message])
        summary_memories[session_id].save_context(
            {"input": user_message_content},
            {"output": ai_reply_output.message}
        )
        
        return {
            "session_id": session_id,
            "reply": {
                "message": ai_reply_output.message,
                "sources": [source.__dict__ for source in ai_reply_output.sources]
            },
            "history": [msg.content for msg in chat_history.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint: Continue an Existing Chat Session ---
@router.post("/chat/message")
async def chat_message(chat_request: ChatMessageRequest):
    try:
        session_id = chat_request.session_id
        chat_history = get_chat_history(session_id)
        
        # Ensure conversation memory exists
        if session_id not in summary_memories:
            summary_memories[session_id] = ConversationSummaryBufferMemory(
                llm=summary_llm,
                memory_key="summary",
                return_messages=True
            )
        
        # If subtopic_id was provided, fetch that subtopic and add it to the prompt
        subtopic_prompt = ""
        if chat_request.subtopic_id:
            with psycopg.connect(conn_info) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT name, text FROM subtopics WHERE id = %s",
                        (chat_request.subtopic_id,)
                    )
                    subtopic = cursor.fetchone()
                    if subtopic:
                        subtopic_name, subtopic_text = subtopic
                        subtopic_prompt = (
                            f"Subtopic Info:\n"
                            f"Name: {subtopic_name}\n"
                            f"Text: {subtopic_text}\n"
                        )
        
        # Append the new user message
        user_message_content = chat_request.message
        if subtopic_prompt:
            # Optionally, add subtopic info to the start of user's message
            user_message_content = f"{chat_request.message}"

        user_message = HumanMessage(content=user_message_content)
        chat_history.add_messages([user_message])
        summary_memories[session_id].save_context(
            {"input": user_message_content},
            {"output": ""}
        )
        
        # Build updated conversation context
        context_text = get_agent_input(session_id)
        prompt = f"{context_text}\nUser: {user_message_content}"
        
        # Run the chat agent
        ai_reply_output = await run_chat_agent(prompt)
        reply_message = AIMessage(content=ai_reply_output.message)
        chat_history.add_messages([reply_message])
        summary_memories[session_id].save_context(
            {"input": user_message_content},
            {"output": ai_reply_output.message}
        )
        
        return {
            "session_id": session_id,
            "reply": {
                "message": ai_reply_output.message,
                "sources": [source.__dict__ for source in ai_reply_output.sources]
            },
            "history": [msg.content for msg in chat_history.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint to Retrieve Messages for an Existing Chat Session ---
@router.get("/chat/messages")
def get_messages(session_id: str):
    try:
        chat_history = get_chat_history(session_id)
        messages = chat_history.messages
        history = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "agent"
            else:
                role = "unknown"
            history.append({"role": role, "content": msg.content})
        return {
            "session_id": session_id,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint: Query Subtopics ---
class SubtopicQueryRequest(BaseModel):
    query: str
    top_n: int = 3

@router.post("/chat/query_subtopics")
def query_subtopics(query_request: SubtopicQueryRequest):
    try:
        results = search_similar_subtopics(query_request.query, query_request.top_n)
        formatted_results = [
            {
                "id": row[0], 
                "name": row[1], 
                "text": row[2], 
                "topicId": row[3], 
                "similarity": row[4]
            }
            for row in results
        ]
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))