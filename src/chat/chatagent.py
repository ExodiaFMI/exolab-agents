import os
import uuid
import psycopg
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import LangChain messages and history manager
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

# NEW: Imports for SQL agent
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent, initialize_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import openai
from langchain.tools import BaseTool




# Initialize FastAPI router
router = APIRouter()

# --- Database and Chat History Setup ---

# Load database credentials from environment variables (with defaults)
DB_NAME = os.getenv("DB_NAME", "langchain")
DB_USER = os.getenv("DB_USER", "langchain")
DB_PASSWORD = os.getenv("DB_PASSWORD", "langchain")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "6024")  # Adjust port if needed

# Build connection string and create a synchronous connection for chat history
conn_info = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
sync_connection = psycopg.connect(conn_info)

# Define table name and create tables if not already created
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

# --- LLM Initialization ---
llm = ChatOpenAI(model_name="gpt-4o")
summary_llm = ChatOpenAI(model_name="gpt-4o-mini")

# Global dictionary to store Conversation Summary Buffer Memory for each session.
summary_memories = {}

# Utility function to get a chat history manager for a given session id
def get_chat_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )

# Helper function to get conversation context using the summary memory.
def get_conversation_context(session_id: str):
    chat_history = get_chat_history(session_id)
    # If a summary memory exists for this session, load its summary.
    if session_id in summary_memories:
        memory_data = summary_memories[session_id].load_memory_variables({})
        summary = memory_data.get("summary", "")
        # For context, combine the summary with the most recent few messages.
        recent_messages = chat_history.messages[-5:]
        # Prepend the summary as a system message.
        context = [SystemMessage(content=f"Conversation Summary: {summary}")] + recent_messages
    else:
        context = chat_history.messages
    return context

# --- Pydantic Models for Chat Requests ---
class ChatCreateRequest(BaseModel):
    message: str

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

# --- Endpoint to create a new chat session with a starting message ---
@router.post("/chat/create")
def create_chat(chat_request: ChatCreateRequest):
    try:
        # Generate a new session ID for the chat
        session_id = str(uuid.uuid4())
        chat_history = get_chat_history(session_id)
        
        # Create and store a new Conversation Summary Buffer Memory instance for this session.
        summary_memories[session_id] = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="summary",
            return_messages=True
        )
        
        # Optionally add a system prompt as the first message if the history is empty
        if not chat_history.messages:
            system_msg = SystemMessage(content="Welcome to the chat!")
            chat_history.add_messages([system_msg])
            # Optionally, update the memory with the system prompt.
            summary_memories[session_id].save_context(
                {"input": system_msg.content},
                {"output": system_msg.content}
            )
        
        # Append the user's starting message
        user_message = HumanMessage(content=chat_request.message)
        chat_history.add_messages([user_message])
        
        # Update memory with the new user message.
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": ""}
        )
        
        # Retrieve the conversation context with a summary (if available)
        conversation = get_conversation_context(session_id)
        
        # Call the LLM with the conversation context
        ai_reply = llm(conversation)
        if hasattr(ai_reply, "content"):
            reply_message = AIMessage(content=ai_reply.content)
        else:
            reply_message = AIMessage(content=str(ai_reply))
        
        # Save the LLM's reply in the chat history
        chat_history.add_messages([reply_message])
        
        # Update the memory with the response (completing the exchange)
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": reply_message.content}
        )
        
        # Return the session id, the LLM reply, and the full conversation history
        return {
            "session_id": session_id,
            "reply": reply_message.content,
            "history": [msg.content for msg in chat_history.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint for sending a new message in an existing chat session ---
@router.post("/chat/message")
def chat_message(chat_request: ChatMessageRequest):
    try:
        session_id = chat_request.session_id
        chat_history = get_chat_history(session_id)
        
        # Ensure the session has an associated summary memory
        if session_id not in summary_memories:
            summary_memories[session_id] = ConversationSummaryBufferMemory(
                llm=summary_llm,
                memory_key="summary",
                return_messages=True
            )
        
        # Append the user's message to the chat history
        user_message = HumanMessage(content=chat_request.message)
        chat_history.add_messages([user_message])
        
        # Update the memory with the new user message.
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": ""}
        )
        
        # Retrieve the conversation context with the summary
        conversation = get_conversation_context(session_id)
        
        # Call the LLM with the updated conversation context
        ai_reply = llm(conversation)
        if hasattr(ai_reply, "content"):
            reply_message = AIMessage(content=ai_reply.content)
        else:
            reply_message = AIMessage(content=str(ai_reply))
        
        # Save the LLM's reply in the chat history
        chat_history.add_messages([reply_message])
        
        # Update the memory with the response
        summary_memories[session_id].save_context(
            {"input": chat_request.message},
            {"output": reply_message.content}
        )
        
        # Return the session id, the LLM reply, and the full conversation history
        return {
            "session_id": session_id,
            "reply": reply_message.content,
            "history": [msg.content for msg in chat_history.messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint to retrieve messages for an existing chat session ---
@router.get("/chat/messages")
def get_messages(session_id: str):
    try:
        chat_history = get_chat_history(session_id)
        messages = chat_history.messages
        history = []
        for msg in messages:
            # Determine the role based on the message type
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
    


def get_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    # Extract the vector from the response.
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

class SubtopicQueryRequest(BaseModel):
    query: str
    top_n: int = 3

@router.post("/chat/query_subtopics")
def query_subtopics(query_request: SubtopicQueryRequest):
    try:
        results = search_similar_subtopics(query_request.query, query_request.top_n)
        formatted_results = [
            {"id": row[0], "name": row[1], "text": row[2], "topicId": row[3], "similarity": row[4]}
            for row in results
        ]
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class SubtopicsQueryTool(BaseTool):
    name: str = "SubtopicsQueryTool"
    description: str = (
        "Useful for querying the subtopics table to find similar subtopics based on a natural language query. "
        "Input should be a query string, and it returns the query results as a string."
    )

    def _run(self, query: str) -> str:
        results = search_similar_subtopics(query, top_n=3)
        formatted_results = [
            {"id": row[0], "name": row[1], "text": row[2], "topicId": row[3], "similarity": row[4]}
            for row in results
        ]
        return str(formatted_results)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")


@router.post("/chat/query_subtopics_agent")
def query_subtopics_agent(query_request: SubtopicQueryRequest):
    try:
        # Create an instance of the custom tool.
        tool = SubtopicsQueryTool()
        # Initialize a zero-shot agent with the custom tool.
        agent = initialize_agent(
            tools=[tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
        # Run the agent with the provided natural language query.
        result = agent.run(query_request.query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))