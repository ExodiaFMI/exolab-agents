from fastapi import APIRouter, HTTPException, Body
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel
from aiolimiter import AsyncLimiter  # Import the rate limiter

# For the purpose of this endpoint, we define a simple Pydantic model representing a lecture subtopics item.
class LectureSubtopicsOutputModel(BaseModel):
    topic: str
    subtopics: list[str]

# This is our FastAPI router for explanations.
router = APIRouter()

# Define the output data structure for subtopic explanations.
@dataclass
class SubtopicExplanationOutput:
    topic: str
    subtopic: str
    explanation: str

# Set up the agent for generating subtopic explanations.
agent_explanations = Agent(
    name="Subtopic Explainer",
    output_type=SubtopicExplanationOutput,
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.7),
    instructions='''Write a detailed explanation of the given subtopic in Markdown format.
   
              Explain in detail what this subtopic is about, but **do not** discuss the other subtopics of the topic (they will be provided).
              Focus only on this subtopic and provide clear and structured information.
              
              Output the result as a JSON object with "topic", "subtopic", and "explanation".'''
)

# Set up a new agent for redacting the explanation.
agent_explanation_redactor = Agent(
    name="Explanation Redactor",
    output_type=SubtopicExplanationOutput,
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.7),
    instructions='''You are provided with an explanation text along with a user prompt that specifies a change to be made.
    
Your task is to produce a revised version of the explanation where you only modify the part specified in the user prompt, keeping the rest of the text exactly the same.

Output the result as a JSON object with "topic", "subtopic", and "explanation".'''
)

# Create an AsyncLimiter instance.
# For example, if you want to allow 60 requests per minute:
limiter = AsyncLimiter(max_rate=60, time_period=60)

# Async function to generate an explanation for a given subtopic.
async def generate_explanation(topic: str, subtopic: str, other_subtopics: list[str]) -> SubtopicExplanationOutput:
    """Generate a detailed explanation for a given subtopic while avoiding other subtopics."""
    prompt_context = f"""
    Topic: {topic}
    Subtopic: {subtopic}
    Other Subtopics: {', '.join(other_subtopics)}
    """
    # Acquire the rate limiter before making the API call.
    async with limiter:
        result = await Runner.run(agent_explanations, prompt_context)
    return result.final_output

# Async function to redact an explanation based on a user prompt.
async def redact_explanation(
    topic: str,
    subtopic: str,
    explanation: str,
    user_prompt: str
) -> SubtopicExplanationOutput:
    """Redact the given explanation by applying only the change specified in the user prompt."""
    prompt_context = f"""
    Topic: {topic}
    Subtopic: {subtopic}
    Original Explanation: {explanation}
    
    User Prompt: {user_prompt}
    
    Revise the explanation by modifying only the part requested by the user, leaving the rest unchanged.
    """
    # Acquire the rate limiter before making the API call.
    async with limiter:
        result = await Runner.run(agent_explanation_redactor, prompt_context)
    return result.final_output

# Helper function to run explanation generation in parallel.
async def run_explanation_generation(subtopics_list: list[LectureSubtopicsOutputModel]) -> list[SubtopicExplanationOutput]:
    """Run explanation generation in parallel for all subtopics."""
    tasks = [
        generate_explanation(item.topic, sub, item.subtopics)
        for item in subtopics_list
        for sub in item.subtopics
    ]
    results = await asyncio.gather(*tasks)
    return results

# Define a request body model that accepts a list of lecture subtopics.
class ExplanationRequest(BaseModel):
    data: list[LectureSubtopicsOutputModel]

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "topic": "Cell theory/definition of life",
                        "subtopics": [
                            "Historical development of cell theory",
                            "The three main tenets of cell theory"
                        ]
                    }
                ]
            }
        }

# Define a new request body model for redacting an explanation.
class RedactExplanationRequest(BaseModel):
    topic: str
    subtopic: str
    explanation: str
    user_prompt: str

    class Config:
        schema_extra = {
            "example": {
                "topic": "Cell theory/definition of life",
                "subtopic": "Historical development of cell theory",
                "explanation": "The historical development of cell theory is a critical component in understanding the foundations of biology. ...",
                "user_prompt": "Please update the explanation to include more details about the contributions of Robert Hooke."
            }
        }

# Create a POST endpoint to generate subtopic explanations.
@router.post("/explanations/generate", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "explanations": [
                                     {
                                         "topic": "Cell theory/definition of life",
                                         "subtopic": "Historical development of cell theory",
                                         "explanation": "Detailed explanation text..."
                                     },
                                     {
                                         "topic": "Cell theory/definition of life",
                                         "subtopic": "The three main tenets of cell theory",
                                         "explanation": "Detailed explanation text..."
                                     }
                                 ]
                             }
                         }
                     }
                 }
             })
async def generate_subtopic_explanations(data: ExplanationRequest = Body(...)):
    """
    Generate detailed explanations for each subtopic provided in the request.
    """
    try:
        explanations = await run_explanation_generation(data.data)
        # Convert each explanation dataclass to a dict for JSON serialization.
        return {"explanations": [exp.__dict__ for exp in explanations]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create a POST endpoint to redact an explanation based on a user prompt.
@router.post("/explanations/redact", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "topic": "Cell theory/definition of life",
                                 "subtopic": "Historical development of cell theory",
                                 "explanation": "Revised explanation text with the updated details about Robert Hooke..."
                             }
                         }
                     }
                 }
             })
async def redact_explanation_endpoint(request: RedactExplanationRequest = Body(...)):
    """
    Redact an existing explanation based on a user prompt. The endpoint receives the topic, subtopic, original explanation,
    and a prompt that indicates what change is desired. It returns a revised explanation where only the specified part is updated.

    **Request Example:**
    ```json
    {
        "topic": "Cell theory/definition of life",
        "subtopic": "Historical development of cell theory",
        "explanation": "The historical development of cell theory is a critical component in understanding the foundations of biology. ...",
        "user_prompt": "Please update the explanation to include more details about the contributions of Robert Hooke."
    }
    ```

    **Response Example:**
    ```json
    {
        "topic": "Cell theory/definition of life",
        "subtopic": "Historical development of cell theory",
        "explanation": "Revised explanation text with the updated details about Robert Hooke..."
    }
    ```
    """
    try:
        result = await redact_explanation(
            topic=request.topic,
            subtopic=request.subtopic,
            explanation=request.explanation,
            user_prompt=request.user_prompt
        )
        # Return the redacted explanation as a dictionary.
        return result.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))