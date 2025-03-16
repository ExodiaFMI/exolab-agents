from fastapi import APIRouter, HTTPException, Body
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings, WebSearchTool
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
    model="gpt-4o",
    tools=[WebSearchTool()],
    model_settings=ModelSettings(temperature=0.1),
    instructions='''Write a detailed explanation of the given subtopic in Markdown format.
   
              Explain in detail what this subtopic is about, but **do not** discuss the other subtopics of the topic (they will be provided).
              Focus only on this subtopic and provide clear and structured information.
              
              Output the result as a JSON object with "topic", "subtopic", and "explanation".'''
)

# Set up a new agent for redacting the explanation.
agent_explanation_redactor = Agent(
    name="Explanation Redactor",
    output_type=SubtopicExplanationOutput,
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.7),
    instructions='''You are provided with an explanation text along with a user prompt that specifies a change to be made.
    
Your task is to produce a revised version of the explanation where you only modify the part specified in the user prompt, keeping the rest of the text exactly the same.

Output the result as a JSON object with "topic", "subtopic", and "explanation".'''
)

# Create an AsyncLimiter instance.
limiter = AsyncLimiter(max_rate=60, time_period=60)

# Updated function to generate an explanation now accepts a resources parameter.
async def generate_explanation(topic: str, subtopic: str, other_subtopics: list[str], resources: list[str]) -> SubtopicExplanationOutput:
    """Generate a detailed explanation for a given subtopic while avoiding other subtopics.
    Resources are also included in the prompt context."""
    prompt_context = f"""
    Topic: {topic}
    Subtopic: {subtopic}
    Other Subtopics: {', '.join(other_subtopics)}
    Resources: {', '.join(resources)}
    """
    async with limiter:
        result = await Runner.run(agent_explanations, prompt_context)
    return result.final_output

# Helper function updated to accept resources.
async def run_explanation_generation(subtopics_list: list[LectureSubtopicsOutputModel], resources: list[str]) -> list[SubtopicExplanationOutput]:
    """Run explanation generation in parallel for all subtopics, passing the resources for each request."""
    tasks = [
        generate_explanation(item.topic, sub, item.subtopics, resources)
        for item in subtopics_list
        for sub in item.subtopics
    ]
    results = await asyncio.gather(*tasks)
    return results

# Updated request body model to include resources.
class ExplanationRequest(BaseModel):
    data: list[LectureSubtopicsOutputModel]
    resources: list[str] = []

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
                ],
                "resources": [
                    "Chapter 1 of Biology Textbook",
                    "Relevant research paper on cell theory"
                ]
            }
        }

# The redaction request model remains unchanged.
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
    Generate detailed explanations for each subtopic provided in the request,
    taking into account any additional resources provided.
    """
    try:
        explanations = await run_explanation_generation(data.data, data.resources)
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
    Redact an existing explanation based on a user prompt.
    The endpoint receives the topic, subtopic, original explanation,
    and a prompt that indicates what change is desired.
    """
    try:
        result = await redact_explanation(
            topic=request.topic,
            subtopic=request.subtopic,
            explanation=request.explanation,
            user_prompt=request.user_prompt
        )
        return result.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))