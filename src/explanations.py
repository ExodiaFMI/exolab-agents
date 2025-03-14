from fastapi import APIRouter, HTTPException
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel

# Optionally import LectureSubtopicsOutput from your subtopics module if available.
# from src.subtopics import LectureSubtopicsOutput

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

# Async function to generate an explanation for a given subtopic.
async def generate_explanation(topic: str, subtopic: str, other_subtopics: list[str]) -> SubtopicExplanationOutput:
    """Generate a detailed explanation for a given subtopic while avoiding other subtopics."""
    prompt_context = f"""
    Topic: {topic}
    Subtopic: {subtopic}
    Other Subtopics: {', '.join(other_subtopics)}
    """
    result = await Runner.run(agent_explanations, prompt_context)
    return result.final_output

# Helper function to run explanation generation in parallel.
async def run_explanation_generation(subtopics_list: list[LectureSubtopicsOutputModel]) -> list[SubtopicExplanationOutput]:
    """Run explanation generation in parallel for all subtopics, limited to one task."""
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

# Create a POST endpoint to generate subtopic explanations.
@router.post("/explanations/generate")
async def generate_subtopic_explanations(data: ExplanationRequest):
    try:
        explanations = await run_explanation_generation(data.data)
        # Convert each explanation dataclass to a dict for JSON serialization.
        return {"explanations": [exp.__dict__ for exp in explanations]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))