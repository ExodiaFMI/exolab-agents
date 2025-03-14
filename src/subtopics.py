from fastapi import APIRouter, HTTPException
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema
from dataclasses_json import dataclass_json
from pydantic import BaseModel

router = APIRouter()

# Define the output data structure for lecture subtopics.
@dataclass
class LectureSubtopicsOutput:
    topic: str
    subtopics: list[str]

output_schema_subtopics = AgentOutputSchema(output_type=LectureSubtopicsOutput)

# Create an Agent with the necessary settings and instructions for subtopics.
agent_subtopics = Agent(
    name="Subtopics Extractor", 
    output_type=LectureSubtopicsOutput,
    model="gpt-4o-mini",
    model_settings=ModelSettings(
        temperature=0.1,
    ),
    instructions='''For the given lecture topic, extract relevant subtopics.
              Output the result as a JSON object with "topic" as the main topic name
              and "subtopics" as a list of subtopics.'''
)

# Define an async function that runs the agent for a single topic.
async def extract_subtopics(topic: str) -> LectureSubtopicsOutput:
    """Run the agent to extract subtopics for a given topic."""
    result = await Runner.run(agent_subtopics, topic)
    return result.final_output

# Define a helper function to run subtopic extraction in parallel.
async def run_subtopic_extraction(topics: list[str]) -> list[LectureSubtopicsOutput]:
    """Run subtopic extraction in parallel for all topics."""
    tasks = [extract_subtopics(topic) for topic in topics]
    results = await asyncio.gather(*tasks)
    return results

# Define a request body model that accepts a list of topics.
class TopicsList(BaseModel):
    topics: list[str]

# Create a POST endpoint that accepts a list of topics and returns extracted subtopics.
@router.post("/subtopics/extract")
async def extract_lecture_subtopics(data: TopicsList):
    try:
        subtopics_list = await run_subtopic_extraction(data.topics)
        # Convert the dataclass instances to dicts for JSON serialization.
        return {"data": [sub.__dict__ for sub in subtopics_list]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))