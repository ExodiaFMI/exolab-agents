from fastapi import APIRouter, HTTPException
from agents import Agent, Runner, ModelSettings, RunContextWrapper
from openai.types.responses import ResponseTextDeltaEvent
import openai
import asyncio
import json
from dataclasses import dataclass
from agents.agent_output import AgentOutputSchema
from dataclasses_json import dataclass_json
from pydantic import BaseModel

router = APIRouter()

# Define the output data structure for lecture topics.
@dataclass_json
@dataclass
class LectureTopicsOutput:
    topics: list[str]

output_schema = AgentOutputSchema(output_type=LectureTopicsOutput)

# Create an Agent with the necessary settings and instructions.
agent = Agent(
    name="Course Schedule Extractor", 
    output_type=LectureTopicsOutput,
    model="gpt-4o-mini",
    model_settings=ModelSettings(
        temperature=0.1,
    ),
    instructions='''Extract only the lecture topics from the following schedule,
              excluding exams, holidays, non-course material, course introductions, discussion events, etc.
              Output the result as a JSON list where each entry is a topic name'''
)

# Define a request body model that accepts the content.
class LectureContent(BaseModel):
    content: str

# Define an async function that runs the agent with the provided content.
async def run_course_agent(content: str):
    result = await Runner.run(agent, content)
    return result.final_output.topics

# Create a POST endpoint that accepts the lecture content and returns extracted topics.
@router.post("/topics/extract")
async def extract_lecture_topics(data: LectureContent):
    try:
        topics = await run_course_agent(data.content)
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))