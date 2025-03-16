from fastapi import APIRouter, HTTPException, Body
from agents import Agent, Runner, ModelSettings, RunContextWrapper, WebSearchTool
from openai.types.responses import ResponseTextDeltaEvent
import openai
import asyncio
import json
from dataclasses import dataclass
from agents.agent_output import AgentOutputSchema
from dataclasses_json import dataclass_json
from pydantic import BaseModel

router = APIRouter()

# Updated output data structure to include topics, description, and reading materials.
@dataclass_json
@dataclass
class CourseContentOutput:
    topics: list[str]
    description: str
    recourses: list[str]

output_schema = AgentOutputSchema(output_type=CourseContentOutput)

# Updated Agent instructions to extract the additional fields.
agent = Agent(
    name="Course Schedule Extractor", 
    output_type=CourseContentOutput,
    model="gpt-4o",
    tools=[WebSearchTool()],
    model_settings=ModelSettings(
        temperature=0.1,
    ),
    instructions='''Extract the following details from the provided course schedule:
1. Lecture topics as a JSON list of strings (exclude exams, holidays, non-course material, course introductions, discussion events, etc.).
2. A short description of the course summarizing its content.
3. Reading materials as a JSON array of strings containing recommended texts.
GIVE MAX 10 topics.
Output the result as a JSON object with the keys "topics", "description", and "recourses".'''
)

# Updated request body model with an example including the additional details.
class LectureContent(BaseModel):
    content: str

    class Config:
        schema_extra = {
            "example": {
                "content": (
                    "May 23: Course Introduction; Cell theory/definition of life; Chemistry of Life\n"
                    "May 24: Nucleic Acids; Carbohydrates\n"
                    "Course Description: This course covers the fundamentals of biology and chemistry in living organisms.\n"
                    "Reading Materials: 'Biology 101', 'Chemistry Basics'"
                )
            }
        }

# Define an async function that runs the agent with the provided content.
async def run_course_agent(content: str):
    result = await Runner.run(agent, content)
    return result.final_output

# Create a POST endpoint that accepts the lecture content and returns extracted details.
@router.post("/topics/extract", response_model=dict, 
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "topics": [
                                     "Cell theory/definition of life",
                                     "Chemistry of Life",
                                     "Nucleic Acids"
                                 ],
                                 "description": "This course covers the fundamentals of biology and chemistry in living organisms.",
                                 "recourses": [
                                     "Biology 101",
                                     "Chemistry Basics"
                                 ]
                             }
                         }
                     }
                 }
             })
async def extract_lecture_topics(data: LectureContent = Body(...)):
    """
    Extract lecture topics, a course description, and reading materials from the provided schedule.

    **Request Example:**
    ```json
    {
        "content": "May 23: Course Introduction; Cell theory/definition of life; Chemistry of Life\nMay 24: Nucleic Acids; Carbohydrates\nCourse Description: This course covers the fundamentals of biology and chemistry in living organisms.\nReading Materials: 'Biology 101', 'Chemistry Basics'"
    }
    ```

    **Response Example:**
    ```json
    {
        "topics": [
            "Cell theory/definition of life",
            "Chemistry of Life",
            "Nucleic Acids"
        ],
        "description": "This course covers the fundamentals of biology and chemistry in living organisms.",
        "recourses": [
            "Biology 101",
            "Chemistry Basics"
        ]
    }
    ```
    """
    try:
        output = await run_course_agent(data.content)
        return {
            "topics": output.topics[:10],
            "description": output.description,
            "recourses": output.recourses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))