from fastapi import APIRouter, HTTPException, Body
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel
from aiolimiter import AsyncLimiter  # Rate limiter

# Define the output dataclass for the axodraw diagram
@dataclass
class AxodrawDiagramOutput:
    document_content: str
    diagram_width: float  # Width in points (pt)
    diagram_height: float  # Height in points (pt)

# Set up the agent for generating the axodraw2 diagram.
agent_axodraw = Agent(
    name="Axodraw Diagram Generator",
    output_type=AxodrawDiagramOutput,
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.7),
    instructions='''You are given a prompt related to physics, chemistry, or biology. Your task is to generate a complete LaTeX document that uses the axodraw2 package to create a diagram corresponding to the given prompt.

The document must include:
- A proper LaTeX preamble with the axodraw2 package.
- The diagram inside a picture environment using axodraw2 commands.
- A clearly defined diagram of an appropriate size.
- At the very end of the document, include a comment in the following format:
  %% Diagram Size: WIDTH x HEIGHT
  where WIDTH and HEIGHT are numerical values in points (pt).

Output your result as a JSON object with the following keys:
"document_content": string containing the entire LaTeX document,
"diagram_width": a number representing the width of the diagram in points,
"diagram_height": a number representing the height of the diagram in points.
Do not include any additional keys or commentary.'''
)

# Create an AsyncLimiter instance (e.g., allowing 60 requests per minute)
limiter = AsyncLimiter(max_rate=60, time_period=60)

# Async function to generate the axodraw diagram based on a prompt.
async def generate_axodraw_diagram(prompt: str) -> AxodrawDiagramOutput:
    prompt_context = f"Prompt: {prompt}"
    async with limiter:
        result = await Runner.run(agent_axodraw, prompt_context)
    return result.final_output

# Define a request body model for axodraw diagram generation.
class AxodrawDiagramRequest(BaseModel):
    prompt: str

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Create a Feynman diagram representing electron-positron annihilation."
            }
        }

# Create a FastAPI router for the axodraw diagram generation endpoint.
router = APIRouter()

@router.post("/diagram/generate", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "document_content": "% LaTeX document content with axodraw2 diagram...",
                                 "diagram_width": 300,
                                 "diagram_height": 200
                             }
                         }
                     }
                 }
             })
async def generate_diagram(request: AxodrawDiagramRequest = Body(...)):
    """
    Generate a LaTeX document with an axodraw2 diagram based on the provided prompt.
    The response includes the full LaTeX document content and the dimensions of the diagram in points (pt).
    """
    try:
        diagram_output = await generate_axodraw_diagram(request.prompt)
        return {
            "document_content": diagram_output.document_content,
            "diagram_width": diagram_output.diagram_width,
            "diagram_height": diagram_output.diagram_height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))