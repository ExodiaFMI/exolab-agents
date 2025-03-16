from fastapi import APIRouter, HTTPException, Body
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings, WebSearchTool
from agents.agent_output import AgentOutputSchema
from dataclasses_json import dataclass_json
from pydantic import BaseModel
from aiolimiter import AsyncLimiter  # Import the rate limiter

router = APIRouter()

# Define the output data structure for lecture subtopics.
@dataclass
class LectureSubtopicsOutput:
    topic: str
    subtopics: list[str]

output_schema_subtopics = AgentOutputSchema(output_type=LectureSubtopicsOutput)

# Create an Agent with the necessary settings and instructions for subtopics.
# We add an instruction to use resources if relevant.
agent_subtopics = Agent(
    name="Subtopics Extractor", 
    output_type=LectureSubtopicsOutput,
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.1,
    ),
    tools=[WebSearchTool()],
    instructions='''You are given a single lecture topic and a list of related resources.
Use the resources if they are helpful, and extract relevant subtopics for the topic. By searching info about them.
GIVE MAX 10 Subtopics per topic.
Output the result as a JSON object with:
  "topic"    : the main topic
  "subtopics": a list of relevant subtopics.
    '''
)

# Create an AsyncLimiter instance.
# For example, if you want to allow 60 requests per minute:
limiter = AsyncLimiter(max_rate=60, time_period=60)

# Define an async function that runs the agent for a single topic + resources.
async def extract_subtopics(topic: str, resources: list[str]) -> LectureSubtopicsOutput:
    """
    Run the agent to extract subtopics for a given topic.
    We pass both the topic and the resources to the agent context.
    """
    # Acquire the rate limiter before making the API call.
    async with limiter:
        # You can combine the topic and resources into a single input string,
        # or pass them as structured data for your agent, depending on how your
        # framework reads the "input" parameter.
        prompt_text = f"Topic: {topic}\nResources:\n- " + "\n- ".join(resources)
        
        result = await Runner.run(agent_subtopics, prompt_text)
    return result.final_output

# Define a helper function to run subtopic extraction in parallel.
async def run_subtopic_extraction(topics: list[str], resources: list[str]) -> list[LectureSubtopicsOutput]:
    """
    Run subtopic extraction in parallel for all topics.
    The same list of resources is applied to each topic here.
    If you need a 1-to-1 mapping of topics to resources, adjust accordingly.
    """
    tasks = [extract_subtopics(topic, resources) for topic in topics]
    results = await asyncio.gather(*tasks)
    return results

# Define a request body model that accepts a list of topics and a list of resources.
class TopicsAndResources(BaseModel):
    topics: list[str]
    resources: list[str]  = [] 

    class Config:
        schema_extra = {
            "example": {
                "topics": [
                    "Cell theory/definition of life",
                    "Enzymes and biochemical reactions"
                ],
                "resources": [
                    "Chapter 1 of Biology Textbook",
                    "Wikipedia article on Cell Theory",
                    "Peer-reviewed journal article on enzyme kinetics"
                ]
            }
        }

# Create a POST endpoint that accepts a list of topics and resources, returns extracted subtopics.
@router.post("/subtopics/extract", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "data": [
                                     {
                                         "topic": "Cell theory/definition of life",
                                         "subtopics": [
                                             "Historical development of cell theory",
                                             "The three main tenets of cell theory",
                                             "Comparison of prokaryotic and eukaryotic cells",
                                             "Cell structure and function",
                                             "The role of cells in multicellular organisms",
                                             "Cell metabolism and energy production",
                                             "Cell division: mitosis and meiosis",
                                             "The relationship between cells and the definition of life",
                                             "Applications of cell theory in modern biology",
                                             "Implications of cell theory in medicine and biotechnology"
                                         ]
                                     },
                                     {
                                         "topic": "Enzymes and biochemical reactions",
                                         "subtopics": [
                                             "Enzyme structure and function",
                                             "Activation energy and the role of catalysts",
                                             "Substrate specificity and enzyme active sites",
                                             "Michaelis-Menten kinetics",
                                             "Factors affecting enzyme activity",
                                             "Inhibition and regulation of enzyme activity",
                                             "Allosteric regulation",
                                             "Applications of enzymes in industry and medicine"
                                         ]
                                     }
                                 ]
                             }
                         }
                     }
                 }
             })
async def extract_lecture_subtopics(data: TopicsAndResources = Body(...)):
    try:
        subtopics_list = await run_subtopic_extraction(data.topics, data.resources)
        # Convert the dataclass instances to dicts for JSON serialization.
        return {"data": [sub.__dict__ for sub in subtopics_list[:10]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))