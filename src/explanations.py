from fastapi import APIRouter, HTTPException, Body
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel

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
                                         "explanation": "The historical development of cell theory is a critical component in understanding the foundations of biology. The theory itself describes the basic unit of life as the cell, which has significant implications for biology as a whole. Here’s a detailed overview of the key milestones in the development of cell theory:\n\n### 1. Early Discoveries (17th Century)\n- **Robert Hooke (1665)**: The term 'cell' was first coined by English scientist Robert Hooke. He observed thin slices of cork through a microscope and noted that they resembled small rooms or 'cells' occupied by monks. This observation was fundamental in recognizing that life could be compartmentalized into smaller units.\n\n### 2. The Advancement of Microscopy\n- **Antonie van Leeuwenhoek (1670s)**: Leeuwenhoek improved the microscope and was the first to observe living cells, including bacteria and protozoa. His meticulous observations laid the groundwork for studying cellular structures and functions.\n\n### 3. The 19th Century Breakthroughs\n- **Matthias Schleiden (1838)**: Schleiden, a botanist, proposed that all plant tissues are composed of cells. He emphasized that the cell is the basic building block of plant life, reinforcing Hooke's early findings.\n- **Theodor Schwann (1839)**: Schwann extended the cell concept to animals, asserting that all animal tissues are also made up of cells. His collaboration with Schleiden led to the formulation of the first unified cell theory.\n\n### 4. The Formulation of Cell Theory (1855)\n- **Rudolf Virchow**: Virchow contributed to cell theory with his famous quote \"Omnis cellula e cellula\" (all cells come from cells). His work emphasized that new cells arise from the division of existing cells, which was a significant advancement in the understanding of cellular reproduction.\n\n### 5. The Three Main Tenets of Cell Theory\nBy the mid-19th century, the combined efforts of these scientists established the three main tenets of cell theory:\n  1. All living organisms are made up of one or more cells.\n  2. The cell is the basic unit of life.\n  3. All cells arise from pre-existing cells.\n\n### 6. Impact and Modern Developments\n- The development of cell theory has significantly influenced various fields, including genetics, microbiology, and biochemistry. Advances in molecular biology, such as the discovery of DNA and the understanding of cellular processes, continue to build upon the foundation established by cell theory.\n\n### Conclusion\nThe historical development of cell theory represents a crucial evolution in biological sciences, shifting the perspective from a focus on whole organisms to the cellular level. This shift has enabled scientists to explore life with a depth and clarity that has transformed our understanding of biology and medicine."
                                     },
                                     {
                                         "topic": "Cell theory/definition of life",
                                         "subtopic": "The three main tenets of cell theory",
                                         "explanation": "Cell theory is a fundamental concept in biology that describes the properties of cells, which are the basic building blocks of all living organisms. The theory consists of three main tenets that outline the essential roles and characteristics of cells:\n\n1. **All living organisms are composed of one or more cells.**  \n   This tenet states that every living organism, from the smallest bacteria to the largest whale, is made up of cells. Cells serve as the structural and functional units of life. This concept emphasizes that even the simplest life forms, such as unicellular organisms, are fully functional entities that operate on cellular processes.\n\n2. **The cell is the basic unit of life.**  \n   According to this tenet, the cell is the smallest unit capable of performing all life processes. This means that all the necessary functions required for life—such as metabolism, growth, and reproduction—are carried out within cells. This principle highlights the importance of cellular organization and indicates that life cannot exist in a non-cellular form. Even in multicellular organisms, all functions can ultimately be traced back to cellular activities.\n\n3. **All cells arise from pre-existing cells.**  \n   The third tenet states that new cells are produced from existing cells through the process of cell division. This principle underscores the continuity of life, as it illustrates how cells replicate and how genetic information is passed down from one generation of cells to the next. This tenet also implies that the processes of growth and healing in living organisms are fundamentally linked to cellular replication.\n\nTogether, these three tenets of cell theory provide a comprehensive framework for understanding the biological significance of cells in the context of life. They serve as a foundation for modern biology and have influenced various fields, including genetics, microbiology, and cellular biology. The clarity brought by these tenets has also facilitated advancements in medical science, biotechnology, and our overall understanding of life processes."
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

    **Request Example:**
    ```json
    {
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
    ```

    **Response Example:**
    ```json
    {
        "explanations": [
            {
                "topic": "Cell theory/definition of life",
                "subtopic": "Historical development of cell theory",
                "explanation": "The historical development of cell"
                },
            {
                "topic": "Cell theory/definition of life",
                "subtopic": "The three main tenets of cell theory",
                "explanation": "Cell theory is a fundamental concept in biology that "
                }
        ]
    }
}
    """
    try:
        explanations = await run_explanation_generation(data.data)
        # Convert each explanation dataclass to a dict for JSON serialization.
        return {"explanations": [exp.__dict__ for exp in explanations]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))