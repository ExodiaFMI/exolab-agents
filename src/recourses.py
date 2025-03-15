from fastapi import APIRouter, HTTPException, Body
from agents import Agent, Runner, ModelSettings, WebSearchTool
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pydantic import BaseModel

router = APIRouter()

# Define the output data structure for the book table of contents.
@dataclass_json
@dataclass
class BookOutput:
    book: str
    table_of_contents: list[str]

# Create an agent that uses the WebSearchTool to find the table of contents for a given book title.
agent_toc_searcher = Agent(
    name="Book TOC Searcher",
    model="gpt-4o",
    output_type=BookOutput,
    model_settings=ModelSettings(temperature=0.7),
    instructions="""
        You are a book table of contents finder.
        When given a book title, search the web for its table of contents.
        Return a JSON object with:
          - "book": the given book title.
          - "table_of_contents": a list of chapter titles (if available).
    """,
    tools=[WebSearchTool()]
)

# Define a request body model with an example.
class BookRequest(BaseModel):
    title: str

    class Config:
        schema_extra = {
            "example": {
                "title": "How Life Works 3rd edition Morris et al editors."
            }
        }

# Define an async function to run the agent.
async def run_toc_agent(title: str) -> BookOutput:
    result = await Runner.run(agent_toc_searcher, title)
    return result.final_output

# Create a POST endpoint that accepts a book title and returns the table of contents.
@router.post("/books/toc", response_model=dict, 
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "book": "How Life Works 3rd edition Morris et al editors.",
                                 "table_of_contents": [
                                     "Chapter 1: Introduction",
                                     "Chapter 2: The Nature of Life",
                                     "Chapter 3: Evolution and Adaptation"
                                 ]
                             }
                         }
                     }
                 }
             })
async def extract_book_toc(data: BookRequest = Body(...)):
    """
    Extract the table of contents for a given book title.

    **Request Example:**
    ```json
    {
        "title": "How Life Works 3rd edition Morris et al editors."
    }
    ```

    **Response Example:**
    ```json
    {
        "book": "How Life Works 3rd edition Morris et al editors.",
        "table_of_contents": [
            "Chapter 1: Introduction",
            "Chapter 2: The Nature of Life",
            "Chapter 3: Evolution and Adaptation"
        ]
    }
    ```
    """
    try:
        output = await run_toc_agent(data.title)
        return {
            "book": output.book,
            "table_of_contents": output.table_of_contents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))