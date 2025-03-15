from fastapi import APIRouter, HTTPException, Body
import platform
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pydantic import BaseModel
from openai import OpenAI 
from agents import Agent, Runner, ModelSettings, WebSearchTool

router = APIRouter()

class ImageRequest(BaseModel):
    prompt: str

    class Config:
        schema_extra = {
            "example": {
                "prompt": "A futuristic 3D rendering of a quantum computer with neon lights and intricate circuitry"
            }
        }
# Define the output data structure for the image search result.
@dataclass_json
@dataclass
class ImageOutput:
    image_url: str

@router.post("/images/generate", response_model=dict)
async def generate_image(request: ImageRequest = Body(...)):
    """
    Generate a realistic 3D scientific image based on the user's prompt.
    
    This endpoint appends system information (such as OS and processor details) to enhance the prompt for a more realistic
    and scientifically nuanced 3D rendering.
    """
    try:
        # Retrieve system information
        system_info = (
            f"System: {platform.system()}, "
            f"Release: {platform.release()}, "
            f"Processor: {platform.processor()}"
        )
        
        # Combine user prompt with system info and additional instructions
        full_prompt = (
            f"{request.prompt}. {system_info}. "
            "Render as a realistic 3D image suitable for scientific visualization."
        )
        
        # Instantiate the OpenAI client (API key is picked from environment variables)
        client = OpenAI()
        
        # Call the new image generation method using the client instance
        response = client.images.generate(
            model="dall-e-3",  # optionally, specify the model (e.g., "dall-e-3")
            prompt=full_prompt,
            n=1,
            size="512x512"
        )
        
        # Extract the URL of the generated image from the pydantic model response
        image_url = response.data[0].url
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Create an agent that uses the WebSearchTool to search for an image URL based on the prompt.
agent_image_searcher = Agent(
    name="Image Searcher",
    model="gpt-4o",
    output_type=ImageOutput,
    model_settings=ModelSettings(temperature=0.7),
    instructions="""
        You are an image searcher agent.
        When given an image description prompt, search the web for a relevant image.
        Return a JSON object with:
          - "image_url": a direct URL link to an image that best matches the description.
    """,
    tools=[WebSearchTool()]
)

# Define a request body model with an example.
class ImageSearchRequest(BaseModel):
    prompt: str

    class Config:
        schema_extra = {
            "example": {
                "prompt": "A futuristic 3D rendering of a quantum computer with neon lights and intricate circuitry"
            }
        }

# Define an async function to run the image search agent.
async def run_image_search_agent(prompt: str) -> ImageOutput:
    result = await Runner.run(agent_image_searcher, prompt)
    return result.final_output

# Create a POST endpoint that accepts an image description prompt and returns the image URL.
@router.post("/images/search", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "image_url": "https://example.com/path/to/your/searched/image.png"
                             }
                         }
                     }
                 }
             })
async def search_image(data: ImageSearchRequest = Body(...)):
    """
    Search for an image URL based on the user's prompt.

    **Request Example:**
    ```json
    {
        "prompt": "A futuristic 3D rendering of a quantum computer with neon lights and intricate circuitry"
    }
    ```

    **Response Example:**
    ```json
    {
        "image_url": "https://example.com/path/to/your/searched/image.png"
    }
    ```
    """
    try:
        output = await run_image_search_agent(data.prompt)
        return {"image_url": output.image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))