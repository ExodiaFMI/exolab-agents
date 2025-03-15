import os
import asyncio
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from lumaai import AsyncLumaAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Load database credentials from environment variables
luma_key = os.getenv("luma_key")

router = APIRouter()

class VideoRequest(BaseModel):
    prompt: str
    model: str = "ray-2"         # You can adjust the model (e.g., "ray-2", "text-to-video", etc.)
    resolution: str = "720p"     # Supported values: "540p", "720p", "1080p", "4k"
    duration: str = "5s"         # e.g., "5s", "10s", etc.
    loop: bool = False           # Optionally create a looping video

    class Config:
        schema_extra = {
            "example": {
                "prompt": "A futuristic cityscape with flying cars and neon lights",
                "model": "ray-2",
                "resolution": "720p",
                "duration": "5s",
                "loop": False
            }
        }

class VideoResponse(BaseModel):
    video_url: str

@router.post("/videos/generate", response_model=VideoResponse, tags=["Videos"])
async def generate_video(request: VideoRequest = Body(...)):
    """
    Generate a video based on the user's prompt.

    **Request Example:**
    ```json
    {
        "prompt": "A futuristic cityscape with flying cars and neon lights",
        "model": "ray-2",
        "resolution": "720p",
        "duration": "5s",
        "loop": false
    }
    ```

    **Response Example:**
    ```json
    {
        "video_url": "https://example.com/path/to/generated/video.mp4"
    }
    ```

    This endpoint uses the LumaAI SDK to create a video generation request and polls until the video is ready.
    """
    try:
        # Initialize the async LumaAI client using the API key from environment variables.
        client = AsyncLumaAI(
            auth_token=os.environ.get("luma_key"),
        )
        
        # Create a video generation request.
        generation = await client.generations.create(
            prompt=request.prompt,
            model=request.model,
            resolution=request.resolution,
            duration=request.duration,
            loop=request.loop
        )
        
        # Poll for the generation to be completed.
        completed = False
        while not completed:
            await asyncio.sleep(3)  # Wait 3 seconds between polls.
            generation = await client.generations.get(id=generation.id)
            if generation.state == "completed":
                completed = True
            elif generation.state == "failed":
                raise HTTPException(status_code=500, detail=f"Generation failed: {generation.failure_reason}")
            # Optionally log or print status here (e.g. "Dreaming...")
        
        video_url = generation.assets.video
        return VideoResponse(video_url=video_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))