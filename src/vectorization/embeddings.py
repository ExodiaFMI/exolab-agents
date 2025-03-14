import openai
import psycopg2
import numpy as np
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI Router
router = APIRouter()

# Define request model with an example
class TextInput(BaseModel):
    text: str = Field(..., example="Biology is cool!")

# Function to get embedding from OpenAI
def get_embedding(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding  # Extract the vector
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# FastAPI endpoint with response example
@router.post("/vectorize", response_model=dict, response_model_exclude_unset=True)
def vectorize(input_data: TextInput):
    embedding = get_embedding(input_data.text)
    return {
        "embedding": embedding
    }

# Example response for Swagger UI
@router.post(
    "/vectorize",
    response_model=dict,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "embedding": [
                            0.003965024836361408,
                            0.011244193650782108,
                            0.01604875922203064
                        ]
                    }
                }
            }
        }
    }
)
def vectorize(input_data: TextInput):
    embedding = get_embedding(input_data.text)
    return {"embedding": embedding}