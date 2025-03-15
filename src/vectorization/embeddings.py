import openai
import psycopg2
import numpy as np
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Initialize FastAPI Router
router = APIRouter()

# Define request model for single text input
class TextInput(BaseModel):
    text: str = Field(..., example="Biology is cool!")

# Define request model for array input
class TextInputObject(BaseModel):
    id: str = Field(..., example="123")
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

# FastAPI endpoint for single text vectorization
@router.post("/vectorize", response_model=dict, response_model_exclude_unset=True)
def vectorize(input_data: TextInput):
    embedding = get_embedding(input_data.text)
    return {"embedding": embedding}

# Example response for Swagger UI for single text vectorization
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

# New endpoint for processing an array of objects containing id and text
@router.post("/vectorize/array", response_model=dict)
def vectorize_array(input_data: List[TextInputObject]):
    results = []
    for item in input_data:
        embedding = get_embedding(item.text)
        results.append({
            "id": item.id,
            "embedding": embedding
        })
    return {"results": results}

# Example response for Swagger UI for array vectorization
@router.post(
    "/vectorize/array",
    response_model=dict,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "id": "123",
                                "embedding": [
                                    0.003965024836361408,
                                    0.011244193650782108,
                                    0.01604875922203064
                                ]
                            },
                            {
                                "id": "456",
                                "embedding": [
                                    0.005965024836361408,
                                    0.013244193650782108,
                                    0.01804875922203064
                                ]
                            }
                        ]
                    }
                }
            }
        }
    }
)
def vectorize_array(input_data: List[TextInputObject]):
    results = []
    for item in input_data:
        embedding = get_embedding(item.text)
        results.append({
            "id": item.id,
            "embedding": embedding
        })
    return {"results": results}