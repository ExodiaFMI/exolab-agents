from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from src.topics import router as topics_router  # Import the router
from src.subtopics import router as subtopics_router
from src.explanations import router as explanations_router
from src.vectorization.embeddings import router as vectorization_router

app = FastAPI()

# Include the router from topics.py
app.include_router(topics_router, tags=["Topic Generation"])
app.include_router(subtopics_router, tags=["Topic Generation"])
app.include_router(explanations_router, tags=["Topic Generation"])
app.include_router(vectorization_router, tags=["Vectorization"])  # Add the new router under a new tag



def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ExoLab Agents",
        version="0.0.0",
        description="Here's a longer description of the custom **OpenAPI** schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi