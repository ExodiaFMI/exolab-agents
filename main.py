from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from src.topics import router as topics_router  # Import the router
from src.subtopics import router as subtopics_router

app = FastAPI()

# Include the router from topics.py
app.include_router(topics_router)
app.include_router(subtopics_router)


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