from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from src.database.database import engine


from src.topics import router as topics_router
from src.subtopics import router as subtopics_router
from src.explanations import router as explanations_router
from src.vectorization.embeddings import router as vectorization_router
from src.questions.questions import router as questions_router  
from src.chat.chatagent import router as chatagent_router
from src.recourses import router as recourses_router
from src.images.images import router as images_router
from src.videos.videos import router as videos_router
from src.diagrams.diagrams import router as diagrams_router
from src.chat.newchatagent import router as newchat_router



app = FastAPI()

# Include the router from topics.py
app.include_router(topics_router, tags=["Topic Generation"])
app.include_router(subtopics_router, tags=["Topic Generation"])
app.include_router(explanations_router, tags=["Topic Generation"])
app.include_router(recourses_router, tags=["Topic Generation"])
app.include_router(vectorization_router, tags=["Vectorization"]) 
app.include_router(questions_router, tags=["Questions"])
app.include_router(chatagent_router, tags=["Chat"])
app.include_router(images_router, tags=["Images"])
app.include_router(videos_router, tags=["Videos"])
app.include_router(diagrams_router, tags=["Diagrams"])
app.include_router(newchat_router, tags=["NewChat"])

@app.on_event("startup")
async def startup():
    print("Application is starting...")

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()  # Close database connections on shutdown
    print("Application is shutting down...")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ExoLab Agents",
        version="0.0.0",
        description="Python Microservice, design to guide the ai agents used inside the **ExoLab Product**",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi