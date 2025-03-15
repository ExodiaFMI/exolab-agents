from fastapi import APIRouter, HTTPException, Body
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pydantic import BaseModel
from agents import Agent, Runner, ModelSettings

# For the purpose of this endpoint, we define a simple Pydantic model representing a lecture subtopics item.
class LectureSubtopicsOutputModel(BaseModel):
    topic: str
    subtopics: List[str]

# Define a Pydantic model for explanation items.
class ExplanationItem(BaseModel):
    topic: str
    subtopic: str
    explanation: str

# Define a request model for question generation that includes both subtopics and explanations.
class QuestionGenerationRequest(BaseModel):
    data: List[LectureSubtopicsOutputModel]
    explanations: List[ExplanationItem]

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
                ],
                "explanations": [
                    {
                        "topic": "Cell theory/definition of life",
                        "subtopic": "Historical development of cell theory",
                        "explanation": "Detailed explanation about the historical development of cell theory..."
                    },
                    {
                        "topic": "Cell theory/definition of life",
                        "subtopic": "The three main tenets of cell theory",
                        "explanation": "Detailed explanation about the three main tenets of cell theory..."
                    }
                ]
            }
        }

# This is our FastAPI router for question generation.
router = APIRouter()

# Define the output data structure for question generation.
@dataclass
class QuestionGenerationOutput:
    topic: str
    subtopic: str
    difficulty: str
    question_type: str
    question: str
    correct_answer: str
    explanation: str  # explanation of the correct answer

# Set up the agent for generating questions.
agent_questions = Agent(
    name="Question Generator",
    output_type=QuestionGenerationOutput,
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.7),
    instructions='''Write a detailed question on the given subtopic in Markdown format.

The question should be designed for the provided difficulty level (Easy, Medium, or Hard) and of the specified type (Multiple Choice with options labeled A, B, C, etc. or Open Answer).

Include the following in your output:
- The question text.
- The correct answer.
- An explanation of why the correct answer is correct.

Other subtopics will be provided but should not be discussed; focus solely on the current subtopic.
Additionally, use the provided explanation of the subtopic as context to craft the question.

Output the result as a JSON object with "topic", "subtopic", "difficulty", "question_type", "question", "correct_answer", and "explanation".'''
)

async def generate_question(
    topic: str,
    subtopic: str,
    other_subtopics: List[str],
    difficulty: str,
    question_type: str,
    subtopic_explanation: str
) -> QuestionGenerationOutput:
    """Generate a detailed question for a given subtopic, difficulty, and question type,
       using the subtopic explanation as context."""
    prompt_context = f"""
Topic: {topic}
Subtopic: {subtopic}
Explanation: {subtopic_explanation}
Other Subtopics: {', '.join(other_subtopics)}
Difficulty: {difficulty}
Question Type: {question_type}
"""
    result = await Runner.run(agent_questions, prompt_context)
    return result.final_output

async def run_question_generation(
    subtopics_list: List[LectureSubtopicsOutputModel],
    explanation_map: Dict[Tuple[str, str], str]
) -> List[QuestionGenerationOutput]:
    """Run question generation in parallel for all subtopics."""
    difficulties = ["Easy", "Medium", "Hard"]
    question_types = ["Multiple Choice", "Open Answer"]

    tasks = []
    for subtopic_item in subtopics_list:
        for sub in subtopic_item.subtopics:
            sub_explanation = explanation_map.get((subtopic_item.topic, sub), "")
            for difficulty in difficulties:
                for q_type in question_types:
                    tasks.append(
                        generate_question(
                            topic=subtopic_item.topic,
                            subtopic=sub,
                            other_subtopics=subtopic_item.subtopics,
                            difficulty=difficulty,
                            question_type=q_type,
                            subtopic_explanation=sub_explanation
                        )
                    )
    results = await asyncio.gather(*tasks)
    return results

# Create a POST endpoint to generate questions.
@router.post("/questions/generate", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "questions": [
                                     {
                                         "topic": "Cell theory/definition of life",
                                         "subtopic": "Historical development of cell theory",
                                         "difficulty": "Easy",
                                         "question_type": "Multiple Choice",
                                         "question": "What is the significance of Robert Hooke's discovery in the context of cell theory?",
                                         "correct_answer": "A. It was the first observation of cells.",
                                         "explanation": "Robert Hooke's observation of cork cells marked the first identification of cells, establishing a foundation for cell theory."
                                     },
                                     {
                                         "topic": "Cell theory/definition of life",
                                         "subtopic": "The three main tenets of cell theory",
                                         "difficulty": "Medium",
                                         "question_type": "Open Answer",
                                         "question": "Explain the three main tenets of cell theory.",
                                         "correct_answer": "All living organisms are made of cells, the cell is the basic unit of life, and all cells come from pre-existing cells.",
                                         "explanation": "These tenets summarize the fundamental principles that form the basis of cell theory."
                                     }
                                 ]
                             }
                         }
                     }
                 }
             })
async def generate_questions(request: QuestionGenerationRequest = Body(...)):
    """
    Generate detailed questions for each subtopic provided in the request.

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
        ],
        "explanations": [
            {
                "topic": "Cell theory/definition of life",
                "subtopic": "Historical development of cell theory",
                "explanation": "Detailed explanation about the historical development of cell theory..."
            },
            {
                "topic": "Cell theory/definition of life",
                "subtopic": "The three main tenets of cell theory",
                "explanation": "Detailed explanation about the three main tenets of cell theory..."
            }
        ]
    }
    ```

    **Response Example:**
    ```json
    {
        "questions": [
            {
                "topic": "Cell theory/definition of life",
                "subtopic": "Historical development of cell theory",
                "difficulty": "Easy",
                "question_type": "Multiple Choice",
                "question": "What is the significance of Robert Hooke's discovery in the context of cell theory?",
                "correct_answer": "A. It was the first observation of cells.",
                "explanation": "Robert Hooke's observation of cork cells marked the first identification of cells, establishing a foundation for cell theory."
            },
            {
                "topic": "Cell theory/definition of life",
                "subtopic": "The three main tenets of cell theory",
                "difficulty": "Medium",
                "question_type": "Open Answer",
                "question": "Explain the three main tenets of cell theory.",
                "correct_answer": "All living organisms are made of cells, the cell is the basic unit of life, and all cells come from pre-existing cells.",
                "explanation": "These tenets summarize the fundamental principles that form the basis of cell theory."
            }
        ]
    }
    ```
    """
    try:
        # Build explanation_map from request.explanations
        explanation_map = { (item.topic, item.subtopic): item.explanation for item in request.explanations }
        questions = await run_question_generation(request.data, explanation_map)
        # Convert each question dataclass to a dict for JSON serialization.
        return {"questions": [q.__dict__ for q in questions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))