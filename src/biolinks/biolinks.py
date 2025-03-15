import os
import psycopg
import openai
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from bs4 import BeautifulSoup

# Initialize FastAPI router
router = APIRouter()

# --- Database Connection Setup ---
DB_NAME = os.getenv("DB_NAME", "langchain")
DB_USER = os.getenv("DB_USER", "langchain")
DB_PASSWORD = os.getenv("DB_PASSWORD", "langchain")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "6024")  # Adjust if needed

conn_info = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

# Create the biolinks table if it does not exist.
def create_biolinks_table():
    with psycopg.connect(conn_info) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS biolinks (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    href TEXT,
                    vector VECTOR(1536)
                )
            """)
        conn.commit()

# Run table creation at startup
create_biolinks_table()

# --- Embedding & Extraction Functions ---

# Function to generate OpenAI embedding
def get_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding  # Extract the vector

# Function to extract gallery card data from an HTML file
def extract_gallery_card_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    gallery_items = soup.find_all('li', class_='gallery-cards__item')
    
    extracted_data = []
    for item in gallery_items:
        link = item.find('a', {'role': 'button'})
        if link:
            name = link.get('aria-label', 'No Name')
            href = link.get('href', 'No Link')
            full_href = f"https://human.biodigital.com{href}"
            vector = get_embedding(name)
            extracted_data.append({'name': name, 'href': full_href, 'vector': vector})
    
    return extracted_data

# --- Endpoints ---

# Pydantic request model for extraction
class ExtractRequest(BaseModel):
    file_path: str

@router.post("/biolinks/extract", response_model=dict)
async def extract_biolinks(request: ExtractRequest = Body(...)):
    """
    Extract biolinks data from an HTML file and insert into the database.
    """
    try:
        data = extract_gallery_card_data(request.file_path)
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cursor:
                for entry in data:
                    cursor.execute(
                        "INSERT INTO biolinks (name, href, vector) VALUES (%s, %s, %s)",
                        (entry['name'], entry['href'], entry['vector'])
                    )
            conn.commit()
        return {"message": f"Inserted {len(data)} records."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic request model for search requests
class BioSearchRequest(BaseModel):
    query_text: str
    top_n: int = 3

# Function to search for similar biolinks using vector similarity
def bio_search_similar_text(query_text: str, top_n: int = 3):
    query_vector = get_embedding(query_text)
    with psycopg.connect(conn_info) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT name, href, vector <#> %s::vector AS similarity
                FROM biolinks
                ORDER BY similarity
                LIMIT %s;
                """,
                (query_vector, top_n)
            )
            results = cursor.fetchall()
    return results

@router.post("/biolinks/search", response_model=dict)
async def search_biolinks(request: BioSearchRequest = Body(...)):
    """
    Search for similar biolinks using vector similarity.
    """
    try:
        results = bio_search_similar_text(request.query_text, request.top_n)
        return {"results": [{"name": r[0], "href": r[1], "similarity": r[2]} for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
