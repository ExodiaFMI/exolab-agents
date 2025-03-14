import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

# Load database credentials from environment variables
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Construct DATABASE_URL dynamically
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create a connection pool (engine)
engine = create_async_engine(DATABASE_URL, echo=True, pool_size=10, max_overflow=20)

# Create a session factory
async_session_maker = sessionmaker(
bind=engine, class_=AsyncSession, expire_on_commit=False
)

# Dependency to get the session
async def get_db():
    async with async_session_maker() as session:
        yield session 