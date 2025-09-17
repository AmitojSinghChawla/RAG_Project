import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Data and DB paths
DATA_PATH = "data/"
DB_PATH = "vectorstore/"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini setup
GEMINI_MODEL = "gemini-1.5-flash"   # or gemini-1.5-pro if you need more accuracy
GEMINI_API_KEY = os.getenv("AIzaSyAzUHUpWd0YFjJvnB8ZM5iSPbrj-ARedkg")

if not GEMINI_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
