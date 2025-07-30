import os
from dotenv import load_dotenv
load_dotenv()

from src.ocr_with_claude import main
# Get Anthropic API key from .env
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found .env")  
main(key = ANTHROPIC_API_KEY )