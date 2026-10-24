# CLI AI Assistant (Gemini)

A small command-line assistant that sends a user's question to the Google Gemini API and prints the response.

## Setup
1. Clone the repo and create a virtual environment
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Gemini API key from Google AI Studio
4. Run: `python assistant.py`

## How it works
User input → API request (with API key) → Gemini model → API response → printed in terminal.

## Security
The API key is stored in `.env`, which is excluded from Git via `.gitignore`.

## Task 2: Embeddings and Semantic Search

Scripts using the Gemini embedding model (`gemini-embedding-001`):

- `embedding_basics.py`: generates an embedding for one sentence and prints the vector length and first values
- `similarity_experiment.py`: compares 8 sentences with cosine similarity
- `semantic_search.py`: semantic search over 10 facts about a fictional academy, tested with sample questions

Run any script with `python <filename>.py`.