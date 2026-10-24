# GenAI Developer Projects

A collection of beginner Generative AI projects built in Python with the Google Gemini API. The projects move step by step from calling an LLM, to understanding embeddings, to storing and searching them in a vector database.

## Projects

| # | Project | What it does | Files |
|---|---------|--------------|-------|
| 1 | **CLI AI Assistant** | Takes a question in the terminal, sends it to the Gemini API, and prints the response | `assistant.py`, `first_request.py`, `list_models.py` |
| 2 | **Embeddings and Semantic Search** | Generates embeddings, compares sentences with cosine similarity, and runs a small semantic search | `embedding_basics.py`, `similarity_experiment.py`, `semantic_search.py` |
| 3 | **Vector Database Search** | Stores embeddings with metadata in a local Chroma database and searches it with natural-language questions | `build_db.py`, `search_db.py` |

## Tech Stack

- Python 3.9+
- [Google Gemini API](https://ai.google.dev/) (`google-genai` SDK)
- [Chroma](https://www.trychroma.com/) (local vector database)
- NumPy
- python-dotenv

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```
- Windows (PowerShell): `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Get a free API key from [Google AI Studio](https://aistudio.google.com/), then copy the example file and add your key:
```bash
cp .env.example .env        # Windows: copy .env.example .env
```
Edit `.env`:
```
GEMINI_API_KEY=your-key-here
```

## Project 1: CLI AI Assistant

A small command-line assistant. The user types a question, the program sends it to a Gemini model, and the response is streamed back to the terminal.

```bash
python list_models.py       # list the models available to your key
python first_request.py     # make a single test request
python assistant.py         # start the assistant (type 'quit' to exit)
```

**Request flow:**
```
user input → API request (with API key) → Gemini model → API response → printed in terminal
```

Model names change over time. If you get a "model not found" error, run `list_models.py` and update the `MODEL` variable in the script.

## Project 2: Embeddings and Semantic Search

Experiments with the `gemini-embedding-001` model.

```bash
python embedding_basics.py        # print the vector length and first values of one embedding
python similarity_experiment.py   # compare 8 sentences using cosine similarity
python semantic_search.py         # search 10 facts about a fictional academy by meaning
```

**What it shows:**
- An embedding is a vector of 3072 numbers that represents the meaning of text.
- Sentences with similar meaning score higher in cosine similarity, even when they share few words.
- Scores for unrelated text are not close to zero, so results should be compared by ranking instead of by absolute score.

## Project 3: Vector Database Search (Chroma)

Ten fictional university FAQs are embedded with Gemini and stored in a local Chroma database together with metadata (title, category, source). Questions written in different words from the stored text are then used to retrieve the most relevant records.

```bash
python build_db.py     # embed the FAQs and store them (run once)
python search_db.py    # run test questions, a category-filtered search, then interactive mode
```

**How it works:**
```
FAQ text → Gemini embedding → vector → Chroma (with metadata)
question → Gemini embedding → similarity search → top matching FAQs
```

Each stored record contains an ID, the vector, the original text, and metadata. `search_db.py` also demonstrates metadata filtering, for example searching only the "Campus Life" category.

**Observations:**
- Questions like "Is there any help for stress and anxiety?" correctly found the counseling FAQ without sharing its keywords.
- Lower-ranked results were often irrelevant, so similarity does not guarantee correctness.
- Real applications improve retrieval with chunking, re-ranking, hybrid search, and score thresholds.

## Project Structure

```
.
├── assistant.py
├── first_request.py
├── list_models.py
├── embedding_basics.py
├── similarity_experiment.py
├── semantic_search.py
├── build_db.py
├── search_db.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The `chroma_db/` folder is created locally when you run `build_db.py` and is not committed.

## Security

- The API key is stored in a `.env` file, which is excluded from Git through `.gitignore`.
- Never commit your key, share it in screenshots, or put it in frontend code.
- If a key is ever exposed, delete it in Google AI Studio and create a new one.

## What I Learned

- How LLM APIs work: request, model, response, and how to call them from code
- How API keys authenticate requests and how to keep them private
- How embeddings turn text into vectors that capture meaning
- How semantic search differs from keyword search
- How vector databases store embeddings with metadata and retrieve similar records
- Why embeddings and vector search are the foundation of RAG applications


## Project 4: RAG Document Q&A

A simple Retrieval-Augmented Generation system over a fictional university handbook.

- `knowledge_base.txt`: the knowledge source (11 sections)
- `rag_build.py`: splits the handbook into chunks, embeds them with Gemini, and stores them in Chroma
- `rag_qa.py`: retrieves the most relevant chunks for a question and sends them to Gemini to generate an answer

Run `python rag_build.py` once, then `python rag_qa.py`.

The prompt instructs the model to answer only from the retrieved context and to say it doesn't have the information when the answer isn't in the handbook.

## Project 5: Function Calling (Tool Use)

A small store assistant that lets Gemini call local Python functions.

- `tool_assistant.py`: defines two tools (`check_order_status`, `get_product_price`) over fictional data, sends their declarations to Gemini, executes the tool the model requests, and returns the result so the model can write the final answer

Run `python tool_assistant.py`.

The model only requests tools. The application decides what runs: only registered functions can be executed, and errors such as unknown orders are returned to the model instead of being invented.