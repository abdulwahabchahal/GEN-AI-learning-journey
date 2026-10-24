import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
gemini = genai.Client()
EMBED_MODEL = "gemini-embedding-001"

def load_chunks(path):
    """Split the knowledge base into chunks: one section = one chunk."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chunks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        title = block.split("\n")[0].lstrip("# ").strip()
        chunks.append({"title": title, "text": block})
    return chunks

chunks = load_chunks("knowledge_base.txt")
print(f"Created {len(chunks)} chunks")

# Embed all chunks (as documents to be searched)
result = gemini.models.embed_content(
    model=EMBED_MODEL,
    contents=[c["text"] for c in chunks],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
)
vectors = [e.values for e in result.embeddings]

# Store in Chroma (a separate collection from Task 3)
chroma = chromadb.PersistentClient(path="chroma_db")
try:
    chroma.delete_collection("rag_knowledge_base")
except Exception:
    pass

collection = chroma.create_collection(
    name="rag_knowledge_base",
    metadata={"hnsw:space": "cosine"},
)
collection.add(
    ids=[f"chunk-{i + 1:02d}" for i in range(len(chunks))],
    documents=[c["text"] for c in chunks],
    embeddings=vectors,
    metadatas=[{"title": c["title"], "source": "knowledge_base.txt", "chunk_number": i + 1}
               for i, c in enumerate(chunks)],
)

print(f"Stored {collection.count()} chunks in the vector database")
for i, c in enumerate(chunks, 1):
    print(f"  chunk {i}: {c['title']}")