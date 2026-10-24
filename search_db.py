import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
gemini = genai.Client()
EMBED_MODEL = "gemini-embedding-001"

chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.get_collection("university_faqs")

def search(question, n_results=3, category=None):
    q_vec = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    ).embeddings[0].values

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_results,
        where={"category": category} if category else None,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQUESTION: {question}" + (f"  [filter: {category}]" if category else ""))
    for rank in range(len(results["ids"][0])):
        similarity = 1 - results["distances"][0][rank]   # cosine distance -> similarity
        meta = results["metadatas"][0][rank]
        print(f"  {rank + 1}. ({similarity:.3f}) [{meta['category']}] {meta['title']}")
        print(f"     {results['documents'][0][rank]}")

test_questions = [
    "How can I pay less for my studies?",
    "Where can I stay if I live far from the university?",
    "What happens if my grades are too low?",
    "Is there any help for stress and anxiety?",
    "When do I need to apply?",
]

for q in test_questions:
    search(q)

# Bonus: metadata filtering (only search one category)
search("How do I get to campus?", category="Campus Life")

# Interactive mode
print("\n--- Ask your own questions (type 'quit' to exit) ---")
while True:
    q = input("\nYou: ").strip()
    if q.lower() in ("quit", "exit"):
        break
    if q:
        search(q)