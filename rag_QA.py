import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
gemini = genai.Client()

EMBED_MODEL = "gemini-embedding-001"
MODEL = "gemini-3.8-flash"  

chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.get_collection("rag_knowledge_base")

SYSTEM_PROMPT = """You are a helpful assistant for Northfield University students.
Answer the question using ONLY the context provided.
If the context does not contain the answer, reply exactly:
"I don't have that information in the university handbook."
Do not guess, and do not use outside knowledge. Keep answers short and clear."""

def retrieve(question, n_results=3):
    """RETRIEVAL: embed the question and find the closest chunks."""
    q_vec = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    ).embeddings[0].values

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "title": meta["title"], "score": 1 - dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

def answer(question):
    # 1. Retrieval
    chunks = retrieve(question)

    # 2. Augmentation: put the retrieved chunks into the prompt
    context = "\n\n".join(f"[Source {i + 1}]\n{c['text']}" for i, c in enumerate(chunks))
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # 3. Generation
    response = gemini.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,   # low = more factual and consistent
        ),
    )

    print(f"\nQUESTION: {question}")
    print("Retrieved chunks:")
    for c in chunks:
        print(f"   ({c['score']:.3f}) {c['title']}")
    print(f"ANSWER: {response.text.strip()}")

def answer_without_rag(question):
    """For comparison: ask the LLM directly, with no retrieved context."""
    response = gemini.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    print(f"\nWITHOUT RAG: {question}")
    print(f"ANSWER: {response.text.strip()}")

if __name__ == "__main__":
    # Questions worded differently from the handbook
    for q in [
        "How much do I have to pay each term?",
        "What documents do I need to apply?",
        "Can I borrow a computer from the university?",
        "What happens if I miss too many classes?",
        "I failed a subject. Can I try it again?",
    ]:
        try:
            answer(q)
        except Exception as e:
            print("Error:", e)

    # Questions NOT covered by the knowledge base (Step 12)
    for q in [
        "What is the tuition fee for the MBA program?",
        "Does the university have a swimming pool?",
    ]:
        try:
            answer(q)
        except Exception as e:
            print("Error:", e)

    # RAG vs asking the LLM directly
    answer_without_rag("How much is tuition per semester at Northfield University?")
    answer("How much is tuition per semester at Northfield University?")

    # Interactive mode
    print("\n--- Ask your own questions (type 'quit' to exit) ---")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if q:
            try:
                answer(q)
            except Exception as e:
                print("Error:", e)