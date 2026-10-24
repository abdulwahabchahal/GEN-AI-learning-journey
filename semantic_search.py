import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()
MODEL = "gemini-embedding-001"

documents = [
    "BrightPath Academy offers online courses in Python, web development, and data science.",
    "Tuition for a full course is $200, and students can pay in three monthly installments.",
    "New batches start on the first Monday of every month.",
    "Every course includes live weekly mentor sessions and recorded lectures.",
    "Students who pass the final project receive a verified digital certificate.",
    "Refunds are available within 14 days of enrollment if you are not satisfied.",
    "Our support team is available by email from 9 AM to 6 PM, Monday to Friday.",
    "Scholarships cover up to 50% of tuition for students with financial need.",
    "Graduates get access to our career center for resume reviews and job referrals.",
    "The platform works on laptops, tablets, and mobile phones.",
]

questions = [
    "How much does it cost to enroll?",
    "Can I get my money back if I don't like it?",
    "Will I get a document proving I finished?",
    "Can I study from my phone?",
    "I can't afford the fees, is there any help?",
    "When can I begin?",
]

def embed(texts):
    result = client.models.embed_content(model=MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])

def cosine_scores(query_vec, doc_vecs):
    return doc_vecs @ query_vec / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
    )

doc_vectors = embed(documents)      # embed the knowledge base once
q_vectors = embed(questions)        # embed all questions

for q, qv in zip(questions, q_vectors):
    scores = cosine_scores(qv, doc_vectors)
    top = np.argsort(scores)[::-1][:3]      # top 3 matches
    print(f"\nQUESTION: {q}")
    for rank, idx in enumerate(top, 1):
        print(f"  {rank}. ({scores[idx]:.3f}) {documents[idx]}")

        




def keyword_search(query, docs):
    q_words = set(query.lower().replace("?", "").split())
    scored = [(len(q_words & set(d.lower().replace(".", "").split())), d) for d in docs]
    return max(scored)

print("\n--- Keyword search comparison ---")
for q in questions:
    hits, doc = keyword_search(q, documents)
    print(f"{q}\n   -> {hits} matching words: {doc}")