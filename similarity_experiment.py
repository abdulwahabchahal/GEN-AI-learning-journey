import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()
MODEL = "gemini-embedding-001"

sentences = [
    # 5 original sentences (different topics)
    "I enjoy learning Python programming.",                    # 0 programming
    "The university offers scholarships to top students.",     # 1 education
    "Spicy chicken biryani is my favorite dish.",              # 2 food
    "The cricket team won the final match yesterday.",         # 3 sports
    "We booked a week-long trip to the northern mountains.",   # 4 travel
    # 3 related sentences, different words
    "I like studying software development with Python.",       # 5 ~ 0
    "Financial aid is available for outstanding students.",    # 6 ~ 1
    "Our squad took the championship game last night.",        # 7 ~ 3
]

def embed(texts):
    result = client.models.embed_content(model=MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

vectors = embed(sentences)

# Compare each of the 3 related sentences to all 5 originals
for i in (5, 6, 7):
    print(f"\n'{sentences[i]}'")
    scores = [(j, cosine(vectors[i], vectors[j])) for j in range(5)]
    for j, s in sorted(scores, key=lambda x: -x[1]):
        print(f"   {s:.3f}  {sentences[j]}")