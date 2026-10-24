from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="I enjoy learning Python programming",
)

vector = result.embeddings[0].values
print("Vector length:", len(vector))
print("First 8 numbers:", vector[:8])