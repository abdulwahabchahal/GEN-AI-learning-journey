from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.8-flash"

user_text = "Explain APIs in one line."
print("1. Input:", user_text)

response = client.models.generate_content(model=MODEL, contents=user_text)

print("2. Model version:", response.model_version)
print("3. Tokens in/out:",
      response.usage_metadata.prompt_token_count, "/",
      response.usage_metadata.candidates_token_count)
print("4. Output:", response.text)