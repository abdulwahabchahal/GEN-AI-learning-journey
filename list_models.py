from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()   # reads GEMINI_API_KEY from the environment

models = list(client.models.list())
if models:
    for model in models:
        print(model.name)
else:
    print("No models returned.")