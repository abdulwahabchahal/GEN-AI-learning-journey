from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.8-flash"   

print("=== Tiny CLI AI Assistant (Gemini) ===")
print("Type your question. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if not user_input:
        print("Please type something.\n")
        continue

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful, concise assistant.",
                temperature=0.7,
                max_output_tokens=800,
            ),
        )
        print("\nAI:", response.text, "\n")
    except Exception as e:
        print("\nSomething went wrong:", e, "\n")