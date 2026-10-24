import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()
MODEL = "gemini-3.8-flash"   

# ---------- 1. Fictional data (no real database) ----------
ORDERS = {
    "ORD-101": {"status": "Shipped", "carrier": "FastShip",
                "estimated_delivery": "2026-10-30", "items": ["Wireless Mouse"]},
    "ORD-102": {"status": "Processing", "estimated_delivery": "2026-11-02",
                "items": ["USB-C Hub", "Laptop Stand"]},
    "ORD-103": {"status": "Delivered", "delivered_on": "2026-10-20",
                "items": ["Mechanical Keyboard"]},
}

PRODUCTS = {
    "wireless mouse":      {"price_usd": 24.99, "in_stock": True},
    "usb-c hub":           {"price_usd": 39.50, "in_stock": True},
    "laptop stand":        {"price_usd": 29.00, "in_stock": False},
    "mechanical keyboard": {"price_usd": 89.99, "in_stock": True},
}

# ---------- 2. The tool functions (our application code) ----------
def check_order_status(order_id: str) -> dict:
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"No order found with ID {order_id}."}
    return {"order_id": order_id.strip().upper(), **order}

def get_product_price(product_name: str) -> dict:
    product = PRODUCTS.get(product_name.strip().lower())
    if product is None:
        return {"error": f"No product named '{product_name}' was found."}
    return {"product": product_name.strip().lower(), **product}

# ---------- 3. Tool declarations (what the model is told about) ----------
check_order_declaration = {
    "name": "check_order_status",
    "description": "Look up the status of a customer order by its order ID (format ORD-123). "
                   "Use only when the user asks about an order.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID, for example ORD-101"},
        },
        "required": ["order_id"],
    },
}

product_price_declaration = {
    "name": "get_product_price",
    "description": "Get the price and stock status of a product from the store catalog by product name.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "description": "The product name, for example 'wireless mouse'"},
        },
        "required": ["product_name"],
    },
}

TOOLS = types.Tool(function_declarations=[check_order_declaration, product_price_declaration])

# Only these functions can ever be executed (safety: the app controls the tools)
AVAILABLE_FUNCTIONS = {
    "check_order_status": check_order_status,
    "get_product_price": get_product_price,
}

CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You are a helpful store assistant. Use the provided tools to look up order status "
        "and product prices. Never invent order or price information. If a tool returns an "
        "error, tell the user clearly what was not found. If the user asks about an order but "
        "gives no order ID, ask for it. Answer general questions normally, without tools. "
        "Keep answers short."
    ),
    tools=[TOOLS],
    temperature=0.2,
)

# ---------- 4. Helpers ----------
def generate_with_retry(**kwargs):
    """Retry if Google's servers are busy (503) or rate-limited (429)."""
    for attempt in range(5):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 4:
                wait = 2 ** (attempt + 1)
                print(f"  (server busy, retrying in {wait}s...)")
                time.sleep(wait)
            else:
                raise

def execute_tool(name, args):
    """The APPLICATION runs the function, never the model."""
    func = AVAILABLE_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**args)
    except Exception as e:
        return {"error": f"Tool failed: {e}"}

# ---------- 5. The tool-calling loop ----------
def ask(question):
    print(f"\n{'=' * 60}\n[1] USER: {question}")
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    used_tool = False

    for _ in range(5):   # safety limit on tool-call rounds
        response = generate_with_retry(model=MODEL, contents=contents, config=CONFIG)
        calls = response.function_calls

        if not calls:   # model produced a normal text answer
            if not used_tool:
                print("[2] AI decided NO tool was needed")
            print(f"[4] FINAL ANSWER: {(response.text or '').strip()}")
            return

        used_tool = True
        contents.append(response.candidates[0].content)   # keep the model's tool request

        result_parts = []
        for call in calls:
            args = dict(call.args)
            print(f"[2] AI REQUESTS TOOL: {call.name}({json.dumps(args)})")
            result = execute_tool(call.name, args)
            print(f"[3] APP EXECUTED, RESULT: {json.dumps(result)}")
            result_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=result_parts))

    print("Stopped: too many tool-call rounds.")

if __name__ == "__main__":
    test_questions = [
        "Where is order ORD-101?",                          # needs order tool
        "How much does the wireless mouse cost?",           # needs price tool
        "Is the laptop stand available?",                   # needs price tool (stock)
        "What is the capital of France?",                   # NO tool needed
        "What is the status of order ORD-999?",             # invalid order ID
        "How much is the gaming chair?",                    # unknown product
        "Where is my order?",                               # missing argument
    ]
    for q in test_questions:
        try:
            ask(q)
        except Exception as e:
            print("Error:", e)

    print("\n--- Ask your own questions (type 'quit' to exit) ---")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if q:
            try:
                ask(q)
            except Exception as e:
                print("Error:", e)