"""Compare a chat model (text in, text out) with an embedding model (text in, vector out) using the OpenAI SDK.

Run with: uv run src/langchain_rag_playground/embeddings/chat_vs_embedding.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load API keys from .env


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Optional: the SDK reads OPENAI_API_KEY by itself

# 1. Chat model: text in -> text out
conversation = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Greece?"},
    ],
)

print("CHAT OUTPUT:")
print(f"Raw response: {conversation}")
print(f"Model's response: {conversation.choices[0].message.content}")

# 2. Embedding model: text in -> vector of numbers out
response = client.embeddings.create(
    input="What is the capital of Greece?",
    model="text-embedding-3-small",
)

vector = response.data[0].embedding
print("\nEMBEDDING OUTPUT:")
print(f"Length: {len(vector)}")  # 1536 for text-embedding-3-small
print(f"First 5 values: {vector[:5]}")
print(f"Full Vector: {vector}")  # Long output: all 1536 numbers