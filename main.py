"""Main entry point for the RAG demo application.

Run with: uv run main.py
"""

from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


print(f"langgraph version: {version('langgraph')}")
print(f"langchain-core version: {version('langchain-core')}")
print(f"langchain-openai version: {version('langchain-openai')}")
print(f"langchain-anthropic version: {version('langchain-anthropic')}")


def main():

    # Test openai
    openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    openai_response = openai_model.invoke("Say 'all good' in one world")
    print(f"\nOpenAI response: {openai_response}")

    # Test anthropic
    anthropic_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    anthropic_response = anthropic_model.invoke("Say 'all good' in one world")
    print(f"\nAnthropic response: {anthropic_response}")

    print("\nSetup complete!")


if __name__ == "__main__":
    main()