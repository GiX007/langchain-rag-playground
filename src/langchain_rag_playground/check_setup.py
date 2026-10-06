"""Check the setup: print package versions and send one test prompt to OpenAI and Anthropic.

Run with: uv run src/langchain_rag_playground/check_setup.py
"""

from importlib.metadata import version

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()  # Load API keys from .env


# Installed versions, useful when debugging version mismatches
print(f"langgraph version: {version('langgraph')}")
print(f"langchain-core version: {version('langchain-core')}")
print(f"langchain-openai version: {version('langchain-openai')}")
print(f"langchain-anthropic version: {version('langchain-anthropic')}")


def main() -> None:
    """Send one short prompt to each provider and print the responses."""

    # Test openai (temperature=0: same answer on every run)
    openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    openai_response = openai_model.invoke("Say 'all good' in one word")
    print(f"\nOpenAI response: {openai_response}")

    # Test anthropic
    anthropic_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    anthropic_response = anthropic_model.invoke("Say 'all good' in one word")
    print(f"\nAnthropic response: {anthropic_response}")

    print("\nSetup complete!")


if __name__ == "__main__":
    main()