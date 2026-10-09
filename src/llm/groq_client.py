import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def get_groq_client():
    """
    Create and return a Groq client.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to the .env file."
        )

    return Groq(api_key=api_key)


def generate_response(
    prompt: str,
    model: str = "openai/gpt-oss-20b",
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    """
    Send a prompt to the Groq LLM and return the response.
    """

    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content