"""LLM integration with prompt templates for RAG-based Q&A.

Implements context-grounded answering with source attribution,
using OpenAI's chat completions API with streaming support.
"""

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL
from retrieval import RetrievalResult

SYSTEM_PROMPT = """You are an expert assistant specializing in SBA (Small Business Administration) 7(a) loan regulations and guidelines. Your role is to provide accurate, helpful answers to questions about SBA lending programs.

IMPORTANT RULES:
1. Answer ONLY based on the provided context documents. Do not use prior knowledge.
2. Always cite your sources using the format [Source: document name, section, page].
3. If the context does not contain enough information to answer the question, say: "I don't have sufficient information in the available documents to answer that question. Try rephrasing or asking about a specific SBA 7(a) topic."
4. Be precise and professional. This information may be used for lending decisions.
5. When citing specific numbers (loan amounts, rates, percentages), always include the source.
6. If multiple sources provide different information, note the discrepancy."""

CONTEXT_TEMPLATE = """Here are the relevant document excerpts to use when answering:

{context}

---
Based ONLY on the above context, answer the following question. Include source citations.

Question: {question}"""


def format_context(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a context string for the LLM."""
    if not results:
        return "No relevant documents found."

    context_parts = []
    for i, result in enumerate(results, 1):
        citation = result.citation
        context_parts.append(
            f"[Document {i}] ({citation})\n{result.text}"
        )

    return "\n\n".join(context_parts)


def build_messages(
    question: str,
    results: list[RetrievalResult],
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """Build the message list for the OpenAI chat completion."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent chat history for conversational context
    if chat_history:
        for msg in chat_history[-6:]:  # Last 3 exchanges
            messages.append(msg)

    # Build the user message with context
    context_str = format_context(results)
    user_content = CONTEXT_TEMPLATE.format(context=context_str, question=question)
    messages.append({"role": "user", "content": user_content})

    return messages


def generate_response(
    question: str,
    results: list[RetrievalResult],
    chat_history: list[dict] | None = None,
) -> str:
    """Generate a non-streaming response."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = build_messages(question, results, chat_history)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def generate_response_stream(
    question: str,
    results: list[RetrievalResult],
    chat_history: list[dict] | None = None,
):
    """Generate a streaming response (yields chunks of text).

    Use this for the Streamlit UI to show tokens as they arrive.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = build_messages(question, results, chat_history)

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
