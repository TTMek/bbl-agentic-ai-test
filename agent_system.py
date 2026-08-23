"""Two-agent RAG system built with the OpenAI Agents SDK.

Architecture (agent-as-tool pattern):

    User query
        -> Report Generator agent
             -> calls Data Retriever agent (exposed as a tool)
                  -> calls search_knowledge_base() custom tool
             <- receives raw snippets
        -> synthesises the final answer

The Data Retriever never answers the user directly; it only returns raw
snippets from knowledge_base.txt.
"""

from __future__ import annotations

import os

from agents import Agent, function_tool, set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

from retrieval import search_knowledge_base

load_dotenv()

MODEL_NAME = os.getenv("BBL_MODEL", "gpt-5-mini")


def configure_llm_client() -> None:
    """Point the Agents SDK at the provided Azure API Management endpoint.

    The gateway authenticates with an `api-key` header rather than a bearer
    token, so the key is supplied through `default_headers`. Tracing is
    disabled because no platform.openai.com key is available.
    """
    api_key = os.getenv("BBL_API_KEY")
    endpoint = os.getenv("BBL_ENDPOINT")

    if not api_key or not endpoint:
        raise RuntimeError("BBL_API_KEY and BBL_ENDPOINT must be set in .env")

    client = AsyncOpenAI(
        base_url=endpoint,
        api_key=api_key,
        default_headers={"api-key": api_key},
    )
    set_default_openai_client(client, use_for_tracing=False)
    set_tracing_disabled(True)


@function_tool
def search_documents(query: str) -> str:
    """Search the internal policy knowledge base for relevant text snippets.

    Args:
        query: Keywords or a question describing the information needed.
    """
    return search_knowledge_base(query)


data_retriever_agent = Agent(
    name="Data Retriever",
    instructions=(
        "You are an information retrieval specialist. "
        "Call the search_documents tool to find every snippet relevant to the "
        "request. Return the retrieved snippets verbatim. "
        "Never answer the question, summarise, interpret, or add commentary."
    ),
    tools=[search_documents],
    model=MODEL_NAME,
)


report_generator_agent = Agent(
    name="Report Generator",
    instructions=(
        "You are an expert writer. For every user question, first call the "
        "retrieve_information tool to obtain source snippets. "
        "Then write the final answer using only those snippets. "
        "Discard irrelevant snippets, merge overlapping details, and never "
        "repeat the same fact twice. Use short markdown sections or bullet "
        "points, and keep the answer concise. If the snippets do not cover "
        "the question, say so plainly instead of guessing."
    ),
    tools=[
        data_retriever_agent.as_tool(
            tool_name="retrieve_information",
            tool_description=(
                "Retrieve raw, relevant snippets from the policy knowledge base."
            ),
        )
    ],
    model=MODEL_NAME,
)

"""Two-agent RAG system built with the OpenAI Agents SDK.

Architecture (agent-as-tool pattern):

    User query
        -> Report Generator agent
             -> calls Data Retriever agent (exposed as a tool)
                  -> calls search_knowledge_base() custom tool
             <- receives raw snippets
        -> synthesises the final answer
"""

from __future__ import annotations
import os

from agents import Agent, function_tool, set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

from retrieval import search_knowledge_base

load_dotenv()

MODEL_NAME = os.getenv("BBL_MODEL", "gpt-5-mini")

def configure_llm_client() -> None:
    api_key = os.getenv("BBL_API_KEY")
    endpoint = os.getenv("BBL_ENDPOINT")

    if not api_key or not endpoint:
        raise RuntimeError("BBL_API_KEY and BBL_ENDPOINT must be set in .env")

    client = AsyncOpenAI(
        base_url=endpoint,
        api_key=api_key,
        default_headers={"api-key": api_key},
    )
    set_default_openai_client(client, use_for_tracing=False)
    set_tracing_disabled(True)

@function_tool
def search_documents(query: str) -> str:
    """Search the internal policy knowledge base for relevant text snippets.

    Args:
        query: Keywords or a question describing the information needed.
    """
    return search_knowledge_base(query)

data_retriever_agent = Agent(
    name="Data Retriever",
    instructions=(
        "You are an expert information retrieval specialist. "
        "Your ONLY task is to call the search_documents tool to find snippets relevant to the user's request. "
        "Return the retrieved snippets EXACTLY as they are provided by the tool. "
        "CRITICAL: Do not answer the question directly, do not summarize, do not interpret, and do not add any commentary."
    ),
    tools=[search_documents],
    model=MODEL_NAME,
)

report_generator_agent = Agent(
    name="Report Generator",
    instructions=(
        "You are an expert corporate communications writer. For every user question, you must first call the "
        "retrieve_information tool to obtain source snippets. "
        "Follow these rules strictly:\n"
        "1. SYNTHESIS: Write the final answer using ONLY the retrieved snippets.\n"
        "2. NO HALLUCINATION: Never invent information, numbers, or policies that are not in the snippets.\n"
        "3. FORMATTING: Use markdown styling (bolding, bullet points) to make the answer clear and easy to read. "
        "Do not repeat the same facts.\n"
        "4. MISSING INFO: If the snippets do not contain the answer, politely state: 'Based on the provided knowledge base, I do not have the information to answer this request.'"
    ),
    tools=[
        data_retriever_agent.as_tool(
            tool_name="retrieve_information",
            tool_description=(
                "Retrieve raw, relevant snippets from the internal policy knowledge base. Always use this tool first."
            ),
        )
    ],
    model=MODEL_NAME,
)