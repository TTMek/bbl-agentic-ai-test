"""Entry point: run the two-agent RAG system against sample queries.

Usage:
    python main.py                      # run the built-in demo queries
    python main.py "your question"      # run a single custom query
"""

from __future__ import annotations

import asyncio
import sys

from agents import Runner

from agent_system import configure_llm_client, report_generator_agent

DEMO_QUERIES = (
    "What is the policy on international travel?",
    "How many days of annual leave can I carry over to next year?",
    "What are the rules for working from home?",
)

# The provided endpoint is limited to 1000 tokens per minute, and each query
# consumes two LLM calls (retriever + generator). Pause between queries so the
# demo run does not trip the rate limit.
DELAY_BETWEEN_QUERIES_SECONDS = 30


async def answer(query: str) -> str:
    """Run the orchestration for a single query and return the final answer."""
    result = await Runner.run(report_generator_agent, query)
    return result.final_output


async def main() -> None:
    configure_llm_client()

    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else list(DEMO_QUERIES)

    for index, query in enumerate(queries):
        print(f"\n{'=' * 72}")
        print(f"QUERY: {query}")
        print(f"{'=' * 72}\n")

        try:
            print(await answer(query))
        except Exception as error:  # noqa: BLE001 - surface any API/config failure
            print(f"[error] {type(error).__name__}: {error}")

        if index < len(queries) - 1:
            print(f"\n[waiting {DELAY_BETWEEN_QUERIES_SECONDS}s to respect rate limit]")
            await asyncio.sleep(DELAY_BETWEEN_QUERIES_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())