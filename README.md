# Agentic AI: Multi-Agent RAG System

This project is an implementation of a multi-agent orchestrated system using Retrieval-Augmented Generation (RAG). It fulfills the requirements for the AI Engineer programming test.

## Architecture
The system employs an **Agent-as-Tool** pattern using the OpenAI Agents SDK:
1. **Data Retriever Agent**: Specializes in information retrieval. It executes a custom Python tool to parse `knowledge_base.txt` using a normalized term-frequency scoring algorithm to extract relevant snippets.
2. **Report Generator Agent**: Acts as the orchestrator. It receives user queries, calls the Data Retriever to gather factual context, and synthesizes a polished, well-formatted response.

## Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt