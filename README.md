# Agentic AI: Multi-Agent RAG System

This project is an implementation of a multi-agent orchestrated system using Retrieval-Augmented Generation (RAG). It fulfills the requirements for the Bangkok Bank (BBL) AI Engineer programming test.

## System Architecture

The system employs an **Agent-as-Tool** pattern using the OpenAI Agents SDK to orchestrate two specialized agents:

1. **Data Retriever Agent (RAG):** 
   - Acts as an information retrieval specialist.
   - Executes a custom Python tool (`search_knowledge_base`) to parse the local `knowledge_base.txt` file.
   - Uses a custom **Term-Frequency (TF) overlap scoring algorithm** (without relying on heavy Vector DBs) to extract the most relevant snippets.
   - Strictly returns raw snippets without answering the user directly.

2. **Report Generator Agent:** 
   - Acts as the orchestrator and expert corporate synthesizer.
   - Receives the user query, calls the Data Retriever to gather factual context, and synthesizes a polished, well-formatted markdown response.
   - Enforces a "Zero Hallucination" policy—if the information is not in the knowledge base, it will politely decline to answer.

## Setup & Installation

**1. Install dependencies:**
Install the required standard libraries and SDKs:

```bash
pip install -r requirements.txt
```

**2. Environment Variables (.env):**
Create a `.env` file in the root directory to configure the Azure API Management endpoint. *Note: The system is configured to authenticate via the api-key header.*

```env
BBL_API_KEY=your_provided_api_key_here
BBL_ENDPOINT=[https://apimsdbxcandidate01.azure-api.net/llm](https://apimsdbxcandidate01.azure-api.net/llm)
BBL_MODEL=gpt-5-mini
```

## Usage & Execution

**Handling API Rate Limits:**
The provided endpoint has a strict limit of 1,000 tokens per minute. To prevent rate-limit errors during continuous execution, the system implements a deliberate 30-second delay between demo queries.

**1. Run Built-in Demo Queries:**
This will sequentially execute a predefined set of test queries against the knowledge base.

```bash
python main.py
```

**2. Run a Custom Query:**
You can pass any specific question as a command-line argument.

```bash
python main.py "What is the policy on cybersecurity?"
```

## Evaluation Highlights
- **Framework & Pattern:** Correctly utilized the OpenAI Agents SDK with an "Agent-as-Tool" handoff pattern.
- **RAG Implementation:** Built a lightweight, custom text-chunking and scoring algorithm using standard Python libraries (re, pathlib).
- **Resilience:** Implemented asynchronous sleeps to respect the strict 1000 TPM rate limit provided in the updated instructions.
- **Output Quality:** Prompts are engineered to force concise, non-redundant, markdown-formatted outputs while strictly preventing hallucinations.

## Sample Execution Outputs
*(See the attached screenshots in this repository for output verification)*

**test_results_demo.png** - Output from running the sequential demo queries.

**test_results_custom.png** - Output from running a specific custom query.