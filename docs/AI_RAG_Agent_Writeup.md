# HRMS-AI: RAG-Powered HR Assistant with Agentic Tool Orchestration

## Project Overview

HRMS-AI is a demonstration HR assistant that combines Retrieval-Augmented
Generation (RAG) with deterministic tool-based agent orchestration. It
addresses a practical HR information problem: users may need answers from HR
policy documents or employee-specific leave information, and those two types
of information are best handled differently.

The system retrieves relevant HR policy content and sends the retrieved
context to Gemini for a grounded answer. For questions that include a
specific employee ID, the agent routes the request to a structured employee
leave tool backed by fictional demo data.

## Key Features

- HR document ingestion
- Document chunking with overlap
- Local sentence-transformer embeddings
- ChromaDB vector search
- Grounded Gemini responses
- Retrieved source document and chunk metadata
- Deterministic agent routing
- `PolicySearchTool`
- `EmployeeLeaveTool`
- FastAPI REST APIs
- React and TypeScript frontend
- Automated backend tests

## Architecture

```text
React Frontend
      |
      v
FastAPI
      |
      v
HRAgent
   /       \
  v         v
Policy    Employee
Search    Leave Tool
  |           |
  v           v
RAG        Demo HR Data
  |
  v
Retrieval
  |
  v
ChromaDB
  |
  v
Gemini
```

`EmployeeLeaveTool` does not use ChromaDB or Gemini. It reads structured
fictional demo records directly. `PolicySearchTool` delegates to the existing
RAG service, which owns retrieval and Gemini generation.

## RAG Pipeline

The policy question path follows this flow:

```text
HR document
    → chunking
    → embedding
    → ChromaDB
    → query embedding
    → similarity retrieval
    → retrieved context
    → Gemini
    → grounded answer
```

During ingestion, `IngestionService` loads a document, splits it into chunks,
creates embeddings, and stores the chunks, embeddings, and source metadata in
local ChromaDB. For a question, `RetrievalService` embeds the query and
requests the most relevant chunks from `VectorStoreService`.

`RAGService` passes those chunks to `LLMService`. The Gemini prompt instructs
the model to answer only from the supplied HR document context, not invent HR
policies or facts, and clearly state when the information is unavailable in
the provided documents.

RAG is used instead of relying on general LLM knowledge because HR policies
should be grounded in the organization's supplied documents. Retrieval also
allows the response to include the source document and chunk index used as
context.

## Agentic Orchestration

The current agent is a first-stage deterministic tool-based agent. `HRAgent`
receives a question and selects one of two injected tools:

- `PolicySearchTool`
- `EmployeeLeaveTool`

If the question contains an employee ID matching the `E###` pattern, such as
`E001`, `HRAgent` routes it to `EmployeeLeaveTool`. Questions without an
employee ID currently route to `PolicySearchTool`.

`PolicySearchTool` reuses the existing `RAGService`. `EmployeeLeaveTool`
looks up fictional structured demo data containing fields such as employee
name, years of experience, annual leave days, and emergency leave days.

> The current implementation intentionally uses deterministic routing rather
> than pretending to perform LLM reasoning. The tool boundary is designed so
> that LLM-based tool calling can be introduced later.

This is a tool-based agentic system, not an autonomous multi-agent system.

## Example Questions

### Policy question

```text
How many annual leave days are employees entitled to?
```

This follows the policy/RAG path:

```text
HRAgent → PolicySearchTool → RAGService → RetrievalService → ChromaDB → Gemini
```

### Employee-specific question

```text
How much leave does E001 have?
```

This follows the structured-data path:

```text
HRAgent → EmployeeLeaveTool → Demo employee data
```

### Emergency leave policy question

```text
What is the emergency leave entitlement?
```

Because it does not contain an employee ID, this follows the policy/RAG path.

The employee records and policy documents used by this project are demo
inputs; the employee data is fictional and not production HR data.

## API Endpoints

### `POST /api/rag/ingest`

Indexes a document through the existing ingestion pipeline.

```json
{
  "filename": "leave_policy.txt"
}
```

### `POST /api/rag/ask`

Runs the RAG pipeline directly for a policy question and returns the answer
with retrieved source metadata.

```json
{
  "question": "How many annual leave days are employees entitled to?"
}
```

### `POST /api/agent/ask`

Runs the deterministic agent router. This is the primary frontend
demonstration endpoint.

```json
{
  "question": "How much leave does E001 have?"
}
```

## Frontend

The React frontend provides:

- A question textarea
- Example question buttons
- Loading and error states
- Question and answer display
- An explanation of the agentic routing architecture
- A simple RAG processing pipeline overview

The frontend communicates with FastAPI through the `VITE_API_BASE_URL`
environment variable and sends questions to `POST /api/agent/ask`. A
`frontend/.env.example` file provides the local backend URL configuration.

## Testing

The latest verified backend test run reported:

```text
28 passed
```

The latest frontend production build completed successfully with:

```powershell
npm run build
```

The automated suite covers document processing, embeddings, ingestion,
retrieval, RAG orchestration, API behavior, and agent routing. Relevant API,
RAG, and agent tests use fake services where appropriate, so they do not make
real Gemini API requests.

## Security / Secret Handling

- The Gemini API key is stored in backend environment configuration.
- `.env` files are ignored by Git.
- `.env.example` files contain configuration placeholders only.
- The React frontend does not contain the Gemini API key.
- API keys and other secrets must never be committed.

No secret values are included in this document.

## Technology Stack

### Frontend

- React
- TypeScript
- Vite
- Plain CSS

### Backend

- Python
- FastAPI

### AI / RAG

- Sentence Transformers
- ChromaDB
- Google Gemini

### Testing

- Pytest
- FastAPI test client

### Version control

- Git

## My Contribution

I designed and implemented the RAG pipeline, including document ingestion,
chunking, embeddings, vector retrieval, and Gemini-based grounded generation.
I also implemented the deterministic HR agent, tool orchestration, FastAPI
APIs, automated tests, and React demonstration interface.

## Limitations

- `EmployeeLeaveTool` currently uses fictional demo data.
- Agent routing is deterministic rather than LLM-based tool calling.
- This is a demonstration/portfolio implementation rather than a production
  HR system.
- The current knowledge base is limited to the supplied HR documents.
- ChromaDB is configured as a local vector store.
- Authentication and authorization are not implemented.

## Future Improvements

- Connect `EmployeeLeaveTool` to the actual HRMS database.
- Introduce LLM-based function/tool calling.
- Add additional HR tools, such as employee directory or benefits lookup.
- Add authentication and authorization.
- Support multiple HR documents and document management.
- Improve retrieval, ranking, and relevance evaluation.
- Add conversation history.
- Deploy the frontend and backend.

## How I would explain this project in an interview

I would describe HRMS-AI as a focused demonstration of how an HR assistant
can combine retrieval with structured tools. For policy questions, the
application ingests documents, creates embeddings, retrieves relevant chunks
from ChromaDB, and gives that context to Gemini so the answer is grounded in
the supplied policy rather than general model knowledge.

On top of that RAG pipeline, I added a small deterministic agent layer. The
agent checks for an employee ID and chooses either the employee leave tool or
the policy-search tool. I intentionally kept this routing transparent rather
than presenting keyword matching as LLM reasoning. Because the tools have
clear interfaces, an LLM-based tool-calling router could be introduced later
without rewriting the underlying RAG or employee-data tools.

## Portfolio Summary

- Implemented an HR policy RAG pipeline with document chunking, embeddings,
  ChromaDB retrieval, and grounded Gemini generation.
- Added source-aware answers that identify retrieved policy documents and
  chunks.
- Built a deterministic tool-based HR agent with policy and employee leave
  tools.
- Exposed the pipeline through FastAPI and connected it to a React/Vite
  demonstration interface.
- Added automated tests using fake services so core tests do not require real
  Gemini requests.
