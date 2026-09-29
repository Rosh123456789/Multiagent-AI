# DeepShield AI

DeepShield AI is a local multi-agent operations prototype that demonstrates diagnosis, knowledge retrieval, safe remediation, and verification for a simple payment API + MySQL environment.

## Project vision

The prototype follows a self-healing operational loop:

Deploy -> Pre-check -> Diagnose -> Search knowledge base -> Remediate -> Verify -> Resolve

The demo focuses on a realistic failure: the payment API cannot connect to MySQL.

## Architecture

- Supervisor Agent: decides the next workflow step.
- Diagnosis Agent: inspects deployment status and identifies likely root causes.
- Knowledge/RAG Agent: retrieves runbooks from the local knowledge base.
- Remediation Agent: chooses an approved action.
- Verification Agent: confirms whether the service is healthy again.

## Repository structure

- `agents/`: agent implementations and workflow orchestration.
- `api/`: FastAPI service for exposing the workflow.
- `docker/`: Docker Compose and demo app definitions.
- `frontend/`: placeholder for the React dashboard.
- `knowledge/`: operational runbooks and troubleshooting guidance.
- `rag/`: vector-style local knowledge retrieval layer.
- `tools/`: docker-safe operational utilities.
- `main.py`: entry point for the FastAPI server.

## Demo scenario

The included Docker stack intentionally models a broken MySQL dependency for the Payment API. The workflow can analyze that issue, locate the matching runbook, and propose a safe remediation action.

## Run the API

1. Create a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start the API:
   `python main.py`
4. Open `http://localhost:8000/docs` for the FastAPI interactive docs.

## Example request

```json
{
  "application": "Payment API",
  "issue": "Payment API cannot connect to MySQL",
  "deployment": {"status": "unhealthy"}
}
```

POST it to `/workflow/run`.

## Docker demo stack

The `docker/` folder contains a local compose stack:

- `mysql`
- `payment-api`

This mirrors the intended hackathon demo and demonstrates how DeepShield can isolate, diagnose, and remediate cluster issues in a controlled environment.

## Notes

- The prototype does not execute arbitrary commands from the LLM.
- All remediation actions are intentionally bounded and approved.
- The workflow is designed to be extensible toward Ollama + LangGraph + ChromaDB in a full local deployment.
