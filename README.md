# Vera Deterministic Message Engine

A deterministic, context-grounded FastAPI bot for the magicpin Vera AI Challenge.

## Approach
- Store category, merchant, customer and trigger contexts by version.
- Reject stale/replayed context versions.
- On `/v1/tick`, select only triggers with a grounded merchant/customer opportunity.
- Prefer concrete trigger signals and real active offers.
- Suppress recent/repeated outreach and never invent offers or facts.
- On `/v1/reply`, handle positive intent, opt-out, canned auto-replies and off-topic requests.

## Run
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
