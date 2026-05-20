PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev lint format test services-up services-down services-reset services-logs smoke-openrouter smoke-memory inspect-backends smoke-healthcare-backends

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev,llm,api,postgres,neo4j]

lint:
	ruff check src tests demos scripts

format:
	ruff format src tests demos scripts

test:
	pytest tests/ -q

services-up:
	docker compose up -d postgres neo4j

services-down:
	docker compose down

services-reset:
	docker compose down -v

services-logs:
	docker compose logs -f --tail=100 postgres neo4j

smoke-openrouter:
	@set -a; [ -f .env ] && . ./.env || true; set +a; \
	MODEL="$${OPENROUTER_MODEL:-$${LLM_MODEL:-z-ai/glm-4.5-air:free}}"; \
	PAYLOAD=$$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply with exactly: OPENROUTER_OK"}],"max_tokens":128,"temperature":0}' "$$MODEL"); \
	RESP=$$(curl -sS -w "\nHTTP_STATUS:%{http_code}" https://openrouter.ai/api/v1/chat/completions \
		-H "Authorization: Bearer $$OPENROUTER_API_KEY" \
		-H "Content-Type: application/json" \
		-H "HTTP-Referer: https://github.com/ground-memory-core" \
		-H "X-Title: GroundedMemory" \
		-d "$$PAYLOAD"); \
	STATUS=$$(printf "%s" "$$RESP" | tail -n1 | cut -d: -f2); \
	BODY=$$(printf "%s" "$$RESP" | sed '$$d'); \
	echo "status=$$STATUS"; \
	if command -v jq >/dev/null 2>&1; then printf "%s" "$$BODY" | jq -r '.choices[0].message.content // .choices[0].message.reasoning // .error.message // "<no-output>"'; else echo "$$BODY"; fi

smoke-memory:
	@set -a; [ -f .env ] && . ./.env || true; set +a; \
	PYTHONPATH="$$PWD/src" $(PYTHON) -c 'from gmem import Memory; m=Memory(adapter="generic", storage_backend="memory"); result=m.add("My project codename is Atlas.", user_id="smoke-user"); print("add_result_type=", type(result).__name__); results=m.search("What is my project codename?", user_id="smoke-user", limit=3); print("search_count=", len(results)); [print(item) for item in results[:2]]; m.close()'

inspect-backends:
	@set -a; [ -f .env ] && . ./.env || true; set +a; \
	PYTHONPATH="$$PWD/src" $(PYTHON) scripts/smoke_and_inspect_backends.py

smoke-healthcare-backends:
	@set -a; [ -f .env ] && . ./.env || true; set +a; \
	PYTHONPATH="$$PWD/src" $(PYTHON) scripts/healthcare_backend_smoke.py
