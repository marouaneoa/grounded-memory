# HPC Deployment and Validation

This guide packages `grounded-memory` like a typical memory SDK (compatibility import surface)
and installs it in a remote HPC environment for model-side testing.

## 1) Build distributable artifacts locally

From the repo root:

```bash
.venv/bin/python -m pip install -U build
.venv/bin/python -m build
```

Artifacts are created in `dist/`:

- `grounded_memory-<version>-py3-none-any.whl`
- `grounded_memory-<version>.tar.gz`

## 2) Transfer wheel to HPC

```bash
scp dist/grounded_memory-*.whl <user>@<hpc-host>:~/packages/
```

## 3) Install on HPC environment

```bash
python -m venv ~/venvs/gms
~/venvs/gms/bin/python -m pip install -U pip
~/venvs/gms/bin/python -m pip install ~/packages/grounded_memory-*.whl
```

Optional runtime extras:

```bash
pip install pydantic-ai
pip install asyncpg psycopg2-binary neo4j
```

## 4) Validate compatibility import and usage

Run this directly on HPC:

```bash
python - <<'PY'
from grounded_memory import Memory

m = Memory(domain_profile="generic", storage_backend="memory")
m.remember("Model endpoint is served on hpc-inference-gw")
hits = m.retrieve("where is model endpoint served", limit=3)

print("hits", len(hits))
print("ok")
PY
```

Expected output includes:

- `hits` with non-negative integer
- `ok`

## 5) Quick troubleshooting

- `ModuleNotFoundError: grounded_memory`: verify wheel installed into the active venv.
- PostgreSQL connection errors: set `POSTGRES_*` env vars or run with in-memory settings.
- Neo4j connection errors: use `storage_backend="memory"` for first smoke test.
