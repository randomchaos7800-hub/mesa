# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

MESA (Memory Eval Suite for Agents) is a pure Python benchmark framework for personal AI memory systems. No external services, databases, or Docker containers are required.

### Dependencies

The build backend in `pyproject.toml` (`setuptools.backends.legacy:build`) requires a recent setuptools. If `pip install -e ".[dev]"` fails with `ModuleNotFoundError: No module named 'setuptools.backends'`, upgrade setuptools first (`pip install --upgrade setuptools`) or install deps directly: `pip install rouge-score openai pytest jsonschema`.

Since packages install to `~/.local`, ensure `~/.local/bin` is on `PATH` and `/workspace` is on `PYTHONPATH` before running commands:

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/workspace:$PYTHONPATH"
```

### Running tests

```bash
pytest tests/ -v
```

All 150 tests should pass (6 skipped for optional adapter deps `chromadb` and `mem0ai`).

### Running the benchmark CLI

See `README.md` for full usage. Quick smoke tests:

```bash
# v1 legacy path
python3 -m mesa.runner --adapter examples.simple_adapter.NullAdapter --dataset dataset/fixtures/sample.json --no-llm-judge --schema-version 1

# v2 official path
python3 -m mesa.runner --adapter examples.simple_adapter.EchoAdapter --dataset dataset/fixtures/sample_v2.json --schema-version 2 --limit 3
```

### Lint

No linter is configured in this repository.

### Notes

- LLM-based adapters (KeywordAdapter, ChromaAdapter, Mem0Adapter) require an OpenAI-compatible API endpoint and will not work without one. The test suite and example adapters (EchoAdapter, NullAdapter, DictAdapter) are fully self-contained.
- The `--no-llm-judge` flag is required for v1 runs without an LLM endpoint.
