# Agent Guidance for selenium-llm-engine

## General purpose
This document defines the expected behavior of agents working on this repository.

1. Check and verify documentation
   - Read and align with the instructions and features defined in:
     - `README.md`
     - `DEVELOPERS.md`
   - Ensure implementations and APIs follow documented usage flows.

2. Diagnose LOGs
   - read container logs if the container is running

3. Fix behavior
   - If a bug is reported (user request), apply the minimal required fix + regression test.
   - Update existing tests or add new tests in `tests/test_api_endpoints.py` to cover behavior.

4. Formatting and style
   - Use idiomatic Python; specifically, avoid JSON-style constructs (`true`/`false`) in Python code.

5. Always cehck your code
   - After any change, run quality checks: `ruff check .`, `python -m py_compile ...`, `pytest -q`.
   - Do not deliver any broken code

6. Never `git commit` or `git add` if not explicitly requested.

## Mandatory checks during review
- `ruff check .` must pass.
- If the change involves APIs, add request tests with valid and invalid payloads.
- Document the solution
