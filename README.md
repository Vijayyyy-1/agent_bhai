# agent-gitv1

AI-powered Git CLI that uses Model Context Protocol (MCP) plus your configured LLM provider to automate commit workflows.

Supported providers:
- Google Gemini
- OpenAI
- Ollama (local or same-network)

## Features

- `agent config` to set provider and model
- Smart staging with auto-grouping (UI/backend/config/etc.)
- `agent commit` to generate commit messages from git diff
- Multiple commit suggestions (`--suggestions`)
- Repo-aware style learning from last ~20 commit subjects
- `agent explain` for change/intent/risk summaries
- Ollama model auto-detection on local network (`/24`, port `11434`)
- `agent push` to push to remote

## Prerequisites

- Python 3.10+
- `uvx` (`pip install uv`)
- Git
- Provider credentials (Gemini/OpenAI) or running Ollama server

## Installation

```bash
cd "d:\Vijay Projects\Agent_bhai"
pip install -e .
```

After install, use command:

```bash
agent --help
```

## Configuration

Start with:

```bash
agent config
```

Provider notes:
- Gemini: use `GEMINI_API_KEY` or save key in config
- OpenAI: use `OPENAI_API_KEY` or save key in config
- Ollama: can auto-detect servers like `http://192.168.1.35:11434`

## Commands

### `agent --help`
Shows all available commands and options.

### `agent config`
Configure provider and model.

### `agent commit`
Smart-stage changes, generate repo-aware suggestions, choose messages, and commit.

Examples:

```bash
agent commit
agent commit --repo /path/to/repo
agent commit --suggestions 3
agent commit --no-smart-stage
agent commit --verbose
```

When smart staging is enabled (default), the CLI detects logical change groups and asks if it should create multiple commits.

### `agent explain`
Explain current changes with:
- what changed
- why it likely changed
- risk areas
- files impacted

Example:

```bash
agent explain
agent explain --repo /path/to/repo
```

### `agent push`
Push commits to remote.

Examples:

```bash
agent push
agent push --remote origin --branch main
```

## Environment Variables

- `GEMINI_API_KEY` (Gemini)
- `OPENAI_API_KEY` (OpenAI)

## Publish to PyPI

1. Bump version in `pyproject.toml` and `agent.py`
2. Build and validate:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

3. Upload:

```bash
python -m twine upload dist/*
```

## Project Structure

```text
Agent_bhai/
|- agent.py
|- pyproject.toml
|- README.md
```

## License

MIT
