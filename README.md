# agent-gitv1

AI-powered Git CLI for commit workflows using MCP plus your chosen LLM provider.

Supported providers:
- Google Gemini
- OpenAI
- Ollama (local or same-network)

## Quick Start

1. Install:

```bash
pip install agent-gitv1
```

2. Configure provider/model:

```bash
agent config
```

3. Use it in any git repository:

```bash
agent commit
agent explain
agent push
```

## Features

- Provider setup with `agent config`
- Smart staging with auto-grouping (UI/backend/config/etc.)
- Multi-suggestion commit messages (`--suggestions`)
- Repo-aware commit style learning (recent commits)
- `agent explain` for change summary, intent, and risk areas
- Ollama local-network detection (same `/24`, port `11434`)
- Loading/thinking spinner during long-running tasks
- `agent push` with live command output streaming
- LLM-driven push-failure diagnosis with adaptive live git evidence collection

## Requirements

- Python 3.10+
- Git
- `uvx` (`pip install uv`)
- Provider credentials (Gemini/OpenAI) or running Ollama server

## Install

```bash
pip install agent-gitv1
```

Verify:

```bash
agent --version
agent --help
```

## Configure

```bash
agent config
```

Provider notes:
- Gemini: set `GEMINI_API_KEY` or enter key in config
- OpenAI: set `OPENAI_API_KEY` or enter key in config
- Ollama: can auto-detect endpoints like `http://192.168.1.35:11434`

## Commands

### `agent config`
Configure LLM provider and model.

### `agent commit`
Smart-stage changes, generate repo-aware suggestions, choose messages, and commit.

Examples:

```bash
agent commit
agent commit --suggestions 3
agent commit --no-smart-stage
agent commit --repo /path/to/repo
agent commit --verbose
```

### `agent explain`
Explain current uncommitted changes:
- what changed
- why it likely changed
- risk areas
- files impacted

Examples:

```bash
agent explain
agent explain --repo /path/to/repo
```

### `agent push`
Push to remote with live output.

Examples:

```bash
agent push
agent push --remote origin --branch main
```

When `--branch` is omitted, `agent` auto-detects and pushes your current local branch name.

If push fails, `agent` first uses the LLM to choose evidence depth, then runs diagnosis commands live.
For divergence/history problems it collects full branch evidence (`fetch`, logs, ahead/behind, merge-base).
For auth/permission issues it uses a lightweight evidence set to avoid noisy output.
If the current branch has no upstream, `agent` offers to run `git push --set-upstream` automatically.

### `agent --help`
Show all commands and options.

## Environment Variables

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

## Developer Setup (optional)

Only for contributors working on this codebase:

```bash
git clone <repo-url>
cd Agent_bhai
pip install -e .
```

## Publish to PyPI

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## License

MIT
