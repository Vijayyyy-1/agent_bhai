# agent-gitv1

AI-powered Git CLI for commit, push, explain, and pull-request workflows using MCP + LLMs.

## What It Does

`agent-gitv1` helps developers move faster in git workflows by generating commit messages, explaining large diffs, diagnosing push failures, and drafting PR titles/descriptions.

Supported LLM providers:
- Google Gemini
- OpenAI
- Ollama (local or same network)

## Quick Start

1. Install:

```bash
pip install agent-gitv1
```

2. Configure your provider:

```bash
agent config
```

3. Run in any git repo:

```bash
agent commit
agent explain
agent pr
agent push
```

## Key Features

- AI-assisted commit messages with multiple suggestions
- Smart staging mode that can split work into logical commit groups
- Repository-aware commit style adaptation from recent commit history
- `agent explain` for change intent, risk, and impacted files
- `agent push` with live command output and LLM-driven failure diagnosis
- Adaptive push diagnostics (light or full evidence collection)
- Interactive PR drafting with remote detection and branch selection
- Optional PR creation via GitHub CLI (`gh pr create`)
- Ollama LAN auto-discovery (`/24` subnet on port `11434`)

## Requirements

- Python 3.10+
- Git
- `uvx` (install with `pip install uv`)
- Provider credentials for Gemini/OpenAI, or a reachable Ollama instance
- GitHub CLI (`gh`) if you want `agent pr --create`

## Installation

```bash
pip install agent-gitv1
```

Verify installation:

```bash
agent --version
agent --help
```

## Configuration

```bash
agent config
```

Provider notes:
- Gemini: uses `GEMINI_API_KEY` env var or key entered during config
- OpenAI: uses `OPENAI_API_KEY` env var or key entered during config
- Ollama: can auto-detect URLs such as `http://192.168.1.35:11434`

## Commands

### `agent commit`
Generate AI commit messages and commit changes.

Examples:

```bash
agent commit
agent commit --suggestions 3
agent commit --no-smart-stage
agent commit --repo /path/to/repo
agent commit --verbose
```

Behavior highlights:
- Uses MCP git diff for normal commit flow
- Can auto-group changed files and create multiple commits
- Adapts commit style to recent repository history

### `agent explain`
Explain current uncommitted changes in a developer-friendly format.

Examples:

```bash
agent explain
agent explain --repo /path/to/repo
```

Output includes:
- What changed
- Why it likely changed
- Risk areas
- Files impacted

### `agent pr`
Generate PR title/body from commits + diff, then optionally create PR via GitHub CLI.

Examples:

```bash
agent pr
agent pr --create
agent pr --draft --create
agent pr --target-remote upstream
agent pr --target-remote upstream --base develop --head feature/my-branch --create
```

Interactive flow (default):
- Detect remotes (`origin`, `upstream`, etc.)
- Let user choose PR target remote if multiple exist
- Fetch and let user choose base branch
- Compare ahead/behind status and warn if branch is behind target
- Generate PR title and description with LLM
- Optionally create PR with `gh pr create`

### `agent push`
Push current branch (or explicit branch) with live output.

Examples:

```bash
agent push
agent push --branch main
agent push --remote origin --branch feature/my-branch
```

Behavior highlights:
- Auto-detects and pushes current local branch when `--branch` is omitted
- If no upstream exists, offers `git push --set-upstream`
- On failure, collects live git evidence and generates LLM diagnosis with fix commands

### `agent config`
Configure provider, model, and Ollama endpoint/model.

### `agent --help`
List all commands and options.

## Environment Variables

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

## GitHub CLI Requirement for PR Creation

`agent pr --create` requires authenticated GitHub CLI.

```bash
gh auth status
gh auth login
```

## Developer Setup (Optional)

Only if you are modifying this codebase locally:

```bash
git clone <repo-url>
cd Agent_bhai
pip install -e .
```

## Build and Publish (PyPI)

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## License

MIT
