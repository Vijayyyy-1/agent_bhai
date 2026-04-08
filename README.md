# agent-gitv1 🤖

> **AI-powered Git CLI** — Uses [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) + Google Gemini to automatically generate professional commit messages from your diff.

---

## How It Works

```
Your Code Changes
      │
      ▼
[MCP Client] ──stdio──► [mcp-server-git]
      │                        │
      │◄── git diff ───────────┘
      │
      ▼
[Gemini AI] ──► Generate commit message
      │
      ▼
[MCP Client] ──► git add . ──► git commit
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python ≥ 3.10 | [python.org](https://python.org) |
| `uvx` (uv tool runner) | `pip install uv` |
| `mcp-server-git` | Auto-fetched by `uvx` |
| Gemini API Key | [aistudio.google.com](https://aistudio.google.com) |

---

## Installation

```bash
# 1. Clone / navigate to the project
cd "d:\Vijay Projects\Agent_bhai"

# 2. Install in editable mode (creates the `agent` command globally)
pip install -e .

# 3. Set your Gemini API key
set GEMINI_API_KEY=your_gemini_api_key_here     # Windows CMD
$env:GEMINI_API_KEY="your_key"                  # Windows PowerShell
export GEMINI_API_KEY=your_gemini_api_key_here  # Linux / Mac
```

---

## Usage

### `agent config` — Configure LLM Provider (Start Here!)

Run this to configure OpenAI, Google Gemini, or Ollama. 

```bash
agent config
```

What you can configure:
- **Google Gemini**: Uses your API key and a model like `gemini-2.0-flash`.
- **OpenAI (ChatGPT)**: Uses your API key and a model like `gpt-4o-mini`.
- **Ollama (Local/Network)**: Provide the base URL (e.g. `http://localhost:11434` or `http://192.168.1.50:11434`). The CLI will automatically fetch your downloaded models and let you choose one!

---

### `agent commit` — Stage + AI commit message + commit

```bash
# In any git repo:
agent commit

# Specify a repo path:
agent commit --repo /path/to/repo

# Generate multiple suggestions (pick one or type your own):
agent commit --suggestions 3

# Verbose mode (shows diff preview + available MCP tools):
agent commit --verbose
agent commit -v
```

`agent commit` is now history-aware:
- It uses recent commit messages from your repo to align tone/style.
- It includes changed file names in the LLM prompt for better scoped messages.

### `agent push` — Push to remote

```bash
agent push
agent push --remote origin --branch main
```

### Help

```bash
agent --help
agent config --help
agent commit --help
agent push --help
```

---

## Example Session

```
🗂  Repository: D:\my-project

🔌 Connecting to mcp-server-git...
✅ MCP session initialized.

📂 Fetching git diff (unstaged changes)...
   Diff captured (1240 chars).

🤖 Generating commit message with Gemini...

💬 Commit Message: feat(auth): add JWT token refresh endpoint

Proceed with git add + commit? [Y/n]: y

📦 Staging all changes (git add .)...
   Files staged.

✍️  Committing...
   [main a3f12bc] feat(auth): add JWT token refresh endpoint

🚀 Done! Changes committed successfully.
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Your Google Gemini API key |

---

## Project Structure

```
Agent_bhai/
├── agent.py          # Main CLI + MCP client logic
├── pyproject.toml    # Packaging + entry point config
└── README.md         # This file
```

---

## License

MIT
