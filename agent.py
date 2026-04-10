"""
agent-gitv1: A CLI tool that uses MCP + Local/Cloud LLMs to automate git workflows.
"""

import asyncio
import sys
import os
import json
import re
import subprocess
import socket
import threading
import time
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import click
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG_FILE = Path.home() / ".agent_git_config.json"
DEFAULT_DIFF_TARGET = "HEAD"
MAX_DIFF_CHARS_FOR_LLM = 28000
MAX_FILES_FOR_LLM_DIFF = 40
MAX_CHARS_PER_FILE_SECTION = 1400
MAX_HISTORY_COMMITS = 12
STYLE_HISTORY_COMMITS = 20
MAX_HISTORY_CHARS = 2400

COMMIT_SYSTEM_PROMPT = """
You are an expert software engineer. Your ONLY job is to write a concise, professional Git commit message.

Rules:
- Use the Conventional Commits format: <type>(<optional scope>): <short summary>
- Types: feat, fix, docs, style, refactor, test, chore, ci, perf
- Summary must be in imperative mood (e.g., "add feature" not "added feature")
- Keep the summary under 72 characters
- Do NOT include explanations, markdown, backticks, or any extra text
- Output ONLY the commit message, nothing else

Examples:
  feat(auth): add JWT token refresh endpoint
  fix(ui): resolve button alignment on mobile devices
  docs: update README with installation steps
  refactor(api): simplify error handling middleware
"""

EXPLAIN_SYSTEM_PROMPT = """
You are a senior engineer explaining a code diff.
Return plain text with exactly these sections:
1) What Changed
2) Why It Likely Changed
3) Risk Areas
4) Files Impacted

Be concrete and concise. Do not use markdown tables.
"""

PUSH_FAILURE_SYSTEM_PROMPT = """
You are a senior Git assistant. Diagnose a failed `git push`.
Return plain text with exactly these sections (no markdown):
1) Why It Failed
2) Most Likely Fix
3) Commands To Run
4) Safety Notes

Rules:
- Do not use markdown headings, code fences, bullet symbols, or tables.
- Use simple lines only.
- Commands must be valid PowerShell commands and copy-paste ready.
- Put one command per line.
- Keep it practical and concise.
- On Windows auth issues, prefer `git credential-manager github logout`
  over generic erase commands.
- Avoid destructive commands unless absolutely necessary.
"""

PUSH_EVIDENCE_ROUTER_PROMPT = """
You classify how much git evidence is needed to diagnose a failed push.
Return exactly one word:
FULL
or
LIGHT

Use FULL for branch divergence/history sync issues (non-fast-forward, fetch first, behind remote, rejected updates).
Use LIGHT for auth/permission/token/network/remote-url issues where heavy history logs are unnecessary.
No extra text.
"""

COMMIT_TYPE_PATTERN = r"(?:feat|fix|docs|style|refactor|test|chore|ci|perf)"
COMMIT_MESSAGE_PATTERN = re.compile(
    rf"^{COMMIT_TYPE_PATTERN}(?:\([^)]+\))?: .+"
)

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

@contextmanager
def loading_spinner(message: str):
    """
    Lightweight terminal spinner for long-running steps.
    Falls back to a plain message when stdout is not a TTY.
    """
    if not sys.stdout.isatty():
        click.echo(f"{message}...")
        yield
        return

    frames = ["[=   ]", "[==  ]", "[=== ]", "[ ===]", "[  ==]", "[   =]", "[  ==]", "[ ===]"]
    stop_event = threading.Event()

    def _spin():
        idx = 0
        while not stop_event.is_set():
            frame = frames[idx % len(frames)]
            sys.stdout.write(f"\r{message} {frame}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.12)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=0.5)
        sys.stdout.write("\r" + (" " * (len(message) + 12)) + "\r")
        sys.stdout.flush()

def run_git(repo_path: str, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        text=False,
        capture_output=True,
    )
    stdout = (result.stdout or b"").decode("utf-8", errors="replace")
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise click.ClickException(stderr.strip() or f"git {' '.join(args)} failed.")
    return stdout

def run_command_live(cmd: list[str], cwd: str) -> tuple[int, str, str]:
    """
    Run a command with live terminal streaming while also capturing output.
    Returns (returncode, stdout_text, stderr_text).
    """
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _reader(stream, collector: list[str], is_err: bool):
        if stream is None:
            return
        for chunk in iter(stream.readline, b""):
            text = chunk.decode("utf-8", errors="replace")
            collector.append(text)
            click.echo(text.rstrip("\n"), err=is_err)
        stream.close()

    t_out = threading.Thread(target=_reader, args=(process.stdout, stdout_chunks, False), daemon=True)
    t_err = threading.Thread(target=_reader, args=(process.stderr, stderr_chunks, True), daemon=True)
    t_out.start()
    t_err.start()

    returncode = process.wait()
    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)

    return returncode, "".join(stdout_chunks).strip(), "".join(stderr_chunks).strip()


# ─────────────────────────────────────────────
# AI Providers
# ─────────────────────────────────────────────

def get_ollama_models(base_url: str):
    import requests
    try:
        url = base_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []

def get_local_ipv4() -> str:
    """
    Best-effort local IPv4 detection for subnet discovery.
    Uses UDP connect trick (no packet sent) to determine outbound interface IP.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return ""

def discover_ollama_servers(port: int = 11434, timeout: float = 0.6) -> list[tuple[str, list[str]]]:
    """
    Discover Ollama servers in the same /24 subnet plus localhost.
    Returns list of (base_url, models) for reachable Ollama instances.
    """
    local_ip = get_local_ipv4()
    candidate_urls = {
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
    }

    if local_ip and "." in local_ip:
        subnet = ".".join(local_ip.split(".")[:3])
        for i in range(1, 255):
            candidate_urls.add(f"http://{subnet}.{i}:{port}")

    found: list[tuple[str, list[str]]] = []

    def probe(url: str):
        models = get_ollama_models(url)
        if models:
            return (url, models)
        return None

    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(probe, url) for url in sorted(candidate_urls)]
        for future in as_completed(futures):
            result = future.result()
            if result:
                found.append(result)

    # Keep localhost entries first, then sort by URL.
    localhost_first = sorted(
        found,
        key=lambda item: (0 if ("localhost" in item[0] or "127.0.0.1" in item[0]) else 1, item[0]),
    )
    return localhost_first

def generate_with_gemini(diff: str, config: dict) -> str:
    from google import genai
    from google.genai import types
    api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise click.ClickException("Gemini API key is not configured. Run 'agent config' or set GEMINI_API_KEY.")
    
    client = genai.Client(api_key=api_key)
    user_prompt = diff
    response = client.models.generate_content(
        model=config.get("model", "gemini-2.0-flash"),
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=COMMIT_SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text.strip()

def generate_with_openai(diff: str, config: dict) -> str:
    from openai import OpenAI
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise click.ClickException("OpenAI API key is not configured. Run 'agent config' or set OPENAI_API_KEY.")
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config.get("model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": COMMIT_SYSTEM_PROMPT},
            {"role": "user", "content": diff}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def generate_with_ollama(diff: str, config: dict) -> str:
    from openai import OpenAI
    base_url = config.get("base_url", "http://localhost:11434")
    # Ollama uses the OpenAI client but needs the /v1 suffix if not provided
    openai_base_url = base_url.rstrip("/") + "/v1"
    
    # Fake API key for Ollama compatibility
    client = OpenAI(base_url=openai_base_url, api_key="ollama-local")
    try:
        response = client.chat.completions.create(
            model=config.get("model", "llama3:latest"),
            messages=[
                {"role": "system", "content": COMMIT_SYSTEM_PROMPT},
                {"role": "user", "content": diff}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise click.ClickException(f"Ollama generation failed: {e}\nIs Ollama running at {base_url}?")

def generate_commit_message(diff: str) -> str:
    """Send the git diff to the configured AI and get back a commit message."""
    config = load_config()
    provider = config.get("provider", "gemini") # Default fallback
    
    if provider == "gemini":
        return generate_with_gemini(diff, config)
    elif provider == "openai":
        return generate_with_openai(diff, config)
    elif provider == "ollama":
        return generate_with_ollama(diff, config)
    else:
        raise click.ClickException(f"Unknown provider '{provider}'. Run 'agent config'.")

def build_commit_system_prompt(suggestions: int) -> str:
    if suggestions <= 1:
        return COMMIT_SYSTEM_PROMPT

    return f"""
You are an expert software engineer. Your ONLY job is to write concise, professional Git commit messages.

Rules:
- Use the Conventional Commits format: <type>(<optional scope>): <short summary>
- Types: feat, fix, docs, style, refactor, test, chore, ci, perf
- Summary must be in imperative mood (e.g., "add feature" not "added feature")
- Keep each summary under 72 characters
- Return exactly {suggestions} options
- Each option must be on a new line with numbering: "1. ...", "2. ..."
- Do NOT include explanations, markdown, backticks, or any extra text
"""

def extract_changed_files(diff_text: str, max_files: int = 20) -> list[str]:
    files = []
    for line in diff_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            path = parts[3]
            if path.startswith("b/"):
                path = path[2:]
            if path not in files:
                files.append(path)
        if len(files) >= max_files:
            break
    return files

def get_recent_commit_history(repo_path: str, max_commits: int = MAX_HISTORY_COMMITS) -> str:
    try:
        cmd = [
            "git",
            "log",
            f"-n{max_commits}",
            "--pretty=format:%h %s",
        ]
        output = subprocess.check_output(
            cmd,
            cwd=repo_path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""

    if not output:
        return ""

    if len(output) > MAX_HISTORY_CHARS:
        return output[:MAX_HISTORY_CHARS] + "\n... [history truncated]"
    return output

def get_recent_commit_subjects(repo_path: str, max_commits: int = STYLE_HISTORY_COMMITS) -> list[str]:
    try:
        output = run_git(repo_path, ["log", f"-n{max_commits}", "--pretty=format:%s"]).strip()
    except Exception:
        return []
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]

def build_repo_style_profile(repo_path: str) -> str:
    subjects = get_recent_commit_subjects(repo_path, STYLE_HISTORY_COMMITS)
    if not subjects:
        return "No commit history style available."

    prefixes = []
    with_scope = 0
    lengths = []
    for msg in subjects:
        lengths.append(len(msg))
        m = re.match(rf"^({COMMIT_TYPE_PATTERN})(\([^)]+\))?:", msg)
        if m:
            prefixes.append(m.group(1).lower())
            if m.group(2):
                with_scope += 1

    prefix_counts = Counter(prefixes)
    top_prefixes = ", ".join(f"{k}({v})" for k, v in prefix_counts.most_common(3)) or "none detected"
    avg_len = int(sum(lengths) / len(lengths))
    scope_ratio = int((with_scope / len(subjects)) * 100)

    return (
        "Repository commit style profile:\n"
        f"- Recent commits analyzed: {len(subjects)}\n"
        f"- Common prefixes: {top_prefixes}\n"
        f"- Average subject length: {avg_len} chars\n"
        f"- Scope usage: {scope_ratio}% of commits\n"
        f"- Examples:\n  - " + "\n  - ".join(subjects[:5])
    )

def build_commit_user_prompt(
    diff_text: str,
    history_text: str,
    changed_files: list[str],
    style_profile: str,
) -> str:
    files_block = "\n".join(f"- {name}" for name in changed_files) if changed_files else "- (not detected)"
    history_block = history_text if history_text else "(No recent commits available)"
    return (
        "Write commit message suggestion(s) for the following change.\n\n"
        "Recent commit messages from this repository (for style/context):\n"
        f"{history_block}\n\n"
        "Derived style guidance from repository history:\n"
        f"{style_profile}\n\n"
        "Changed files:\n"
        f"{files_block}\n\n"
        "Git diff:\n"
        f"{diff_text}"
    )

def sanitize_commit_candidate(text: str) -> str:
    cleaned = text.strip().strip("`").strip("\"").strip("'")
    cleaned = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", cleaned).strip()
    return cleaned

def parse_commit_suggestions(raw_text: str, expected: int) -> list[str]:
    suggestions = []
    for line in raw_text.splitlines():
        candidate = sanitize_commit_candidate(line)
        if COMMIT_MESSAGE_PATTERN.match(candidate):
            if candidate not in suggestions:
                suggestions.append(candidate)
        if len(suggestions) >= expected:
            return suggestions

    matches = re.findall(
        rf"(?:^|\n)\s*(?:\d+\s*[\.\)\-:]\s*)?({COMMIT_TYPE_PATTERN}(?:\([^)]+\))?: [^\n]+)",
        raw_text,
        flags=re.IGNORECASE,
    )
    for match in matches:
        candidate = sanitize_commit_candidate(match)
        if COMMIT_MESSAGE_PATTERN.match(candidate) and candidate not in suggestions:
            suggestions.append(candidate)
        if len(suggestions) >= expected:
            return suggestions

    # Fallback: take first non-empty line even if model ignored format rules.
    if not suggestions:
        for line in raw_text.splitlines():
            candidate = sanitize_commit_candidate(line)
            if candidate:
                suggestions.append(candidate)
                break
    return suggestions

def choose_commit_message(suggestions: list[str], title: str = "Suggested Commit Messages") -> str:
    click.echo(click.style(f"\n💬 {title}:", fg="green"))
    for idx, suggestion in enumerate(suggestions, start=1):
        click.echo(click.style(f"   {idx}. {suggestion}", fg="bright_white"))

    selection = click.prompt(
        click.style("\nSelect message number or type a custom message", fg="yellow"),
        default="1",
        type=str,
    ).strip()
    if selection.isdigit() and 1 <= int(selection) <= len(suggestions):
        return suggestions[int(selection) - 1]
    if selection:
        return selection
    return suggestions[0]

def llm_generate_text(system_prompt: str, user_prompt: str, config: dict, temperature: float = 0.2) -> str:
    provider = config.get("provider", "gemini")
    if provider == "gemini":
        from google import genai
        from google.genai import types
        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise click.ClickException("Gemini API key is not configured. Run 'agent config' or set GEMINI_API_KEY.")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.get("model", "gemini-2.0-flash"),
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )
        return (response.text or "").strip()

    if provider in ("openai", "ollama"):
        from openai import OpenAI
        if provider == "openai":
            api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise click.ClickException("OpenAI API key is not configured. Run 'agent config' or set OPENAI_API_KEY.")
            client = OpenAI(api_key=api_key)
            model = config.get("model", "gpt-4o-mini")
        else:
            base_url = config.get("base_url", "http://localhost:11434").rstrip("/") + "/v1"
            client = OpenAI(base_url=base_url, api_key="ollama-local")
            model = config.get("model", "llama3:latest")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if provider == "ollama":
                raise click.ClickException(f"Ollama generation failed: {e}\nIs Ollama running at {config.get('base_url', 'http://localhost:11434')}?")
            raise

    raise click.ClickException(f"Unknown provider '{provider}'. Run 'agent config'.")

def generate_commit_suggestions(
    diff_text: str,
    repo_path: str,
    suggestions: int,
) -> list[str]:
    config = load_config()

    history_text = get_recent_commit_history(repo_path)
    style_profile = build_repo_style_profile(repo_path)
    changed_files = extract_changed_files(diff_text)
    user_prompt = build_commit_user_prompt(diff_text, history_text, changed_files, style_profile)
    system_prompt = build_commit_system_prompt(suggestions)
    raw_text = llm_generate_text(system_prompt, user_prompt, config=config, temperature=0.2)

    parsed = parse_commit_suggestions(raw_text, suggestions)
    if not parsed:
        raise click.ClickException("Model returned no usable commit message suggestions.")
    return parsed

def extract_tool_text(result) -> str:
    """Collect text blocks from an MCP tool response."""
    output = ""
    for block in result.content:
        if hasattr(block, "text"):
            output += block.text
    return output.strip()

def split_diff_sections(diff_text: str):
    """
    Split unified diff into per-file sections starting at 'diff --git'.
    If no markers exist, return the original text as one section.
    """
    lines = diff_text.splitlines(keepends=True)
    sections = []
    current = []

    for line in lines:
        if line.startswith("diff --git "):
            if current:
                sections.append("".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("".join(current))

    if not sections:
        return [diff_text]
    return sections

def build_llm_diff_payload(diff_text: str):
    """
    Keep prompt size bounded for better LLM latency/quality on very large diffs.
    Returns (payload_text, was_truncated).
    """
    if len(diff_text) <= MAX_DIFF_CHARS_FOR_LLM:
        return diff_text, False

    sections = split_diff_sections(diff_text)
    compact_sections = []
    total_files = len(sections)

    for idx, section in enumerate(sections):
        if idx >= MAX_FILES_FOR_LLM_DIFF:
            break
        if len(section) > MAX_CHARS_PER_FILE_SECTION:
            compact_sections.append(
                section[:MAX_CHARS_PER_FILE_SECTION]
                + f"\n... [section truncated at {MAX_CHARS_PER_FILE_SECTION} chars]\n"
            )
        else:
            compact_sections.append(section)

    remaining_files = max(0, total_files - len(compact_sections))
    header = (
        f"[Truncated diff for AI input: original={len(diff_text)} chars, "
        f"included_files={len(compact_sections)}, omitted_files={remaining_files}]\n\n"
    )
    payload = header + "".join(compact_sections)

    if len(payload) > MAX_DIFF_CHARS_FOR_LLM:
        payload = payload[:MAX_DIFF_CHARS_FOR_LLM] + "\n... [overall diff payload truncated]\n"

    return payload, True

def list_changed_files(repo_path: str) -> list[str]:
    output = run_git(repo_path, ["status", "--porcelain"])
    files = []
    for line in output.splitlines():
        if not line:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path and path not in files:
            files.append(path)
    return files

def list_staged_files(repo_path: str) -> list[str]:
    output = run_git(repo_path, ["diff", "--cached", "--name-only"])
    return [line.strip() for line in output.splitlines() if line.strip()]

def classify_change_group(path: str) -> str:
    p = path.replace("\\", "/").lower()
    base = os.path.basename(p)

    config_names = {
        ".env", ".env.example", "dockerfile", "docker-compose.yml",
        "package.json", "package-lock.json", "pyproject.toml", "requirements.txt",
        "tsconfig.json", "vite.config.ts", "webpack.config.js",
    }
    if base in config_names or p.endswith((".yml", ".yaml", ".toml", ".ini", ".cfg")):
        return "config updates"

    if any(token in p for token in ("ui/", "frontend/", "web/", "client/", "components/", "pages/", "styles/")):
        return "ui changes"
    if p.endswith((".tsx", ".jsx", ".css", ".scss", ".sass", ".html", ".vue")):
        return "ui changes"

    if any(token in p for token in ("api/", "backend/", "server/", "services/", "controllers/", "models/", "db/")):
        return "backend logic"
    if p.endswith((".py", ".go", ".java", ".kt", ".rb", ".php", ".cs", ".rs")):
        return "backend logic"

    if p.startswith("docs/") or base.endswith(".md"):
        return "docs and content"

    return "other changes"

def build_change_groups(files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in files:
        group = classify_change_group(path)
        groups.setdefault(group, []).append(path)
    return groups

def build_files_diff(repo_path: str, files: list[str]) -> str:
    if not files:
        return ""
    return run_git(repo_path, ["diff", DEFAULT_DIFF_TARGET, "--", *files])

def build_explain_user_prompt(diff_text: str, changed_files: list[str]) -> str:
    files_block = "\n".join(f"- {name}" for name in changed_files) if changed_files else "- (not detected)"
    return (
        "Explain this repository diff.\n\n"
        "Changed files:\n"
        f"{files_block}\n\n"
        "Diff:\n"
        f"{diff_text}\n"
    )

def get_current_branch(repo_path: str) -> str:
    return run_git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()

def build_push_fallback_diagnosis(remote: str, branch: str) -> str:
    return (
        "Why It Failed:\n"
        "- Push failed, but dynamic diagnosis is unavailable right now.\n\n"
        "Most Likely Fix:\n"
        "- Inspect branch tracking and reconcile local and remote history.\n\n"
        "Commands To Run:\n"
        "- git status\n"
        "- git branch -vv\n"
        f"- git fetch {remote}\n"
        f"- git pull --rebase {remote} {branch}\n"
        f"- git push {remote} {branch}\n\n"
        "Safety Notes:\n"
        "- Review any conflict resolutions before continuing."
    )

def normalize_plain_diagnosis(text: str) -> str:
    """
    Enforce plain terminal output if model returns markdown.
    """
    if not text:
        return ""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("```"):
            continue
        if line.startswith("|") and line.endswith("|"):
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^\*\*(.+)\*\*$", r"\1", line)
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        lines.append(line)

    # Collapse excessive blank lines.
    normalized = []
    blank = False
    for line in lines:
        if line == "":
            if blank:
                continue
            blank = True
            normalized.append(line)
        else:
            blank = False
            normalized.append(line)
    return "\n".join(normalized).strip()

def diagnosis_is_clear(dynamic_context: str) -> bool:
    """
    Detect if git evidence is decisive enough for concise mode.
    """
    m = re.search(r"git rev-list --left-right --count HEAD\.\.\.[^\n`]+`?:\n([0-9]+)\s+([0-9]+)", dynamic_context)
    if not m:
        return False
    ahead = int(m.group(1))
    behind = int(m.group(2))
    # Any non-equal state is already enough for direct, concise guidance.
    return (ahead == 0 and behind > 0) or (ahead > 0 and behind == 0) or (ahead > 0 and behind > 0)

def trim_diagnosis_for_clarity(text: str, concise: bool) -> str:
    if not concise or not text:
        return text

    # Keep sections but trim verbose paragraphs to first 2 non-empty lines each.
    headers = ["Why It Failed", "Most Likely Fix", "Commands To Run", "Safety Notes"]
    sections = {h: [] for h in headers}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        matched = next((h for h in headers if stripped.lower() == h.lower()), None)
        if matched:
            current = matched
            sections[current].append(matched)
            continue
        if current:
            sections[current].append(line)

    out_lines = []
    for h in headers:
        block = sections[h]
        if not block:
            continue
        out_lines.append(h)
        kept = []
        non_empty = 0
        for line in block[1:]:
            s = line.strip()
            if h == "Commands To Run":
                # Keep all command lines in concise mode.
                if s:
                    kept.append(s)
                continue
            if not s:
                continue
            kept.append(s)
            non_empty += 1
            if non_empty >= 2:
                break
        out_lines.extend(kept)
        out_lines.append("")
    return "\n".join(out_lines).strip()

def collect_push_diagnosis_context(repo_path: str, remote: str, branch: str) -> str:
    snippets = []
    snippets.append(f"Repo path: {repo_path}")
    snippets.append(f"Remote: {remote}")
    snippets.append(f"Branch: {branch}")

    def try_git_live(args: list[str], label: str):
        cmd_display = "git " + " ".join(args)
        click.echo(click.style(f"  [diagnose] Running: {cmd_display}", fg="bright_black"))
        code, out, err = run_command_live(["git", *args], cwd=repo_path)
        combined = "\n".join(part for part in [out, err] if part).strip()
        if code == 0:
            snippets.append(f"{label}:\n{combined or '(empty)'}")
        else:
            snippets.append(f"{label}:\n(command failed, exit={code})\n{combined or '(no output)'}")

    # Default: full diagnostic evidence.
    try_git_live(["fetch", remote, branch], f"`git fetch {remote} {branch}`")
    try_git_live(["status", "--short", "--branch"], "`git status --short --branch`")
    try_git_live(["branch", "-vv"], "`git branch -vv`")
    try_git_live(["remote", "-v"], "`git remote -v`")
    try_git_live(["rev-list", "--left-right", "--count", f"HEAD...{remote}/{branch}"], f"`git rev-list --left-right --count HEAD...{remote}/{branch}`")
    try_git_live(["merge-base", "HEAD", f"{remote}/{branch}"], f"`git merge-base HEAD {remote}/{branch}`")
    try_git_live(["log", "--oneline", "-10", "HEAD"], "`git log --oneline -10 HEAD`")
    try_git_live(["log", "--oneline", "-10", f"{remote}/{branch}"], f"`git log --oneline -10 {remote}/{branch}`")

    return "\n\n".join(snippets)

def collect_push_diagnosis_context_light(repo_path: str, remote: str, branch: str) -> str:
    snippets = []
    snippets.append(f"Repo path: {repo_path}")
    snippets.append(f"Remote: {remote}")
    snippets.append(f"Branch: {branch}")

    def try_git_live(args: list[str], label: str):
        cmd_display = "git " + " ".join(args)
        click.echo(click.style(f"  [diagnose] Running: {cmd_display}", fg="bright_black"))
        code, out, err = run_command_live(["git", *args], cwd=repo_path)
        combined = "\n".join(part for part in [out, err] if part).strip()
        if code == 0:
            snippets.append(f"{label}:\n{combined or '(empty)'}")
        else:
            snippets.append(f"{label}:\n(command failed, exit={code})\n{combined or '(no output)'}")

    # Lightweight evidence for auth/permission/url/network style issues.
    try_git_live(["status", "--short", "--branch"], "`git status --short --branch`")
    try_git_live(["remote", "-v"], "`git remote -v`")
    try_git_live(["config", "--get", "credential.helper"], "`git config --get credential.helper`")
    try_git_live(["config", "--show-origin", "--get-all", "credential.helper"], "`git config --show-origin --get-all credential.helper`")
    return "\n\n".join(snippets)

def route_push_evidence_mode(error_text: str, config: dict) -> str:
    prompt = (
        "Classify this failed push using only the error output below.\n\n"
        "Error output:\n"
        f"{error_text}\n"
    )
    try:
        verdict = llm_generate_text(PUSH_EVIDENCE_ROUTER_PROMPT, prompt, config=config, temperature=0.0).strip().upper()
        if verdict in ("FULL", "LIGHT"):
            return verdict
    except Exception:
        pass
    # Safe default keeps current strong behavior.
    return "FULL"

def build_push_failure_user_prompt(
    repo_path: str,
    remote: str,
    branch: str,
    error_text: str,
    dynamic_context: str,
    concise_mode: bool,
) -> str:
    return (
        "Diagnose this failed git push.\n\n"
        f"Target remote: {remote}\n"
        f"Target branch: {branch}\n\n"
        f"Diagnosis mode: {'concise' if concise_mode else 'detailed'}\n\n"
        "Git error output:\n"
        f"{error_text}\n\n"
        "Repository evidence collected live:\n"
        f"{dynamic_context}\n\n"
        "Task:\n"
        "- Diagnose dynamically from this exact failure and evidence.\n"
        "- Explicitly use ahead/behind, merge-base, and recent commit logs when present.\n"
        "- Prefer safest non-destructive fix first.\n"
        "- Provide optional destructive fix only as a secondary option.\n"
        "- Commands must be PowerShell-ready.\n"
        "- If diagnosis mode is concise, keep each section short and avoid long explanations.\n"
    )

def diagnose_push_failure(repo_path: str, remote: str, branch: str, error_text: str) -> str:
    fallback = build_push_fallback_diagnosis(remote, branch)
    config = load_config()
    if not config.get("provider"):
        return fallback

    try:
        mode = route_push_evidence_mode(error_text, config)
        with loading_spinner(f"Collecting git evidence ({mode.lower()})"):
            if mode == "LIGHT":
                dynamic_context = collect_push_diagnosis_context_light(repo_path, remote, branch)
            else:
                dynamic_context = collect_push_diagnosis_context(repo_path, remote, branch)
        concise_mode = diagnosis_is_clear(dynamic_context)
        llm_prompt = build_push_failure_user_prompt(
            repo_path,
            remote,
            branch,
            error_text,
            dynamic_context,
            concise_mode,
        )
        with loading_spinner("Analyzing push failure"):
            diagnosis = llm_generate_text(PUSH_FAILURE_SYSTEM_PROMPT, llm_prompt, config=config, temperature=0.1)
            diagnosis = normalize_plain_diagnosis(diagnosis)
            return trim_diagnosis_for_clarity(diagnosis, concise=concise_mode) or fallback
    except Exception:
        return fallback


# ─────────────────────────────────────────────
# MCP Core Logic
# ─────────────────────────────────────────────

async def run_agent_commit(repo_path: str, verbose: bool, suggestion_count: int, smart_stage: bool):
    """
    Full MCP lifecycle:
      1. Initialize MCP session with mcp-server-git
      2. Call git_diff to get uncommitted changes
      3. Generate commit message via selected AI Provider
      4. Call git_add then git_commit via MCP
    """
    if smart_stage:
        changed_files = list_changed_files(repo_path)
        if changed_files:
            groups = build_change_groups(changed_files)
            if len(groups) > 1:
                click.echo(click.style("\n🧠 Smart staging detected logical groups:", fg="cyan"))
                for idx, (name, files) in enumerate(groups.items(), start=1):
                    click.echo(click.style(f"   {idx}. {name} ({len(files)} files)", fg="bright_white"))
                    if verbose:
                        for file_path in files[:5]:
                            click.echo(click.style(f"      - {file_path}", fg="bright_black"))
                        if len(files) > 5:
                            click.echo(click.style(f"      ... +{len(files)-5} more", fg="bright_black"))

                if click.confirm(click.style(f"\nCreate {len(groups)} commits from these groups?", fg="yellow"), default=True):
                    staged_before = list_staged_files(repo_path)
                    if staged_before:
                        click.echo(click.style(
                            f"⚠️  Found {len(staged_before)} pre-staged file(s). Smart staging will unstage them first.",
                            fg="yellow",
                        ))
                        if not click.confirm(click.style("Proceed and reset index (keeps working tree changes)?", fg="yellow"), default=True):
                            click.echo(click.style("❌ Aborted by user.", fg="red"))
                            return

                    # Ensure staged index starts clean so each group commit is isolated.
                    run_git(repo_path, ["reset"])

                    for group_name, files in groups.items():
                        group_diff = build_files_diff(repo_path, files)
                        if not group_diff.strip():
                            continue

                        click.echo(click.style(f"\n📂 Group: {group_name}", fg="cyan"))
                        llm_diff_payload, was_truncated = build_llm_diff_payload(group_diff)
                        if was_truncated and verbose:
                            click.echo(click.style(
                                f"   Large group diff. Using compact payload ({len(llm_diff_payload)} chars).",
                                fg="yellow",
                            ))

                        with loading_spinner(f"Thinking for {group_name}"):
                            suggestions = generate_commit_suggestions(
                                llm_diff_payload,
                                repo_path,
                                suggestion_count,
                            )
                        commit_message = choose_commit_message(
                            suggestions,
                            title=f"Suggested messages for {group_name}",
                        )
                        click.echo(click.style(
                            f"\n✅ Selected Commit Message: {click.style(commit_message, bold=True, fg='bright_white')}",
                            fg="green"
                        ))
                        if not click.confirm(click.style("Commit this group now?", fg="yellow"), default=True):
                            click.echo(click.style("   Skipped group.", fg="yellow"))
                            run_git(repo_path, ["reset"])
                            continue

                        run_git(repo_path, ["reset"])
                        run_git(repo_path, ["add", "-A", "--", *files])
                        commit_output = run_git(repo_path, ["commit", "-m", commit_message]).strip()
                        click.echo(click.style(f"   {commit_output}", fg="green"))

                    click.echo(click.style(
                        "\n🚀 Done! Smart-staged commits created successfully.",
                        fg="bright_green", bold=True
                    ))
                    return

    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-git", "--repository", repo_path],
        env=None,
    )

    click.echo(click.style("🔌 Connecting to mcp-server-git...", fg="cyan"))

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # ── Step 1: Initialize ──────────────────────────
            await session.initialize()
            click.echo(click.style("✅ MCP session initialized.", fg="green"))

            if verbose:
                tools = await session.list_tools()
                click.echo(click.style(
                    f"   Available tools: {[t.name for t in tools.tools]}",
                    fg="bright_black"
                ))

            # ── Step 2: Get git diff ────────────────────────
            click.echo(click.style("\n📂 Fetching git diff (changes against HEAD)...", fg="cyan"))

            diff_result = await session.call_tool(
                "git_diff",
                arguments={
                    "repo_path": repo_path,
                    "target": DEFAULT_DIFF_TARGET,
                },
            )

            diff_text = extract_tool_text(diff_result)

            if getattr(diff_result, "isError", False):
                raise click.ClickException(
                    f"git_diff failed: {diff_text or 'Unknown MCP tool error.'}"
                )

            if not diff_text.strip():
                click.echo(click.style(
                    "⚠️  No changes detected. Nothing to commit.",
                    fg="yellow"
                ))
                return

            if verbose:
                click.echo(click.style("\n── Diff Preview ──", fg="bright_black"))
                preview = diff_text[:1500] + ("..." if len(diff_text) > 1500 else "")
                click.echo(click.style(preview, fg="bright_black"))

            click.echo(click.style(
                f"   Diff captured ({len(diff_text)} chars).", fg="green"
            ))

            # ── Step 3: Generate commit message via LLM ─────
            config = load_config()
            provider = config.get("provider", "gemini")
            click.echo(click.style(f"\n🤖 Generating commit message suggestions with {provider.upper()}...", fg="cyan"))

            llm_diff_payload, was_truncated = build_llm_diff_payload(diff_text)
            if was_truncated:
                click.echo(click.style(
                    f"   Large diff detected. Sending compacted payload ({len(llm_diff_payload)} chars) to AI.",
                    fg="yellow"
                ))

            with loading_spinner("Thinking about commit messages"):
                suggestions = generate_commit_suggestions(
                    llm_diff_payload,
                    repo_path,
                    suggestion_count,
                )
            if len(suggestions) < suggestion_count and verbose:
                click.echo(click.style(
                    f"   Provider returned {len(suggestions)} valid suggestion(s).",
                    fg="yellow",
                ))

            commit_message = choose_commit_message(suggestions)

            click.echo(click.style(
                f"\n✅ Selected Commit Message: {click.style(commit_message, bold=True, fg='bright_white')}",
                fg="green"
            ))

            # ── Confirmation Prompt ─────────────────────────
            if not click.confirm(
                click.style("\nProceed with git add + commit?", fg="yellow"),
                default=True,
            ):
                click.echo(click.style("❌ Aborted by user.", fg="red"))
                return

            # ── Step 4: git add . ───────────────────────────
            click.echo(click.style("\n📦 Staging all changes (git add .)...", fg="cyan"))

            add_result = await session.call_tool(
                "git_add",
                arguments={
                    "repo_path": repo_path,
                    "files": ["."],
                },
            )

            add_output = ""
            for block in add_result.content:
                if hasattr(block, "text"):
                    add_output += block.text
            click.echo(click.style(f"   {add_output.strip() or 'Files staged.'}", fg="green"))

            # ── Step 5: git commit ──────────────────────────
            click.echo(click.style("\n✍️  Committing...", fg="cyan"))

            commit_result = await session.call_tool(
                "git_commit",
                arguments={
                    "repo_path": repo_path,
                    "message": commit_message,
                },
            )

            commit_output = ""
            for block in commit_result.content:
                if hasattr(block, "text"):
                    commit_output += block.text
            click.echo(click.style(f"   {commit_output.strip()}", fg="green"))

            click.echo(click.style(
                "\n🚀 Done! Changes committed successfully.",
                fg="bright_green", bold=True
            ))


# ─────────────────────────────────────────────
# CLI Entry Points
# ─────────────────────────────────────────────

@click.group()
@click.version_option(
    version="0.1.3",
    prog_name="agent-gitv1",
    message="%(prog)s %(version)s | Author: Vijayyyy-01",
)
def cli():
    """
    agent-gitv1: MCP + Multi-LLM Git Assistant

    Automate your Git workflow with AI-generated commit messages.
    Supports OpenAI, Gemini, and local Ollama models.

    Available commands:
      config  Configure AI provider and model
      commit  Smart-stage changes and create AI-assisted commit(s)
      explain Explain what changed, why, risks, and impacted files
      push    Push committed changes to remote

    Run `agent <command> --help` for command-specific options.
    """
    pass

@cli.command("config")
def config_command():
    """Configure your preferred AI Provider (OpenAI, Gemini, or Ollama)."""
    click.echo(click.style("🛠  Agent Git Setup\n", bold=True, fg="cyan"))
    
    provider = click.prompt(
        "Select AI Provider", 
        type=click.Choice(["gemini", "openai", "ollama"]), 
        default="ollama"
    )
    
    config = {"provider": provider}
    
    if provider == "gemini":
        config["api_key"] = click.prompt("Gemini API Key (Leave empty to use GEMINI_API_KEY env var)", default="", hide_input=True)
        config["model"] = click.prompt("Model", default="gemini-2.0-flash")
        
    elif provider == "openai":
        config["api_key"] = click.prompt("OpenAI API Key (Leave empty to use OPENAI_API_KEY env var)", default="", hide_input=True)
        config["model"] = click.prompt("Model", default="gpt-4o-mini")
        
    elif provider == "ollama":
        base_url = "http://localhost:11434"

        if click.confirm("Auto-detect Ollama servers on your local network?", default=True):
            with loading_spinner("Scanning local network for Ollama servers"):
                discovered = discover_ollama_servers()
            if discovered:
                click.echo(click.style("\nDetected Ollama Servers:", fg="green"))
                for i, (url, models) in enumerate(discovered, start=1):
                    preview = ", ".join(models[:3])
                    more = f" (+{len(models)-3} more)" if len(models) > 3 else ""
                    click.echo(f"  {i}. {url}  ->  {preview}{more}")

                choice = click.prompt(
                    "\nSelect server by number or type custom base URL",
                    type=str,
                    default="1",
                ).strip()
                if choice.isdigit() and 1 <= int(choice) <= len(discovered):
                    base_url = discovered[int(choice) - 1][0]
                elif choice:
                    base_url = choice
            else:
                click.echo(click.style("No Ollama servers auto-detected.", fg="yellow"))
                base_url = click.prompt("Ollama Base URL", default="http://localhost:11434")
        else:
            base_url = click.prompt("Ollama Base URL", default="http://localhost:11434")

        config["base_url"] = base_url
        
        click.echo("Fetching available models from Ollama...")
        models = get_ollama_models(base_url)
        
        if models:
            click.echo(click.style("\nAvailable Ollama Models:", fg="green"))
            for i, m in enumerate(models):
                click.echo(f"  {i+1}. {m}")
            
            choice = click.prompt("\nSelect a model by number or type model name manually", type=str)
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                config["model"] = models[int(choice)-1]
            else:
                config["model"] = choice
        else:
            click.echo(click.style("Could not auto-fetch models.", fg="yellow"))
            config["model"] = click.prompt("Model Name (e.g. llama3:8b, mistral, deepseek-r1)", default="llama3:latest")
            
    save_config(config)
    click.echo(click.style(f"\n✅ Configuration saved successfully to {CONFIG_FILE}!", fg="bright_green", bold=True))
    click.echo(click.style(f"Current setup -> Provider: {provider}, Model: {config['model']}", fg="green"))


@cli.command("commit")
@click.option(
    "--repo",
    "-r",
    default=".",
    show_default=True,
    help="Path to the git repository.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show extra debug information.",
)
@click.option(
    "--suggestions",
    "-s",
    type=click.IntRange(1, 5),
    default=3,
    show_default=True,
    help="How many AI commit message suggestions to generate.",
)
@click.option(
    "--smart-stage/--no-smart-stage",
    default=True,
    show_default=True,
    help="Auto-group changes (ui/backend/config/etc.) and optionally create multiple commits.",
)
def commit_command(repo: str, verbose: bool, suggestions: int, smart_stage: bool):
    """Smart-stage changes, generate AI messages, and create commit(s)."""
    repo_path = os.path.abspath(repo)

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        raise click.ClickException(
            f"'{repo_path}' is not a valid git repository (no .git folder found)."
        )

    click.echo(click.style(
        f"\n🗂  Repository: {repo_path}", fg="bright_cyan", bold=True
    ))

    try:
        asyncio.run(run_agent_commit(repo_path, verbose, suggestions, smart_stage))
    except KeyboardInterrupt:
        click.echo(click.style("\n\n⚡ Interrupted by user.", fg="yellow"))
        sys.exit(0)
    except Exception as e:
        raise click.ClickException(str(e))

@cli.command("explain")
@click.option("--repo", "-r", default=".", show_default=True, help="Path to the git repository.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show extra debug information.")
def explain_command(repo: str, verbose: bool):
    """Explain current changes: what changed, why, risks, and impacted files."""
    repo_path = os.path.abspath(repo)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        raise click.ClickException(f"'{repo_path}' is not a valid git repository (no .git folder found).")

    click.echo(click.style(f"\n🗂  Repository: {repo_path}", fg="bright_cyan", bold=True))
    click.echo(click.style("🔎 Analyzing repository changes...", fg="cyan"))

    diff_text = run_git(repo_path, ["diff", DEFAULT_DIFF_TARGET])
    if not diff_text.strip():
        click.echo(click.style("⚠️  No changes detected to explain.", fg="yellow"))
        return

    changed_files = list_changed_files(repo_path)
    payload, truncated = build_llm_diff_payload(diff_text)
    if truncated and verbose:
        click.echo(click.style(f"   Large diff detected. Using compacted payload ({len(payload)} chars).", fg="yellow"))

    config = load_config()
    user_prompt = build_explain_user_prompt(payload, changed_files)
    with loading_spinner("Analyzing changes"):
        explanation = llm_generate_text(EXPLAIN_SYSTEM_PROMPT, user_prompt, config=config, temperature=0.1)

    click.echo(click.style("\n📘 Change Explanation", fg="green", bold=True))
    click.echo(explanation.strip() or "No explanation generated.")


@cli.command("push")
@click.option("--remote", default="origin", show_default=True, help="Remote name.")
@click.option("--branch", default=None, help="Branch to push (defaults to current).")
@click.option("--repo", "-r", default=".", show_default=True, help="Repo path.")
def push_command(remote: str, branch: str, repo: str):
    """Push committed changes to the remote repository."""
    import subprocess
    repo_path = os.path.abspath(repo)

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        raise click.ClickException(f"'{repo_path}' is not a valid git repository.")

    # Check if a remote exists
    try:
        remotes = subprocess.check_output(["git", "remote", "-v"], cwd=repo_path, text=True)
        if not remotes.strip():
            raise click.ClickException(
                "No git remotes configured! You need to add a remote first.\n"
                "Example: git remote add origin https://github.com/user/repo.git"
            )
    except subprocess.CalledProcessError:
        raise click.ClickException("Failed to check git remotes.")

    try:
        target_branch = branch or get_current_branch(repo_path)
    except Exception:
        target_branch = branch or "(current)"

    click.echo(click.style(
        f"🚀 Pushing to {remote} {target_branch}...", fg="cyan"
    ))

    # Always push an explicit target branch to avoid no-upstream failures on new branches.
    cmd = ["git", "push", remote, target_branch]

    try:
        click.echo(click.style(f"Running: {' '.join(cmd)}", fg="bright_black"))
        returncode, stdout, stderr = run_command_live(cmd, cwd=repo_path)

        if returncode == 0:
            click.echo(click.style("\n✅ Push complete!", fg="bright_green", bold=True))
        else:
            error_text = f"{stdout}\n{stderr}".strip()

            if "has no upstream branch" in error_text.lower():
                click.echo(click.style("\nℹ️  No upstream branch detected.", fg="yellow"))
                upstream_cmd = ["git", "push", "--set-upstream", remote, target_branch]
                if click.confirm(click.style(f"Run: {' '.join(upstream_cmd)} ?", fg="yellow"), default=True):
                    click.echo(click.style(f"Running: {' '.join(upstream_cmd)}", fg="bright_black"))
                    up_code, up_out, up_err = run_command_live(upstream_cmd, cwd=repo_path)
                    if up_code == 0:
                        click.echo(click.style("\n✅ Upstream configured and push complete!", fg="bright_green", bold=True))
                        return
                    error_text = f"{up_out}\n{up_err}".strip()

            diagnosis = diagnose_push_failure(repo_path, remote, target_branch, error_text)
            click.echo(click.style("\n🩺 Push Failure Diagnosis", fg="yellow", bold=True))
            click.echo(diagnosis.strip())
            sys.exit(returncode)
    except KeyboardInterrupt:
        click.echo(click.style("\n⚡ Interrupted.", fg="yellow"))
    except Exception as e:
        raise click.ClickException(f"Failed to execute git push: {e}")


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
