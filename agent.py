"""
agent-gitv1: A CLI tool that uses MCP + Local/Cloud LLMs to automate git workflows.
"""

import asyncio
import sys
import os
import json
import re
import subprocess
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

def build_commit_user_prompt(
    diff_text: str,
    history_text: str,
    changed_files: list[str],
) -> str:
    files_block = "\n".join(f"- {name}" for name in changed_files) if changed_files else "- (not detected)"
    history_block = history_text if history_text else "(No recent commits available)"
    return (
        "Write commit message suggestion(s) for the following change.\n\n"
        "Recent commit messages from this repository (for style/context):\n"
        f"{history_block}\n\n"
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

def generate_commit_suggestions(
    diff_text: str,
    repo_path: str,
    suggestions: int,
) -> list[str]:
    config = load_config()
    provider = config.get("provider", "gemini") # Default fallback

    history_text = get_recent_commit_history(repo_path)
    changed_files = extract_changed_files(diff_text)
    user_prompt = build_commit_user_prompt(diff_text, history_text, changed_files)
    system_prompt = build_commit_system_prompt(suggestions)

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
                temperature=0.2,
            ),
        )
        raw_text = (response.text or "").strip()
    elif provider in ("openai", "ollama"):
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
                temperature=0.2,
            )
        except Exception as e:
            if provider == "ollama":
                raise click.ClickException(f"Ollama generation failed: {e}\nIs Ollama running at {config.get('base_url', 'http://localhost:11434')}?")
            raise
        raw_text = (response.choices[0].message.content or "").strip()
    else:
        raise click.ClickException(f"Unknown provider '{provider}'. Run 'agent config'.")

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


# ─────────────────────────────────────────────
# MCP Core Logic
# ─────────────────────────────────────────────

async def run_agent_commit(repo_path: str, verbose: bool, suggestion_count: int):
    """
    Full MCP lifecycle:
      1. Initialize MCP session with mcp-server-git
      2. Call git_diff to get uncommitted changes
      3. Generate commit message via selected AI Provider
      4. Call git_add then git_commit via MCP
    """
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

            click.echo(click.style("\n💬 Suggested Commit Messages:", fg="green"))
            for idx, suggestion in enumerate(suggestions, start=1):
                click.echo(click.style(f"   {idx}. {suggestion}", fg="bright_white"))

            selection = click.prompt(
                click.style("\nSelect message number or type a custom message", fg="yellow"),
                default="1",
                type=str,
            ).strip()
            if selection.isdigit() and 1 <= int(selection) <= len(suggestions):
                commit_message = suggestions[int(selection) - 1]
            elif selection:
                commit_message = selection
            else:
                commit_message = suggestions[0]

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
@click.version_option(version="0.1.1", prog_name="agent-gitv1")
def cli():
    """
    agent-gitv1: MCP + Multi-LLM Git Assistant

    Automate your Git workflow with AI-generated commit messages.
    Supports OpenAI, Gemini, and local Ollama models.

    Available commands:
      config  Configure AI provider and model
      commit  Stage changes and create an AI-assisted commit
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
def commit_command(repo: str, verbose: bool, suggestions: int):
    """Stage, generate a commit message, and commit all changes."""
    repo_path = os.path.abspath(repo)

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        raise click.ClickException(
            f"'{repo_path}' is not a valid git repository (no .git folder found)."
        )

    click.echo(click.style(
        f"\n🗂  Repository: {repo_path}", fg="bright_cyan", bold=True
    ))

    try:
        asyncio.run(run_agent_commit(repo_path, verbose, suggestions))
    except KeyboardInterrupt:
        click.echo(click.style("\n\n⚡ Interrupted by user.", fg="yellow"))
        sys.exit(0)
    except Exception as e:
        raise click.ClickException(str(e))


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

    click.echo(click.style(
        f"🚀 Pushing to {remote} {branch or '(current branch)'}...", fg="cyan"
    ))

    cmd = ["git", "push", remote]
    if branch:
        cmd.append(branch)

    try:
        # We run it directly so SSH / password prompts work nicely in the terminal
        result = subprocess.run(cmd, cwd=repo_path)
        if result.returncode == 0:
            click.echo(click.style("\n✅ Push complete!", fg="bright_green", bold=True))
        else:
            sys.exit(result.returncode)
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
