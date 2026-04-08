"""
agent-git: A CLI tool that uses MCP + Local/Cloud LLMs to automate git workflows.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
import click
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG_FILE = Path.home() / ".agent_git_config.json"

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
    user_prompt = f"Write a commit message for the following git diff:\n\n{diff}"
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
            {"role": "user", "content": f"Write a commit message for the following git diff:\n\n{diff}"}
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
                {"role": "user", "content": f"Write a commit message for the following git diff:\n\n{diff}"}
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


# ─────────────────────────────────────────────
# MCP Core Logic
# ─────────────────────────────────────────────

async def run_agent_commit(repo_path: str, verbose: bool):
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
            click.echo(click.style("\n📂 Fetching git diff (unstaged changes)...", fg="cyan"))

            diff_result = await session.call_tool(
                "git_diff",
                arguments={"repo_path": repo_path},
            )

            diff_text = ""
            for block in diff_result.content:
                if hasattr(block, "text"):
                    diff_text += block.text

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
            click.echo(click.style(f"\n🤖 Generating commit message with {provider.upper()}...", fg="cyan"))
            
            commit_message = generate_commit_message(diff_text)
            
            click.echo(click.style(
                f"\n💬 Commit Message: {click.style(commit_message, bold=True, fg='bright_white')}",
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
@click.version_option(version="0.1.0", prog_name="agent-git")
def cli():
    """
    \b
    ╔═══════════════════════════════════════════╗
    ║  agent-git  🤖  MCP + Multi-LLM Commits  ║
    ╚═══════════════════════════════════════════╝

    Automate your Git workflow with AI-generated commit messages.
    Supports OpenAI, Gemini, and Local Ollama models!
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
def commit_command(repo: str, verbose: bool):
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
        asyncio.run(run_agent_commit(repo_path, verbose))
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
    async def _push():
        repo_path = os.path.abspath(repo)
        server_params = StdioServerParameters(
            command="uvx",
            args=["mcp-server-git", "--repository", repo_path],
            env=None,
        )

        click.echo(click.style("🔌 Connecting to mcp-server-git...", fg="cyan"))
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                push_args = {"repo_path": repo_path, "remote": remote}
                if branch:
                    push_args["branch"] = branch

                click.echo(click.style(
                    f"🚀 Pushing to {remote}/{branch or 'current branch'}...", fg="cyan"
                ))
                result = await session.call_tool("git_push", arguments=push_args)

                output = "".join(
                    block.text for block in result.content if hasattr(block, "text")
                )
                click.echo(click.style(output.strip(), fg="green"))
                click.echo(click.style("\n✅ Push complete!", fg="bright_green", bold=True))

    try:
        asyncio.run(_push())
    except KeyboardInterrupt:
        click.echo(click.style("\n⚡ Interrupted.", fg="yellow"))
    except Exception as e:
        raise click.ClickException(str(e))


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
