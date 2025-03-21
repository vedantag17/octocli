import typer
from github import Github
from rich.console import Console
from rich.tree import Tree
import os
from typing import Optional, Dict, List
import ast
from radon.complexity import cc_visit
import base64
import openai
import anthropic
from pathlib import Path
import json
import configparser
import click

app = typer.Typer()
console = Console()

CONFIG_FILE = "octo_config.ini"

class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        # Create config directory in user's home directory
        self.config_dir = os.path.expanduser("~/.octocli")
        self.config_file = os.path.join(self.config_dir, "config.ini")
        self.load_config()

    def load_config(self):
        try:
            # Create config directory if it doesn't exist
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)
            
            # Load existing config or create new one
            if os.path.exists(self.config_file):
                self.config.read(self.config_file)
            
            # Ensure DEFAULT section exists
            if 'DEFAULT' not in self.config:
                self.config['DEFAULT'] = {
                    'github_token': '',
                    'llm_type': '',
                    'llm_key': '',
                    'current_repo': '',
                    'last_analysis_file': ''
                }
                self.save_config()
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config: {str(e)}[/yellow]")
            # Ensure we have a DEFAULT section even if file operations fail
            self.config['DEFAULT'] = {
                'github_token': '',
                'llm_type': '',
                'llm_key': '',
                'current_repo': '',
                'last_analysis_file': ''
            }

    def save_config(self):
        try:
            # Create config directory if it doesn't exist
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)
            
            # Save config
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save config: {str(e)}[/yellow]")

    def get(self, key, default=None):
        try:
            return self.config['DEFAULT'].get(key, default)
        except:
            return default

    def set(self, key, value):
        if 'DEFAULT' not in self.config:
            self.config['DEFAULT'] = {}
        self.config['DEFAULT'][key] = value
        self.save_config()

class GitHubAnalyzer:
    def __init__(self, token: Optional[str] = None):
        self.github = Github(token)
        self.current_repo = None
        self.markdown_content = {}
        self.code_structure = {}
        self.config = Config()

    def init_repository(self, repo_url: str):
        parts = repo_url.rstrip('/').split('/')
        repo_name = f"{parts[-2]}/{parts[-1]}"
        self.current_repo = self.github.get_repo(repo_name)
        return self.current_repo

    def get_issue(self, issue_number: int):
        if not self.current_repo:
            repo_url = self.config.get('current_repo')
            if not repo_url:
                raise Exception("No repository configured. Use 'octo config --repo URL' first.")
            self.init_repository(repo_url)
        return self.current_repo.get_issue(issue_number)

    def analyze_markdown_content(self, content: str, file_path: str) -> Dict:
        """Analyze markdown file content"""
        try:
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            sections = []
            current_section = {"title": "Main", "content": [], "subsections": []}
            current_level = 0

            for line in content.split('\n'):
                if line.strip().startswith('#'):
                    # Count heading level
                    level = len(line.split()[0])
                    title = line.lstrip('#').strip()
                    
                    if level == 1:
                        if current_section["content"]:
                            sections.append(current_section)
                        current_section = {"title": title, "content": [], "subsections": []}
                    else:
                        current_section["subsections"].append({
                            "level": level,
                            "title": title,
                            "content": []
                        })
                else:
                    if line.strip():
                        if current_section["subsections"]:
                            current_section["subsections"][-1]["content"].append(line)
                        else:
                            current_section["content"].append(line)

            if current_section["content"] or current_section["subsections"]:
                sections.append(current_section)

            return {
                "path": file_path,
                "type": "markdown",
                "sections": sections
            }
        except Exception as e:
            return {"path": file_path, "type": "markdown", "error": str(e)}

    def analyze_file_content(self, content, file_path: str) -> Dict:
        """Analyze content of a single file"""
        try:
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # Handle markdown files
            if file_path.lower().endswith('.md'):
                return self.analyze_markdown_content(content, file_path)

            # Handle code files
            file_info = {
                "path": file_path,
                "type": "code",
                "functions": [],
                "classes": [],
                "imports": [],
                "description": ""
            }

            # Analyze Python files
            if file_path.endswith('.py'):
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_info = {
                                "name": node.name,
                                "line_number": node.lineno,
                                "docstring": ast.get_docstring(node) or "No documentation"
                            }
                            file_info["functions"].append(func_info)
                        elif isinstance(node, ast.ClassDef):
                            class_info = {
                                "name": node.name,
                                "line_number": node.lineno,
                                "docstring": ast.get_docstring(node) or "No documentation"
                            }
                            file_info["classes"].append(class_info)
                        elif isinstance(node, (ast.Import, ast.ImportFrom)):
                            if isinstance(node, ast.Import):
                                for name in node.names:
                                    file_info["imports"].append(name.name)
                            else:
                                module = node.module or ""
                                for name in node.names:
                                    file_info["imports"].append(f"{module}.{name.name}")
                except SyntaxError:
                    file_info["error"] = "Could not parse Python file"

            return file_info
        except Exception as e:
            return {"path": file_path, "type": "unknown", "error": str(e)}

    def generate_file_summary(self, file_info: Dict) -> str:
        """Generate a markdown summary for a file"""
        summary = [f"## {file_info['path']}\n"]
        
        if file_info.get('type') == 'markdown':
            summary.append("### Content Sections\n")
            for section in file_info.get('sections', []):
                summary.append(f"#### {section['title']}\n")
                if section['content']:
                    summary.append("".join(section['content']) + "\n")
                for subsection in section.get('subsections', []):
                    summary.append(f"{'#' * subsection['level']} {subsection['title']}\n")
                    if subsection['content']:
                        summary.append("".join(subsection['content']) + "\n")
        else:
            if file_info.get("classes"):
                summary.append("\n### Classes\n")
                for cls in file_info["classes"]:
                    summary.append(f"- **{cls['name']}** (line {cls['line_number']})")
                    if cls['docstring'] != "No documentation":
                        summary.append(f"  - {cls['docstring']}")

            if file_info.get("functions"):
                summary.append("\n### Functions\n")
                for func in file_info["functions"]:
                    summary.append(f"- **{func['name']}** (line {func['line_number']})")
                    if func['docstring'] != "No documentation":
                        summary.append(f"  - {func['docstring']}")

            if file_info.get("imports"):
                summary.append("\n### Imports\n")
                for imp in file_info["imports"]:
                    summary.append(f"- {imp}")

        return "\n".join(summary)

    def generate_tree_structure(self, structure: Dict) -> str:
        """Generate a tree representation of the repository structure"""
        def _generate_tree(struct, prefix=""):
            result = []
            items = list(struct.items())
            for i, (name, content) in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                if isinstance(content, dict):
                    result.append(f"{prefix}{current_prefix}📁 {name}")
                    result.extend(_generate_tree(content, prefix + next_prefix))
                else:
                    result.append(f"{prefix}{current_prefix}📄 {name}")
            return result
        
        return "\n".join(_generate_tree(structure))

    def query_llm(self, prompt: str, llm_type: str, api_key: Optional[str] = None) -> str:
        """Query the chosen LLM"""
        try:
            if llm_type == "openai":
                openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful code analysis assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            elif llm_type == "anthropic":
                client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
                response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                return response.content[0].text
            else:
                return "Unsupported LLM type"
        except Exception as e:
            return f"Error querying LLM: {str(e)}"

class CodebaseFetcher:
    def __init__(self, token: Optional[str] = None):
        self.github = Github(token)
        self.current_repo = None

    def init_repository(self, repo_url: str):
        parts = repo_url.rstrip('/').split('/')
        repo_name = f"{parts[-2]}/{parts[-1]}"
        self.current_repo = self.github.get_repo(repo_name)
        return self.current_repo

    def fetch_codebase(self, repo_url: str) -> List[Dict]:
        """Fetch the entire codebase excluding JSON and Markdown files."""
        self.init_repository(repo_url)
        contents = self.current_repo.get_contents("")
        code_files = []

        with console.status("[bold green]Fetching codebase...") as status:
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    try:
                        dir_contents = self.current_repo.get_contents(file_content.path)
                        contents.extend(dir_contents)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not read directory {file_content.path}: {str(e)}[/yellow]")
                        continue
                else:
                    # Exclude JSON and Markdown files
                    if not (file_content.path.endswith('.json') or file_content.path.endswith('.md')):
                        try:
                            code_files.append({
                                "path": file_content.path,
                                "content": file_content.decoded_content.decode('utf-8') if file_content.content else None
                            })
                            console.print(f"Fetched: {file_content.path}")
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not fetch {file_content.path}: {str(e)}[/yellow]")
                            continue

        return code_files

@app.command()
def config(
    github_token: str = typer.Option(None, "--github-token", help="GitHub API token"),
    llm_type: str = typer.Option(None, "--llm-type", help="LLM type (openai/anthropic)"),
    llm_key: str = typer.Option(None, "--llm-key", help="LLM API key"),
    repo: str = typer.Option(None, "--repo", help="Default GitHub repository URL")
):
    """Configure OctoCLI settings"""
    config = Config()
    
    if github_token:
        config.set('github_token', github_token)
        console.print("[green]GitHub token updated successfully![/green]")
    
    if llm_type:
        if llm_type not in ['openai', 'anthropic']:
            console.print("[red]Invalid LLM type. Use 'openai' or 'anthropic'[/red]")
            raise typer.Exit(1)
        config.set('llm_type', llm_type)
        console.print(f"[green]LLM type set to {llm_type}![/green]")
    
    if llm_key:
        config.set('llm_key', llm_key)
        console.print("[green]LLM API key updated successfully![/green]")
    
    if repo:
        config.set('current_repo', repo)
        console.print(f"[green]Default repository set to {repo}![/green]")

    # Show current configuration
    console.print("\n[bold]Current Configuration:[/bold]")
    console.print(f"GitHub Token: {'*' * 8 if config.get('github_token') else 'Not set'}")
    console.print(f"LLM Type: {config.get('llm_type') or 'Not set'}")
    console.print(f"LLM Key: {'*' * 8 if config.get('llm_key') else 'Not set'}")
    console.print(f"Default Repository: {config.get('current_repo') or 'Not set'}")

@app.command()
def analyze_codebase(
    repo_url: str = typer.Option(None, help="GitHub repository URL (optional if default repo is set)"),
):
    """Analyze codebase and generate comprehensive documentation"""
    config = Config()
    
    if not repo_url:
        repo_url = config.get('current_repo')
        if not repo_url:
            console.print("[red]No repository specified. Use --repo-url URL[/red]")
            raise typer.Exit(1)

    try:
        console.print(f"[bold blue]Analyzing repository: {repo_url}[/bold blue]")
        
        analyzer = GitHubAnalyzer(config.get('github_token'))
        repo = analyzer.init_repository(repo_url)
        
        # Get all files
        contents = repo.get_contents("")
        all_files_info = []
        structure = {}

        with console.status("[bold green]Analyzing files...") as status:
            while contents:
                try:
                    file_content = contents.pop(0)
                    
                    # Skip if file_content is None
                    if not file_content:
                        continue

                    # Safely get file path
                    file_path = getattr(file_content, 'path', None)
                    if not file_path:
                        continue

                    # Build structure tree
                    current_dict = structure
                    path_parts = file_path.split('/')
                    
                    # Handle the directory structure
                    for part in path_parts[:-1]:
                        if part not in current_dict:
                            current_dict[part] = {}
                        current_dict = current_dict[part]
                    
                    # Add the file/directory to the structure
                    if file_content.type == "dir":
                        if path_parts[-1] not in current_dict:
                            current_dict[path_parts[-1]] = {}
                    else:
                        current_dict[path_parts[-1]] = "file"

                    # Handle directories
                    if file_content.type == "dir":
                        try:
                            dir_contents = repo.get_contents(file_path)
                            if isinstance(dir_contents, list):
                                contents.extend(dir_contents)
                            else:
                                contents.append(dir_contents)
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not read directory {file_path}: {str(e)}[/yellow]")
                            continue
                    else:
                        try:
                            # Only try to analyze text-based files
                            if file_path.lower().endswith(('.py', '.md', '.txt', '.json', '.yml', '.yaml', '.js', '.jsx', '.ts', '.tsx', '.html', '.css')):
                                try:
                                    decoded_content = file_content.decoded_content
                                    file_info = analyzer.analyze_file_content(decoded_content, file_path)
                                    all_files_info.append(file_info)
                                    console.print(f"Analyzed: {file_path}")
                                except Exception as e:
                                    console.print(f"[yellow]Warning: Could not analyze {file_path}: {str(e)}[/yellow]")
                            else:
                                console.print(f"[yellow]Skipping binary or unsupported file: {file_path}[/yellow]")
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not process {file_path}: {str(e)}[/yellow]")
                            continue

                except Exception as e:
                    console.print(f"[yellow]Warning: Unexpected error: {str(e)}[/yellow]")
                    continue

        # Generate analysis content
        analysis_content = ["# Repository Analysis\n"]
        analysis_content.append(f"## Repository: {repo.full_name}\n")
        analysis_content.append(f"Description: {repo.description or 'No description provided'}\n")
        
        # Add repository structure
        analysis_content.append("\n## Repository Structure\n```\n")
        analysis_content.append(analyzer.generate_tree_structure(structure))
        analysis_content.append("```\n")

        # Add documentation analysis
        analysis_content.append("\n## Documentation Analysis\n")
        markdown_files = [f for f in all_files_info if f.get('type') == 'markdown']
        if markdown_files:
            for file_info in markdown_files:
                analysis_content.append(analyzer.generate_file_summary(file_info))
        else:
            analysis_content.append("No markdown documentation files found.\n")

        # Add code analysis
        analysis_content.append("\n## Code Analysis\n")
        code_files = [f for f in all_files_info if f.get('type') == 'code']
        if code_files:
            for file_info in code_files:
                analysis_content.append(analyzer.generate_file_summary(file_info))
        else:
            analysis_content.append("No code files found.\n")

        # Ensure the analysis file is saved in the current directory
        analysis_file = "codebase_analysis.md"
        try:
            # Ensure the content is properly joined
            final_content = "\n".join(str(line) for line in analysis_content if line is not None)
            
            # Check if the file exists, if not create it
            if not os.path.exists(analysis_file):
                with open(analysis_file, "w", encoding='utf-8') as f:
                    f.write(final_content)
            else:
                # If it exists, update the content
                with open(analysis_file, "w", encoding='utf-8') as f:
                    f.write(final_content)
            
            console.print(f"[bold green]Analysis complete! Results updated in {analysis_file}[/bold green]")
            
        except Exception as e:
            console.print(f"[red]Error saving analysis: {str(e)}[/red]")
            raise typer.Exit(1)
        
        return all_files_info

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)

@app.command()
def query(
    question: str = typer.Argument(..., help="Question about the codebase"),
    repo_url: str = typer.Option(None, help="GitHub repository URL (optional if default repo is set)"),
):
    """Ask questions about the analyzed codebase"""
    config = Config()
    
    if not repo_url:
        repo_url = config.get('current_repo')
        if not repo_url:
            console.print("[red]No repository specified. Use --repo or set default with 'octo config --repo URL'[/red]")
            raise typer.Exit(1)

    try:
        # Use the fixed filename
        analysis_file = "codebase_analysis.md"

        # Check if analysis exists
        if not os.path.exists(analysis_file):
            console.print("[yellow]No analysis found. Running analysis first...[/yellow]")
            analyze_codebase(repo_url)
        
        # Read the analysis
        try:
            with open(analysis_file, "r", encoding='utf-8') as f:
                analysis_content = f.read()
        except Exception as e:
            console.print(f"[red]Error reading analysis file: {str(e)}[/red]")
            console.print("[yellow]Running fresh analysis...[/yellow]")
            analyze_codebase(repo_url)
            with open(analysis_file, "r", encoding='utf-8') as f:
                analysis_content = f.read()

        # Initialize analyzer for LLM querying
        analyzer = GitHubAnalyzer(config.get('github_token'))
        
        # Prepare context for the question
        context = f"""
        Based on the following codebase analysis, please answer this question:
        
        Question: {question}

        Codebase Analysis:
        {analysis_content}
        
        Please provide a clear and concise answer based on the information available in the analysis.
        """

        # Query the LLM
        llm_type = config.get('llm_type')
        llm_key = config.get('llm_key')
        
        if not llm_type or not llm_key:
            console.print("[red]LLM configuration missing. Use 'octo config --llm-type TYPE --llm-key KEY'[/red]")
            raise typer.Exit(1)

        response = analyzer.query_llm(context, llm_type, llm_key)
        console.print(f"\n[bold green]Answer:[/bold green]\n{response}")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)

@app.command()
def issue(
    issue_number: int = typer.Argument(..., help="Issue number to analyze"),
    repo_url: str = typer.Option(None, help="GitHub repository URL (optional if default repo is set)"),
):
    """Analyze and suggest solutions for a GitHub issue"""
    config = Config()
    
    if not repo_url:
        repo_url = config.get('current_repo')
        if not repo_url:
            console.print("[red]No repository specified. Use --repo or set default with 'octo config --repo URL'[/red]")
            raise typer.Exit(1)

    try:
        analyzer = GitHubAnalyzer(config.get('github_token'))
        if repo_url:
            analyzer.init_repository(repo_url)
        
        issue = analyzer.get_issue(issue_number)
        
        # Prepare issue context
        issue_context = f"""
        Issue #{issue.number}: {issue.title}
        
        Description:
        {issue.body}
        
        Labels: {', '.join([label.name for label in issue.labels])}
        State: {issue.state}
        Created: {issue.created_at}
        """

        # If there are comments, add them to the context
        comments = list(issue.get_comments())
        if comments:
            issue_context += "\nComments:\n"
            for comment in comments:
                issue_context += f"\n{comment.user.login} commented on {comment.created_at}:\n{comment.body}\n"

        # Prepare prompt for LLM
        prompt = f"""
        Please analyze this GitHub issue and provide:
        1. A summary of the problem
        2. Potential root causes
        3. Suggested solutions
        4. Next steps for resolution

        Issue Details:
        {issue_context}
        """

        # Query LLM for analysis
        response = analyzer.query_llm(prompt, config.get('llm_type'), config.get('llm_key'))
        
        console.print(f"\n[bold blue]Analysis for Issue #{issue.number}[/bold blue]")
        console.print(f"[bold]{issue.title}[/bold]\n")
        console.print(response)

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)

@app.command()
def fetch_codebase(
    repo_url: str = typer.Argument(..., help="GitHub repository URL"),
):
    """Fetch the entire codebase excluding JSON and Markdown files."""
    config = Config()
    github_token = config.get('github_token')
    
    if not github_token:
        console.print("[red]GitHub token not found. Use 'octo config --github-token YOUR_TOKEN'[/red]")
        raise typer.Exit(1)

    fetcher = CodebaseFetcher(github_token)
    
    try:
        code_files = fetcher.fetch_codebase(repo_url)
        
        # Save the fetched code files to a local directory
        output_dir = "fetched_codebase"
        os.makedirs(output_dir, exist_ok=True)

        for file in code_files:
            file_path = os.path.join(output_dir, file['path'])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directories if needed
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(file['content'])
        
        console.print(f"[bold green]Codebase fetched successfully! Files saved in '{output_dir}'[/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]Error fetching codebase: {str(e)}[/bold red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 