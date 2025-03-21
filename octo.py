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
import google.generativeai as genai
from cohere import Client as CohereClient
import mistralai.client
from mistralai.client import MistralClient
from pathlib import Path
import json
import configparser
import click
import requests

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
            elif llm_type == "gemini":
                genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return response.text
            elif llm_type == "cohere":
                cohere_api_key = api_key or os.getenv("COHERE_API_KEY")
                client = CohereClient(cohere_api_key)
                response = client.chat(
                    message=prompt,
                    model="command"
                )
                return response.text
            elif llm_type == "mistral":
                mistral_api_key = api_key or os.getenv("MISTRAL_API_KEY")
                client = MistralClient(api_key=mistral_api_key)
                response = client.chat(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            elif llm_type == "llama":
                llama_api_key = api_key or os.getenv("LLAMA_API_KEY")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {llama_api_key}"
                }
                payload = {
                    "model": "meta-llama/Llama-3-70b-chat-hf",
                    "messages": [
                        {"role": "system", "content": "You are a helpful code analysis assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                response = requests.post(
                    "https://api.together.xyz/inference",
                    headers=headers,
                    json=payload
                )
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Unsupported LLM type: {llm_type}. Please use 'octo setup_model' to configure a supported model."
        except ImportError as e:
            return f"Error: The required package for {llm_type} is not installed. Please install it with 'pip install <package>'. Error details: {str(e)}"
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
    
    def organize_codebase(self, fetched_dir: str = "fetched_codebase") -> Dict[str, Dict]:
        """
        Organize the fetched codebase by file type and prepare it for relevance-based searching.
        
        Args:
            fetched_dir: Directory where fetched codebase is stored
            
        Returns:
            Dict with code files organized by file type and metadata
        """
        if not os.path.exists(fetched_dir):
            console.print("[red]Fetched codebase directory does not exist.[/red]")
            return {}
            
        organized_codebase = {
            "code_files": {},
            "docs_files": {},
            "config_files": {},
            "stats": {
                "total_files": 0,
                "code_files": 0,
                "doc_files": 0,
                "config_files": 0,
                "total_lines": 0
            }
        }
        
        # File type categories
        code_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.java', '.c', '.cpp', '.h', '.go', '.rs', '.php', '.rb', '.swift']
        doc_extensions = ['.md', '.txt', '.rst', '.tex']
        config_extensions = ['.yml', '.yaml', '.json', '.toml', '.ini', '.cfg', '.xml']
        
        total_lines = 0
        
        with console.status("[bold green]Organizing codebase...") as status:
            for root, _, files in os.walk(fetched_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, fetched_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.count('\n') + 1
                            total_lines += lines
                            
                            file_info = {
                                "content": content,
                                "lines": lines,
                                "size_bytes": os.path.getsize(file_path),
                                "extension": os.path.splitext(file_path)[1].lower()
                            }
                            
                            # Categorize by file type
                            if any(rel_path.lower().endswith(ext) for ext in code_extensions):
                                organized_codebase["code_files"][rel_path] = file_info
                                organized_codebase["stats"]["code_files"] += 1
                            elif any(rel_path.lower().endswith(ext) for ext in doc_extensions):
                                organized_codebase["docs_files"][rel_path] = file_info
                                organized_codebase["stats"]["doc_files"] += 1
                            elif any(rel_path.lower().endswith(ext) for ext in config_extensions):
                                organized_codebase["config_files"][rel_path] = file_info
                                organized_codebase["stats"]["config_files"] += 1
                            else:
                                # Other file types - still add to code files dictionary
                                organized_codebase["code_files"][rel_path] = file_info
                                organized_codebase["stats"]["code_files"] += 1
                            
                            organized_codebase["stats"]["total_files"] += 1
                    except UnicodeDecodeError:
                        # Skip binary files
                        continue
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not process {file_path}: {str(e)}[/yellow]")
                        continue
            
            organized_codebase["stats"]["total_lines"] = total_lines
        
        return organized_codebase
    
    def get_relevant_files(self, query: str, organized_codebase: Dict, top_n: int = 5) -> List[str]:
        """
        Find the most relevant files for a given query.
        This is a placeholder for future embedding-based relevance ranking.
        
        Args:
            query: User query
            organized_codebase: Organized codebase from organize_codebase()
            top_n: Number of most relevant files to return
            
        Returns:
            List of file paths ordered by relevance
        """
        # Simple keyword-based relevance for now
        # This would be replaced with embedding-based semantic search in the future
        
        query_terms = query.lower().split()
        file_scores = {}
        
        # Score files based on keyword matches
        for file_category in ["code_files", "docs_files", "config_files"]:
            for file_path, file_info in organized_codebase.get(file_category, {}).items():
                content = file_info.get("content", "").lower()
                score = 0
                
                # Score based on filename match
                for term in query_terms:
                    if term in file_path.lower():
                        score += 5  # Higher weight for filename matches
                    
                    # Score based on content match
                    score += content.count(term)
                
                if score > 0:
                    file_scores[file_path] = score
        
        # Sort files by score
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N file paths
        return [file_path for file_path, _ in sorted_files[:top_n]]

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
def analyze(
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
def tell(
    question: str = typer.Argument(..., help="Question about the codebase"),
    repo_url: str = typer.Option(None, help="GitHub repository URL (optional if default repo is set)"),
    use_fetched: bool = typer.Option(False, "--use-fetched", help="Use the fetched codebase instead of querying GitHub"),
):
    """Ask questions about the analyzed codebase"""
    config = Config()
    
    if use_fetched:
        # Read from the fetched codebase
        fetched_dir = "fetched_codebase"
        if not os.path.exists(fetched_dir):
            console.print("[red]Fetched codebase directory does not exist. Please fetch the codebase first.[/red]")
            raise typer.Exit(1)

        # Read all files in the fetched codebase
        codebase_content = ""
        code_files_dict = {}
        
        # First, categorize files by type for better context
        for root, _, files in os.walk(fetched_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, fetched_dir)
                
                # Include more file extensions for better code coverage
                if file.lower().endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.java', '.c', '.cpp', '.h', '.go', '.rs', '.php')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            code_files_dict[rel_path] = content
                    except UnicodeDecodeError:
                        # Skip binary files
                        continue
                # Still include docs but with lower priority
                elif file.lower().endswith(('.md', '.txt', '.yml', '.yaml', '.json')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            code_files_dict[rel_path] = content
                    except UnicodeDecodeError:
                        continue
        
        # Determine which files are most relevant to the question
        # For now, a simple approach - we'll include all code files
        # In a more advanced version, we could use embeddings to find relevant files
        
        # Build content with the most important files first
        important_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.go', '.rs']
        
        # Add the most likely relevant code files first
        for path, content in code_files_dict.items():
            if any(path.endswith(ext) for ext in important_extensions):
                codebase_content += f"\n\n### FILE: {path}\n```\n{content}\n```\n"
        
        # Add documentation files
        for path, content in code_files_dict.items():
            if path.endswith(('.md', '.txt', '.yml', '.yaml', '.json')):
                codebase_content += f"\n\n### DOCUMENTATION: {path}\n```\n{content}\n```\n"
        
        # Limit context size if too large
        max_chars = 100000  # Adjust based on your LLM's context window
        if len(codebase_content) > max_chars:
            codebase_content = codebase_content[:max_chars] + "\n... (codebase truncated due to size)"

        # Prepare context for the question
        context = f"""
        You are an expert code assistant analyzing a GitHub repository codebase. Answer the following question with specific code examples whenever possible.
        
        Question: {question}

        Codebase:
        {codebase_content}
        
        When answering:
        1. Reference specific parts of the codebase in your explanation
        2. If applicable, provide code suggestions or improvements
        3. If the user is asking how to implement something, provide example code based on the codebase's patterns and style
        4. Format any code snippets with the appropriate language markers (e.g., ```python```)
        """

    else:
        if not repo_url:
            repo_url = config.get('current_repo')
            if not repo_url:
                console.print("[red]No repository specified. Use --repo-url URL[/red]")
                raise typer.Exit(1)

        # Initialize analyzer for LLM querying
        github_token = config.get('github_token')
        if not github_token:
            console.print("[red]GitHub token not found. Use 'octo config --github-token YOUR_TOKEN'[/red]")
            raise typer.Exit(1)

        analyzer = GitHubAnalyzer(github_token)

        # Use the analysis file in the current directory
        analysis_file = "codebase_analysis.md"

        # Check if analysis exists
        if not os.path.exists(analysis_file):
            console.print("[yellow]No analysis found. Running analysis first...[/yellow]")
            analyze(repo_url)
        
        # Read the analysis
        try:
            with open(analysis_file, "r", encoding='utf-8') as f:
                analysis_content = f.read()
        except Exception as e:
            console.print(f"[red]Error reading analysis file: {str(e)}[/red]")
            console.print("[yellow]Running fresh analysis...[/yellow]")
            analyze(repo_url)
            with open(analysis_file, "r", encoding='utf-8') as f:
                analysis_content = f.read()

        # Prepare context for the question
        context = f"""
        You are an expert code assistant analyzing a GitHub repository codebase. Answer the following question with specific code examples whenever possible.
        
        Question: {question}

        Codebase Analysis:
        {analysis_content}
        
        When answering:
        1. Reference specific parts of the codebase in your explanation
        2. If applicable, provide code suggestions or improvements
        3. If the user is asking how to implement something, provide example code based on the codebase's patterns and style
        4. Format any code snippets with the appropriate language markers (e.g., ```python```)
        """

    # Query the LLM
    llm_type = config.get('llm_type')
    llm_key = config.get('llm_key')
    
    if not llm_type or not llm_key:
        console.print("[red]LLM configuration missing. Use 'octo config --llm-type TYPE --llm-key KEY'[/red]")
        raise typer.Exit(1)

    analyzer = GitHubAnalyzer(config.get('github_token'))
    console.print(f"[bold blue]Analyzing your question using the {'fetched codebase' if use_fetched else 'repository analysis'}...[/bold blue]")
    response = analyzer.query_llm(context, llm_type, llm_key)
    console.print(f"\n[bold green]Answer:[/bold green]\n{response}")

@app.command()
def list_issues(
    repo_url: str = typer.Option(None, help="GitHub repository URL (optional if default repo is set)"),
    count: int = typer.Option(10, help="Number of issues to display"),
    state: str = typer.Option("open", help="Issue state (open, closed, all)")
):
    """Display issues from the repository"""
    config = Config()
    
    if not repo_url:
        repo_url = config.get('current_repo')
        if not repo_url:
            console.print("[red]No repository specified. Use --repo-url URL[/red]")
            raise typer.Exit(1)
    
    github_token = config.get('github_token')
    if not github_token:
        console.print("[red]GitHub token not found. Use 'octo config --github-token YOUR_TOKEN'[/red]")
        raise typer.Exit(1)
    
    try:
        analyzer = GitHubAnalyzer(github_token)
        repo = analyzer.init_repository(repo_url)
        
        # Validate state parameter
        if state not in ["open", "closed", "all"]:
            console.print("[red]Invalid state. Use 'open', 'closed', or 'all'[/red]")
            raise typer.Exit(1)
        
        # Get issues
        issues = repo.get_issues(state=state)
        
        # Display issues
        console.print(f"\n[bold blue]Issues for {repo.full_name} ({state})[/bold blue]\n")
        
        for i, issue in enumerate(issues[:count]):
            console.print(f"[bold]#{issue.number}: {issue.title}[/bold]")
            console.print(f"State: {issue.state}")
            console.print(f"Created: {issue.created_at}")
            console.print(f"Labels: {', '.join([label.name for label in issue.labels]) or 'None'}")
            console.print(f"URL: {issue.html_url}")
            console.print("")
            
            if i == count - 1:
                break
        
        # Check if there are more issues
        total_issues_count = repo.get_issues(state=state).totalCount
        if total_issues_count > count:
            console.print(f"Showing {count} of {total_issues_count} issues.")
            console.print(f"To see more issues, use: octo list-issues --repo-url {repo_url} --count <number>")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)

@app.command()
def issue(
    issue_number: int = typer.Argument(..., help="Issue number to analyze"),
    repo_url: str = typer.Option(None, help="GitHub repository URL (optional if default repo is set)"),
    use_fetched: bool = typer.Option(False, "--use-fetched", help="Use the fetched codebase instead of querying GitHub"),
):
    """Analyze and suggest solutions for a GitHub issue"""
    config = Config()
    analyzer = GitHubAnalyzer(config.get('github_token'))
    
    if use_fetched:
        # Read from the fetched codebase
        fetched_dir = "fetched_codebase"
        if not os.path.exists(fetched_dir):
            console.print("[red]Fetched codebase directory does not exist. Please fetch the codebase first.[/red]")
            raise typer.Exit(1)

        if not repo_url:
            repo_url = config.get('current_repo')
            if not repo_url:
                console.print("[yellow]No repository specified. Using simulated issue context.[/yellow]")
                # Prepare simulated issue context
                issue_context = f"Issue #{issue_number}:\n"
                issue_context += "This is a simulated issue context based on the fetched codebase.\n"
            else:
                # Try to get real issue details even when using fetched codebase
                try:
                    analyzer.init_repository(repo_url)
                    issue = analyzer.get_issue(issue_number)
                    issue_context = f"""
                    Issue #{issue.number}: {issue.title}
                    
                    Description:
                    {issue.body}
                    
                    Labels: {', '.join([label.name for label in issue.labels])}
                    State: {issue.state}
                    Created: {issue.created_at}
                    """
                except Exception as e:
                    console.print(f"[yellow]Could not fetch issue details: {str(e)}. Using simulated issue context.[/yellow]")
                    issue_context = f"Issue #{issue_number}:\n"
                    issue_context += "This is a simulated issue context based on the fetched codebase.\n"

        # Read all files in the fetched codebase
        codebase_content = ""
        code_files_dict = {}
        
        # First, categorize files by type for better context
        for root, _, files in os.walk(fetched_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, fetched_dir)
                
                # Include more file extensions for better code coverage
                if file.lower().endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.java', '.c', '.cpp', '.h', '.go', '.rs', '.php')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            code_files_dict[rel_path] = content
                    except UnicodeDecodeError:
                        # Skip binary files
                        continue
                # Still include docs but with lower priority
                elif file.lower().endswith(('.md', '.txt', '.yml', '.yaml', '.json')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            code_files_dict[rel_path] = content
                    except UnicodeDecodeError:
                        continue
        
        # Build content with the most important files first
        important_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.go', '.rs']
        
        # Add the most likely relevant code files first
        for path, content in code_files_dict.items():
            if any(path.endswith(ext) for ext in important_extensions):
                codebase_content += f"\n\n### FILE: {path}\n```\n{content}\n```\n"
        
        # Add documentation files
        for path, content in code_files_dict.items():
            if path.endswith(('.md', '.txt', '.yml', '.yaml', '.json')):
                codebase_content += f"\n\n### DOCUMENTATION: {path}\n```\n{content}\n```\n"
        
        # Limit context size if too large
        max_chars = 100000  # Adjust based on your LLM's context window
        if len(codebase_content) > max_chars:
            codebase_content = codebase_content[:max_chars] + "\n... (codebase truncated due to size)"

        # Prepare prompt for LLM
        prompt = f"""
        You are an expert code assistant helping with GitHub issues. Based on the issue details and codebase, provide:
        
        1. A summary of the problem
        2. Potential root causes with specific references to the codebase
        3. Detailed code solutions with implementation examples
        4. Next steps for resolution

        Issue Details:
        {issue_context}

        Codebase:
        {codebase_content}
        
        When providing solutions:
        1. Reference the exact files and code locations that need to be modified
        2. Provide complete code examples that can be directly implemented
        3. Format any code snippets with the appropriate language markers (e.g., ```python```)
        4. If multiple approaches are possible, list them in order of recommendation
        """

        # Query LLM for analysis
        console.print(f"[bold blue]Analyzing issue #{issue_number} using the fetched codebase...[/bold blue]")
        response = analyzer.query_llm(prompt, config.get('llm_type'), config.get('llm_key'))
        
        console.print(f"\n[bold blue]Analysis for Issue #{issue_number}[/bold blue]")
        console.print(f"[bold]{issue_context}[/bold]\n")
        console.print(response)

    else:
        if not repo_url:
            repo_url = config.get('current_repo')
            if not repo_url:
                console.print("[red]No repository specified. Use --repo-url or set default with 'octo config --repo URL'[/red]")
                raise typer.Exit(1)

        try:
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

            # Use the analysis file
            analysis_file = "codebase_analysis.md"

            # Check if analysis exists
            if not os.path.exists(analysis_file):
                console.print("[yellow]No analysis found. Running analysis first...[/yellow]")
                analyze(repo_url)
            
            # Read the analysis
            try:
                with open(analysis_file, "r", encoding='utf-8') as f:
                    analysis_content = f.read()
            except Exception as e:
                console.print(f"[red]Error reading analysis file: {str(e)}[/red]")
                console.print("[yellow]Running fresh analysis...[/yellow]")
                analyze(repo_url)
                with open(analysis_file, "r", encoding='utf-8') as f:
                    analysis_content = f.read()

            # Prepare prompt for LLM
            prompt = f"""
            You are an expert code assistant helping with GitHub issues. Based on the issue details and codebase analysis, provide:
            
            1. A summary of the problem
            2. Potential root causes with specific references to the codebase
            3. Detailed code solutions with implementation examples
            4. Next steps for resolution

            Issue Details:
            {issue_context}

            Codebase Analysis:
            {analysis_content}
            
            When providing solutions:
            1. Reference the exact files and code locations that need to be modified
            2. Provide complete code examples that can be directly implemented
            3. Format any code snippets with the appropriate language markers (e.g., ```python```)
            4. If multiple approaches are possible, list them in order of recommendation
            """

            # Query LLM for analysis
            console.print(f"[bold blue]Analyzing issue #{issue.number} using repository analysis...[/bold blue]")
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

@app.command()
def setup_model():
    """Interactive setup for choosing and configuring an LLM model"""
    config = Config()
    
    # Present expanded model options
    console.print("[bold]Available LLM Models:[/bold]")
    console.print("1. OpenAI (GPT-4)")
    console.print("2. Anthropic (Claude)")
    console.print("3. Google (Gemini)")
    console.print("4. Cohere (Command)")
    console.print("5. Mistral AI")
    console.print("6. Meta (Llama)")
    
    # Get user choice
    valid_choices = ["1", "2", "3", "4", "5", "6"]
    choice = ""
    while choice not in valid_choices:
        choice = typer.prompt(f"Select a model (1-{len(valid_choices)})")
        if choice not in valid_choices:
            console.print(f"[red]Invalid choice. Please select a number between 1 and {len(valid_choices)}.[/red]")
    
    # Set model type based on choice
    model_types = {
        "1": "openai",
        "2": "anthropic",
        "3": "gemini",
        "4": "cohere",
        "5": "mistral",
        "6": "llama"
    }
    
    model_names = {
        "1": "OpenAI (GPT-4)",
        "2": "Anthropic (Claude)",
        "3": "Google (Gemini)",
        "4": "Cohere (Command)",
        "5": "Mistral AI",
        "6": "Meta (Llama)"
    }
    
    model_packages = {
        "1": "openai",
        "2": "anthropic",
        "3": "google-generativeai",
        "4": "cohere",
        "5": "mistralai",
        "6": "requests (already installed)"
    }
    
    model_type = model_types[choice]
    console.print(f"\n[bold]Selected: {model_names[choice]}[/bold]")
    
    # Show package installation instructions
    package_name = model_packages[choice]
    console.print(f"\n[bold]Required package: {package_name}[/bold]")
    console.print(f"If not already installed, use: pip install {package_name}")
    
    # Ask for API key
    console.print(f"\n[bold]Please enter your {model_type.upper()} API key:[/bold]")
    api_key = typer.prompt("API Key", hide_input=True)
    
    # Save configuration
    config.set('llm_type', model_type)
    config.set('llm_key', api_key)
    
    console.print(f"\n[bold green]✓ {model_type.upper()} configured successfully![/bold green]")
    console.print("You can now use the model with commands like 'octo tell' and 'octo issue'")
    
    # Information about API key environment variables
    console.print(f"\n[bold]API Key Environment Variable:[/bold] {model_type.upper()}_API_KEY")
    console.print("You can also set this environment variable instead of storing the key in the config file.")

@app.command()
def codebase_info():
    """Display information about the fetched codebase"""
    config = Config()
    
    fetched_dir = "fetched_codebase"
    if not os.path.exists(fetched_dir):
        console.print("[red]Fetched codebase directory does not exist. Please fetch the codebase first.[/red]")
        raise typer.Exit(1)
    
    # Use the fetcher to organize the codebase
    fetcher = CodebaseFetcher(config.get('github_token'))
    console.print("[bold blue]Analyzing fetched codebase...[/bold blue]")
    
    organized_codebase = fetcher.organize_codebase(fetched_dir)
    stats = organized_codebase.get("stats", {})
    
    # Display codebase information
    console.print("\n[bold green]Fetched Codebase Information[/bold green]")
    console.print(f"Total files: {stats.get('total_files', 0)}")
    console.print(f"Code files: {stats.get('code_files', 0)}")
    console.print(f"Documentation files: {stats.get('doc_files', 0)}")
    console.print(f"Configuration files: {stats.get('config_files', 0)}")
    console.print(f"Total lines of code: {stats.get('total_lines', 0):,}")
    
    # Show most significant code files (by size)
    console.print("\n[bold]Top 10 largest code files:[/bold]")
    code_files = organized_codebase.get("code_files", {})
    sorted_by_size = sorted(code_files.items(), key=lambda x: x[1].get("size_bytes", 0), reverse=True)
    
    for i, (file_path, file_info) in enumerate(sorted_by_size[:10]):
        size_kb = file_info.get("size_bytes", 0) / 1024
        console.print(f"{i+1}. {file_path} - {size_kb:.2f} KB, {file_info.get('lines', 0)} lines")
    
    # List main file types
    extensions = {}
    for file_path, file_info in code_files.items():
        ext = file_info.get("extension", "")
        if ext:
            extensions[ext] = extensions.get(ext, 0) + 1
    
    console.print("\n[bold]File types:[/bold]")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        console.print(f"{ext}: {count} files")
    
    console.print("\n[bold]Using the fetched codebase:[/bold]")
    console.print("Run 'octo tell --use-fetched \"<your question>\"' to query the codebase")
    console.print("Run 'octo issue --use-fetched <issue_number>' to analyze GitHub issues with the fetched codebase")
    
    return organized_codebase

if __name__ == "__main__":
    app() 