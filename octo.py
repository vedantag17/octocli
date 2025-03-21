import typer
from github import Github
from rich.console import Console
from rich.tree import Tree
import os
from typing import Optional, Dict, List, Tuple
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
import shutil
from dotenv import load_dotenv
import re
from octocli.services.analyzer_service import AnalyzerService

app = typer.Typer()
console = Console()

CONFIG_FILE = "octo_config.ini"

# Token estimation constants - rough approximations
CHARS_PER_TOKEN = 4  # Approximation for token estimation
MAX_TOKENS_DEFAULT = 128000  # Default maximum tokens
MAX_TOKENS_SAFETY_MARGIN = 5000  # Safety margin to stay under limits

class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        # Create config directory in user's home directory
        self.config_dir = os.path.expanduser("~/.octocli")
        self.config_file = os.path.join(self.config_dir, "config.ini")
        
        # Load environment variables from .env file if it exists
        load_dotenv()
        
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
                    'last_analysis_file': '',
                    'azure_endpoint': '',
                    'azure_deployment': '',
                    'azure_api_version': ''
                }
                self.save_config()
                
            # Set environment variables for Azure OpenAI if configured
            if self.config['DEFAULT'].get('llm_type') == 'azure':
                azure_endpoint = self.config['DEFAULT'].get('azure_endpoint', '')
                if azure_endpoint:
                    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint
                
                azure_deployment = self.config['DEFAULT'].get('azure_deployment', '')
                if azure_deployment:
                    os.environ['AZURE_OPENAI_DEPLOYMENT'] = azure_deployment
                
                azure_api_version = self.config['DEFAULT'].get('azure_api_version', '')
                if azure_api_version:
                    os.environ['AZURE_OPENAI_API_VERSION'] = azure_api_version
                
                # Set API key if available
                azure_api_key = self.config['DEFAULT'].get('llm_key', '')
                if azure_api_key:
                    os.environ['AZURE_OPENAI_API_KEY'] = azure_api_key
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config: {str(e)}[/yellow]")
            # Ensure we have a DEFAULT section even if file operations fail
            self.config['DEFAULT'] = {
                'github_token': '',
                'llm_type': '',
                'llm_key': '',
                'current_repo': '',
                'last_analysis_file': '',
                'azure_endpoint': '',
                'azure_deployment': '',
                'azure_api_version': ''
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
            elif llm_type == "azure":
                # Azure OpenAI endpoint integration
                import openai
                
                # Get Azure specific settings from environment, .env file, or config
                azure_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or self.config.get('azure_endpoint')
                azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or self.config.get('azure_deployment')
                azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15") or self.config.get('azure_api_version', "2023-05-15")
                
                try:
                    # Try new OpenAI client format (>=1.0.0)
                    from openai import AzureOpenAI
                    client = AzureOpenAI(
                        api_key=azure_api_key,
                        api_version=azure_api_version,
                        azure_endpoint=azure_endpoint
                    )
                    response = client.chat.completions.create(
                        model=azure_deployment,
                        messages=[
                            {"role": "system", "content": "You are a helpful code analysis assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.choices[0].message.content
                except (ImportError, AttributeError):
                    # Fall back to legacy format for older OpenAI versions
                    openai.api_type = "azure"
                    openai.api_key = azure_api_key
                    openai.api_base = azure_endpoint
                    openai.api_version = azure_api_version
                    
                    response = openai.ChatCompletion.create(
                        deployment_id=azure_deployment,
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
                            # Handle binary files properly
                            try:
                                if file_content.content:
                                    decoded_content = file_content.decoded_content.decode('utf-8')
                                else:
                                    decoded_content = None
                            except UnicodeDecodeError:
                                # Skip binary files that can't be decoded as UTF-8
                                console.print(f"[yellow]Skipping binary file: {file_content.path}[/yellow]")
                                decoded_content = None
                            
                            code_files.append({
                                "path": file_content.path,
                                "content": decoded_content
                            })
                            
                            if decoded_content is not None:
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
    local_path: str = typer.Option(None, help="Local path to repository (for offline analysis)"),
    enhanced_readme: bool = typer.Option(True, help="Generate enhanced readme with code examples for LLMs"),
    max_tokens: int = typer.Option(MAX_TOKENS_DEFAULT, help="Maximum tokens to include in analysis file"),
    summary_only: bool = typer.Option(False, help="Generate only high-level summary without detailed code"),
    prioritize_code: bool = typer.Option(True, help="Prioritize code over documentation when reducing size")
):
    """Analyze codebase and generate comprehensive documentation"""
    config = Config()
    
    # Check if we're analyzing a remote or local repository
    if local_path and os.path.exists(local_path):
        try:
            console.print(f"[bold blue]Analyzing local repository at: {local_path}[/bold blue]")
            
            # Import the analyzer service
            try:
                analyzer_service = AnalyzerService(local_path)
                analysis_results = analyzer_service.analyze_repository()
                
                # Generate enhanced README with code examples if requested
                if enhanced_readme:
                    readme_content = analyzer_service.generate_codebase_readme(
                        analysis_results, 
                        max_tokens=max_tokens,
                        summary_only=summary_only
                    )
                    analysis_file = "codebase_analysis.md"
                    
                    with open(analysis_file, "w", encoding='utf-8') as f:
                        f.write(readme_content)
                    
                    # Estimate token count of the final file
                    estimated_tokens = estimate_tokens(readme_content)
                    
                    console.print(f"[bold green]Enhanced analysis complete! Results saved in {analysis_file}[/bold green]")
                    console.print(f"[yellow]Estimated token count: {estimated_tokens:,}[/yellow]")
                    console.print("[yellow]This enhanced README includes code snippets to help LLMs understand your codebase.[/yellow]")
                    
                return analysis_results
                
            except ImportError:
                console.print("[yellow]Could not import AnalyzerService. Falling back to basic analysis.[/yellow]")
                # Continue with basic analysis if service isn't available
            
        except Exception as e:
            console.print(f"[bold red]Error analyzing local repository: {str(e)}[/bold red]")
            raise typer.Exit(1)
    
    # Remote repository analysis
    if not repo_url:
        repo_url = config.get('current_repo')
        if not repo_url:
            console.print("[red]No repository specified. Use --repo-url URL or --local-path PATH[/red]")
            raise typer.Exit(1)

    try:
        console.print(f"[bold blue]Analyzing repository: {repo_url}[/bold blue]")
        
        analyzer = GitHubAnalyzer(config.get('github_token'))
        repo = analyzer.init_repository(repo_url)
        
        # Get all files
        contents = repo.get_contents("")
        all_files_info = []
        structure = {}
        
        # Prepare for extra analysis if enhanced_readme is enabled
        important_functions = {}
        file_stats = {}

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
                            supported_extensions = ('.py', '.md', '.txt', '.json', '.yml', '.yaml', '.js', '.jsx', '.ts', '.tsx', '.html', '.css')
                            if file_path.lower().endswith(supported_extensions):
                                try:
                                    decoded_content = file_content.decoded_content
                                    file_info = analyzer.analyze_file_content(decoded_content, file_path)
                                    all_files_info.append(file_info)
                                    console.print(f"Analyzed: {file_path}")
                                    
                                    # If enhanced readme is requested, perform extra analysis
                                    if enhanced_readme and file_path.lower().endswith(('.py', '.js', '.jsx', '.ts')):
                                        # Compute basic stats for the file
                                        content_str = decoded_content.decode('utf-8')
                                        file_stats[file_path] = {
                                            'lines': content_str.count('\n') + 1,
                                            'size': len(content_str)
                                        }
                                        
                                        # For Python files, extract important functions and classes
                                        if file_path.endswith('.py'):
                                            try:
                                                temp_file = os.path.join(os.path.dirname(__file__), "temp_code.py")
                                                with open(temp_file, 'w', encoding='utf-8') as f:
                                                    f.write(content_str)
                                                
                                                # Import the analyzer service
                                                try:
                                                    from octocli.services.analyzer_service import AnalyzerService
                                                    temp_analyzer = AnalyzerService(os.path.dirname(__file__))
                                                    important_funcs = temp_analyzer.extract_important_functions(temp_file)
                                                    if important_funcs:
                                                        important_functions[file_path] = important_funcs
                                                except ImportError:
                                                    pass
                                                
                                                # Clean up temp file
                                                if os.path.exists(temp_file):
                                                    os.remove(temp_file)
                                            except Exception as e:
                                                console.print(f"[yellow]Warning: Could not extract functions from {file_path}: {str(e)}[/yellow]")
                                    
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
        if enhanced_readme and len(important_functions) > 0:
            # Use the advanced readme generation if we've extracted functions
            analysis_content = ["# Repository Analysis - Enhanced for LLM Understanding\n"]
            analysis_content.append(f"## Repository: {repo.full_name}\n")
            analysis_content.append(f"Description: {repo.description or 'No description provided'}\n")
            
            # Add repository structure
            analysis_content.append("\n## Repository Structure\n```\n")
            analysis_content.append(analyzer.generate_tree_structure(structure))
            analysis_content.append("```\n")
            
            # Estimate current token count
            current_content = "\n".join(str(line) for line in analysis_content if line is not None)
            current_tokens = estimate_tokens(current_content)
            
            # Maximum tokens for the markdown/code sections
            remaining_tokens = max_tokens - current_tokens - MAX_TOKENS_SAFETY_MARGIN
            
            # Add documentation analysis (from basic analysis)
            if remaining_tokens > 0 and not summary_only:
                analysis_content.append("\n## Documentation\n")
                markdown_files = [f for f in all_files_info if f.get('type') == 'markdown']
                
                if markdown_files:
                    # Sort markdown files by potential importance (README.md should come first)
                    sorted_md_files = sorted(markdown_files, key=lambda f: 
                                           0 if f['path'].lower() == 'readme.md' else 
                                           1 if 'readme' in f['path'].lower() else 
                                           2 if 'docs' in f['path'].lower() else 3)
                    
                    # Prioritize based on available tokens
                    md_limit = min(len(sorted_md_files), 3 if prioritize_code else 5)
                    for file_info in sorted_md_files[:md_limit]:
                        doc_summary = analyzer.generate_file_summary(file_info)
                        # Check if adding this would exceed token limit
                        if estimate_tokens(doc_summary) < remaining_tokens:
                            analysis_content.append(doc_summary)
                            remaining_tokens -= estimate_tokens(doc_summary)
                        else:
                            # Add truncated version
                            truncated_summary = truncate_content(doc_summary, remaining_tokens)
                            analysis_content.append(truncated_summary)
                            remaining_tokens = 0
                            break
                else:
                    analysis_content.append("No markdown documentation files found.\n")
            
            # Add core components section with code snippets
            if remaining_tokens > 0 and not summary_only:
                analysis_content.append("\n## Core Components and Code\n")
                
                # Sort functions by importance (prioritize larger files with more functions)
                sorted_files = sorted(
                    important_functions.items(), 
                    key=lambda x: (
                        # Main app files first
                        0 if 'app.py' in x[0] or 'main.py' in x[0] or '__main__.py' in x[0] else
                        1 if 'cli.py' in x[0] or 'commands' in x[0] else
                        2,
                        # Then by number of functions/classes (descending)
                        -len(x[1]),
                        # Then by file path alphabetically
                        x[0]
                    )
                )
                
                for file_path, functions in sorted_files:
                    if not functions or remaining_tokens <= 0:
                        continue
                    
                    file_header = f"\n### {file_path}\n"
                    analysis_content.append(file_header)
                    remaining_tokens -= estimate_tokens(file_header)
                    
                    # Group by type
                    classes = [f for f in functions if f.get('type') == 'class']
                    standalone_functions = [f for f in functions if f.get('type') == 'function']
                    
                    # Add classes first - they're usually more important
                    for class_info in classes:
                        if remaining_tokens <= 0:
                            break
                            
                        class_name = class_info.get('name', 'Unknown')
                        docstring = class_info.get('docstring', '').strip()
                        code = class_info.get('code', '')
                        
                        class_section = f"#### Class: {class_name}\n"
                        if docstring:
                            class_section += f"{docstring}\n"
                        
                        # Add class code
                        file_ext = os.path.splitext(file_path)[1].lstrip('.')
                        if not file_ext:
                            file_ext = 'python' if file_path.endswith('.py') else 'javascript'
                        
                        code_block = f"```{file_ext}\n{code}\n```\n"
                        class_section += code_block
                        
                        # Check token count before adding
                        class_tokens = estimate_tokens(class_section)
                        if class_tokens < remaining_tokens:
                            analysis_content.append(class_section)
                            remaining_tokens -= class_tokens
                        else:
                            # Add summary instead of full code
                            summary = f"#### Class: {class_name}\n"
                            if docstring:
                                summary += f"{docstring}\n"
                            summary += f"*Class code omitted to save context space (approx. {class_tokens} tokens)*\n"
                            analysis_content.append(summary)
                            remaining_tokens -= estimate_tokens(summary)
                        
                        # Add methods if available and we have tokens left
                        methods = class_info.get('methods', [])
                        if methods and remaining_tokens > 0:
                            methods_section = "**Methods:**\n"
                            for method in methods:
                                method_name = method.get('name', 'Unknown')
                                method_docstring = method.get('docstring', '').strip()
                                method_line = f"- `{method_name}`: {method_docstring or 'No description'}\n"
                                
                                if estimate_tokens(method_line) < remaining_tokens:
                                    methods_section += method_line
                                    remaining_tokens -= estimate_tokens(method_line)
                                else:
                                    methods_section += "- *Additional methods omitted to save context space*\n"
                                    remaining_tokens -= 50  # Approximate tokens for the omitted line
                                    break
                                    
                            analysis_content.append(methods_section)
                
                    # Add standalone functions if we have tokens left
                    if standalone_functions and remaining_tokens > 0:
                        functions_header = "\n#### Key Functions\n"
                        analysis_content.append(functions_header)
                        remaining_tokens -= estimate_tokens(functions_header)
                        
                        # Sort functions by importance (name length as a heuristic)
                        sorted_functions = sorted(
                            standalone_functions,
                            key=lambda x: (
                                # 'main' functions first
                                0 if x.get('name', '') == 'main' else
                                # Then by docstring length (more documented = more important)
                                -len(x.get('docstring', '')),
                                # Then by code length (longer = more complex)
                                -len(x.get('code', ''))
                            )
                        )
                        
                        for func_info in sorted_functions:
                            if remaining_tokens <= 0:
                                break
                                
                            func_name = func_info.get('name', 'Unknown')
                            docstring = func_info.get('docstring', '').strip()
                            code = func_info.get('code', '')
                            
                            func_section = f"##### `{func_name}`\n"
                            if docstring:
                                func_section += f"{docstring}\n"
                            
                            # Add function code
                            file_ext = os.path.splitext(file_path)[1].lstrip('.')
                            if not file_ext:
                                file_ext = 'python' if file_path.endswith('.py') else 'javascript'
                            
                            code_block = f"```{file_ext}\n{code}\n```\n"
                            func_section += code_block
                            
                            # Check token count before adding
                            func_tokens = estimate_tokens(func_section)
                            if func_tokens < remaining_tokens:
                                analysis_content.append(func_section)
                                remaining_tokens -= func_tokens
                            else:
                                # Add summary instead of full code
                                summary = f"##### `{func_name}`\n"
                                if docstring:
                                    summary += f"{docstring}\n"
                                summary += f"*Function code omitted to save context space (approx. {func_tokens} tokens)*\n"
                                analysis_content.append(summary)
                                remaining_tokens -= estimate_tokens(summary)
            
            # Add usage guidelines if we have tokens left
            if remaining_tokens > 0:
                analysis_content.append("\n## Usage Guidelines\n")
                analysis_content.append("This section shows how to use the main components of the codebase.\n")
                
                # Look for main entry points
                main_files = []
                for file_path in file_stats.keys():
                    if file_path.endswith(('main.py', 'app.py', 'cli.py', 'index.js')):
                        main_files.append(file_path)
                
                if main_files and remaining_tokens > 200:  # Only add if we have enough tokens left
                    analysis_content.append("### Getting Started\n")
                    for main_file in main_files:
                        if main_file.endswith('.py'):
                            example = f"```python\n# Example usage of {main_file}\nimport {os.path.splitext(os.path.basename(main_file))[0]}\n\n# See class and function documentation above for details\n```"
                            if estimate_tokens(example) < remaining_tokens:
                                analysis_content.append(example)
                                remaining_tokens -= estimate_tokens(example)
                        elif main_file.endswith('.js'):
                            example = f"```javascript\n// Example usage of {main_file}\nconst app = require('./{os.path.splitext(os.path.basename(main_file))[0]}');\n\n// See class and function documentation above for details\n```"
                            if estimate_tokens(example) < remaining_tokens:
                                analysis_content.append(example)
                                remaining_tokens -= estimate_tokens(example)
        else:
            # Use the original format if we don't have enhanced data, but still apply token limits
            analysis_content = ["# Repository Analysis\n"]
            analysis_content.append(f"## Repository: {repo.full_name}\n")
            analysis_content.append(f"Description: {repo.description or 'No description provided'}\n")
            
            # Add repository structure
            analysis_content.append("\n## Repository Structure\n```\n")
            analysis_content.append(analyzer.generate_tree_structure(structure))
            analysis_content.append("```\n")
            
            # Estimate current token count
            current_content = "\n".join(str(line) for line in analysis_content if line is not None)
            current_tokens = estimate_tokens(current_content)
            remaining_tokens = max_tokens - current_tokens - MAX_TOKENS_SAFETY_MARGIN
            
            # Add documentation analysis
            if remaining_tokens > 0 and not summary_only:
                analysis_content.append("\n## Documentation Analysis\n")
                markdown_files = [f for f in all_files_info if f.get('type') == 'markdown']
                if markdown_files:
                    # Sort by importance (README.md first)
                    sorted_md_files = sorted(markdown_files, key=lambda f: 
                                          0 if f['path'].lower() == 'readme.md' else 
                                          1 if 'readme' in f['path'].lower() else 
                                          2)
                                          
                    for file_info in sorted_md_files:
                        if remaining_tokens <= 0:
                            break
                            
                        file_summary = analyzer.generate_file_summary(file_info)
                        summary_tokens = estimate_tokens(file_summary)
                        
                        if summary_tokens < remaining_tokens:
                            analysis_content.append(file_summary)
                            remaining_tokens -= summary_tokens
                        else:
                            # Add truncated summary
                            truncated = truncate_content(file_summary, remaining_tokens)
                            analysis_content.append(truncated)
                            remaining_tokens = 0
                            break
                else:
                    analysis_content.append("No markdown documentation files found.\n")

            # Add code analysis
            if remaining_tokens > 0 and not summary_only:
                analysis_content.append("\n## Code Analysis\n")
                code_files = [f for f in all_files_info if f.get('type') == 'code']
                if code_files:
                    # Sort by size or number of functions
                    sorted_code_files = sorted(code_files, 
                                            key=lambda f: 
                                            len(f.get('functions', [])) + len(f.get('classes', [])),
                                            reverse=True)
                                            
                    for file_info in sorted_code_files:
                        if remaining_tokens <= 0:
                            break
                            
                        file_summary = analyzer.generate_file_summary(file_info)
                        summary_tokens = estimate_tokens(file_summary)
                        
                        if summary_tokens < remaining_tokens:
                            analysis_content.append(file_summary)
                            remaining_tokens -= summary_tokens
                        else:
                            # Add truncated summary
                            truncated = truncate_content(file_summary, remaining_tokens)
                            analysis_content.append(truncated)
                            remaining_tokens = 0
                            break
                else:
                    analysis_content.append("No code files found.\n")
            
            # Add token count information at the end
            estimated_total = max_tokens - remaining_tokens
            analysis_content.append(f"\n\n*Analysis contains approximately {estimated_total:,} tokens (max allowed: {max_tokens:,})*\n")

        # Ensure the analysis file is saved in the current directory
        analysis_file = "codebase_analysis.md"
        try:
            # Ensure the content is properly joined
            final_content = "\n".join(str(line) for line in analysis_content if line is not None)
            
            # Estimate final token count
            final_tokens = estimate_tokens(final_content)
            
            # Check if we're under the limit, if not add a warning
            if final_tokens > max_tokens:
                warning = f"\n\n**WARNING: This analysis may exceed the {max_tokens:,} token limit with approximately {final_tokens:,} tokens.**\n"
                warning += "Consider using the --max-tokens option to set a lower limit or --summary-only for a briefer analysis.\n"
                final_content += warning
            
            # Check if the file exists, if not create it
            if not os.path.exists(analysis_file):
                with open(analysis_file, "w", encoding='utf-8') as f:
                    f.write(final_content)
            else:
                # If it exists, update the content
                with open(analysis_file, "w", encoding='utf-8') as f:
                    f.write(final_content)
            
            console.print(f"[bold green]Analysis complete! Results updated in {analysis_file}[/bold green]")
            console.print(f"[yellow]Estimated token count: {final_tokens:,} tokens[/yellow]")
            if enhanced_readme:
                console.print("[yellow]The enhanced README includes code snippets to help LLMs understand your codebase.[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error saving analysis: {str(e)}[/red]")
            raise typer.Exit(1)
        
        return all_files_info

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    This is a rough approximation based on character count.
    """
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN

def truncate_content(content: str, max_tokens: int) -> str:
    """
    Truncate content to fit within token limit.
    Preserves markdown structure where possible.
    """
    if estimate_tokens(content) <= max_tokens:
        return content
    
    # Convert tokens to characters for truncation
    max_chars = max_tokens * CHARS_PER_TOKEN
    
    # If content is very small, just return the first part
    if len(content) < max_chars * 2:
        return content[:max_chars] + "...\n*(content truncated to fit token limit)*"
    
    # Try to preserve structure - split by headers
    headers = re.split(r'(^##+ .*$)', content, flags=re.MULTILINE)
    
    # Build the truncated content
    result = []
    current_length = 0
    
    # Always include the first section (usually the title)
    if headers and not headers[0].startswith('#'):
        result.append(headers[0])
        current_length += len(headers[0])
    
    # Process header sections
    i = 1
    while i < len(headers):
        # Each header section consists of a header (i) and content (i+1)
        if i+1 < len(headers):
            header = headers[i]
            content = headers[i+1]
            section_length = len(header) + len(content)
            
            # If adding this section would exceed the limit
            if current_length + section_length > max_chars:
                # If it's just slightly over, try to include a truncated version
                if current_length + len(header) + 100 < max_chars:
                    chars_left = max_chars - current_length - len(header)
                    result.append(header)
                    result.append(content[:chars_left] + "...\n*(content truncated)*")
                else:
                    # Otherwise just note that sections were omitted
                    result.append("\n\n*(additional sections omitted to fit token limit)*")
                break
            
            result.append(header)
            result.append(content)
            current_length += section_length
        else:
            # Handle odd number of splits (last item might be a header with no content)
            if current_length + len(headers[i]) < max_chars:
                result.append(headers[i])
            break
        
        i += 2
    
    return "".join(result)

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
    """Fetch the entire codebase from a GitHub repository"""
    config = Config()
    github_token = config.get('github_token')
    if not github_token:
        console.print("[red]GitHub token not found. Use 'octo config --github-token YOUR_TOKEN'[/red]")
        raise typer.Exit(1)

    fetcher = CodebaseFetcher(github_token)
    try:
        console.print(f"[bold blue]Fetching codebase from {repo_url}...[/bold blue]")
        code_files = fetcher.fetch_codebase(repo_url)
        
        # Create directory to store the fetched codebase
        fetched_dir = "fetched_codebase"
        if os.path.exists(fetched_dir):
            shutil.rmtree(fetched_dir)
        os.makedirs(fetched_dir)
        
        # Write the files to the directory
        for file_info in code_files:
            file_path = os.path.join(fetched_dir, file_info["path"])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Check if content is None before writing
            file_content = file_info.get("content")
            if file_content is not None:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
            else:
                # Log skipped files
                console.print(f"[yellow]Skipping binary or empty file: {file_info['path']}[/yellow]")
        
        console.print(f"[bold green]Successfully fetched {len(code_files)} files from {repo_url}[/bold green]")
        console.print(f"The codebase is saved in the '{fetched_dir}' directory")
        
        # Update current repo in config
        config.set('current_repo', repo_url)
        config.save()
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
    console.print("2. Azure OpenAI (Deployed Models)")
    console.print("3. Anthropic (Claude)")
    console.print("4. Google (Gemini)")
    console.print("5. Cohere (Command)")
    console.print("6. Mistral AI")
    console.print("7. Meta (Llama)")
    
    # Get user choice
    valid_choices = ["1", "2", "3", "4", "5", "6", "7"]
    choice = ""
    while choice not in valid_choices:
        choice = typer.prompt(f"Select a model (1-{len(valid_choices)})")
        if choice not in valid_choices:
            console.print(f"[red]Invalid choice. Please select a number between 1 and {len(valid_choices)}.[/red]")
    
    # Set model type based on choice
    model_types = {
        "1": "openai",
        "2": "azure",
        "3": "anthropic",
        "4": "gemini",
        "5": "cohere",
        "6": "mistral",
        "7": "llama"
    }
    
    model_names = {
        "1": "OpenAI (GPT-4)",
        "2": "Azure OpenAI",
        "3": "Anthropic (Claude)",
        "4": "Google (Gemini)",
        "5": "Cohere (Command)",
        "6": "Mistral AI",
        "7": "Meta (Llama)"
    }
    
    model_packages = {
        "1": "openai",
        "2": "openai",
        "3": "anthropic",
        "4": "google-generativeai",
        "5": "cohere",
        "6": "mistralai",
        "7": "requests (already installed)"
    }
    
    model_type = model_types[choice]
    console.print(f"\n[bold]Selected: {model_names[choice]}[/bold]")
    
    # Show package installation instructions
    package_name = model_packages[choice]
    console.print(f"\n[bold]Required package: {package_name}[/bold]")
    console.print(f"If not already installed, use: pip install {package_name}")
    
    # For Azure, suggest python-dotenv
    if model_type == "azure":
        console.print(f"For Azure OpenAI, you may also want to install: pip install python-dotenv")
    
    # Ask for API key
    console.print(f"\n[bold]Please enter your {model_type.upper()} API key:[/bold]")
    api_key = typer.prompt("API Key", hide_input=True)
    
    # For Azure, collect additional required information
    if model_type == "azure":
        console.print("\n[bold]Azure OpenAI requires additional configuration:[/bold]")
        azure_endpoint = typer.prompt("Enter your Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com)")
        azure_deployment = typer.prompt("Enter your Azure OpenAI deployment name")
        azure_api_version = typer.prompt("Enter API version (default: 2023-05-15)", default="2023-05-15")
        
        # Save Azure-specific configuration
        config.set('azure_endpoint', azure_endpoint)
        config.set('azure_deployment', azure_deployment)
        config.set('azure_api_version', azure_api_version)
        
        # Set environment variables
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = azure_deployment
        os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version
    
    # Save configuration
    config.set('llm_type', model_type)
    config.set('llm_key', api_key)
    
    console.print(f"\n[bold green]✓ {model_type.upper()} configured successfully![/bold green]")
    console.print("You can now use the model with commands like 'octo tell' and 'octo issue'")
    
    # Information about API key environment variables
    env_var_name = f"{model_type.upper()}_API_KEY"
    if model_type == "azure":
        env_var_name = "AZURE_OPENAI_API_KEY"
        console.print(f"\n[bold]Azure OpenAI Environment Variables:[/bold]")
        console.print(f"- {env_var_name}: Your Azure OpenAI API key")
        console.print(f"- AZURE_OPENAI_ENDPOINT: {config.get('azure_endpoint')}")
        console.print(f"- AZURE_OPENAI_DEPLOYMENT: {config.get('azure_deployment')}")
        console.print(f"- AZURE_OPENAI_API_VERSION: {config.get('azure_api_version')}")
        
        # Add information about .env file
        console.print("\n[bold]Using .env files:[/bold]")
        console.print("You can also store these values in a .env file in your project directory:")
        console.print("""
Example .env file:
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2023-05-15
        """)
    else:
        console.print(f"\n[bold]API Key Environment Variable:[/bold] {env_var_name}")
    
    console.print("You can set these environment variables instead of storing values in the config file.")

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

@app.command()
def list_models():
    """List available models for the configured LLM provider"""
    config = Config()
    llm_type = config.get('llm_type')
    
    if not llm_type:
        console.print("[red]No LLM type configured. Use 'octo setup_model' first.[/red]")
        raise typer.Exit(1)
        
    console.print(f"[bold]Listing models for {llm_type}...[/bold]")
    
    try:
        if llm_type == "openai":
            import openai
            openai.api_key = config.get('llm_key')
            models = openai.Model.list()
            
            console.print("\n[bold]Available OpenAI Models:[/bold]")
            for model in models['data']:
                console.print(f"- {model['id']}")
                
        elif llm_type == "azure":
            # For Azure OpenAI, we need to fetch models directly from the deployment
            azure_api_key = config.get('llm_key')
            azure_endpoint = config.get('azure_endpoint')
            azure_api_version = config.get('azure_api_version', '2023-05-15')
            
            if not azure_endpoint:
                console.print("[red]Azure OpenAI endpoint not configured. Use 'octo setup_model' first.[/red]")
                raise typer.Exit(1)
                
            try:
                # Try using the newer OpenAI client
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    azure_endpoint=azure_endpoint
                )
                
                console.print("\n[bold]Available Azure OpenAI Deployments:[/bold]")
                
                # For newer clients, directly list deployments
                try:
                    models = client.models.list()
                    for model in models.data:
                        console.print(f"- {model.id}")
                except Exception as e:
                    # If direct model listing fails, display the configured deployment
                    azure_deployment = config.get('azure_deployment')
                    console.print(f"- {azure_deployment} (configured deployment)")
                    console.print("\n[yellow]Note: Could not list all deployments. Make sure your API key has proper permissions.[/yellow]")
                    console.print(f"[yellow]Error: {str(e)}[/yellow]")
                
            except (ImportError, AttributeError):
                # Fall back to legacy client
                import openai
                openai.api_type = "azure"
                openai.api_key = azure_api_key
                openai.api_base = azure_endpoint
                openai.api_version = azure_api_version
                
                try:
                    # Try to list models, but this might not work in all Azure setups
                    models = openai.Model.list()
                    console.print("\n[bold]Available Azure OpenAI Models:[/bold]")
                    for model in models['data']:
                        console.print(f"- {model['id']}")
                except Exception as e:
                    # If that fails, show the configured deployment
                    azure_deployment = config.get('azure_deployment')
                    console.print(f"- {azure_deployment} (configured deployment)")
                    console.print("\n[yellow]Note: Could not list all models. Make sure your API key has proper permissions.[/yellow]")
                    console.print(f"[yellow]Error: {str(e)}[/yellow]")
                
            # Show the current configuration
            console.print("\n[bold]Current Azure OpenAI Configuration:[/bold]")
            console.print(f"Endpoint: {azure_endpoint}")
            console.print(f"Deployment: {config.get('azure_deployment')}")
            console.print(f"API Version: {azure_api_version}")
            
        elif llm_type == "anthropic":
            console.print("\n[bold]Available Anthropic Models:[/bold]")
            console.print("- claude-3-opus-20240229")
            console.print("- claude-3-sonnet-20240229")
            console.print("- claude-3-haiku-20240307")
            console.print("- claude-2.1")
            console.print("- claude-2.0")
            console.print("- claude-instant-1.2")
            
        elif llm_type == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config.get('llm_key'))
            
            console.print("\n[bold]Available Google Gemini Models:[/bold]")
            try:
                models = genai.list_models()
                for model in models:
                    console.print(f"- {model.name}")
            except Exception as e:
                # Fallback to showing known models
                console.print("- gemini-pro")
                console.print("- gemini-pro-vision")
                console.print("- gemini-ultra (if available in your region)")
                console.print(f"\n[yellow]Error listing models: {str(e)}[/yellow]")
                
        elif llm_type == "cohere":
            console.print("\n[bold]Available Cohere Models:[/bold]")
            console.print("- command")
            console.print("- command-light")
            console.print("- command-r")
            console.print("- command-r-plus")
            
        elif llm_type == "mistral":
            console.print("\n[bold]Available Mistral AI Models:[/bold]")
            console.print("- mistral-tiny")
            console.print("- mistral-small")
            console.print("- mistral-medium")
            console.print("- mistral-large-latest")
            
        elif llm_type == "llama":
            console.print("\n[bold]Available Llama Models (via Together API):[/bold]")
            console.print("- meta-llama/Llama-3-70b-chat-hf")
            console.print("- meta-llama/Llama-3-8b-chat-hf")
            console.print("- meta-llama/Llama-2-70b-chat-hf")
        
        else:
            console.print(f"[red]Unsupported LLM type: {llm_type}[/red]")
            
    except ImportError as e:
        console.print(f"[red]Error: The required package for {llm_type} is not installed.[/red]")
        console.print(f"[yellow]Details: {str(e)}[/yellow]")
        console.print(f"[yellow]Try installing the package: pip install {llm_type}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error listing models: {str(e)}[/red]")

if __name__ == "__main__":
    app() 