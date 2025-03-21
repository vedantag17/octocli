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

app = typer.Typer()
console = Console()

class GitHubAnalyzer:
    def __init__(self, token: Optional[str] = None):
        self.github = Github(token)
        self.current_repo = None

    def init_repository(self, repo_url: str):
        parts = repo_url.rstrip('/').split('/')
        repo_name = f"{parts[-2]}/{parts[-1]}"
        self.current_repo = self.github.get_repo(repo_name)
        return self.current_repo

    def get_file_structure(self):
        if not self.current_repo:
            raise Exception("Repository not initialized")
        
        def get_contents(path: str):
            structure = {}
            contents = self.current_repo.get_contents(path)
            
            for content in contents:
                if content.type == "dir":
                    structure[content.name] = get_contents(content.path)
                else:
                    structure[content.name] = content.type
            return structure
            
        return get_contents("")

    def analyze_file_content(self, content, file_path: str) -> Dict:
        """Analyze content of a single file"""
        try:
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            file_info = {
                "path": file_path,
                "functions": [],
                "classes": [],
                "imports": [],
                "description": ""
            }

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
            return {"error": str(e)}

    def generate_file_summary(self, file_info: Dict) -> str:
        """Generate a markdown summary for a file"""
        summary = [f"## {file_info['path']}\n"]
        
        if file_info.get("description"):
            summary.append(f"{file_info['description']}\n")

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

@app.command()
def analyze_codebase(
    repo_url: str = typer.Argument(..., help="GitHub repository URL"),
    token: Optional[str] = typer.Option(None, help="GitHub personal access token"),
    llm_type: str = typer.Option("openai", help="LLM type (openai or anthropic)"),
    llm_key: Optional[str] = typer.Option(None, help="LLM API key")
):
    """Analyze codebase and generate comprehensive documentation"""
    try:
        console.print(f"[bold blue]Analyzing repository: {repo_url}[/bold blue]")
        
        analyzer = GitHubAnalyzer(token or os.getenv("GITHUB_TOKEN"))
        repo = analyzer.init_repository(repo_url)
        
        # Get all files
        contents = repo.get_contents("")
        all_files_info = []

        with console.status("[bold green]Analyzing files...") as status:
            while contents:
                try:
                    file_content = contents.pop(0)
                    
                    # Safely get the path
                    file_path = getattr(file_content, 'path', None)
                    if not file_path:
                        console.print("[yellow]Warning: Found item with no path, skipping...[/yellow]")
                        continue

                    if file_content.type == "dir":
                        try:
                            dir_contents = repo.get_contents(file_path)
                            contents.extend(dir_contents)
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not read directory {file_path}: {str(e)}[/yellow]")
                            continue
                    else:
                        try:
                            decoded_content = file_content.decoded_content
                            file_info = analyzer.analyze_file_content(decoded_content, file_path)
                            all_files_info.append(file_info)
                            console.print(f"Analyzed: {file_path}")
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not analyze {file_path}: {str(e)}[/yellow]")
                            continue

                except AttributeError:
                    console.print("[yellow]Warning: Found invalid file entry, skipping...[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"[yellow]Warning: Unexpected error: {str(e)}[/yellow]")
                    continue

        # Generate README content
        readme_content = ["# Repository Analysis\n"]
        readme_content.append(f"## Repository: {repo.full_name}\n")
        readme_content.append(f"Description: {repo.description}\n")
        
        for file_info in all_files_info:
            if not file_info.get("error"):
                readme_content.append(analyzer.generate_file_summary(file_info))

        # Save analysis results
        with open("codebase_analysis.md", "w", encoding='utf-8') as f:
            f.write("\n".join(readme_content))

        console.print("[bold green]Analysis complete! Results saved to codebase_analysis.md[/bold green]")
        
        return all_files_info

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)

@app.command()
def query(
    question: str = typer.Argument(..., help="Question about the codebase"),
    repo_url: str = typer.Argument(..., help="GitHub repository URL"),
    llm_type: str = typer.Option("openai", help="LLM type (openai or anthropic)"),
    llm_key: Optional[str] = typer.Option(None, help="LLM API key")
):
    """Ask questions about the analyzed codebase using the existing analysis"""
    try:
        # Check if analysis file exists
        if not os.path.exists("codebase_analysis.md"):
            console.print("[yellow]No existing analysis found. Running analysis first...[/yellow]")
            analyze_codebase(repo_url)
        
        try:
            # Read the existing analysis
            with open("codebase_analysis.md", "r", encoding='utf-8') as f:
                analysis_content = f.read()
        except Exception as e:
            console.print(f"[red]Error reading analysis file: {str(e)}[/red]")
            console.print("[yellow]Running fresh analysis...[/yellow]")
            analyze_codebase(repo_url)
            with open("codebase_analysis.md", "r", encoding='utf-8') as f:
                analysis_content = f.read()

        # Initialize analyzer for LLM querying
        analyzer = GitHubAnalyzer(os.getenv("GITHUB_TOKEN"))
        
        # Prepare context for the question
        context = f"""
        Based on the following codebase analysis, please answer this question:
        
        Question: {question}

        Codebase Analysis:
        {analysis_content}
        
        Please provide a clear and concise answer based on the information available in the analysis.
        """

        # Query the LLM
        response = analyzer.query_llm(context, llm_type, llm_key)
        console.print(f"\n[bold green]Answer:[/bold green]\n{response}")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 