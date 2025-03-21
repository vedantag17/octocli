from rich.tree import Tree
from rich import print
from typing import Dict
import os

def create_file_tree(structure: Dict, tree: Tree = None) -> Tree:
    """Create a visual tree representation of the file structure"""
    if tree is None:
        tree = Tree("📁 Repository")

    for name, content in sorted(structure.items()):
        if isinstance(content, dict):
            branch = tree.add(f"📁 {name}")
            create_file_tree(content, branch)
        else:
            icon = "📄" if content == "file" else "🔗"
            tree.add(f"{icon} {name}")

    return tree

def print_analysis_results(results: Dict):
    """Print analysis results in a formatted way"""
    print("\n[bold blue]Repository Analysis Results[/bold blue]")
    
    if results.get('complexity'):
        print("\n[yellow]Code Complexity:[/yellow]")
        for file_path, complexity in results['complexity'].items():
            print(f"\n📄 {file_path}")
            print(f"  Average Complexity: {complexity['average_complexity']:.2f}")
            print(f"  Total Complexity: {complexity['total_complexity']}")
            
            if complexity['modules']:
                print("  Complex Functions:")
                for module in complexity['modules']:
                    if module['complexity'] > 5:  # Show only complex functions
                        print(f"    - {module['name']} (Complexity: {module['complexity']})")

    if results.get('unused_imports'):
        print("\n[yellow]Unused Imports:[/yellow]")
        for file_path, imports in results['unused_imports'].items():
            if imports:
                print(f"\n📄 {file_path}")
                for imp in imports:
                    print(f"  - {imp}")