import ast
import os
from typing import Dict, List
from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor

class AnalyzerService:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def analyze_complexity(self, file_path: str) -> Dict:
        """Analyze code complexity of a Python file"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            complexity = ComplexityVisitor.from_code(code)
            return {
                'average_complexity': complexity.average_complexity,
                'total_complexity': complexity.total_complexity,
                'modules': [
                    {
                        'name': block.name,
                        'complexity': block.complexity,
                        'line_number': block.lineno
                    }
                    for block in complexity.functions
                ]
            }
        except Exception:
            return {}

    def find_unused_imports(self, file_path: str) -> List[str]:
        """Find unused imports in a Python file"""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())

            imports = []
            used_names = set()

            # Collect imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)

            return [imp for imp in imports if imp not in used_names]
        except Exception:
            return []

    def analyze_repository(self) -> Dict:
        """Analyze entire repository"""
        results = {
            'complexity': {},
            'unused_imports': {},
            'large_functions': []
        }

        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_path)
                    
                    # Analyze complexity
                    complexity = self.analyze_complexity(file_path)
                    if complexity:
                        results['complexity'][relative_path] = complexity

                    # Find unused imports
                    unused = self.find_unused_imports(file_path)
                    if unused:
                        results['unused_imports'][relative_path] = unused

        return results