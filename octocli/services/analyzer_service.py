import ast
import os
import re
from typing import Dict, List, Set, Tuple, Optional
from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor
from radon.raw import analyze

class AnalyzerService:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        # Track files by language for better organization
        self.language_extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx'],
            'web': ['.html', '.css'],
            'config': ['.json', '.yaml', '.yml', '.toml', '.ini', '.xml'],
            'documentation': ['.md', '.rst', '.txt'],
            'shell': ['.sh', '.bash'],
            'other': []
        }

    def analyze_complexity(self, file_path: str) -> Dict:
        """Analyze code complexity of a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
        except Exception as e:
            return {"error": str(e)}

    def find_unused_imports(self, file_path: str) -> List[str]:
        """Find unused imports in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
        except Exception as e:
            return [f"Error analyzing imports: {str(e)}"]

    def extract_important_functions(self, file_path: str) -> List[Dict]:
        """
        Extract important functions from a file based on heuristics:
        - Functions with docstrings
        - Functions with high complexity
        - Functions with meaningful names (not helpers like _)
        - Classes and their methods
        """
        important_functions = []

        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle Python files
        if file_ext in self.language_extensions['python']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    
                tree = ast.parse(code)
                
                # Get complexity metrics for all functions
                complexity_data = self.analyze_complexity(file_path)
                complexity_map = {}
                if 'modules' in complexity_data:
                    for module in complexity_data['modules']:
                        complexity_map[module['name']] = module['complexity']
                
                # Create parent mapping to distinguish standalone functions from methods
                parent_map = {}
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        parent_map[child] = node
                
                # First, extract all classes
                class_nodes = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_nodes[node] = node.name
                        class_info = {
                            'type': 'class',
                            'name': node.name,
                            'line_number': node.lineno,
                            'docstring': ast.get_docstring(node) or '',
                            'methods': [],
                            'code': self._get_node_source(code, node)
                        }
                        
                        # Get all methods in this class
                        for child_node in ast.iter_child_nodes(node):
                            if isinstance(child_node, ast.FunctionDef):
                                method_docstring = ast.get_docstring(child_node) or ''
                                method_complexity = complexity_map.get(f"{node.name}.{child_node.name}", 0)
                                
                                method_info = {
                                    'type': 'method',
                                    'name': child_node.name,
                                    'line_number': child_node.lineno,
                                    'docstring': method_docstring,
                                    'complexity': method_complexity,
                                    'code': self._get_node_source(code, child_node)
                                }
                                
                                # Consider important if it has a docstring, high complexity, or is not a private method
                                if (method_docstring or 
                                    method_complexity > 5 or 
                                    not child_node.name.startswith('_')):
                                    class_info['methods'].append(method_info)
                        
                        important_functions.append(class_info)
                
                # Then extract standalone functions (excluding class methods)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip methods (they're handled in class processing)
                        parent = parent_map.get(node)
                        if isinstance(parent, ast.ClassDef):
                            continue
                            
                        func_docstring = ast.get_docstring(node) or ''
                        func_complexity = complexity_map.get(node.name, 0)
                        
                        # Consider important if it has a docstring, high complexity, or is not a utility function
                        # Lower the complexity threshold to catch more functions
                        if (func_docstring or 
                            func_complexity > 3 or 
                            (not node.name.startswith('_') and len(node.name) > 2)):
                            
                            function_info = {
                                'type': 'function',
                                'name': node.name,
                                'line_number': node.lineno,
                                'docstring': func_docstring,
                                'complexity': func_complexity,
                                'code': self._get_node_source(code, node)
                            }
                            important_functions.append(function_info)
                            
            except Exception as e:
                important_functions.append({
                    'type': 'error',
                    'file': file_path,
                    'message': f"Error extracting functions: {str(e)}"
                })
                
        # Handle JavaScript/TypeScript files
        elif file_ext in self.language_extensions['javascript']:
            # Basic regex pattern for JS functions (limited but usable)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Look for class definitions
                class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*{'
                for match in re.finditer(class_pattern, code):
                    class_name = match.group(1)
                    start_pos = match.start()
                    # Find class end (simplistic approach)
                    brace_count = 0
                    end_pos = start_pos
                    for i in range(start_pos, len(code)):
                        if code[i] == '{':
                            brace_count += 1
                        elif code[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    class_code = code[start_pos:end_pos]
                    
                    # Extract JSDoc if present
                    jsdoc = self._extract_jsdoc(code, start_pos)
                    
                    important_functions.append({
                        'type': 'class',
                        'name': class_name,
                        'docstring': jsdoc,
                        'code': class_code
                    })
                
                # Function patterns for various JS function syntaxes
                function_patterns = [
                    # Regular functions: function name() {}
                    r'function\s+(\w+)\s*\([^)]*\)\s*{',
                    # Arrow functions with const: const name = () => {}
                    r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{',
                    # Class methods: methodName() {}
                    r'\s+(\w+)\s*\([^)]*\)\s*{'
                ]
                
                for pattern in function_patterns:
                    for match in re.finditer(pattern, code):
                        func_name = match.group(1)
                        start_pos = match.start()
                        
                        # Skip if this is a common name like constructor
                        if func_name in ['constructor', 'render', 'componentDidMount']:
                            continue
                            
                        # Extract JSDoc if present
                        jsdoc = self._extract_jsdoc(code, start_pos)
                        
                        # Only include functions with JSDoc or meaningful names
                        if jsdoc or len(func_name) > 3:
                            # Find function end (simplistic approach)
                            brace_count = 0
                            in_function = False
                            end_pos = start_pos
                            
                            for i in range(start_pos, min(start_pos + 5000, len(code))):
                                if code[i] == '{':
                                    in_function = True
                                    brace_count += 1
                                elif code[i] == '}':
                                    brace_count -= 1
                                    if in_function and brace_count == 0:
                                        end_pos = i + 1
                                        break
                            
                            # Safety check
                            if end_pos > start_pos:
                                func_code = code[start_pos:end_pos]
                                important_functions.append({
                                    'type': 'function',
                                    'name': func_name,
                                    'docstring': jsdoc,
                                    'code': func_code
                                })
            except Exception as e:
                important_functions.append({
                    'type': 'error',
                    'file': file_path,
                    'message': f"Error extracting JavaScript functions: {str(e)}"
                })
                
        return important_functions
    
    def _extract_jsdoc(self, code: str, position: int) -> str:
        """Extract JSDoc comment before the given position"""
        # Look for JSDoc pattern before the position
        jsdoc_pattern = r'/\*\*[\s\S]*?\*/'
        for match in re.finditer(jsdoc_pattern, code[:position]):
            if match.end() + 10 >= position:  # If JSDoc is close to the function
                return match.group(0)
        return ""
    
    def _get_node_source(self, source_code: str, node) -> str:
        """Extract source code for an AST node"""
        try:
            lines = source_code.splitlines()
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                # For Python 3.8+ with end_lineno attribute
                return "\n".join(lines[node.lineno - 1:node.end_lineno])
            elif hasattr(node, 'lineno'):
                # For older Python where we need to estimate the end
                start_line = node.lineno - 1
                # Simple heuristic: for functions/classes, find the next line with less indentation
                indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                end_line = start_line
                
                for i in range(start_line + 1, min(start_line + 200, len(lines))):
                    if i >= len(lines):
                        break
                    line = lines[i]
                    if line.strip() and len(line) - len(line.lstrip()) <= indent:
                        end_line = i - 1
                        break
                    end_line = i
                
                return "\n".join(lines[start_line:end_line + 1])
            return "# Source code extraction not supported for this node"
        except Exception as e:
            return f"# Error extracting source: {str(e)}"
    
    def get_file_stats(self, file_path: str) -> Dict:
        """Get basic stats for a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            stats = {
                'lines': content.count('\n') + 1,
                'size': os.path.getsize(file_path),
                'extension': os.path.splitext(file_path)[1].lower()
            }
            
            # For Python files, get more detailed stats
            if file_path.endswith('.py'):
                try:
                    raw_stats = analyze(content)
                    stats.update({
                        'loc': raw_stats.loc,  # Lines of code
                        'lloc': raw_stats.lloc,  # Logical lines of code
                        'comments': raw_stats.comments,  # Comment lines
                        'blank': raw_stats.blank,  # Blank lines
                        'single_comments': raw_stats.single_comments  # Single-line comments
                    })
                except:
                    pass
                    
            return stats
        except Exception as e:
            return {'error': str(e)}

    def analyze_repository(self) -> Dict:
        """Analyze entire repository"""
        results = {
            'complexity': {},
            'unused_imports': {},
            'large_functions': [],
            'stats': {
                'total_files': 0,
                'by_language': {},
                'total_code_lines': 0
            },
            'important_files': [],
            'important_functions': {}
        }
        
        # Track language stats
        language_stats = {lang: 0 for lang in self.language_extensions.keys()}
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                
                # Skip .git and virtual environments
                if '.git' in file_path or 'venv' in file_path or '__pycache__' in file_path:
                    continue
                
                # Categorize file by extension
                ext = os.path.splitext(file)[1].lower()
                file_language = 'other'
                for lang, extensions in self.language_extensions.items():
                    if ext in extensions:
                        file_language = lang
                        break
                
                # Update stats
                results['stats']['total_files'] += 1
                if file_language in language_stats:
                    language_stats[file_language] += 1
                
                # Get file stats
                file_stats = self.get_file_stats(file_path)
                
                # Extract important functions
                important_funcs = self.extract_important_functions(file_path)
                if important_funcs:
                    results['important_functions'][relative_path] = important_funcs
                
                # For Python files, do additional analysis
                if file.endswith('.py'):
                    # Analyze complexity
                    complexity = self.analyze_complexity(file_path)
                    if complexity and 'error' not in complexity:
                        results['complexity'][relative_path] = complexity
                        
                        # Update code lines count
                        if 'loc' in file_stats:
                            results['stats']['total_code_lines'] += file_stats['loc']

                    # Find unused imports
                    unused = self.find_unused_imports(file_path)
                    if unused:
                        results['unused_imports'][relative_path] = unused
                        
                    # Identify large or complex functions
                    if 'modules' in complexity:
                        for module in complexity['modules']:
                            if module['complexity'] > 10:
                                results['large_functions'].append({
                                    'file': relative_path,
                                    'function': module['name'],
                                    'complexity': module['complexity'],
                                    'line': module['line_number']
                                })
                                
                # Track important files based on various heuristics
                is_important = False
                if file.endswith(('main.py', 'app.py', 'index.js', 'server.py', 'cli.py')):
                    is_important = True
                elif 'api' in file.lower() or 'service' in file.lower() or 'model' in file.lower():
                    is_important = True
                elif file.startswith('README') or file.startswith('CONTRIBUTING'):
                    is_important = True
                
                if is_important:
                    results['important_files'].append({
                        'path': relative_path,
                        'stats': file_stats
                    })
        
        # Update language stats
        results['stats']['by_language'] = {k: v for k, v in language_stats.items() if v > 0}
        
        return results
    
    def generate_codebase_readme(self, analysis_results: Dict) -> str:
        """
        Generate a comprehensive README with code examples to help LLMs understand the codebase
        """
        readme = [
            "# Codebase Analysis",
            "\nThis document provides a comprehensive analysis of the codebase with important code snippets to help understand the project structure and functionality.\n",
            "## Overview\n",
            "This analysis is designed to help LLMs understand the structure and functionality of the codebase.",
            "It includes key components, important functions, and code snippets that represent the core functionality.",
            "Use this document as a reference when working with this codebase.\n"
        ]
        
        # Add repository statistics
        readme.append("## Repository Statistics\n")
        stats = analysis_results.get('stats', {})
        readme.append(f"- Total files: {stats.get('total_files', 0)}")
        readme.append(f"- Total lines of code: {stats.get('total_code_lines', 0)}")
        
        # Add language breakdown
        if 'by_language' in stats:
            readme.append("\n### Languages\n")
            for lang, count in stats['by_language'].items():
                if count > 0:
                    readme.append(f"- {lang.capitalize()}: {count} files")
        
        # Add file structure section
        readme.append("\n## Directory Structure\n")
        readme.append("```")
        readme.append(self._generate_directory_tree())
        readme.append("```\n")
        
        # Add important files section
        if 'important_files' in analysis_results and analysis_results['important_files']:
            readme.append("\n## Key Files\n")
            for file_info in analysis_results['important_files']:
                path = file_info['path']
                file_stats = file_info.get('stats', {})
                lines = file_stats.get('lines', 'N/A')
                readme.append(f"- **{path}** ({lines} lines)")
        
        # Add important functions section
        if 'important_functions' in analysis_results:
            readme.append("\n## Core Components and Code\n")
            
            for file_path, functions in analysis_results['important_functions'].items():
                if not functions:
                    continue
                    
                readme.append(f"\n### {file_path}\n")
                
                # Group by type
                classes = [f for f in functions if f.get('type') == 'class']
                standalone_functions = [f for f in functions if f.get('type') == 'function']
                
                # Add classes first
                for class_info in classes:
                    class_name = class_info.get('name', 'Unknown')
                    docstring = class_info.get('docstring', '').strip()
                    code = class_info.get('code', '')
                    
                    readme.append(f"#### Class: {class_name}\n")
                    if docstring:
                        readme.append(f"{docstring}\n")
                    
                    # Add class code
                    file_ext = os.path.splitext(file_path)[1].lstrip('.')
                    if not file_ext:
                        file_ext = 'python' if file_path.endswith('.py') else 'javascript'
                    
                    readme.append(f"```{file_ext}")
                    readme.append(code)
                    readme.append("```\n")
                    
                    # Add methods if available
                    methods = class_info.get('methods', [])
                    if methods:
                        readme.append("**Methods:**\n")
                        for method in methods:
                            method_name = method.get('name', 'Unknown')
                            method_docstring = method.get('docstring', '').strip()
                            
                            readme.append(f"- `{method_name}`: {method_docstring or 'No description'}")
                
                # Add standalone functions
                if standalone_functions:
                    readme.append("\n#### Key Functions\n")
                    
                    for func_info in standalone_functions:
                        func_name = func_info.get('name', 'Unknown')
                        docstring = func_info.get('docstring', '').strip()
                        code = func_info.get('code', '')
                        
                        readme.append(f"##### `{func_name}`\n")
                        if docstring:
                            readme.append(f"{docstring}\n")
                        
                        # Add function code
                        file_ext = os.path.splitext(file_path)[1].lstrip('.')
                        if not file_ext:
                            file_ext = 'python' if file_path.endswith('.py') else 'javascript'
                        
                        readme.append(f"```{file_ext}")
                        readme.append(code)
                        readme.append("```\n")
        
        # Add complexity insights
        if 'large_functions' in analysis_results and analysis_results['large_functions']:
            readme.append("\n## Complexity Insights\n")
            readme.append("Functions with high cyclomatic complexity (may need refactoring):\n")
            
            for func in sorted(analysis_results['large_functions'], key=lambda x: x['complexity'], reverse=True)[:10]:
                file_path = func['file']
                func_name = func['function']
                complexity = func['complexity']
                line = func['line']
                
                readme.append(f"- `{func_name}` in {file_path}:line {line} (complexity: {complexity})")
        
        # Add dependencies section if we can detect them
        readme.append("\n## Dependencies\n")
        dependencies = self._detect_dependencies()
        if dependencies:
            readme.append("The codebase appears to use the following libraries and frameworks:\n")
            for dep in dependencies:
                readme.append(f"- {dep}")
        else:
            readme.append("No dependencies detected automatically. Check requirements.txt or package.json for details.")
        
        # Add usage examples section
        readme.append("\n## Usage Guidelines\n")
        readme.append("This section shows how to use the main components of the codebase.\n")
        
        # Look for main entry points and add usage examples
        main_files = [f['path'] for f in analysis_results.get('important_files', []) 
                     if f['path'].endswith(('main.py', 'app.py', 'cli.py', 'index.js'))]
        
        if main_files:
            readme.append("### Getting Started\n")
            for main_file in main_files:
                if main_file.endswith('.py'):
                    readme.append(f"```python\n# Example usage of {main_file}\nimport {os.path.splitext(main_file)[0]}\n\n# See class and function documentation above for details\n```")
                elif main_file.endswith('.js'):
                    readme.append(f"```javascript\n// Example usage of {main_file}\nconst app = require('./{os.path.splitext(main_file)[0]}');\n\n// See class and function documentation above for details\n```")
        
        return "\n".join(readme)
        
    def _generate_directory_tree(self) -> str:
        """Generate a simple directory tree of the repository for README"""
        tree_lines = []
        repo_name = os.path.basename(self.repo_path)
        tree_lines.append(repo_name)
        
        # Skip these directories/files
        skip_patterns = ['.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode']
        
        def add_directory(path, prefix=""):
            entries = sorted([e for e in os.listdir(path) 
                             if not any(skip in e for skip in skip_patterns)])
            
            files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
            dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
            
            # Add directories
            for i, dir_name in enumerate(dirs):
                is_last = (i == len(dirs) - 1 and not files)
                
                if is_last:
                    tree_lines.append(f"{prefix}└── {dir_name}/")
                    next_prefix = prefix + "    "
                else:
                    tree_lines.append(f"{prefix}├── {dir_name}/")
                    next_prefix = prefix + "│   "
                
                add_directory(os.path.join(path, dir_name), next_prefix)
            
            # Add files
            for i, file_name in enumerate(files):
                if i == len(files) - 1:
                    tree_lines.append(f"{prefix}└── {file_name}")
                else:
                    tree_lines.append(f"{prefix}├── {file_name}")
        
        try:
            add_directory(self.repo_path)
            return "\n".join(tree_lines)
        except Exception as e:
            return f"{repo_name}\n└── Error generating directory tree: {str(e)}"
    
    def _detect_dependencies(self) -> List[str]:
        """
        Attempt to detect dependencies used in the codebase by scanning files
        for import statements and checking for dependency files.
        """
        dependencies = set()
        
        # Check for Python dependencies
        requirement_files = ['requirements.txt', 'pyproject.toml', 'setup.py']
        for req_file in requirement_files:
            req_path = os.path.join(self.repo_path, req_file)
            if os.path.exists(req_path):
                try:
                    with open(req_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if req_file == 'requirements.txt':
                            # Parse requirements.txt
                            for line in content.splitlines():
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    # Strip version info
                                    package = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                                    if package:
                                        dependencies.add(package)
                except Exception:
                    pass
                    
        # Check for JavaScript dependencies
        js_dep_files = ['package.json']
        for dep_file in js_dep_files:
            dep_path = os.path.join(self.repo_path, dep_file)
            if os.path.exists(dep_path):
                try:
                    import json
                    with open(dep_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        deps = data.get('dependencies', {})
                        dev_deps = data.get('devDependencies', {})
                        
                        for package in deps.keys():
                            dependencies.add(package)
                        for package in dev_deps.keys():
                            dependencies.add(package + ' (dev)')
                except Exception:
                    pass
                    
        # Scan Python files for import statements
        py_imports = set()
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            tree = ast.parse(content)
                            
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Import):
                                    for name in node.names:
                                        py_imports.add(name.name.split('.')[0])
                                elif isinstance(node, ast.ImportFrom):
                                    if node.module:
                                        py_imports.add(node.module.split('.')[0])
                    except Exception:
                        pass
                        
        # Filter out standard library modules and local imports
        stdlib_modules = set([
            'os', 'sys', 'time', 'datetime', 're', 'math', 'random', 'json', 
            'pickle', 'argparse', 'logging', 'pathlib', 'collections', 'itertools', 
            'functools', 'types', 'typing', 'enum', 'contextlib', 'abc', 'copy',
            'shutil', 'tempfile', 'textwrap', 'string', 'threading', 'multiprocessing',
            'traceback', 'io', 'unittest', 'csv', 'hashlib', 'base64', 'xml', 'html',
            'urllib', 'http', 'sqlite3'
        ])
        
        filtered_imports = py_imports - stdlib_modules
        for imp in filtered_imports:
            if imp not in dependencies and not imp.startswith('_'):
                dependencies.add(imp)
        
        return sorted(list(dependencies))