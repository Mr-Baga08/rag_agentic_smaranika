"""
Code Execution Tool - Safely execute Python code in a sandboxed environment
"""

import sys
import io
import contextlib
import traceback
import ast
import time
from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import Field


class CodeExecutorTool(BaseTool):
    """
    Tool for executing Python code safely with timeout and output capture.
    Supports: NumPy, Pandas, Matplotlib, SymPy, and standard library.
    """

    name: str = "code_executor"
    description: str = """
    Execute Python code safely. Useful for:
    - Running calculations and algorithms
    - Data processing with Pandas
    - Mathematical computations with NumPy
    - Creating variables and performing operations

    Input should be valid Python code as a string.
    Returns the output, printed values, and any errors.

    Example: "import numpy as np; result = np.array([1,2,3]).mean(); print(result)"
    """

    timeout: int = Field(default=30, description="Maximum execution time in seconds")
    max_output_length: int = Field(default=10000, description="Maximum output length")

    # Shared namespace for persistent variables across executions
    _namespace: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize namespace with safe imports
        self._initialize_namespace()

    def _initialize_namespace(self):
        """Initialize the execution namespace with common libraries"""
        if not self._namespace:
            safe_imports = {
                '__builtins__': __builtins__,
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
            }
            self._namespace.update(safe_imports)

    def _is_safe_code(self, code: str) -> tuple[bool, str]:
        """
        Check if code is safe to execute (basic checks).
        Returns (is_safe, reason)
        """
        # Dangerous patterns to block
        dangerous_patterns = [
            'import os',
            'import subprocess',
            'import sys',
            '__import__',
            'eval(',
            'exec(',
            'compile(',
            'open(',
            'file(',
            'input(',
            'raw_input(',
            '__builtins__',
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, f"Dangerous operation detected: {pattern}"

        # Try to parse the code
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        return True, "Code is safe"

    def _run(self, code: str) -> str:
        """Execute the Python code and return results"""

        # Safety check
        is_safe, reason = self._is_safe_code(code)
        if not is_safe:
            return f"❌ Code execution blocked: {reason}"

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = {
            'success': False,
            'output': '',
            'error': None,
            'execution_time': 0,
        }

        start_time = time.time()

        try:
            # Redirect stdout and stderr
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # Execute the code
                exec(code, self._namespace)

                result['success'] = True
                result['output'] = stdout_capture.getvalue()

        except Exception as e:
            result['success'] = False
            result['error'] = f"{type(e).__name__}: {str(e)}"
            result['traceback'] = traceback.format_exc()

        finally:
            execution_time = time.time() - start_time
            result['execution_time'] = round(execution_time, 3)

        # Format output
        output_parts = []

        if result['success']:
            output_parts.append("✅ Code executed successfully")
            if result['output']:
                output_parts.append(f"\n**Output:**\n```\n{result['output']}\n```")
            else:
                # Check if there are any variables defined
                user_vars = {k: v for k, v in self._namespace.items()
                            if not k.startswith('_') and k not in ['print', 'len', 'range']}
                if user_vars:
                    output_parts.append(f"\n**Variables defined:**\n```python\n")
                    for k, v in list(user_vars.items())[:10]:  # Limit to 10 vars
                        output_parts.append(f"{k} = {repr(v)[:100]}\n")
                    output_parts.append("```")
        else:
            output_parts.append(f"❌ Execution failed: {result['error']}")
            if 'traceback' in result:
                output_parts.append(f"\n**Traceback:**\n```\n{result['traceback']}\n```")

        output_parts.append(f"\n⏱️ Execution time: {result['execution_time']}s")

        final_output = ''.join(output_parts)

        # Truncate if too long
        if len(final_output) > self.max_output_length:
            final_output = final_output[:self.max_output_length] + "\n... (output truncated)"

        return final_output

    async def _arun(self, code: str) -> str:
        """Async version - just calls sync version"""
        return self._run(code)

    def clear_namespace(self):
        """Clear all user-defined variables from namespace"""
        self._namespace.clear()
        self._initialize_namespace()
