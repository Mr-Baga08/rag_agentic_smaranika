"""
Mathematical Analysis Tool - Advanced symbolic and numerical mathematics
"""

from __future__ import annotations
from langchain.tools import BaseTool
from pydantic import Field
from typing import Optional, TYPE_CHECKING
import traceback

if TYPE_CHECKING:
    import sympy as sp


class MathematicalAnalysisTool(BaseTool):
    """
    Tool for advanced mathematical analysis using SymPy and NumPy.
    """

    name: str = "mathematical_analysis"
    description: str = """
    Perform advanced mathematical analysis including:
    - Symbolic mathematics (algebra, calculus, equations)
    - Differentiation and integration
    - Solving equations and systems
    - Matrix operations and linear algebra
    - Statistical analysis
    - Trigonometry and complex numbers

    Input should be a mathematical operation or equation as a string.

    Examples:
    - "differentiate x**2 + 3*x + 2 with respect to x"
    - "integrate sin(x)*cos(x) dx"
    - "solve x**2 - 4 = 0"
    - "matrix eigenvalues [[1, 2], [3, 4]]"
    - "factor x**2 + 5*x + 6"
    """

    def _run(self, query: str) -> str:
        """Process mathematical query"""
        try:
            import sympy as sp
            import numpy as np
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

            query_lower = query.lower().strip()

            # Parse different types of math operations

            # 1. Differentiation
            if any(word in query_lower for word in ['differentiate', 'derivative', 'diff']):
                return self._handle_differentiation(query, sp)

            # 2. Integration
            elif any(word in query_lower for word in ['integrate', 'integral']):
                return self._handle_integration(query, sp)

            # 3. Solve equations
            elif 'solve' in query_lower:
                return self._handle_solve(query, sp)

            # 4. Factor
            elif 'factor' in query_lower:
                return self._handle_factor(query, sp)

            # 5. Expand
            elif 'expand' in query_lower:
                return self._handle_expand(query, sp)

            # 6. Simplify
            elif 'simplify' in query_lower:
                return self._handle_simplify(query, sp)

            # 7. Limit
            elif 'limit' in query_lower:
                return self._handle_limit(query, sp)

            # 8. Matrix operations
            elif any(word in query_lower for word in ['matrix', 'eigenvalue', 'eigenvector', 'determinant']):
                return self._handle_matrix(query, sp, np)

            # 9. Series expansion
            elif 'series' in query_lower or 'taylor' in query_lower:
                return self._handle_series(query, sp)

            # Default: try to evaluate the expression
            else:
                return self._handle_evaluate(query, sp)

        except Exception as e:
            return f"❌ Mathematical analysis error: {str(e)}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"

    def _extract_expression(self, query: str, sp) -> sp.Expr:
        """Extract mathematical expression from query"""
        import re
        # Try to find expression in quotes or after keywords
        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'(?:of|expression|equation)\s+(.+?)(?:\s+with|\s*$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                expr_str = match.group(1)
                return sp.sympify(expr_str)

        # If no pattern matches, try the whole query
        return sp.sympify(query)

    def _handle_differentiation(self, query: str, sp) -> str:
        """Handle differentiation queries"""
        import re

        # Extract expression and variable
        expr_match = re.search(r'(?:differentiate|derivative of)\s+(.+?)\s+(?:with respect to|wrt)\s+(\w+)', query, re.IGNORECASE)

        if expr_match:
            expr_str = expr_match.group(1).strip()
            var_str = expr_match.group(2).strip()

            expr = sp.sympify(expr_str)
            var = sp.Symbol(var_str)

            derivative = sp.diff(expr, var)

            return f"""✅ **Differentiation Result**

**Expression:** `{expr}`
**Variable:** `{var}`
**Derivative:** `{derivative}`

**LaTeX:** ${sp.latex(derivative)}$

**Simplified:** `{sp.simplify(derivative)}`
"""
        else:
            return "❌ Please specify expression and variable: 'differentiate [expression] with respect to [variable]'"

    def _handle_integration(self, query: str, sp) -> str:
        """Handle integration queries"""
        import re

        expr_match = re.search(r'integrate\s+(.+?)(?:\s+d(\w+)|\s+with respect to\s+(\w+)|\s*$)', query, re.IGNORECASE)

        if expr_match:
            expr_str = expr_match.group(1).strip()
            var_str = expr_match.group(2) or expr_match.group(3) or 'x'

            expr = sp.sympify(expr_str)
            var = sp.Symbol(var_str.strip())

            integral = sp.integrate(expr, var)

            return f"""✅ **Integration Result**

**Expression:** `{expr}`
**Variable:** `{var}`
**Integral:** `{integral} + C`

**LaTeX:** ${sp.latex(integral)}$ + C
"""
        else:
            return "❌ Please specify expression: 'integrate [expression] dx'"

    def _handle_solve(self, query: str, sp) -> str:
        """Handle equation solving"""
        import re

        expr_match = re.search(r'solve\s+(.+?)(?:\s+for\s+(\w+)|\s*$)', query, re.IGNORECASE)

        if expr_match:
            expr_str = expr_match.group(1).strip()
            var_str = expr_match.group(2) or 'x'

            expr = sp.sympify(expr_str)
            var = sp.Symbol(var_str.strip())

            solutions = sp.solve(expr, var)

            return f"""✅ **Equation Solved**

**Equation:** `{expr} = 0`
**Variable:** `{var}`
**Solutions:** `{solutions}`

**Number of solutions:** {len(solutions)}
"""
        else:
            return "❌ Please specify equation: 'solve [equation]'"

    def _handle_factor(self, query: str, sp) -> str:
        """Handle factorization"""
        import re
        expr_match = re.search(r'factor\s+(.+)', query, re.IGNORECASE)

        if expr_match:
            expr = sp.sympify(expr_match.group(1).strip())
            factored = sp.factor(expr)

            return f"""✅ **Factorization Result**

**Original:** `{expr}`
**Factored:** `{factored}`
**LaTeX:** ${sp.latex(factored)}$
"""
        else:
            return "❌ Please specify expression to factor"

    def _handle_expand(self, query: str, sp) -> str:
        """Handle expansion"""
        import re
        expr_match = re.search(r'expand\s+(.+)', query, re.IGNORECASE)

        if expr_match:
            expr = sp.sympify(expr_match.group(1).strip())
            expanded = sp.expand(expr)

            return f"""✅ **Expansion Result**

**Original:** `{expr}`
**Expanded:** `{expanded}`
**LaTeX:** ${sp.latex(expanded)}$
"""
        else:
            return "❌ Please specify expression to expand"

    def _handle_simplify(self, query: str, sp) -> str:
        """Handle simplification"""
        import re
        expr_match = re.search(r'simplify\s+(.+)', query, re.IGNORECASE)

        if expr_match:
            expr = sp.sympify(expr_match.group(1).strip())
            simplified = sp.simplify(expr)

            return f"""✅ **Simplification Result**

**Original:** `{expr}`
**Simplified:** `{simplified}`
**LaTeX:** ${sp.latex(simplified)}$
"""
        else:
            return "❌ Please specify expression to simplify"

    def _handle_limit(self, query: str, sp) -> str:
        """Handle limits"""
        import re
        # Pattern: limit of f(x) as x approaches a
        match = re.search(r'limit\s+(?:of\s+)?(.+?)\s+as\s+(\w+)\s+(?:approaches|->)\s+(.+)', query, re.IGNORECASE)

        if match:
            expr = sp.sympify(match.group(1).strip())
            var = sp.Symbol(match.group(2).strip())
            point = sp.sympify(match.group(3).strip())

            limit_result = sp.limit(expr, var, point)

            return f"""✅ **Limit Result**

**Expression:** `{expr}`
**Variable:** `{var} → {point}`
**Limit:** `{limit_result}`
"""
        else:
            return "❌ Please specify: 'limit of [expression] as [variable] approaches [point]'"

    def _handle_matrix(self, query: str, sp, np) -> str:
        """Handle matrix operations"""
        import re

        # Extract matrix from query
        matrix_match = re.search(r'\[\[.+\]\]', query)

        if matrix_match:
            matrix_str = matrix_match.group(0)
            matrix = sp.Matrix(eval(matrix_str))

            result_parts = [f"✅ **Matrix Analysis**\n\n**Matrix:**\n```\n{matrix}\n```\n"]

            if 'eigenvalue' in query.lower():
                eigenvals = matrix.eigenvals()
                result_parts.append(f"**Eigenvalues:** `{eigenvals}`\n")

            if 'eigenvector' in query.lower():
                eigenvects = matrix.eigenvects()
                result_parts.append(f"**Eigenvectors:** `{eigenvects}`\n")

            if 'determinant' in query.lower() or 'det' in query.lower():
                det = matrix.det()
                result_parts.append(f"**Determinant:** `{det}`\n")

            if 'inverse' in query.lower():
                try:
                    inv = matrix.inv()
                    result_parts.append(f"**Inverse:**\n```\n{inv}\n```\n")
                except:
                    result_parts.append("**Inverse:** Matrix is singular (not invertible)\n")

            # Default: show basic properties
            if len(result_parts) == 1:
                result_parts.append(f"**Shape:** {matrix.shape}\n")
                result_parts.append(f"**Determinant:** `{matrix.det()}`\n")
                result_parts.append(f"**Trace:** `{matrix.trace()}`\n")
                result_parts.append(f"**Rank:** `{matrix.rank()}`\n")

            return ''.join(result_parts)
        else:
            return "❌ Please provide matrix in format: [[row1], [row2], ...]"

    def _handle_series(self, query: str, sp) -> str:
        """Handle series expansion"""
        import re
        match = re.search(r'(?:series|taylor)\s+(?:of\s+)?(.+?)\s+(?:at|around)\s+(\w+)\s*=\s*(.+?)(?:\s+(?:order|terms)\s+(\d+))?', query, re.IGNORECASE)

        if match:
            expr = sp.sympify(match.group(1).strip())
            var = sp.Symbol(match.group(2).strip())
            point = sp.sympify(match.group(3).strip())
            order = int(match.group(4)) if match.group(4) else 6

            series = sp.series(expr, var, point, order)

            return f"""✅ **Series Expansion**

**Expression:** `{expr}`
**Point:** `{var} = {point}`
**Order:** {order}
**Series:** `{series}`

**LaTeX:** ${sp.latex(series)}$
"""
        else:
            return "❌ Please specify: 'series of [expression] at [variable]=[point]'"

    def _handle_evaluate(self, query: str, sp) -> str:
        """Evaluate mathematical expression"""
        expr = sp.sympify(query)
        result = expr.evalf()

        return f"""✅ **Evaluation Result**

**Expression:** `{expr}`
**Result:** `{result}`
**LaTeX:** ${sp.latex(expr)}$ = {result}
"""

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)
