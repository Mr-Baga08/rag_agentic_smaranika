"""
Data Analysis Tool - Pandas-based data operations and analysis
"""

from langchain.tools import BaseTool
from pydantic import Field, ConfigDict
from typing import Optional, Dict, Any
import traceback
import io
from .shared_context import get_shared_context, SharedContext


class DataAnalysisTool(BaseTool):
    """
    Tool for data analysis using Pandas.
    Can load, manipulate, analyze, and transform data.
    Automatically discovers files from RAG uploads.
    """

    name: str = "data_analysis"
    description: str = """
    Perform data analysis operations using Pandas:
    - Load data from CSV, JSON, or files uploaded to RAG
    - Statistical analysis (mean, median, std, describe)
    - Data filtering and querying
    - Groupby and aggregations
    - Data cleaning and transformation
    - Correlation and pivot tables

    Input should describe the data operation you want to perform.

    Examples:
    - "load csv from data.csv" or "load SX5E.csv"
    - "describe the dataset"
    - "filter rows where column_name > 100"
    - "group by category and calculate mean"
    - "correlation matrix"
    - "show available files"
    """

    # Allow arbitrary types for non-Pydantic objects
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Declare context as a Pydantic field with default_factory
    context: SharedContext = Field(default_factory=get_shared_context)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-discover files on initialization
        self.context.auto_discover_files()

    def _run(self, query: str) -> str:
        """Process data analysis query"""
        try:
            import pandas as pd
            import numpy as np
            import json

            query_lower = query.lower().strip()

            # Parse different types of data operations

            # 0. Show available files
            if any(word in query_lower for word in ['show files', 'list files', 'available files', 'what files']):
                return self._handle_show_files()

            # 1. Load data
            if any(word in query_lower for word in ['load', 'read', 'import']):
                return self._handle_load_data(query, pd)

            # 2. Describe/summary statistics
            elif any(word in query_lower for word in ['describe', 'summary', 'stats', 'info']):
                return self._handle_describe(query, pd)

            # 3. Filter/query
            elif any(word in query_lower for word in ['filter', 'where', 'query', 'select rows']):
                return self._handle_filter(query, pd)

            # 4. Group by and aggregation
            elif 'group' in query_lower or 'aggregate' in query_lower:
                return self._handle_groupby(query, pd)

            # 5. Correlation
            elif 'correlation' in query_lower or 'corr' in query_lower:
                return self._handle_correlation(query, pd)

            # 6. Pivot table
            elif 'pivot' in query_lower:
                return self._handle_pivot(query, pd)

            # 7. Sort
            elif 'sort' in query_lower:
                return self._handle_sort(query, pd)

            # 8. Show/display data
            elif any(word in query_lower for word in ['show', 'display', 'head', 'tail', 'sample']):
                return self._handle_display(query, pd)

            # 9. Missing values
            elif any(word in query_lower for word in ['missing', 'null', 'na', 'fillna']):
                return self._handle_missing(query, pd)

            # 10. Create new column
            elif any(word in query_lower for word in ['create column', 'add column', 'new column']):
                return self._handle_new_column(query, pd)

            else:
                return self._get_help_message()

        except Exception as e:
            return f"❌ Data analysis error: {str(e)}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"

    def _get_current_df(self, pd):
        """Get the current active dataframe from shared context"""
        df = self.context.get_dataframe()
        if df is None:
            # Create sample dataframe
            sample_data = {
                'A': [1, 2, 3, 4, 5],
                'B': [10, 20, 30, 40, 50],
                'Category': ['X', 'Y', 'X', 'Y', 'X']
            }
            df = pd.DataFrame(sample_data)
            self.context.store_dataframe('df', df)
            return df, True  # True = is_sample
        return df, False

    def _handle_show_files(self) -> str:
        """Show available files in RAG uploads"""
        discovered = self.context.file_registry

        if not discovered:
            return "❌ No files found. Upload files to the RAG system first."

        file_list = []
        for name, path in discovered.items():
            import os
            if os.path.exists(path):
                size = os.path.getsize(path)
                file_list.append(f"- **{name}** ({size:,} bytes)\n  Path: `{path}`")
            else:
                file_list.append(f"- **{name}** (file not found)")

        return f"""✅ **Available Files**

{chr(10).join(file_list)}

**Usage:** `load csv from {list(discovered.keys())[0] if discovered else 'filename.csv'}`
"""

    def _handle_load_data(self, query: str, pd) -> str:
        """Handle data loading with automatic file discovery from RAG uploads"""
        import re

        # Check for CSV file
        csv_match = re.search(r'(?:from|file|path)\s+["\']?([^"\']+\.csv)["\']?', query, re.IGNORECASE)
        if csv_match:
            filename = csv_match.group(1)

            # Try to find the file using shared context
            filepath = self.context.get_file_path(filename)

            if filepath is None:
                # Try as direct path
                filepath = filename

            try:
                df = pd.read_csv(filepath)
                self.context.store_dataframe('df', df)
                return f"""✅ **Data Loaded Successfully**

**Source:** `{filename}`
**Actual Path:** `{filepath}`
**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Columns:** {list(df.columns)}
**Memory:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB

**Preview:**
```
{df.head().to_string()}
```

**Next Steps:**
- Describe: `describe the dataset`
- Plot: Use visualization tool with column names
- Filter: `filter where column_name > value`
"""
            except FileNotFoundError:
                # Show available files to help user
                available = list(self.context.file_registry.keys())
                suggestion = f"\n\n**Available files:** {', '.join(available)}" if available else "\n\n**Tip:** Upload files to RAG first, then try again."
                return f"❌ File not found: {filename}{suggestion}"
            except Exception as e:
                return f"❌ Error loading CSV: {str(e)}"

        # Check for JSON
        json_match = re.search(r'(?:from|file)\s+["\']?([^"\']+\.json)["\']?', query, re.IGNORECASE)
        if json_match:
            filepath = json_match.group(1)
            try:
                df = pd.read_json(filepath)
                self._dataframes[self._current_df_name] = df
                return f"""✅ **Data Loaded from JSON**

**Source:** `{filepath}`
**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Preview:**
```
{df.head().to_string()}
```
"""
            except Exception as e:
                return f"❌ Error loading JSON: {str(e)}"

        # Check for inline data (dictionary or list)
        if '{' in query or '[' in query:
            try:
                # Extract JSON-like structure
                data_match = re.search(r'(\{.+\}|\[.+\])', query)
                if data_match:
                    import json
                    data = json.loads(data_match.group(1))
                    df = pd.DataFrame(data)
                    self._dataframes[self._current_df_name] = df
                    return f"""✅ **Data Created from Dictionary**

**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Preview:**
```
{df.to_string()}
```
"""
            except Exception as e:
                return f"❌ Error parsing data: {str(e)}"

        return "❌ Please specify data source: 'load csv from file.csv' or 'load json from file.json'"

    def _handle_describe(self, query: str, pd) -> str:
        """Handle describe/summary statistics"""
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        if 'info' in query.lower():
            # DataFrame info
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()

            return f"""✅ **DataFrame Info**{sample_note}
```
{info_str}
```
"""
        else:
            # Statistical description
            desc = df.describe(include='all')

            return f"""✅ **Statistical Summary**{sample_note}

**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Columns:** {list(df.columns)}

**Statistics:**
```
{desc.to_string()}
```

**Data Types:**
```
{df.dtypes.to_string()}
```

**Missing Values:**
```
{df.isnull().sum().to_string()}
```
"""

    def _handle_filter(self, query: str, pd) -> str:
        """Handle data filtering"""
        import re
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        # Try to extract filter condition
        # Pattern: "where column_name operator value"
        match = re.search(r'where\s+(\w+)\s*([><=!]+)\s*(.+)', query, re.IGNORECASE)

        if match:
            column = match.group(1).strip()
            operator = match.group(2).strip()
            value = match.group(3).strip()

            # Remove quotes if present
            value = value.strip('"\'')

            # Try to convert to number
            try:
                value = float(value)
            except ValueError:
                pass

            try:
                # Apply filter
                if operator == '>':
                    filtered = df[df[column] > value]
                elif operator == '>=':
                    filtered = df[df[column] >= value]
                elif operator == '<':
                    filtered = df[df[column] < value]
                elif operator == '<=':
                    filtered = df[df[column] <= value]
                elif operator == '==':
                    filtered = df[df[column] == value]
                elif operator == '!=':
                    filtered = df[df[column] != value]
                else:
                    return f"❌ Unknown operator: {operator}"

                return f"""✅ **Filtered Data**{sample_note}

**Condition:** `{column} {operator} {value}`
**Result:** {len(filtered)} rows (from {len(df)} total)

**Preview:**
```
{filtered.head(20).to_string()}
```
"""
            except KeyError:
                return f"❌ Column '{column}' not found. Available columns: {list(df.columns)}"
            except Exception as e:
                return f"❌ Filter error: {str(e)}"

        return "❌ Please specify filter condition: 'filter where column_name > value'"

    def _handle_groupby(self, query: str, pd) -> str:
        """Handle group by operations"""
        import re
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        # Pattern: "group by column_name and calculate aggregation"
        match = re.search(r'group\s+by\s+(\w+)(?:\s+(?:and\s+)?(?:calculate|compute|get)\s+(\w+))?', query, re.IGNORECASE)

        if match:
            group_col = match.group(1).strip()
            agg_func = match.group(2).strip() if match.group(2) else 'mean'

            try:
                grouped = df.groupby(group_col)

                if agg_func in ['mean', 'average', 'avg']:
                    result = grouped.mean(numeric_only=True)
                elif agg_func in ['sum', 'total']:
                    result = grouped.sum(numeric_only=True)
                elif agg_func in ['count', 'size']:
                    result = grouped.size()
                elif agg_func in ['min', 'minimum']:
                    result = grouped.min(numeric_only=True)
                elif agg_func in ['max', 'maximum']:
                    result = grouped.max(numeric_only=True)
                elif agg_func in ['std', 'stdev']:
                    result = grouped.std(numeric_only=True)
                else:
                    result = grouped.mean(numeric_only=True)

                return f"""✅ **Group By Result**{sample_note}

**Grouped by:** `{group_col}`
**Aggregation:** `{agg_func}`

**Result:**
```
{result.to_string()}
```
"""
            except KeyError:
                return f"❌ Column '{group_col}' not found. Available columns: {list(df.columns)}"
            except Exception as e:
                return f"❌ Group by error: {str(e)}"

        return "❌ Please specify: 'group by column_name and calculate mean'"

    def _handle_correlation(self, query: str, pd) -> str:
        """Handle correlation analysis"""
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        # Get numeric columns only
        numeric_df = df.select_dtypes(include=['number'])

        if numeric_df.empty:
            return "❌ No numeric columns found for correlation analysis"

        corr = numeric_df.corr()

        return f"""✅ **Correlation Matrix**{sample_note}

**Numeric Columns:** {list(numeric_df.columns)}

**Correlation:**
```
{corr.to_string()}
```

**Strong Correlations (|r| > 0.7):**
```
{self._find_strong_correlations(corr)}
```
"""

    def _find_strong_correlations(self, corr) -> str:
        """Find strong correlations"""
        strong_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    strong_corr.append(f"{corr.columns[i]} ↔ {corr.columns[j]}: {val:.3f}")

        if strong_corr:
            return '\n'.join(strong_corr)
        else:
            return "No strong correlations found"

    def _handle_pivot(self, query: str, pd) -> str:
        """Handle pivot table"""
        import re
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        # Pattern: "pivot with rows=A, columns=B, values=C"
        match = re.search(r'rows?=(\w+).*columns?=(\w+).*values?=(\w+)', query, re.IGNORECASE)

        if match:
            row_col = match.group(1).strip()
            col_col = match.group(2).strip()
            val_col = match.group(3).strip()

            try:
                pivot = pd.pivot_table(df, values=val_col, index=row_col, columns=col_col, aggfunc='mean')

                return f"""✅ **Pivot Table**{sample_note}

**Rows:** `{row_col}`
**Columns:** `{col_col}`
**Values:** `{val_col}` (mean)

**Result:**
```
{pivot.to_string()}
```
"""
            except Exception as e:
                return f"❌ Pivot error: {str(e)}"

        return "❌ Please specify: 'pivot with rows=column1, columns=column2, values=column3'"

    def _handle_sort(self, query: str, pd) -> str:
        """Handle sorting"""
        import re
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        # Pattern: "sort by column_name descending/ascending"
        match = re.search(r'sort\s+by\s+(\w+)(?:\s+(desc|asc|descending|ascending))?', query, re.IGNORECASE)

        if match:
            column = match.group(1).strip()
            order = match.group(2)
            ascending = False if order and 'desc' in order.lower() else True

            try:
                sorted_df = df.sort_values(by=column, ascending=ascending)

                return f"""✅ **Sorted Data**{sample_note}

**Sort by:** `{column}` ({'ascending' if ascending else 'descending'})

**Preview:**
```
{sorted_df.head(20).to_string()}
```
"""
            except KeyError:
                return f"❌ Column '{column}' not found. Available columns: {list(df.columns)}"

        return "❌ Please specify: 'sort by column_name descending'"

    def _handle_display(self, query: str, pd) -> str:
        """Handle display operations"""
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        if 'tail' in query.lower():
            return f"""✅ **Last Rows**{sample_note}
```
{df.tail().to_string()}
```
"""
        elif 'sample' in query.lower():
            return f"""✅ **Random Sample**{sample_note}
```
{df.sample(min(5, len(df))).to_string()}
```
"""
        else:  # head
            return f"""✅ **First Rows**{sample_note}
```
{df.head().to_string()}
```
"""

    def _handle_missing(self, query: str, pd) -> str:
        """Handle missing values analysis"""
        df, is_sample = self._get_current_df(pd)

        sample_note = "\n⚠️ *Using sample data. Load your data first.*\n" if is_sample else ""

        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df)) * 100

        result_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })

        return f"""✅ **Missing Values Analysis**{sample_note}

**Total Rows:** {len(df)}

**Missing Values:**
```
{result_df.to_string()}
```

**Total Missing:** {df.isnull().sum().sum()} cells
"""

    def _handle_new_column(self, query: str, pd) -> str:
        """Handle creating new columns"""
        import re
        df, is_sample = self._get_current_df(pd)

        # Pattern: "create column name = expression"
        match = re.search(r'(?:create|add)\s+column\s+(\w+)\s*=\s*(.+)', query, re.IGNORECASE)

        if match:
            col_name = match.group(1).strip()
            expression = match.group(2).strip()

            try:
                # Evaluate expression in context of dataframe
                df[col_name] = df.eval(expression)
                self._dataframes[self._current_df_name] = df

                return f"""✅ **New Column Created**

**Column Name:** `{col_name}`
**Expression:** `{expression}`

**Preview:**
```
{df.head().to_string()}
```
"""
            except Exception as e:
                return f"❌ Error creating column: {str(e)}"

        return "❌ Please specify: 'create column new_col = expression'"

    def _get_help_message(self) -> str:
        """Return help message"""
        return """❓ **Data Analysis Tool - Available Operations**

**Loading Data:**
- load csv from file.csv
- load json from file.json

**Exploration:**
- describe the dataset
- show head/tail/sample
- info

**Filtering & Selection:**
- filter where column_name > value
- sort by column_name descending

**Aggregation:**
- group by category and calculate mean
- correlation matrix
- pivot with rows=A, columns=B, values=C

**Data Quality:**
- missing values analysis

**Transformation:**
- create column new_col = expression

Use the code_executor tool for custom Pandas operations.
"""

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)

    def clear_data(self):
        """Clear all stored dataframes"""
        self._dataframes.clear()
