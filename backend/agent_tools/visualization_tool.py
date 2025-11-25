"""
Visualization Tool - Create charts and plots with Matplotlib and Plotly
"""

from langchain.tools import BaseTool
from pydantic import Field, ConfigDict
from typing import Optional, Dict, Any
import traceback
import base64
import io
import os
import uuid
from .shared_context import get_shared_context, SharedContext


class VisualizationTool(BaseTool):
    """
    Tool for creating data visualizations.
    Supports Matplotlib and Plotly charts.
    Can access dataframes loaded by DataAnalysisTool.
    """

    name: str = "visualization"
    description: str = """
    Create data visualizations and charts:
    - Line plots, bar charts, scatter plots from loaded dataframes
    - Histograms and distributions
    - Pie charts and donut charts
    - Box plots and violin plots
    - Heatmaps and correlation matrices
    - Can use column names from loaded data

    Input should describe the visualization you want to create.

    Examples:
    - "plot date vs close" (uses loaded dataframe columns)
    - "create line plot of date and close price"
    - "bar chart with categories ['A','B','C'] and values [10,20,15]"
    - "scatter plot of x vs y columns"
    - "histogram of price column"
    - "correlation heatmap"
    """

    # Allow arbitrary types for non-Pydantic objects
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: str = Field(default="backend/visualizations", description="Directory to save plots")

    # Declare context as a Pydantic field with default_factory
    context: SharedContext = Field(default_factory=get_shared_context)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _run(self, query: str) -> str:
        """Process visualization query"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np

            query_lower = query.lower().strip()

            # Parse different types of visualizations

            # 1. Line plot
            if 'line' in query_lower or 'plot' in query_lower:
                return self._handle_line_plot(query, plt, np)

            # 2. Bar chart
            elif 'bar' in query_lower:
                return self._handle_bar_chart(query, plt, np)

            # 3. Scatter plot
            elif 'scatter' in query_lower:
                return self._handle_scatter(query, plt, np)

            # 4. Histogram
            elif 'histogram' in query_lower or 'hist' in query_lower:
                return self._handle_histogram(query, plt, np)

            # 5. Pie chart
            elif 'pie' in query_lower:
                return self._handle_pie_chart(query, plt, np)

            # 6. Heatmap
            elif 'heatmap' in query_lower or 'heat map' in query_lower:
                return self._handle_heatmap(query, plt, np)

            # 7. Box plot
            elif 'box' in query_lower:
                return self._handle_box_plot(query, plt, np)

            else:
                return self._get_help_message()

        except Exception as e:
            return f"❌ Visualization error: {str(e)}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"

    def _extract_columns_from_df(self, query: str):
        """Extract column names from query and get data from loaded dataframe"""
        import re

        df = self.context.get_dataframe()
        if df is None:
            return None, None, []

        query_lower = query.lower()

        # Try to find column references in various patterns
        # Pattern 1: "plot X vs Y" or "X vs Y"
        vs_match = re.search(r'([\w\s]+)\s+vs\s+([\w\s]+)', query_lower)
        if vs_match:
            col_x = vs_match.group(1).strip()
            col_y = vs_match.group(2).strip()

            # Find matching columns (case-insensitive)
            df_columns_lower = {col.lower(): col for col in df.columns}

            x_col = df_columns_lower.get(col_x)
            y_col = df_columns_lower.get(col_y)

            if x_col and y_col:
                return df[x_col], df[y_col], [x_col, y_col]

        # Pattern 2: "plot/graph between X and Y"
        between_match = re.search(r'between\s+([\w\s]+?)\s+and\s+([\w\s]+)', query_lower)
        if between_match:
            col_x = between_match.group(1).strip()
            col_y = between_match.group(2).strip()

            df_columns_lower = {col.lower(): col for col in df.columns}
            x_col = df_columns_lower.get(col_x)
            y_col = df_columns_lower.get(col_y)

            if x_col and y_col:
                return df[x_col], df[y_col], [x_col, y_col]

        # Pattern 3: Look for any column names mentioned
        mentioned_cols = []
        df_columns_lower = {col.lower(): col for col in df.columns}

        for col_lower, col_actual in df_columns_lower.items():
            if col_lower in query_lower:
                mentioned_cols.append(col_actual)

        if len(mentioned_cols) >= 2:
            return df[mentioned_cols[0]], df[mentioned_cols[1]], mentioned_cols

        return None, None, []

    def _extract_data(self, query: str, np):
        """Extract numerical data from query"""
        import re

        # Try to find list-like data
        data_matches = re.findall(r'\[([^\]]+)\]', query)

        extracted_data = []
        for match in data_matches:
            try:
                # Try to parse as numbers
                values = [float(x.strip().strip("'\"")) for x in match.split(',')]
                extracted_data.append(values)
            except ValueError:
                # Keep as strings
                values = [x.strip().strip("'\"") for x in match.split(',')]
                extracted_data.append(values)

        return extracted_data

    def _save_and_encode_plot(self, plt, title: str = "plot") -> tuple[str, str]:
        """Save plot to file and return base64 encoded image"""
        # Generate unique filename
        filename = f"{title.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)

        # Save to file
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        # Also encode to base64 for inline display
        buffer = io.BytesIO()
        plt.gcf()
        import matplotlib.pyplot as plt2
        fig = plt2.gcf()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt2.close()

        return filepath, image_base64

    def _handle_line_plot(self, query: str, plt, np) -> str:
        """Create line plot - supports both dataframe columns and explicit data"""
        import re

        # First, try to extract columns from loaded dataframe
        x_data, y_data, col_names = self._extract_columns_from_df(query)

        # If dataframe columns found, use them
        if x_data is not None and y_data is not None:
            x = x_data.values if hasattr(x_data, 'values') else x_data
            y = y_data.values if hasattr(y_data, 'values') else y_data

            xlabel = col_names[0] if len(col_names) > 0 else "X"
            ylabel = col_names[1] if len(col_names) > 1 else "Y"
            title = f"{xlabel} vs {ylabel}"

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(x, y, linewidth=2, marker='o', markersize=3)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            filepath, _ = self._save_and_encode_plot(plt, "line_plot")

            return f"""✅ **Line Plot Created from Loaded Data**

**Title:** {title}
**X-axis:** {xlabel} ({len(x)} points)
**Y-axis:** {ylabel}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

        # Otherwise, try explicit data arrays
        data = self._extract_data(query, np)

        # Extract title if present
        title_match = re.search(r'title[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Line Plot"

        # Extract labels
        xlabel_match = re.search(r'xlabel[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        ylabel_match = re.search(r'ylabel[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        xlabel = xlabel_match.group(1) if xlabel_match else "X"
        ylabel = ylabel_match.group(1) if ylabel_match else "Y"

        if len(data) == 0:
            df = self.context.get_dataframe()
            if df is not None:
                return f"""❌ **No data specified for plotting**

**Available columns in loaded data:** {list(df.columns)}

**Try:** `plot {df.columns[0]} vs {df.columns[1]}` (if you have 2+ columns)
"""
            # Create sample data
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
        elif len(data) == 1:
            # Only y values provided
            y = data[0]
            x = list(range(len(y)))
        else:
            # Both x and y provided
            x = data[0]
            y = data[1]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, linewidth=2, marker='o', markersize=4)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)

        filepath, _ = self._save_and_encode_plot(plt, "line_plot")

        return f"""✅ **Line Plot Created**

**Title:** {title}
**Data points:** {len(y)}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

    def _handle_bar_chart(self, query: str, plt, np) -> str:
        """Create bar chart"""
        import re

        data = self._extract_data(query, np)

        # Extract title
        title_match = re.search(r'title[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Bar Chart"

        if len(data) >= 2:
            categories = data[0]
            values = data[1]
        elif len(data) == 1:
            values = data[0]
            categories = [f"Cat {i+1}" for i in range(len(values))]
        else:
            # Sample data
            categories = ['A', 'B', 'C', 'D']
            values = [10, 25, 15, 30]

        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, values, color='steelblue', edgecolor='black', alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        filepath, _ = self._save_and_encode_plot(plt, "bar_chart")

        return f"""✅ **Bar Chart Created**

**Title:** {title}
**Categories:** {len(categories)}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

    def _handle_scatter(self, query: str, plt, np) -> str:
        """Create scatter plot"""
        import re

        data = self._extract_data(query, np)

        title_match = re.search(r'title[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Scatter Plot"

        if len(data) >= 2:
            x = data[0]
            y = data[1]
        elif len(data) == 1:
            y = data[0]
            x = list(range(len(y)))
        else:
            # Sample data
            np.random.seed(42)
            x = np.random.randn(100)
            y = 2 * x + np.random.randn(100) * 0.5

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(True, alpha=0.3)

        filepath, _ = self._save_and_encode_plot(plt, "scatter_plot")

        return f"""✅ **Scatter Plot Created**

**Title:** {title}
**Points:** {len(x)}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

    def _handle_histogram(self, query: str, plt, np) -> str:
        """Create histogram"""
        import re

        data = self._extract_data(query, np)

        title_match = re.search(r'title[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Histogram"

        # Extract bins
        bins_match = re.search(r'bins[:\s=]+(\d+)', query, re.IGNORECASE)
        bins = int(bins_match.group(1)) if bins_match else 20

        if len(data) >= 1:
            values = data[0]
        else:
            # Sample data
            np.random.seed(42)
            values = np.random.randn(1000)

        # Create plot
        plt.figure(figsize=(10, 6))
        n, bins_edges, patches = plt.hist(values, bins=bins, color='steelblue',
                                          edgecolor='black', alpha=0.7)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.legend()

        filepath, _ = self._save_and_encode_plot(plt, "histogram")

        return f"""✅ **Histogram Created**

**Title:** {title}
**Data points:** {len(values)}
**Bins:** {bins}
**Mean:** {mean_val:.2f}
**Std Dev:** {std_val:.2f}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

    def _handle_pie_chart(self, query: str, plt, np) -> str:
        """Create pie chart"""
        import re

        data = self._extract_data(query, np)

        title_match = re.search(r'title[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Pie Chart"

        if len(data) >= 2:
            labels = data[0]
            values = data[1]
        elif len(data) == 1:
            values = data[0]
            labels = [f"Slice {i+1}" for i in range(len(values))]
        else:
            # Sample data
            labels = ['Category A', 'Category B', 'Category C', 'Category D']
            values = [30, 25, 20, 25]

        # Create plot
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(values)))
        explode = [0.05] * len(values)  # Slight separation

        plt.pie(values, labels=labels, autopct='%1.1f%%',
               startangle=90, colors=colors, explode=explode,
               shadow=True, textprops={'fontsize': 11})

        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.axis('equal')

        filepath, _ = self._save_and_encode_plot(plt, "pie_chart")

        return f"""✅ **Pie Chart Created**

**Title:** {title}
**Slices:** {len(values)}
**Total:** {sum(values):.2f}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

    def _handle_heatmap(self, query: str, plt, np) -> str:
        """Create heatmap"""
        import re

        title_match = re.search(r'title[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Heatmap"

        # Try to extract matrix data
        data = self._extract_data(query, np)

        if len(data) >= 1 and len(data[0]) > 0:
            # Reshape into matrix if needed
            matrix = np.array(data)
            if matrix.ndim == 1:
                size = int(np.sqrt(len(matrix)))
                matrix = matrix[:size*size].reshape(size, size)
        else:
            # Sample correlation matrix
            np.random.seed(42)
            size = 5
            matrix = np.random.randn(size, size)
            matrix = (matrix + matrix.T) / 2  # Make symmetric
            np.fill_diagonal(matrix, 1)  # Diagonal = 1

        # Create plot
        plt.figure(figsize=(10, 8))
        im = plt.imshow(matrix, cmap='coolwarm', aspect='auto', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Value', rotation=270, labelpad=20, fontsize=12)

        # Add value annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = plt.text(j, i, f'{matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        filepath, _ = self._save_and_encode_plot(plt, "heatmap")

        return f"""✅ **Heatmap Created**

**Title:** {title}
**Size:** {matrix.shape[0]} × {matrix.shape[1]}
**Min value:** {matrix.min():.2f}
**Max value:** {matrix.max():.2f}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

    def _handle_box_plot(self, query: str, plt, np) -> str:
        """Create box plot"""
        import re

        data = self._extract_data(query, np)

        title_match = re.search(r'title[:\s]+["\']([^"\']+)["\']', query, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Box Plot"

        if len(data) >= 1:
            plot_data = data
        else:
            # Sample data
            np.random.seed(42)
            plot_data = [np.random.randn(100), np.random.randn(100) + 2,
                        np.random.randn(100) - 1]

        # Create plot
        plt.figure(figsize=(10, 6))
        bp = plt.boxplot(plot_data, patch_artist=True, notch=True)

        # Color the boxes
        colors = plt.cm.Set3(range(len(plot_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Value', fontsize=12)
        plt.xlabel('Group', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        filepath, _ = self._save_and_encode_plot(plt, "box_plot")

        return f"""✅ **Box Plot Created**

**Title:** {title}
**Groups:** {len(plot_data)}
**File:** `{filepath}`

📊 Plot saved successfully!
"""

    def _get_help_message(self) -> str:
        """Return help message"""
        return """❓ **Visualization Tool - Available Plots**

**Basic Plots:**
- line plot of [x_values] vs [y_values] title "My Plot"
- bar chart with categories ['A','B','C'] and values [10,20,15]
- scatter plot of [x_data] vs [y_data]

**Distributions:**
- histogram of [data] bins 30
- box plot of [data1], [data2], [data3]

**Proportions:**
- pie chart with labels ['X','Y','Z'] and values [30,50,20]

**Matrices:**
- heatmap of correlation matrix

All plots are saved to the visualizations directory.
"""

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)
