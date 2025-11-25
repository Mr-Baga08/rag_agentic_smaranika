# 🤖 Agentic RAG System - Complete Guide

## Overview

Your RAG system has been transformed into an **Agentic RAG** - an intelligent system that can not only retrieve and generate answers from documents, but also:

- **Execute Python code** for complex computations
- **Perform mathematical analysis** using symbolic mathematics
- **Analyze data** with Pandas operations
- **Create visualizations** with charts and plots
- **Intelligently route queries** to the right tools using LangChain

---

## 🏗️ Architecture

```
User Query
    ↓
LangChain Agent (Gemini)
    ↓
┌────────────────┴────────────────┐
│     Tool Selection & Routing     │
└────────────────┬────────────────┘
                 ↓
    ┌────────────┼────────────┬──────────┬──────────┐
    ↓            ↓            ↓          ↓          ↓
RAG Tool   Code Exec   Math Tool  Data Tool  Viz Tool
    ↓            ↓            ↓          ↓          ↓
Documents   Python    SymPy/NumPy  Pandas   Matplotlib
```

---

## 🛠️ Available Tools

### 1. **RAG Query Tool** (`rag_query`)
- Searches through uploaded documents
- Retrieves context-aware answers
- Provides source citations
- Works across all domains (medical, legal, financial, technical, academic)

**Example queries:**
- "What are the symptoms of hypertension from the medical documents?"
- "Summarize the key clauses in the contract"

---

### 2. **Code Executor Tool** (`code_executor`)
- Safely executes Python code
- Supports NumPy, Pandas, Matplotlib, SymPy
- Persistent variable namespace
- Output capture and error handling

**Example queries:**
- "Calculate the factorial of 10 using Python"
- "Create a list of prime numbers up to 100"
- "Process this data: [1,2,3,4,5] and calculate the mean"

**Safety features:**
- Blocks dangerous operations (file I/O, subprocess, eval)
- 30-second timeout
- Sandboxed execution

---

### 3. **Mathematical Analysis Tool** (`mathematical_analysis`)
- Symbolic mathematics with SymPy
- Calculus operations (derivatives, integrals, limits)
- Equation solving
- Matrix operations
- Series expansion

**Example queries:**
- "differentiate x^2 + 3*x + 2 with respect to x"
- "integrate sin(x)*cos(x) dx"
- "solve x^2 - 4 = 0"
- "find the eigenvalues of [[1, 2], [3, 4]]"
- "expand (x + y)^3"
- "simplify (x^2 - 1)/(x - 1)"

---

### 4. **Data Analysis Tool** (`data_analysis`)
- Load data from CSV/JSON
- Statistical analysis and summaries
- Data filtering and querying
- Group by and aggregations
- Correlation analysis
- Pivot tables

**Example queries:**
- "load csv from sales_data.csv"
- "describe the dataset"
- "filter rows where sales > 1000"
- "group by category and calculate mean"
- "show correlation matrix"
- "create pivot table with rows=region, columns=product, values=sales"

---

### 5. **Visualization Tool** (`visualization`)
- Create professional charts and plots
- Multiple chart types supported
- Automatic saving to files

**Chart types:**
- Line plots
- Bar charts
- Scatter plots
- Histograms
- Pie charts
- Heatmaps
- Box plots

**Example queries:**
- "create line plot of [1,2,3,4,5] vs [2,4,6,8,10]"
- "bar chart with categories ['A','B','C'] and values [10,20,15]"
- "histogram of [1,2,2,3,3,3,4,4,5] with 10 bins"
- "pie chart with labels ['X','Y','Z'] and values [30,50,20]"

**Output:** Charts are saved to `backend/visualizations/` directory

---

## 📡 API Endpoints

### 1. **Agent Query** (Standard)

```http
POST /agent/query
```

**Request:**
```json
{
  "query": "Calculate the derivative of x^2 and plot it",
  "domain": "general",
  "conversation_id": "optional_conv_id",
  "use_history": true,
  "return_steps": true
}
```

**Response:**
```json
{
  "success": true,
  "output": "The derivative is 2x. Here's the plot...",
  "intermediate_steps": [
    {
      "tool": "mathematical_analysis",
      "tool_input": "differentiate x^2",
      "observation": "Result: 2*x"
    },
    {
      "tool": "visualization",
      "tool_input": "create line plot...",
      "observation": "Plot saved to backend/visualizations/..."
    }
  ],
  "tools_used": ["mathematical_analysis", "visualization"],
  "num_steps": 2,
  "conversation_id": "agent_conv_abc123"
}
```

---

### 2. **Agent Query** (Streaming)

```http
POST /agent/query/stream
```

Returns Server-Sent Events (SSE) with real-time updates:

```javascript
// Event types:
// - start: Agent execution started
// - tool_call: Tool was called
// - output: Final output chunk
// - complete: Execution completed
// - error: Error occurred
```

---

### 3. **Get Available Tools**

```http
GET /agent/tools?domain=general
```

**Response:**
```json
{
  "domain": "general",
  "count": 5,
  "tools": [
    {
      "name": "rag_query",
      "description": "Query the RAG system..."
    },
    {
      "name": "code_executor",
      "description": "Execute Python code..."
    }
    // ... more tools
  ]
}
```

---

### 4. **Clear Conversation**

```http
DELETE /agent/conversation/{conversation_id}?domain=general
```

---

## 🎯 Example Use Cases

### 1. **Data Science Workflow**
```
Query: "Load the sales data from sales.csv, calculate the average sales by region,
       and create a bar chart showing the results"

Agent will:
1. Use data_analysis to load CSV
2. Use data_analysis to group by region and calculate mean
3. Use visualization to create bar chart
```

---

### 2. **Mathematical Research**
```
Query: "Find the derivative and integral of sin(x)*cos(x), then plot both functions"

Agent will:
1. Use mathematical_analysis to differentiate
2. Use mathematical_analysis to integrate
3. Use code_executor to create data points
4. Use visualization to plot all three functions
```

---

### 3. **Document Analysis + Computation**
```
Query: "What is the projected revenue growth from the financial report?
       Calculate the compound annual growth rate (CAGR) for the next 5 years"

Agent will:
1. Use rag_query to find revenue data from documents
2. Use code_executor to calculate CAGR
3. Return comprehensive answer with calculations
```

---

### 4. **Complex Multi-Step Analysis**
```
Query: "Analyze the patient data: load patient_data.csv, filter patients with
       blood pressure > 140, calculate statistics, and create a histogram of ages"

Agent will:
1. Use data_analysis to load CSV
2. Use data_analysis to filter by condition
3. Use data_analysis to calculate statistics
4. Use visualization to create histogram
```

---

## 🚀 Quick Start

### Backend Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Configure environment variables:**
```bash
# Create .env file
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_key_optional
```

3. **Run the backend:**
```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### Testing the Agent

Using `curl`:

```bash
# Basic query
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "differentiate x^2 + 3x",
    "domain": "general",
    "return_steps": true
  }'

# Get available tools
curl http://localhost:8000/agent/tools?domain=general

# Streaming query
curl -X POST http://localhost:8000/agent/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "calculate fibonacci numbers up to 100",
    "domain": "general"
  }'
```

Using Python:

```python
import requests

# Standard query
response = requests.post(
    "http://localhost:8000/agent/query",
    json={
        "query": "integrate x^2 dx and plot it from 0 to 10",
        "domain": "general",
        "return_steps": True
    }
)

result = response.json()
print(f"Output: {result['output']}")
print(f"Tools used: {result['tools_used']}")
print(f"Steps: {result['num_steps']}")
```

---

## 🔧 Configuration

### Model Configuration

Edit `backend/.env`:

```bash
# LLM Model for agent
GEMINI_TEXT_MODEL=models/gemini-flash-latest

# Alternative: Use Pro for better reasoning
# GEMINI_TEXT_MODEL=models/gemini-pro-latest
```

### Agent Temperature

In `backend/main.py`, adjust the `temperature` parameter:

```python
agent_executors[domain] = create_agentic_rag_executor(
    gemini_api_key=GEMINI_API_KEY,
    temperature=0.7,  # Lower = more focused, Higher = more creative
)
```

---

## 🎨 Frontend Integration

The frontend needs to be updated to support agent interactions. Key changes needed:

1. **Add Agent Mode Toggle** in the UI
2. **Display Tool Calls** in the chat interface
3. **Show Visualizations** inline when generated
4. **Add Agent Settings** panel

Example frontend request:

```javascript
const response = await fetch('http://localhost:8000/agent/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: userInput,
    domain: selectedDomain,
    conversation_id: conversationId,
    return_steps: true
  })
});

const result = await response.json();

// Display output
console.log(result.output);

// Show tools used
result.tools_used.forEach(tool => {
  console.log(`Used: ${tool}`);
});

// Display intermediate steps
result.intermediate_steps.forEach(step => {
  console.log(`Tool: ${step.tool}`);
  console.log(`Input: ${step.tool_input}`);
  console.log(`Result: ${step.observation}`);
});
```

---

## 📊 Performance & Limits

| Feature | Limit |
|---------|-------|
| Code execution timeout | 30 seconds |
| Max output length | 10,000 characters |
| Max agent iterations | 15 steps |
| Max execution time | 5 minutes |
| Conversation history | Last 10 messages |

---

## 🔒 Security

The agentic RAG system includes several security measures:

### Code Executor Safety:
- ❌ **Blocks:** file operations, subprocess, eval, exec
- ✅ **Allows:** NumPy, Pandas, Matplotlib, SymPy, standard library
- ⏱️ **Timeout:** 30 seconds max
- 📦 **Sandboxed:** Isolated namespace

### Agent Safety:
- 🔒 **No persistent file access** (temporary only)
- 🚫 **No network operations** from code
- ⚠️ **Input validation** on all queries
- 📝 **Audit logging** of all tool calls

---

## 🐛 Troubleshooting

### Issue: Agent not responding

**Solution:** Check that:
1. Gemini API key is valid
2. Backend is running
3. All dependencies are installed

### Issue: Tool import errors

**Solution:**
```bash
pip install langchain langchain-core langchain-google-genai sympy numpy pandas matplotlib
```

### Issue: Visualization files not found

**Solution:**
```bash
mkdir -p backend/visualizations
```

---

## 📚 Advanced Usage

### Custom Tool Development

Create your own tools in `backend/agent_tools/`:

```python
from langchain.tools import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "What this tool does..."

    def _run(self, query: str) -> str:
        # Your implementation
        return "Result"

    async def _arun(self, query: str) -> str:
        return self._run(query)
```

Register in `backend/agent_executor.py`:

```python
from agent_tools import MyCustomTool

# In _initialize_tools():
custom_tool = MyCustomTool()
tools.append(custom_tool)
```

---

## 🎓 Best Practices

1. **Be specific in queries** - Clear instructions lead to better tool selection
2. **Use conversational context** - The agent remembers previous interactions
3. **Combine tools** - Complex queries can use multiple tools
4. **Check intermediate steps** - Use `return_steps: true` for debugging
5. **Set appropriate domain** - Domain-specific RAG queries work better

---

## 📖 Examples Gallery

### Example 1: Financial Analysis
```
Query: "Load financial_data.csv, calculate the year-over-year growth rate,
       and create a line chart showing revenue trends"
```

### Example 2: Medical Research
```
Query: "What are the side effects of drug X from the medical literature?
       Then calculate the percentage of patients experiencing each side effect"
```

### Example 3: Mathematical Proof
```
Query: "Prove that the derivative of sin(x) is cos(x) using the limit definition,
       then plot both functions"
```

### Example 4: Data Cleaning
```
Query: "Load messy_data.csv, remove rows with missing values,
       show statistics before and after, and create a box plot"
```

---

## 🔮 Future Enhancements

Potential additions to the agentic RAG:

- [ ] **Web browsing tool** - Fetch real-time data
- [ ] **SQL query tool** - Database operations
- [ ] **Image generation** - Create diagrams and illustrations
- [ ] **PDF generation** - Export reports
- [ ] **Email tool** - Send results via email
- [ ] **File I/O tool** (with safety controls)
- [ ] **Multi-agent collaboration** - Specialized agents for different domains

---

## 📞 Support

For issues or questions:
1. Check the logs: `backend/logs/`
2. Review intermediate steps in API response
3. Test individual tools using `/agent/tools` endpoint

---

## 🎉 Summary

You now have a **fully functional agentic RAG system** that combines:
- ✅ Document retrieval and generation
- ✅ Code execution capabilities
- ✅ Mathematical analysis
- ✅ Data analysis and manipulation
- ✅ Visualization creation
- ✅ Intelligent tool routing via LangChain
- ✅ Multi-domain support
- ✅ Streaming responses

**The system can handle complex, multi-step queries that require multiple tools working together!**

---

*Built with LangChain, Google Gemini, and LightRAG*
