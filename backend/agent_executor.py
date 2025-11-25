"""
Agentic RAG Executor - LangChain-based agent system with tools
"""

import os
from typing import List, Dict, Any, Optional
import asyncio
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our custom tools
from agent_tools import (
    CodeExecutorTool,
    MathematicalAnalysisTool,
    DataAnalysisTool,
    VisualizationTool,
    RAGTool,
)


class AgenticRAGExecutor:
    """
    Main agentic RAG executor that orchestrates multiple tools using LangChain.
    """

    def __init__(
        self,
        gemini_api_key: str,
        rag_instance: Optional[Any] = None,
        domain: str = "general",
        model_name: str = "gemini-1.5-flash-latest",
        temperature: float = 0.7,
    ):
        """
        Initialize the agentic RAG executor.

        Args:
            gemini_api_key: Google Gemini API key
            rag_instance: Reference to the RAG instance
            domain: Current domain for RAG queries
            model_name: Gemini model to use
            temperature: LLM temperature
        """
        self.gemini_api_key = gemini_api_key
        self.rag_instance = rag_instance
        self.domain = domain
        self.model_name = model_name
        self.temperature = temperature

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=temperature,
            convert_system_message_to_human=True,  # Gemini compatibility
        )

        # Initialize tools
        self.tools = self._initialize_tools()

        # Create agent
        self.agent_executor = self._create_agent()

        # Conversation memory
        self.conversation_history: List[Dict[str, str]] = []

    def _initialize_tools(self) -> List:
        """Initialize all available tools"""
        tools = []

        # Code Execution Tool
        code_tool = CodeExecutorTool(
            timeout=30,
            max_output_length=10000,
        )
        tools.append(code_tool)

        # Mathematical Analysis Tool
        math_tool = MathematicalAnalysisTool()
        tools.append(math_tool)

        # Data Analysis Tool
        data_tool = DataAnalysisTool()
        tools.append(data_tool)

        # Visualization Tool
        viz_tool = VisualizationTool(
            output_dir="backend/visualizations"
        )
        tools.append(viz_tool)

        # RAG Tool
        rag_tool = RAGTool(
            rag_instance=self.rag_instance,
            domain=self.domain,
            llm_model=self.llm,
        )
        tools.append(rag_tool)

        return tools

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools"""

        # Create the system prompt
        system_prompt = """You are an advanced AI assistant with access to multiple specialized tools.

**Your capabilities:**

1. **Code Execution**: You can write and execute Python code for:
   - Complex calculations and algorithms
   - Data processing with NumPy and Pandas
   - Custom operations not covered by other tools

2. **Mathematical Analysis**: You can perform symbolic mathematics:
   - Calculus (derivatives, integrals, limits)
   - Algebra (solving equations, factoring, expanding)
   - Linear algebra (matrices, eigenvalues)
   - Series expansions and simplification

3. **Data Analysis**: You can analyze datasets:
   - Load and explore data from CSV/JSON
   - Statistical summaries and descriptions
   - Filtering, grouping, and aggregation
   - Correlation analysis and pivot tables

4. **Visualization**: You can create charts and plots:
   - Line plots, bar charts, scatter plots
   - Histograms and distributions
   - Pie charts and heatmaps
   - Box plots and statistical visualizations

5. **RAG Query**: You can search through uploaded documents:
   - Answer questions from domain-specific knowledge
   - Retrieve context with sources
   - Get verified, cited information

**How to use tools effectively:**

- **Choose the right tool** for each subtask
- **Break down complex requests** into multiple tool calls
- **Combine tools** when needed (e.g., load data → analyze → visualize)
- **Explain your reasoning** before and after using tools
- **Provide clear, formatted output** to the user

**Tool selection guidelines:**

- For document questions → use `rag_query`
- For math equations/calculus → use `mathematical_analysis`
- For data operations → use `data_analysis` or `code_executor`
- For creating charts → use `visualization`
- For custom logic → use `code_executor`

**Important:**
- Always explain what you're doing before calling a tool
- Interpret and summarize tool outputs for the user
- If a tool fails, try an alternative approach
- Combine multiple tools to solve complex problems

Be helpful, accurate, and efficient!
"""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        # Create the executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            max_execution_time=300,  # 5 minutes max
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        return agent_executor

    async def execute(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        use_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a query using the agent.

        Args:
            query: User query
            conversation_id: Optional conversation ID for history
            use_history: Whether to use conversation history

        Returns:
            Dict with output, intermediate_steps, and metadata
        """
        try:
            # Prepare chat history
            chat_history = []
            if use_history and self.conversation_history:
                for msg in self.conversation_history[-10:]:  # Last 10 messages
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

            # Execute the agent
            result = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": chat_history,
            })

            # Extract output
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Update conversation history
            if use_history:
                self.conversation_history.append({
                    "role": "user",
                    "content": query,
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": output,
                })

            # Format intermediate steps for frontend
            formatted_steps = []
            for step in intermediate_steps:
                action, observation = step
                formatted_steps.append({
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": str(observation),
                })

            return {
                "success": True,
                "output": output,
                "intermediate_steps": formatted_steps,
                "tools_used": list(set([s["tool"] for s in formatted_steps])),
                "num_steps": len(formatted_steps),
                "conversation_id": conversation_id,
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "output": f"❌ Agent execution failed: {str(e)}",
                "intermediate_steps": [],
            }

    def execute_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for execute"""
        return asyncio.run(self.execute(query, **kwargs))

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history

    def update_rag_instance(self, rag_instance: Any, domain: str):
        """Update RAG instance and domain"""
        self.rag_instance = rag_instance
        self.domain = domain

        # Update RAG tool
        for tool in self.tools:
            if isinstance(tool, RAGTool):
                tool.set_rag_instance(rag_instance)
                tool.set_domain(domain)

    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get descriptions of all available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self.tools
        ]


# Streaming version for real-time responses
class StreamingAgenticRAGExecutor(AgenticRAGExecutor):
    """
    Streaming version of the agentic RAG executor.
    """

    async def execute_stream(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        use_history: bool = True,
    ):
        """
        Execute a query with streaming output.

        Yields intermediate results as they become available.
        """
        try:
            # Prepare chat history
            chat_history = []
            if use_history and self.conversation_history:
                for msg in self.conversation_history[-10:]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

            # Yield start event
            yield {
                "type": "start",
                "message": "Agent execution started",
                "query": query,
            }

            # Stream agent execution
            intermediate_steps = []
            async for chunk in self.agent_executor.astream({
                "input": query,
                "chat_history": chat_history,
            }):
                # Check for intermediate steps
                if "intermediate_step" in chunk:
                    action, observation = chunk["intermediate_step"]
                    step_info = {
                        "type": "tool_call",
                        "tool": action.tool,
                        "tool_input": action.tool_input,
                        "observation": str(observation),
                    }
                    intermediate_steps.append(step_info)
                    yield step_info

                # Check for output chunks
                if "output" in chunk:
                    yield {
                        "type": "output",
                        "content": chunk["output"],
                    }

            # Yield completion event
            yield {
                "type": "complete",
                "message": "Agent execution completed",
                "num_steps": len(intermediate_steps),
            }

        except Exception as e:
            import traceback
            yield {
                "type": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


# Utility function to create executor
def create_agentic_rag_executor(
    gemini_api_key: str,
    rag_instance: Optional[Any] = None,
    domain: str = "general",
    streaming: bool = False,
    **kwargs,
) -> AgenticRAGExecutor:
    """
    Factory function to create agentic RAG executor.

    Args:
        gemini_api_key: Google Gemini API key
        rag_instance: Reference to RAG instance
        domain: Current domain
        streaming: Whether to create streaming executor
        **kwargs: Additional arguments

    Returns:
        AgenticRAGExecutor instance
    """
    if streaming:
        return StreamingAgenticRAGExecutor(
            gemini_api_key=gemini_api_key,
            rag_instance=rag_instance,
            domain=domain,
            **kwargs,
        )
    else:
        return AgenticRAGExecutor(
            gemini_api_key=gemini_api_key,
            rag_instance=rag_instance,
            domain=domain,
            **kwargs,
        )
