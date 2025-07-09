"""
LangGraph + Mistral Non-Linear Agent Implementation
Built for Lumina Engineering Team Assignment

This agent implements a non-linear graph with routing capabilities:
- Router Node: Analyzes input and decides processing path
- Math Solver Node: Handles mathematical operations
- Text Summarizer Node: Summarizes text content  
- Final Printer Node: Outputs results

Author: G.THIRUMALA HEMANT KUMAR
Date: 08-07-2025
"""

from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from typing import Dict, Any, Optional
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralAgent:
    """Non-linear agent using LangGraph and Mistral via Ollama"""
    
    def __init__(self, model_name: str = "mistral"):
        """Initialize the agent with Mistral model"""
        try:
            self.llm = Ollama(model=model_name)
            logger.info(f"Successfully initialized {model_name} model")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the non-linear agent graph"""
        # Create state graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("math", self.math_node)
        workflow.add_node("summarizer", self.summarizer_node)
        workflow.add_node("final", self.final_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self.decide_next_node,
            {
                "math": "math",
                "summarizer": "summarizer", 
                "fallback": "final"
            }
        )
        
        # Add edges to final node
        workflow.add_edge("math", "final")
        workflow.add_edge("summarizer", "final")
        workflow.add_edge("final", END)
        
        return workflow.compile()
    
    def router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Router node that analyzes input and determines processing path
        
        Routing Logic:
        - Contains math operations (+, -, *, /, calculate) -> math node
        - Contains 'summarize' keyword -> summarizer node
        - Everything else -> fallback to final node
        """
        user_input = state.get("input", "").lower()
        
        # Check for math operations
        math_patterns = ['+', '-', '*', '/', 'calculate', 'solve', 'math', 'equation']
        if any(pattern in user_input for pattern in math_patterns):
            state["next"] = "math"
            state["processing_type"] = "mathematical"
            logger.info("Router: Directing to math node")
        
        # Check for summarization request
        elif any(keyword in user_input for keyword in ['summarize', 'summary', 'sum up']):
            state["next"] = "summarizer"
            state["processing_type"] = "summarization"
            logger.info("Router: Directing to summarizer node")
        
        # Fallback for general queries
        else:
            state["next"] = "fallback"
            state["processing_type"] = "general"
            logger.info("Router: Directing to fallback (final node)")
        
        return state
    
    def decide_next_node(self, state: Dict[str, Any]) -> str:
        """Decision function for conditional edges"""
        return state.get("next", "fallback")
    
    def math_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Math solver node that handles mathematical operations
        Uses Mistral to solve mathematical problems
        """
        user_input = state.get("input", "")
        
        # Create math-focused prompt
        math_prompt = f"""
        You are a mathematical problem solver. Solve the following problem step by step:
        
        Problem: {user_input}
        
        Please provide:
        1. The calculation steps
        2. The final answer
        
        Be precise and show your work clearly.
        """
        
        try:
            # Get response from Mistral
            response = self.llm.invoke(math_prompt)
            state["output"] = f"[MATH] Math Solution:\n{response}"
            logger.info("Math node: Successfully processed mathematical query")
            
        except Exception as e:
            state["output"] = f"[ERROR] Math Error: Could not solve the problem. Error: {str(e)}"
            logger.error(f"Math node error: {e}")
        
        return state
    
    def summarizer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Text summarizer node that creates concise summaries
        Uses Mistral to summarize text content
        """
        user_input = state.get("input", "")
        
        # Extract text to summarize (remove "summarize:" prefix if present)
        text_to_summarize = re.sub(r'^summarize:?\s*', '', user_input, flags=re.IGNORECASE)
        
        # Create summarization prompt
        summary_prompt = f"""
        You are a text summarization expert. Create a concise and informative summary of the following text:
        
        Text to summarize: {text_to_summarize}
        
        Please provide:
        1. A brief summary (2-3 sentences)
        2. Key points if applicable
        
        Keep it clear and concise.
        """
        
        try:
            # Get response from Mistral
            response = self.llm.invoke(summary_prompt)
            state["output"] = f"[SUMMARY] Summary:\n{response}"
            logger.info("Summarizer node: Successfully processed summarization request")
            
        except Exception as e:
            state["output"] = f"[ERROR] Summary Error: Could not summarize the text. Error: {str(e)}"
            logger.error(f"Summarizer node error: {e}")
        
        return state
    
    def final_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final printer node that handles output and fallback cases
        """
        # If no output was generated (fallback case), handle general queries
        if not state.get("output"):
            user_input = state.get("input", "")
            
            general_prompt = f"""
            You are a helpful assistant. Please respond to the following query:
            
            Query: {user_input}
            
            Provide a helpful and informative response.
            """
            
            try:
                response = self.llm.invoke(general_prompt)
                state["output"] = f"[GENERAL] General Response:\n{response}"
                logger.info("Final node: Processed general query")
                
            except Exception as e:
                state["output"] = f"[ERROR] Error: Could not process your request. Error: {str(e)}"
                logger.error(f"Final node error: {e}")
        
        # Print the final output
        print("=" * 50)
        print("AGENT RESPONSE")
        print("=" * 50)
        print(state["output"])
        print("=" * 50)
        print(f"Processing Type: {state.get('processing_type', 'Unknown')}")
        print(f"Route Taken: {state.get('next', 'Unknown')}")
        print("=" * 50)
        
        return state
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Main method to run the agent with user input
        
        Args:
            user_input (str): The user's query
            
        Returns:
            Dict[str, Any]: The final state after processing
        """
        initial_state = {
            "input": user_input,
            "output": "",
            "next": "",
            "processing_type": ""
        }
        
        logger.info(f"Starting agent with input: {user_input}")
        
        try:
            result = self.graph.invoke(initial_state)
            logger.info("Agent execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "input": user_input,
                "output": f"[ERROR] Agent Error: {str(e)}",
                "next": "error",
                "processing_type": "error"
            }

def main():
    """Main function to demonstrate the agent"""
    print("LangGraph + Mistral Non-Linear Agent")
    print("=" * 50)
    
    # Initialize agent
    try:
        agent = MistralAgent()
        print("Agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return
    
    # Test cases
    test_cases = [
        "What is 15 + 25 * 3?",
        "Summarize: how is earth formed?",
        "Hello, how are you doing today?"
    ]
    
    print("\nRunning Test Cases:")
    print("=" * 50)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_input}")
        print("\nProcessing...")
        
        result = agent.run(test_input)
        
        print(f"\nRouting Decision: {result.get('next', 'Unknown')}")
        print(f"Processing Type: {result.get('processing_type', 'Unknown')}")
        
        print("\n" + "="*50)
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nEnter your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter a valid query.")
                continue
            
            agent.run(user_input)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
