# langgraph-mistral-agent


LangGraph + Mistral Non-Linear Agent


A sophisticated non-linear agent implementation using LangGraph orchestration framework with Mistral 7B model via Ollama.

🎯 Features


Non-linear Graph Architecture: Intelligent routing between specialized nodes
Mathematical Problem Solving: Dedicated math solver node for calculations
Text Summarization: Advanced summarization capabilities
Fallback Handling: Graceful handling of general queries
Comprehensive Logging: Detailed execution tracking
Interactive Mode: Real-time testing and interaction

🏗️ Architecture


[User Input] → [Router Node] → [Math/Summary/Fallback] → [Final Output Node]

Node Description:
Router Node: Analyzes input and determines processing path
Math Node: Handles mathematical operations using Mistral
Summarizer Node: Creates concise text summaries
Final Node: Outputs results and handles fallback cases


🚀 Quick Start


Prerequisites

Python 3.10+
Ollama installed and running
Mistral model downloaded

Installation

Install Ollama:
bash# Visit https://ollama.com/download for your OS
# After installation:
ollama pull mistral

Set up Python environment:
bashpython3 -m venv langgraph_env
source langgraph_env/bin/activate  # Windows: langgraph_env\Scripts\activate

Install dependencies:
bashpip install -r requirements.txt


Usage

Run the agent:
python agent_graph.py

Test with sample queries:

Math: "What is 15 + 25 * 3?"
Summary: "Summarize: LangGraph is a powerful tool..."
General: "Hello, how are you?"


📁 Project Structure


Agent/
├── agent_graph.py      
├── test_agent.py        
├── requirements.txt     
├── README.md          
├── test_result_1.txt     
├── test_result_2.txt     
├── test_result_3.txt      
└── screenshots/           
    ├── math_routing.png
    ├── summary_routing.png
    └── terminal_output.png


Common Issues

Ollama Connection Error:

Ensure Ollama service is running
Check if Mistral model is downloaded: ollama list


Import Errors:

Verify virtual environment activation
Reinstall packages: pip install -r requirements.txt


Model Loading Issues:

Check available models: ollama list
Pull Mistral again: ollama pull mistral
