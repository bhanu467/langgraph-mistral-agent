"""
Test script for LangGraph + Mistral Agent
This script helps validate the agent implementation and generate screenshots

Usage: python test_agent.py
"""

import sys
import time
from datetime import datetime
from agent_graph import MistralAgent

def print_separator(title=""):
    """Print a formatted separator"""
    print("\n" + "="*60)
    if title:
        print(f" {title} ")
        print("="*60)
    else:
        print()

def test_routing_logic():
    """Test the routing logic with various inputs"""
    print_separator("ROUTING LOGIC TEST")
    
    test_cases = [
        # Math queries
        ("15 + 25 * 3", "math"),
        ("Calculate 100 / 5", "math"),
        ("What is 2 + 2?", "math"),
        ("Solve this equation: x = 10 * 5", "math"),
        
        # Summary queries
        ("Summarize: This is a test document", "summarizer"),
        ("Please summarize the following text", "summarizer"),
        ("Sum up this information", "summarizer"),
        
        # General queries
        ("Hello, how are you?", "fallback"),
        ("What's the weather like?", "fallback"),
        ("Tell me a joke", "fallback"),
    ]
    
    print("Testing routing decisions...")
    
    try:
        agent = MistralAgent()
        
        for i, (query, expected_route) in enumerate(test_cases, 1):
            print(f"\n--- Test {i} ---")
            print(f"Input: {query}")
            print(f"Expected Route: {expected_route}")
            
            # Test just the router node
            state = {"input": query, "output": "", "next": "", "processing_type": ""}
            result_state = agent.router_node(state)
            actual_route = result_state.get("next", "unknown")
            
            print(f"Actual Route: {actual_route}")
            print(f"Status: {'✅ PASS' if actual_route == expected_route else '❌ FAIL'}")
            
    except Exception as e:
        print(f"❌ Error during routing test: {e}")
        return False
    
    print_separator()
    return True

def test_full_execution():
    """Test full agent execution with screenshot-worthy examples"""
    print_separator("FULL EXECUTION TEST")
    
    test_cases = [
        {
            "name": "Math Problem",
            "input": "What is 15 + 25 * 3?",
            "description": "Testing mathematical problem solving"
        },
        {
            "name": "Text Summary",
            "input": "Summarize: LangGraph is a powerful tool for building agent workflows with non-linear processing capabilities. It enables developers to create sophisticated multi-agent systems.",
            "description": "Testing text summarization"
        },
        {
            "name": "General Query",
            "input": "Hello, how are you today?",
            "description": "Testing general conversation handling"
        }
    ]
    
    try:
        agent = MistralAgent()
        
        for i, test_case in enumerate(test_cases, 1):
            print_separator(f"TEST CASE {i}: {test_case['name']}")
            print(f"Description: {test_case['description']}")
            print(f"Input: {test_case['input']}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\nExecuting agent...")
            start_time = time.time()
            
            result = agent.run(test_case['input'])
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"\nExecution Time: {execution_time:.2f} seconds")
            print(f"Route Taken: {result.get('next', 'Unknown')}")
            print(f"Processing Type: {result.get('processing_type', 'Unknown')}")
            
            # Save results for documentation
            with open(f"test_result_{i}.txt", "w", encoding='utf-8') as f:
                f.write(f"Test Case: {test_case['name']}\n")
                f.write(f"Input: {test_case['input']}\n")
                f.write(f"Route: {result.get('next', 'Unknown')}\n")
                f.write(f"Processing Type: {result.get('processing_type', 'Unknown')}\n")
                f.write(f"Execution Time: {execution_time:.2f}s\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output: {result.get('output', 'No output')}\n")
            
            print(f"Results saved to test_result_{i}.txt")
            
            # Pause between tests
            if i < len(test_cases):
                print("\nPress Enter to continue to next test...")
                input()
            
    except Exception as e:
        print(f"❌ Error during full execution test: {e}")
        return False
    
    print_separator("ALL TESTS COMPLETED")
    return True

def performance_test():
    """Test agent performance with multiple queries"""
    print_separator("PERFORMANCE TEST")
    
    queries = [
        "Calculate 50 + 30 * 2",
        "Summarize: AI agents are becoming increasingly sophisticated",
        "What's your favorite color?",
        "Solve: 100 / 25 + 15",
        "Sum up: Technology is rapidly evolving"
    ]
    
    try:
        agent = MistralAgent()
        total_time = 0
        
        print("Running performance test with 5 queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            
            start_time = time.time()
            result = agent.run(query)
            end_time = time.time()
            
            execution_time = end_time - start_time
            total_time += execution_time
            
            print(f"Time: {execution_time:.2f}s | Route: {result.get('next', 'Unknown')}")
        
        avg_time = total_time / len(queries)
        print(f"\nPerformance Summary:")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Time: {avg_time:.2f}s")
        print(f"Queries per minute: {60/avg_time:.1f}")
        
    except Exception as e:
        print(f"❌ Error during performance test: {e}")
        return False
    
    return True

def main():
    """Main test runner"""
    print("LangGraph + Mistral Agent Test Suite")
    print("=" * 60)
    print("This script will test the agent implementation and generate")
    print("results that can be used for screenshots and documentation.")
    print("=" * 60)
    
    # Check if user wants to run specific tests
    print("\nAvailable tests:")
    print("1. Routing Logic Test")
    print("2. Full Execution Test (for screenshots)")
    print("3. Performance Test")
    print("4. Run All Tests")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        test_routing_logic()
    elif choice == "2":
        test_full_execution()
    elif choice == "3":
        performance_test()
    elif choice == "4":
        print("\n🚀 Running all tests...")
        success = True
        success &= test_routing_logic()
        success &= test_full_execution()
        success &= performance_test()
        
        if success:
            print("\nAll tests completed successfully!")
        else:
            print("\nSome tests failed. Check the output above.")
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    print("\nScreenshot Tips:")
    print("=" * 60)
    print("For assignment submission, take screenshots of:")
    print("1. Math query routing to 'math' node")
    print("2. Summary query routing to 'summarizer' node")
    print("3. Final output showing the complete response")
    print("4. Terminal showing successful execution")
    
    print("\nFiles generated:")
    print("- test_result_1.txt (Math problem results)")
    print("- test_result_2.txt (Summary results)")
    print("- test_result_3.txt (General query results)")
    
    print("\nTesting complete! Ready for submission.")

if __name__ == "__main__":
    main()
