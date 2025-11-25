"""
Test script for the Agentic RAG system
"""

import requests
import json
import sys


def test_agent_tools():
    """Test getting available tools"""
    print("📋 Testing: Get Available Tools")
    print("-" * 50)

    try:
        response = requests.get("http://localhost:8000/agent/tools?domain=general")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {data['count']} tools:")
            for tool in data['tools']:
                print(f"  - {tool['name']}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the backend running?")
        print("   Start with: cd backend && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_agent_query(query: str, description: str):
    """Test an agent query"""
    print(f"\n🤖 Testing: {description}")
    print("-" * 50)
    print(f"Query: {query}")
    print()

    try:
        response = requests.post(
            "http://localhost:8000/agent/query",
            json={
                "query": query,
                "domain": "general",
                "return_steps": True
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()

            if data['success']:
                print("✅ Success!")
                print(f"\n📤 Output:")
                print(data['output'])
                print(f"\n🛠️  Tools used: {', '.join(data['tools_used'])}")
                print(f"📊 Steps taken: {data['num_steps']}")

                if data['intermediate_steps']:
                    print(f"\n🔍 Intermediate steps:")
                    for i, step in enumerate(data['intermediate_steps'], 1):
                        print(f"  {i}. Tool: {step['tool']}")
                        print(f"     Input: {step['tool_input'][:100]}...")

                return True
            else:
                print(f"❌ Agent failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timeout (>60s)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("🚀 Agentic RAG System - Test Suite")
    print("=" * 50 + "\n")

    results = []

    # Test 1: Get tools
    results.append(("Get Tools", test_agent_tools()))

    # Test 2: Simple math query
    results.append((
        "Mathematical Analysis",
        test_agent_query(
            "differentiate x^2 + 3*x + 2 with respect to x",
            "Mathematical Analysis"
        )
    ))

    # Test 3: Code execution
    results.append((
        "Code Execution",
        test_agent_query(
            "calculate the factorial of 10 using Python",
            "Code Execution"
        )
    ))

    # Test 4: Data analysis
    results.append((
        "Data Analysis",
        test_agent_query(
            "create a sample dataset with 5 numbers and calculate the mean and standard deviation",
            "Data Analysis"
        )
    ))

    # Test 5: Visualization
    results.append((
        "Visualization",
        test_agent_query(
            "create a line plot of values [1, 4, 9, 16, 25] with title 'Square Numbers'",
            "Visualization"
        )
    ))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The agentic RAG system is working perfectly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
