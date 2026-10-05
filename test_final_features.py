import asyncio
from react_loop.runner import ReActRunner
from react_loop.llm import build_llm
from react_loop.tools import ALL_TOOLS
from langchain_core.messages import AIMessage, ToolMessage

async def test_hitl():
    print("--- Testing HITL Approval ---")
    llm = build_llm(provider="ollama") # Using ollama as default
    runner = ReActRunner(llm)
    
    async def approval_callback(tool_calls):
        for tc in tool_calls:
            print(f"Requesting approval for: {tc['name']}({tc['args']})")
        return "approve"

    result = await runner.run_async(
        "What is 15 * 3?", 
        approval_callback=approval_callback
    )
    print(f"Final Answer: {result['messages'][-1].content}")

async def test_hitl_edit():
    print("\n--- Testing HITL Edit ---")
    llm = build_llm(provider="ollama")
    runner = ReActRunner(llm)
    
    async def edit_callback(tool_calls):
        print(f"Editing tool call: {tool_calls[0]['name']}")
        # Edit 15*3 to 15*10
        new_calls = [{**tool_calls[0], "args": {"expression": "15 * 10"}}]
        return new_calls

    result = await runner.run_async(
        "What is 15 * 3?", 
        approval_callback=edit_callback
    )
    print(f"Final Answer: {result['messages'][-1].content}")

async def test_delegation():
    print("\n--- Testing Multi-Agent Delegation ---")
    llm = build_llm(provider="ollama")
    runner = ReActRunner(llm)
    
    # We ask the agent to delegate
    result = await runner.run_async(
        "Use the delegate tool to ask a researcher about the capital of France"
    )
    print(f"Final Answer: {result['messages'][-1].content}")

async def main():
    await test_hitl()
    await test_hitl_edit()
    await test_delegation()

if __name__ == "__main__":
    asyncio.run(main())
