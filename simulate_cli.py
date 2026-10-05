import io
import asyncio
from unittest.mock import patch
from react_loop.__main__ import main
from react_loop.llm import ScriptedChatModel
from react_loop.demo import DEMO_SCRIPT
from langchain_core.messages import AIMessage

def run_simulation(scenarios):
    """Run a specific set of inputs and return the captured output."""
    user_inputs = scenarios[:]
    
    class ComprehensiveMockModel(ScriptedChatModel):
        def _next_turn(self, messages):
            last_msg = messages[-1].content.lower()
            
            # 1. Test HITL/Tool use: Math triggers calculator
            if any(op in last_msg for op in ["*", "+", "-", "/"]):
                return AIMessage(
                    content="", 
                    tool_calls=[{"name": "calculator", "args": {"expression": "15 * 3"}, "id": "call_math"}]
                )
            
            # 2. Test Delegation: "delegate" triggers delegation tool
            if "delegate" in last_msg:
                return AIMessage(
                    content="", 
                    tool_calls=[{"name": "delegate", "args": {"agent_name": "researcher", "query": "Eiffel Tower"}, "id": "call_del"}]
                )
            
            # 3. Test Memory: "what did I just ask"
            if "just ask" in last_msg:
                return AIMessage(content="You just asked me about a math problem.")
            
            # Default to demo script
            return super()._next_turn(messages)

    output = io.StringIO()
    
    async def mock_prompt_async(*args, **kwargs):
        if not user_inputs:
            return "/exit"
        return user_inputs.pop(0)

    with patch('sys.stdout', output), \
         patch('sys.stdin.isatty', return_value=True), \
         patch('prompt_toolkit.PromptSession.prompt_async', side_effect=mock_prompt_async), \
         patch('react_loop.__main__.build_llm') as mock_build:
        
        mock_build.return_value = ComprehensiveMockModel(script=list(DEMO_SCRIPT))
        
        try:
            main(["--plain"])
        except SystemExit:
            pass
    
    return output.getvalue()

if __name__ == "__main__":
    # Scenario 1: The "Happy Path" with HITL Approval
    print("\n--- Scenario 1: HITL Approval ---")
    s1_inputs = ["What is 15 * 3?", "a", "/exit"]
    print(run_simulation(s1_inputs))

    # Scenario 2: HITL Editing
    print("\n--- Scenario 2: HITL Edit ---")
    s2_inputs = ["What is 10 * 2?", "e", '{"expression": "10 * 10"}', "/exit"]
    print(run_simulation(s2_inputs))

    # Scenario 3: HITL Cancellation
    print("\n--- Scenario 3: HITL Cancel ---")
    s3_inputs = ["What is 5 * 5?", "c", "/exit"]
    print(run_simulation(s3_inputs))

    # Scenario 4: Multi-Agent Delegation
    print("\n--- Scenario 4: Delegation ---")
    s4_inputs = ["Please delegate a research task to the researcher about the Eiffel Tower", "a", "/exit"]
    print(run_simulation(s4_inputs))

    # Scenario 5: Session Memory & Commands
    print("\n--- Scenario 5: Memory & Commands ---")
    s5_inputs = ["Hello!", "What did I just ask?", "/help", "/clear", "Hi again", "/exit"]
    print(run_simulation(s5_inputs))

    # Scenario 6: Deep Research command
    print("\n--- Scenario 6: Deep Research ---")
    s6_inputs = ["/research Quantum Computing", "a", "/exit"]
    print(run_simulation(s6_inputs))

    print("\n" + "="*40)
    print("All Simulation Scenarios Complete.")
    print("="*40)
