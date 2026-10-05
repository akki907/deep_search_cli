import asyncio
import io
from unittest.mock import patch, MagicMock
from react_loop.__main__ import main
from react_loop.llm import ScriptedChatModel
from react_loop.demo import DEMO_SCRIPT
from langchain_core.messages import AIMessage

def simulate_advanced_cli():
    # 1. Trigger calculator -> Approve
    # 2. Trigger calculator -> Edit
    # 3. Trigger calculator -> Cancel
    # 4. Trigger delegate -> Success
    # 5. Exit
    user_inputs = [
        "What is 15 * 3?", # triggers calculator
        "a",               # Approve
        "What is 10 * 2?", # triggers calculator
        "e",               # Edit
        '{"expression": "10 * 3"}', # New args
        "What is 5 * 5?",   # triggers calculator
        "c",               # Cancel
        "Delegate a research task to the researcher about the Eiffel Tower", # triggers delegate
        "/exit"
    ]
    
    class CustomScriptedModel(ScriptedChatModel):
        def _next_turn(self, messages):
            last_msg = messages[-1].content
            # Trigger calculator for math questions
            if any(op in last_msg for op in ["*", "+", "-", "/"]):
                from langchain_core.messages import AIMessage
                return AIMessage(
                    content="", 
                    tool_calls=[{"name": "calculator", "args": {"expression": "15 * 3"}, "id": "call_1"}]
                )
            # Trigger delegate for delegation requests
            if "delegate" in last_msg.lower():
                from langchain_core.messages import AIMessage
                return AIMessage(
                    content="", 
                    tool_calls=[{"name": "delegate", "args": {"agent_name": "researcher", "query": "Eiffel Tower"}, "id": "call_2"}]
                )
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
        
        mock_build.return_value = CustomScriptedModel(script=list(DEMO_SCRIPT))
        
        try:
            main(["--plain"])
        except SystemExit:
            pass
    
    return output.getvalue()

if __name__ == "__main__":
    # The main() function calls asyncio.run(), so we just wrap the simulation.
    # Since we need to run an async function (simulate_advanced_cli),
    # and main is sync, we can just call simulate_advanced_cli in a wrapper.
    import asyncio
    result = simulate_advanced_cli()
    print(result)
    print("--- END LOG ---")
