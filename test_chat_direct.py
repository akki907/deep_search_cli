import asyncio
from unittest.mock import patch
from react_loop.chat import run_chat, ChatSession
from react_loop.runner import ReActRunner
from react_loop.llm import ScriptedChatModel
from react_loop.demo import DEMO_SCRIPT

async def test_run_chat():
    # Setup
    llm = ScriptedChatModel(script=list(DEMO_SCRIPT))
    runner = ReActRunner(llm)
    session = ChatSession(runner)
    
    # Mock read_line to simulate user input
    inputs = ["What is 2+2?", "/help", "/exit"]
    async def mock_read_line(prompt):
        if not inputs:
            raise EOFError()
        return inputs.pop(0)

    # Mock turn to simulate agent response
    async def mock_turn(sess, text):
        print(f"Turn called with: {text}")
        if text == "What is 2+2?":
            # In a real scenario, session.ask(text) would be called
            pass

    # Capture print output
    import sys
    import io
    captured = io.StringIO()
    
    with patch('sys.stdout', captured):
        code = await run_chat(
            session,
            read_line=mock_read_line,
            turn=mock_turn,
            write=lambda x: print(x)
        )
    
    print(f"Exit code: {code}")
    print("Captured output:\n", captured.getvalue())

if __name__ == "__main__":
    from unittest.mock import patch
    asyncio.run(test_run_chat())
