import io
from unittest.mock import patch, MagicMock
from react_loop.__main__ import main
from react_loop.llm import ScriptedChatModel
from react_loop.demo import DEMO_SCRIPT
from langchain_core.messages import AIMessage

def test_deep_research_flag():
    print("Testing --deep-research flag...")
    
    # We want to verify that the deep research prompt is passed to the runner.
    # Since we don't have a real LLM that reacts to system prompts in a predictable way,
    # we'll mock the ReActRunner to check the system_prompt it was initialized with.
    
    with patch('react_loop.runner.ReActRunner') as MockRunner, \
         patch('react_loop.__main__.build_llm') as mock_build:
        
        mock_build.return_value = MagicMock()
        # Mock the run method to return a dummy result
        MockRunner.return_value.run.return_value = {"messages": [], "steps": 0}
        
        # Run with --deep-research
        try:
            main(["--deep-research", "Quantum Computing"])
        except SystemExit:
            pass
        
        # Check if ReActRunner was initialized with the correct system prompt
        args, kwargs = MockRunner.call_args
        system_prompt = kwargs.get('system_prompt', '')
        assert "professional deep research agent" in system_prompt
        print("✅ --deep-research flag correctly sets system prompt.")

def test_deep_research_command():
    print("Testing /research command...")
    
    # We'll simulate a chat session and check if the turn function receives 
    # the augmented research text.
    
    user_inputs = ["/research Artificial Intelligence", "/exit"]
    
    async def mock_prompt_async(*args, **kwargs):
        if not user_inputs:
            return "/exit"
        return user_inputs.pop(0)

    captured_turns = []
    async def mock_turn(session, text):
        captured_turns.append(text)

    with patch('sys.stdout', io.StringIO()), \
         patch('sys.stdin.isatty', return_value=True), \
         patch('prompt_toolkit.PromptSession.prompt_async', side_effect=mock_prompt_async), \
         patch('react_loop.__main__.build_llm') as mock_build:
        
        mock_build.return_value = ScriptedChatModel(script=list(DEMO_SCRIPT))
        
        # We need to patch the actual turn function used in _run_chat
        # Since _run_chat defines turn internally, we have to be clever.
        # Alternatively, we can just check the output of the turn if we had a way.
        # But since we're testing the logic in chat.py, we can just test run_chat.
        
        from react_loop.chat import run_chat, ChatSession
        from react_loop.runner import ReActRunner
        
        runner = ReActRunner(mock_build.return_value)
        session = ChatSession(runner)
        
        import asyncio
        asyncio.run(run_chat(
            session,
            read_line=mock_prompt_async,
            turn=mock_turn
        ))
    
    # Verify the text passed to turn() includes the research prompt
    last_text = captured_turns[0]
    assert "[RESEARCH MODE]" in last_text
    assert "professional deep research agent" in last_text
    assert "Artificial Intelligence" in last_text
    print("✅ /research command correctly augments input text.")

if __name__ == "__main__":
    test_deep_research_flag()
    test_deep_research_command()
    print("=" * 40)
    print("Deep Research Tests Passed!")
