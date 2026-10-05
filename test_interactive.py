import io
from unittest.mock import patch
from react_loop.__main__ import main
from react_loop.llm import ScriptedChatModel
from react_loop.demo import DEMO_SCRIPT

def test_cli():
    class MockStdin(io.StringIO):
        def isatty(self):
            return True

    input_text = "What is 2+2?\n/help\n/exit\n"
    mock_stdin = MockStdin(input_text)

    with patch('sys.stdin', mock_stdin), \
         patch('sys.stdout', new=io.StringIO()) as mock_stdout:
        
        with patch('prompt_toolkit.PromptSession.prompt_async') as mock_prompt:
            mock_prompt.side_effect = ["What is 2+2?", "/help", "/exit"]
            
            with patch('react_loop.__main__.build_llm') as mock_build:
                mock_build.return_value = ScriptedChatModel(script=list(DEMO_SCRIPT))
                
                try:
                    main(["--plain"])
                except SystemExit:
                    pass
                
                print("--- CAPTURED OUTPUT ---")
                print(mock_stdout.getvalue())
                print("--- END OUTPUT ---")

if __name__ == "__main__":
    test_cli()
