import subprocess
import time
import os
from typing import List, Tuple

# Configuration
PYTHON_EXE = "/Users/akashkumar/workspace/personal/react_loop/.venv/bin/python"
CLI_CMD = [PYTHON_EXE, "-m", "react_loop"]
MODEL_FLAGS = ["--provider", "ollama", "--model", "qwen2.5-coder:7b", "--plain", "--force-interactive"]

def run_interactive_test(question: str, expected_substring: str, approvals: List[str]) -> Tuple[bool, str]:
    print(f"\nTesting: {question}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        CLI_CMD + MODEL_FLAGS,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env
    )

    output = ""
    approval_idx = 0
    
    try:
        # Send the question
        process.stdin.write(question + "\n")
        process.stdin.flush()

        start_time = time.time()
        while time.time() - start_time < 30: # 30s timeout
            if process.poll() is not None:
                break
                
            char = process.stdout.read(1)
            if not char:
                time.sleep(0.1)
                continue
            output += char
            
            if "Approve?" in output and approval_idx < len(approvals):
                process.stdin.write(approvals[approval_idx] + "\n")
                process.stdin.flush()
                approval_idx += 1
                # Clear the prompt from buffer to avoid double trigger
                output = output[output.find("Approve?") + 10:]

            if "Final Answer:" in output or "assistant>" in output:
                # Give it a moment to finish the answer
                time.sleep(1)
                break
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        process.terminate()
        try:
            # Read remaining output
            remaining = process.stdout.read()
            if remaining:
                output += remaining
        except:
            pass

    if expected_substring in output:
        return True, output
    else:
        return False, output

def main():
    test_cases = [
        (
            "What is 2 + 2?", 
            "4", 
            []
        ),
        (
            "Find the sum of primes between 1 and 10 using Python.", 
            "17", 
            ["y"]
        ),
        (
            "Calculate 2**10 using Python.", 
            "1024", 
            ["y"]
        )
    ]

    passed = 0
    for q, exp, apps in test_cases:
        success, res = run_interactive_test(q, exp, apps)
        if success:
            print(f"✅ PASS: {q}")
            passed += 1
        else:
            print(f"❌ FAIL: {q}\nOutput:\n{res}")

    print(f"\nSummary: {passed}/{len(test_cases)} passed.")

if __name__ == "__main__":
    main()
