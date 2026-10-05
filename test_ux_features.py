import subprocess
import time
import os
import re
from typing import List

# Configuration
PYTHON_EXE = "/Users/akashkumar/workspace/personal/react_loop/.venv/bin/python"
CLI_CMD = [PYTHON_EXE, "-m", "react_loop"]
MODEL_FLAGS = ["--provider", "ollama", "--model", "qwen2.5-coder:7b", "--plain", "--force-interactive"]

def simulate_chat(inputs: List[str], timeout: int = 30):
    print(f"Running simulation with inputs: {inputs}")
    process = subprocess.Popen(
        CLI_CMD + MODEL_FLAGS,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    output = ""
    try:
        for inp in inputs:
            try:
                process.stdin.write(inp + "\n")
                process.stdin.flush()
            except BrokenPipeError:
                break
            
            # Wait for the output to contain either "you >" or a tool request
            start_time = time.time()
            while time.time() - start_time < 10:
                char = process.stdout.read(1)
                if not char: break
                output += char
                if "you > " in output or "Approve?" in output or "Step: Agent requested" in output:
                    break
        
        time.sleep(1)
        remaining = process.stdout.read()
        if remaining: output += remaining
    finally:
        process.terminate()
    
    return output

def test_persistence():
    print("\n--- Testing Session Persistence ---")
    out1 = simulate_chat(["Hello! I am a test user.", "/save", "/exit"])
    match = re.search(r"Session saved as ([a-f0-9]+)", out1)
    if not match:
        print(f"❌ Could not find thread_id\nOutput: {out1}")
        return False
    
    tid = match.group(1)
    out2 = simulate_chat([f"/load {tid}", "Who am I?", "/exit"])
    if "test user" in out2.lower():
        print("✅ Persistence PASS")
        return True
    print(f"❌ Persistence FAIL\nOutput: {out2}")
    return False

def test_markdown():
    print("\n--- Testing Markdown Rendering ---")
    # Use a prompt that almost certainly generates a table/list
    out = simulate_chat(["List 3 fruits and their colors in a markdown table.", "/exit"])
    if "|" in out or "**" in out:
        print("✅ Markdown PASS")
        return True
    print(f"❌ Markdown FAIL\nOutput: {out}")
    return False

def test_debugger():
    print("\n--- Testing Interactive Debugger ---")
    # Workflow: /debug -> Force tool -> 's' -> provide steering
    out = simulate_chat([
        "/debug", 
        "What is the 100th prime? Use python_executor.", 
        "s", 
        "The answer is 541.", 
        "/exit"
    ])
    if "🐞 Debug Mode" in out and "Steered" in out:
        print("✅ Debugger PASS")
        return True
    print(f"❌ Debugger FAIL\nOutput: {out}")
    return False

if __name__ == "__main__":
    p = test_persistence()
    m = test_markdown()
    d = test_debugger()
    print(f"\nSummary: {sum([p, m, d])}/3 passed.")
