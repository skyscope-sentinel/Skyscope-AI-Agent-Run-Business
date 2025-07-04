import os
from e2b_code_interpreter import CodeInterpreter, Result
from dotenv import load_dotenv

# Ensure project root is in path for sibling imports
import sys
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from skyscope_sentinel.config import global_config # Use the global_config

# Load .env for local development if API keys are needed
load_dotenv()

def execute_python_code_in_e2b(code: str, timeout: int = 300) -> dict:
    """
    Executes a given Python code string in a secure E2B cloud sandbox.

    Args:
        code (str): The Python code to execute.
        timeout (int): Timeout for the sandbox execution in seconds. Default is 300 seconds (5 minutes).

    Returns:
        dict: A dictionary containing:
              'stdout': The standard output from the code execution.
              'stderr': The standard error output (if any).
              'result_value': The value of the last expression if it's a result (e.g. for notebooks).
              'error': Error message if an exception occurred during execution or setup.
              'artifacts': List of artifacts generated (not fully supported in this basic version).
    """
    print(f"[Tool: E2BSecureExec] Attempting to execute Python code in E2B sandbox.")
    e2b_api_key = global_config.get_e2b_api_key() # Assumes E2B_API_KEY is now in config

    if not e2b_api_key:
        print("[Tool: E2BSecureExec] Error: E2B_API_KEY not found in config or environment.")
        return {
            "stdout": "",
            "stderr": "E2B_API_KEY not configured. Cannot execute code.",
            "result_value": None,
            "error": "E2B_API_KEY not configured.",
            "artifacts": []
        }

    try:
        # The sandbox will be automatically closed when the 'with' block exits.
        # Timeout for the sandbox session itself can be set if E2B supports it directly in constructor or run.
        # The exec_cell timeout is for that specific execution.
        with CodeInterpreter(api_key=e2b_api_key) as sandbox:
            print(f"[Tool: E2BSecureExec] E2B Sandbox started. Executing code (timeout: {timeout}s)...")
            execution: Result = sandbox.notebook.exec_cell(code, timeout=timeout) # timeout in seconds for e2b

            output = {
                "stdout": "\n".join(out.text for out in execution.logs.stdout),
                "stderr": "\n".join(out.text for out in execution.logs.stderr),
                "result_value": execution.text if execution.text else None, # .text is the result of the last expression
                "error": execution.error.name if execution.error else None,
                "artifacts": [] # Basic version, artifact handling can be expanded
            }

            if execution.error:
                output["stderr"] = (output["stderr"] + "\n" + execution.error.traceback_raw).strip()
                print(f"[Tool: E2BSecureExec] Execution error: {execution.error.name} - {execution.error.value}")

            # Example for artifacts if they were generated and accessible:
            # for artifact in execution.artifacts:
            #    output["artifacts"].append({"name": artifact.name, "size": artifact.size, "content_base64": base64.b64encode(artifact.download()).decode()})

            print(f"[Tool: E2BSecureExec] E2B Execution finished. Stdout: {len(output['stdout'])} chars, Stderr: {len(output['stderr'])} chars.")
            return output

    except Exception as e:
        print(f"[Tool: E2BSecureExec] General error during E2B execution: {e}")
        return {
            "stdout": "",
            "stderr": str(e),
            "result_value": None,
            "error": f"E2B execution failed: {str(e)}",
            "artifacts": []
        }

if __name__ == '__main__':
    print("--- Testing E2B Secure Code Execution Tool ---")
    # This test requires an E2B_API_KEY to be set in your .env file or environment
    # and `global_config` to pick it up (or `Config` to load it directly for this test).

    # Ensure global_config is updated for this standalone test if main.py hasn't run
    # This is a bit of a hack for standalone module testing.
    # In the full app, main.py handles SettingsManager -> global_config update.
    if not global_config.get_e2b_api_key():
         print("Attempting to load E2B_API_KEY from .env for standalone test...")
         # Temporarily set it if found in env for the purpose of this test
         manual_e2b_key = os.getenv("E2B_API_KEY")
         if manual_e2b_key:
             global_config.current_e2b_api_key = manual_e2b_key # Directly set for test
             print("E2B_API_KEY loaded for test.")
         else:
             print("WARNING: E2B_API_KEY not found in environment. E2B tests will fail to connect.")


    code_example_1 = "print('Hello from E2B sandbox!')\nimport sys\nprint(f'Python version: {sys.version_info}')\na = 10 + 5\na"
    print(f"\nExecuting code:\n{code_example_1}")
    result_1 = execute_python_code_in_e2b(code_example_1)
    print(f"Result 1:\n{result_1}")

    code_example_2_error = "print(x)\nprint('This should not print if x is not defined')"
    print(f"\nExecuting code with an error:\n{code_example_2_error}")
    result_2 = execute_python_code_in_e2b(code_example_2_error)
    print(f"Result 2:\n{result_2}")

    code_example_3_long = "import time\nprint('Starting long task...')\ntime.sleep(2)\nprint('Long task finished.')\n'Done'"
    print(f"\nExecuting potentially longer code (timeout 5s):\n{code_example_3_long}")
    result_3 = execute_python_code_in_e2b(code_example_3_long, timeout=5)
    print(f"Result 3:\n{result_3}")

    # Example of a timeout
    # code_example_4_timeout = "import time\nprint('Starting task that will timeout...')\ntime.sleep(10)\nprint('This should not be reached.')"
    # print(f"\nExecuting code designed to timeout (timeout 2s):\n{code_example_4_timeout}")
    # result_4 = execute_python_code_in_e2b(code_example_4_timeout, timeout=2)
    # print(f"Result 4:\n{result_4}")


    print("\n--- E2B Code Execution Tool Test Complete ---")
    print("Note: Requires E2B_API_KEY environment variable to be set for successful execution.")
