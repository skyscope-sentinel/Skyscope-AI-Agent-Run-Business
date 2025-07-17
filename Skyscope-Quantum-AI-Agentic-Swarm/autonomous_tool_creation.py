import os
import sys
import json
import subprocess
import tempfile
import importlib.util
from typing import Any, Dict, List, Optional, Tuple, Callable

# Mock imports from the project structure for standalone functionality
# In the actual project, these would be real imports.
class Agent:
    def __init__(self, system_prompt: str, model_name: str = "gpt-4o"):
        self.system_prompt = system_prompt
        self.model_name = model_name

    def run(self, task: str, **kwargs) -> str:
        # This is a mock response for generating a tool.
        # A real implementation would call an LLM.
        print(f"Agent running task: {task[:100]}...")
        if "create a tool" in task.lower() and "add two numbers" in task.lower():
            return """
import sys
import json

def add_two_numbers(a: int, b: int) -> int:
    \"\"\"
    Adds two numbers together and returns the result.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The sum of the two numbers.
    \"\"\"
    return a + b

# --- Boilerplate for sandboxed execution ---
if __name__ == '__main__':
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        # Call the function with the provided arguments
        result = add_two_numbers(**input_data)
        # Write output to stdout
        json.dump({"result": result}, sys.stdout)
    except Exception as e:
        # Write error to stderr
        json.dump({"error": str(e)}, sys.stderr)
        sys.exit(1)
"""
        return "def new_tool():\n    pass"

class ToolRegistry:
    def __init__(self):
        self.tools = {}
    def register_tool(self, name: str, func: Callable, description: str, parameters: Optional[Dict[str, Any]] = None):
        print(f"Tool '{name}' registered successfully.")
        self.tools[name] = {"func": func, "description": description, "parameters": parameters}
    def get_tool(self, name: str):
        return self.tools.get(name)

class AgentManager:
    def __init__(self):
        self.tool_registry = ToolRegistry()
    def get_code_generation_agent(self):
        return Agent(system_prompt="You are a master Python programmer who writes single-function tools.")

# --- End of Mock Imports ---


class AutonomousToolCreator:
    """
    A class to autonomously generate, test, and integrate new tools for AI agents.
    """

    def __init__(self, agent_manager: AgentManager, tool_registry: ToolRegistry):
        """
        Initializes the AutonomousToolCreator.

        Args:
            agent_manager (AgentManager): The manager to access AI agents for code generation.
            tool_registry (ToolRegistry): The registry where new tools will be integrated.
        """
        self.agent_manager = agent_manager
        self.tool_registry = tool_registry
        self.code_gen_agent = self.agent_manager.get_code_generation_agent()

    def generate_tool_code(self, task_description: str, tool_name: str) -> Optional[str]:
        """
        Generates Python code for a new tool based on a task description.

        Args:
            task_description (str): A natural language description of what the tool should do.
            tool_name (str): The desired function name for the new tool.

        Returns:
            Optional[str]: The generated Python code as a string, or None if generation fails.
        """
        prompt = f"""
        You are an expert Python programmer. Your task is to create a tool as a single Python function.
        The function should be named `{tool_name}`.
        The tool must accomplish the following task: "{task_description}".

        RULES:
        1. The output must be a single, complete Python script.
        2. The script must contain only one function: `{tool_name}`.
        3. The function must have clear type hints for all arguments and the return value.
        4. The function must have a comprehensive docstring explaining what it does, its arguments, and what it returns.
        5. The script MUST include a boilerplate `if __name__ == '__main__':` block.
        6. This block must read keyword arguments from a single JSON object from `sys.stdin`.
        7. It must call the `{tool_name}` function with these arguments.
        8. It must write the function's return value as a JSON object `{{'result': ...}}` to `sys.stdout`.
        9. If an error occurs, it must write a JSON object `{{'error': '...'}}` to `sys.stderr` and exit with a non-zero status.
        10. Do not include any code outside of the function definition and the `if __name__ == '__main__':` block, except for necessary imports.
        11. The function should not have side effects like printing to the console or modifying files unless that is its explicit purpose.

        Example of the required structure:
        ```python
        import sys
        import json
        # ... any other necessary imports

        def {tool_name}(arg1: str, arg2: int) -> dict:
            \"\"\"
            Docstring explaining the tool.
            \"\"\"
            # function logic here
            return {{"status": "success", "data": f"{{arg1}} {{arg2}}"}}

        if __name__ == '__main__':
            try:
                input_data = json.load(sys.stdin)
                result = {tool_name}(**input_data)
                json.dump({{"result": result}}, sys.stdout)
            except Exception as e:
                json.dump({{"error": str(e)}}, sys.stderr)
                sys.exit(1)
        ```
        """
        print("Generating tool code...")
        try:
            code = self.code_gen_agent.run(prompt)
            # Basic validation to ensure the generated code seems plausible
            if f"def {tool_name}(" in code and "if __name__ == '__main__':" in code:
                print("Tool code generated successfully.")
                return code
            else:
                print("Error: Generated code does not conform to the required structure.")
                return None
        except Exception as e:
            print(f"An error occurred during code generation: {e}")
            return None

    def _execute_sandboxed(self, code: str, test_inputs: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """
        Executes the provided Python code in a sandboxed subprocess.

        Args:
            code (str): The Python code to execute.
            test_inputs (Dict[str, Any]): A dictionary of keyword arguments to pass to the tool function.

        Returns:
            Tuple[bool, Any, str]: A tuple containing:
                - success (bool): True if execution was successful, False otherwise.
                - result (Any): The deserialized JSON result from stdout, or None on failure.
                - error_message (str): The error message from stderr, or an empty string on success.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            tmp_file.write(code)
            tmp_file_path = tmp_file.name

        try:
            process = subprocess.Popen(
                [sys.executable, tmp_file_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Serialize inputs and send to stdin
            input_json = json.dumps(test_inputs)
            stdout, stderr = process.communicate(input=input_json, timeout=30)
            
            if process.returncode == 0:
                try:
                    output_data = json.loads(stdout)
                    return True, output_data.get('result'), ""
                except json.JSONDecodeError:
                    return False, None, "Failed to decode JSON from tool output."
            else:
                try:
                    error_data = json.loads(stderr)
                    return False, None, error_data.get('error', 'Unknown execution error.')
                except json.JSONDecodeError:
                    return False, None, stderr or "Execution failed with no specific error message."

        except subprocess.TimeoutExpired:
            return False, None, "Execution timed out."
        except Exception as e:
            return False, None, f"An unexpected error occurred during sandboxed execution: {e}"
        finally:
            os.remove(tmp_file_path)

    def test_and_validate_tool(
        self,
        tool_name: str,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> bool:
        """
        Tests the generated tool code against a series of test cases.

        Args:
            tool_name (str): The name of the tool function.
            code (str): The Python code for the tool.
            test_cases (List[Dict[str, Any]]): A list of test cases. Each case is a dictionary
                                               containing 'inputs' and an optional 'expected_output'.

        Returns:
            bool: True if all test cases pass, False otherwise.
        """
        if not test_cases:
            print("Warning: No test cases provided for tool validation.")
            return True # Or False, depending on desired strictness

        print(f"Validating tool '{tool_name}' with {len(test_cases)} test case(s)...")
        for i, case in enumerate(test_cases):
            inputs = case.get('inputs', {})
            expected_output = case.get('expected_output')

            success, result, error_message = self._execute_sandboxed(code, inputs)

            if not success:
                print(f"Test case {i+1} failed: Execution error -> {error_message}")
                return False
            
            if expected_output is not None and result != expected_output:
                print(f"Test case {i+1} failed: Output mismatch.")
                print(f"  Input: {inputs}")
                print(f"  Expected: {expected_output}")
                print(f"  Got: {result}")
                return False
        
        print(f"All {len(test_cases)} test cases passed.")
        return True

    def create_and_register_tool(
        self,
        task_description: str,
        tool_name: str,
        test_cases: List[Dict[str, Any]]
    ) -> bool:
        """
        Orchestrates the full process of creating, testing, and registering a new tool.

        Args:
            task_description (str): A natural language description of the tool's purpose.
            tool_name (str): The desired name for the new tool's function.
            test_cases (List[Dict[str, Any]]): Test cases to validate the tool.

        Returns:
            bool: True if the tool was successfully created and registered, False otherwise.
        """
        # Step 1: Generate the tool's code
        code = self.generate_tool_code(task_description, tool_name)
        if not code:
            print("Tool creation failed at code generation step.")
            return False

        # Step 2: Test and validate the generated code
        if not self.test_and_validate_tool(tool_name, code, test_cases):
            print("Tool creation failed at validation step.")
            return False

        # Step 3: Integrate the tool into the registry
        try:
            # Create a temporary module to load the function
            spec = importlib.util.spec_from_loader(tool_name, loader=None)
            if spec is None:
                raise ImportError("Could not create module spec.")
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute the code in the module's namespace
            exec(code, module.__dict__)
            
            # Get the function from the module
            tool_function = getattr(module, tool_name)
            
            # Extract docstring for description
            description = tool_function.__doc__.strip() if tool_function.__doc__ else task_description
            
            # Register the dynamically loaded function
            self.tool_registry.register_tool(
                name=tool_name,
                func=tool_function,
                description=description
            )
            print(f"Tool '{tool_name}' has been successfully created and registered.")
            return True
        except Exception as e:
            print(f"Tool creation failed at integration step: {e}")
            return False

if __name__ == '__main__':
    print("--- AutonomousToolCreator Demonstration ---")
    
    # Setup mock managers
    agent_mgr = AgentManager()
    tool_reg = agent_mgr.tool_registry
    
    # Initialize the creator
    tool_creator = AutonomousToolCreator(agent_manager=agent_mgr, tool_registry=tool_reg)
    
    # Define the task for the new tool
    task = "create a tool that adds two numbers."
    new_tool_name = "add_two_numbers"
    
    # Define test cases for validation
    tests = [
        {"inputs": {"a": 5, "b": 10}, "expected_output": 15},
        {"inputs": {"a": -3, "b": 3}, "expected_output": 0},
        {"inputs": {"a": 0, "b": 0}, "expected_output": 0},
    ]
    
    # Run the creation process
    success = tool_creator.create_and_register_tool(
        task_description=task,
        tool_name=new_tool_name,
        test_cases=tests
    )
    
    if success:
        print("\n--- Verifying the newly created tool ---")
        # Verify that the tool is in the registry
        newly_registered_tool = tool_reg.get_tool(new_tool_name)
        if newly_registered_tool:
            print(f"Tool '{new_tool_name}' found in registry.")
            print(f"Description: {newly_registered_tool['description']}")
            
            # Test the registered function directly
            result = newly_registered_tool['func'](a=100, b=200)
            print(f"Direct call test: 100 + 200 = {result}")
            assert result == 300
            print("Direct call test passed!")
        else:
            print(f"Error: Tool '{new_tool_name}' not found in registry after creation.")
    else:
        print("\nTool creation process failed.")
