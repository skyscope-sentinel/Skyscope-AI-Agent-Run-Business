import os
import shutil
from pathlib import Path

# Define a base workspace for agent file operations.
# This should be configurable and sandboxed in a real, secure application.
# For now, using a subdirectory in the user's home or a project-local directory.
# Using a project-local directory for easier cleanup during development.
AGENT_WORKSPACE_DIR = Path(os.getcwd()) / "skyscope_agent_workspace"
AGENT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

print(f"[FileIOTools] Agent workspace initialized at: {AGENT_WORKSPACE_DIR.resolve()}")

def _resolve_path(file_path: str) -> Path:
    """
    Resolves a given file path against the agent workspace directory
    and ensures it does not traverse outside the workspace.
    """
    # Normalize the path to prevent directory traversal tricks like ".."
    resolved_base = AGENT_WORKSPACE_DIR.resolve()
    full_path = (resolved_base / file_path).resolve()

    # Check if the resolved path is within the workspace
    if resolved_base not in full_path.parents and full_path != resolved_base:
        if not str(full_path).startswith(str(resolved_base)): # Additional check for exact prefix
            raisePermissionError(f"Path traversal detected. Operation outside workspace '{resolved_base}' is not allowed for path '{file_path}'. Resolved to '{full_path}'")

    # For paths that are exactly the workspace dir (e.g. listing root), allow it.
    # For paths that are children, ensure they start with resolved_base.
    if full_path != resolved_base and not str(full_path).startswith(str(resolved_base) + os.sep):
         raise PermissionError(f"Path traversal detected. Operation outside workspace '{resolved_base}' is not allowed for path '{file_path}'. Resolved to '{full_path}'")

    return full_path

def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Writes content to a file within the agent's workspace.
    Args:
        file_path (str): The relative path to the file within the agent workspace.
        content (str): The content to write to the file.
        overwrite (bool): Whether to overwrite the file if it already exists. Defaults to False.
    Returns:
        str: A confirmation message or an error message.
    """
    try:
        full_path = _resolve_path(file_path)

        if full_path.exists() and not overwrite:
            return f"Error: File '{file_path}' already exists and overwrite is False."

        full_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to file '{file_path}' in agent workspace."
    except PermissionError as e:
        return f"Error writing file: {str(e)}"
    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"

def read_file(file_path: str) -> str:
    """
    Reads content from a file within the agent's workspace.
    Args:
        file_path (str): The relative path to the file within the agent workspace.
    Returns:
        str: The content of the file or an error message.
    """
    try:
        full_path = _resolve_path(file_path)
        if not full_path.is_file():
            return f"Error: File '{file_path}' not found or is not a file in agent workspace."

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except PermissionError as e:
        return f"Error reading file: {str(e)}"
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"

def list_files(directory_path: str = ".") -> str:
    """
    Lists files and directories within a specified path in the agent's workspace.
    Args:
        directory_path (str): The relative path to the directory within the agent workspace.
                              Defaults to the root of the agent workspace.
    Returns:
        str: A string listing files and directories, or an error message.
    """
    try:
        full_path = _resolve_path(directory_path)
        if not full_path.is_dir():
            return f"Error: Directory '{directory_path}' not found or is not a directory in agent workspace."

        items = []
        for item in full_path.iterdir():
            item_type = "DIR" if item.is_dir() else "FILE"
            items.append(f"{item_type}: {item.name}")

        if not items:
            return f"Directory '{directory_path}' is empty."
        return f"Contents of '{directory_path}':\n" + "\n".join(items)
    except PermissionError as e:
        return f"Error listing files: {str(e)}"
    except Exception as e:
        return f"Error listing files in '{directory_path}': {str(e)}"

def delete_file_or_directory(path: str) -> str:
    """
    Deletes a file or directory within the agent's workspace.
    For safety, directories must be empty to be deleted unless a specific flag is added later.
    Args:
        path (str): The relative path to the file or directory to delete.
    Returns:
        str: Confirmation or error message.
    """
    try:
        full_path = _resolve_path(path)
        if not full_path.exists():
            return f"Error: Path '{path}' does not exist in agent workspace."

        if full_path.is_dir():
            # For safety, only delete empty directories for now.
            # Could add a recursive delete flag later if needed.
            if any(full_path.iterdir()):
                return f"Error: Directory '{path}' is not empty. Cannot delete non-empty directory for safety."
            shutil.rmtree(full_path) # More robust for dirs, though os.rmdir works for empty
            return f"Successfully deleted directory '{path}' from agent workspace."
        elif full_path.is_file():
            full_path.unlink()
            return f"Successfully deleted file '{path}' from agent workspace."
        else:
            return f"Error: Path '{path}' is neither a file nor a directory."
    except PermissionError as e:
        return f"Error deleting path: {str(e)}"
    except Exception as e:
        return f"Error deleting path '{path}': {str(e)}"


if __name__ == "__main__":
    print("--- Testing File I/O Tools ---")
    # Ensure the workspace is clean for repeatable tests, or handle existing files.
    if AGENT_WORKSPACE_DIR.exists():
        # Simple cleanup for test: remove and recreate
        # In a real app, be more careful.
        try:
            shutil.rmtree(AGENT_WORKSPACE_DIR)
            print(f"Cleaned up existing workspace: {AGENT_WORKSPACE_DIR}")
        except Exception as e:
            print(f"Could not cleanup workspace: {e}")
    AGENT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


    # Test write_file
    print("\nTesting write_file...")
    write_result1 = write_file("test_file_1.txt", "Hello from Skyscope Agent!")
    print(write_result1)
    write_result2 = write_file("subdir/test_file_2.txt", "This is in a subdirectory.")
    print(write_result2)
    write_result3 = write_file("test_file_1.txt", "Trying to overwrite.", overwrite=False) # Should fail
    print(write_result3)
    write_result4 = write_file("test_file_1.txt", "Overwriting now.", overwrite=True) # Should succeed
    print(write_result4)
    write_result_traversal = write_file("../outside_file.txt", "Attempting traversal")
    print(f"Traversal write attempt: {write_result_traversal}")


    # Test list_files
    print("\nTesting list_files...")
    list_result1 = list_files(".")
    print(list_result1)
    list_result2 = list_files("subdir")
    print(list_result2)
    list_result_nonexistent = list_files("nonexistent_dir")
    print(list_result_nonexistent)

    # Test read_file
    print("\nTesting read_file...")
    read_result1 = read_file("test_file_1.txt")
    print(f"Content of test_file_1.txt: '{read_result1}'")
    read_result2 = read_file("subdir/test_file_2.txt")
    print(f"Content of subdir/test_file_2.txt: '{read_result2}'")
    read_result_nonexistent = read_file("nonexistent.txt")
    print(read_result_nonexistent)
    read_result_traversal = read_file("../config.py") # Attempt to read outside workspace
    print(f"Traversal read attempt: {read_result_traversal}")


    # Test delete_file_or_directory
    print("\nTesting delete_file_or_directory...")
    # Create a file to delete
    write_file("to_delete.txt", "This file will be deleted.")
    print(list_files("."))
    delete_result1 = delete_file_or_directory("to_delete.txt")
    print(delete_result1)
    print(list_files("."))

    # Test deleting a directory (must be empty)
    delete_dir_result1 = delete_file_or_directory("subdir") # Should fail if test_file_2.txt is in it
    print(f"Attempt to delete non-empty subdir: {delete_dir_result1}")
    delete_file_in_subdir = delete_file_or_directory("subdir/test_file_2.txt")
    print(f"Deleting file in subdir: {delete_file_in_subdir}")
    delete_dir_result2 = delete_file_or_directory("subdir") # Should succeed now
    print(f"Attempt to delete empty subdir: {delete_dir_result2}")
    print(list_files("."))

    delete_result_traversal = delete_file_or_directory("../requirements_gui.txt")
    print(f"Traversal delete attempt: {delete_result_traversal}")


    print("\n--- File I/O Tools Test Complete ---")
    # Consider cleaning up AGENT_WORKSPACE_DIR after tests if desired,
    # but for inspection, leaving it might be useful.
    # shutil.rmtree(AGENT_WORKSPACE_DIR)
    # print(f"Cleaned up workspace: {AGENT_WORKSPACE_DIR}")
