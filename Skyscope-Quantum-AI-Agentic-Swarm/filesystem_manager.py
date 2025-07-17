import os
import shutil
from pathlib import Path
from typing import List, Union, Optional, Set

class FileSystemManager:
    """
    A class to safely manage filesystem operations for AI agents.

    This manager provides a controlled interface for reading, writing, and
    manipulating files and directories, enforcing security constraints to prevent
    unauthorized access to the filesystem. Access is restricted to a set of
    pre-approved directories and file extensions.
    """

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        allowed_extensions: Optional[List[str]] = None,
    ):
        """
        Initializes the FileSystemManager with security constraints.

        Args:
            allowed_paths (Optional[List[str]]): A list of directory paths where
                operations are permitted. If None, defaults to a 'workspace'
                subdirectory in the current working directory.
            allowed_extensions (Optional[List[str]]): A list of file extensions
                that are permissible for read/write operations (e.g., ['.txt', '.py']).
                If None, all extensions are allowed within the permitted paths.
        """
        if allowed_paths is None:
            # Default to a safe 'workspace' directory
            default_workspace = Path.cwd() / "workspace"
            self.allowed_paths: Set[Path] = {default_workspace.resolve()}
            # Automatically create the workspace if it doesn't exist
            self.create_directory(str(default_workspace), exist_ok=True)
        else:
            self.allowed_paths = {Path(p).resolve() for p in allowed_paths}

        self.allowed_extensions: Optional[Set[str]] = (
            set(ext.lower() for ext in allowed_extensions)
            if allowed_extensions is not None
            else None
        )

    def _is_path_allowed(self, path: Union[str, Path], check_extension: bool = True) -> bool:
        """
        Checks if a given path is within the allowed directories and,
        optionally, has an allowed extension.

        Args:
            path (Union[str, Path]): The path to validate.
            check_extension (bool): Whether to validate the file extension.

        Returns:
            bool: True if the path is allowed, False otherwise.
        """
        try:
            resolved_path = Path(path).resolve()
        except (TypeError, ValueError):
            return False

        # Check if the path is within any of the allowed parent directories
        is_in_allowed_dir = any(
            resolved_path.is_relative_to(allowed_dir)
            for allowed_dir in self.allowed_paths
        )

        if not is_in_allowed_dir:
            return False

        # If extensions are restricted, check the file extension
        if check_extension and self.allowed_extensions is not None:
            if resolved_path.is_file() or resolved_path.suffix:
                return resolved_path.suffix.lower() in self.allowed_extensions
            # If it's a directory, extension check is not applicable
            return True

        return True

    def read_file(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Reads the content of a file.

        Args:
            file_path (str): The path to the file to read.
            encoding (str): The encoding to use.

        Returns:
            str: The content of the file.

        Raises:
            PermissionError: If the path is not in the allowed directories.
            FileNotFoundError: If the file does not exist.
        """
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access to path '{file_path}' is denied.")
        
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"File not found at path: '{file_path}'")
            
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()

    def write_file(self, file_path: str, content: Union[str, bytes], encoding: str = 'utf-8') -> None:
        """
        Writes content to a file. Creates parent directories if they don't exist.

        Args:
            file_path (str): The path to the file to write to.
            content (Union[str, bytes]): The content to write.
            encoding (str): The encoding to use if content is a string.

        Raises:
            PermissionError: If the path is not in the allowed directories.
        """
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access to path '{file_path}' is denied.")

        # Ensure parent directories exist
        parent_dir = Path(file_path).parent
        self.create_directory(str(parent_dir), exist_ok=True)
        
        mode = 'wb' if isinstance(content, bytes) else 'w'
        write_args = {'encoding': encoding} if mode == 'w' else {}
        
        with open(file_path, mode, **write_args) as f:
            f.write(content)

    def delete_file(self, file_path: str) -> bool:
        """
        Deletes a file.

        Args:
            file_path (str): The path to the file to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.

        Raises:
            PermissionError: If the path is not in the allowed directories.
        """
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access to path '{file_path}' is denied.")
        
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            print(f"Error deleting file '{file_path}': {e}")
            return False

    def file_exists(self, file_path: str) -> bool:
        """
        Checks if a file exists at the given path.

        Args:
            file_path (str): The path to check.

        Returns:
            bool: True if a file exists, False otherwise.
        """
        if not self._is_path_allowed(file_path, check_extension=False):
            return False
        return Path(file_path).is_file()

    def create_directory(self, dir_path: str, exist_ok: bool = True) -> None:
        """
        Creates a directory.

        Args:
            dir_path (str): The path of the directory to create.
            exist_ok (bool): If True, no error is raised if the directory already exists.

        Raises:
            PermissionError: If the path is not in the allowed directories.
        """
        if not self._is_path_allowed(dir_path, check_extension=False):
            raise PermissionError(f"Access to path '{dir_path}' is denied.")
        
        Path(dir_path).mkdir(parents=True, exist_ok=exist_ok)

    def delete_directory(self, dir_path: str, recursive: bool = False) -> bool:
        """
        Deletes a directory.

        Args:
            dir_path (str): The path of the directory to delete.
            recursive (bool): If True, deletes the directory and all its contents.
                              If False, only deletes an empty directory.

        Returns:
            bool: True if deletion was successful, False otherwise.

        Raises:
            PermissionError: If the path is not in the allowed directories.
        """
        if not self._is_path_allowed(dir_path, check_extension=False):
            raise PermissionError(f"Access to path '{dir_path}' is denied.")
        
        path_obj = Path(dir_path)
        if not path_obj.is_dir():
            return False

        try:
            if recursive:
                shutil.rmtree(path_obj)
            else:
                os.rmdir(path_obj)
            return True
        except OSError as e:
            print(f"Error deleting directory '{dir_path}': {e}")
            return False

    def list_directory_contents(self, dir_path: str) -> List[str]:
        """
        Lists the contents of a directory.

        Args:
            dir_path (str): The path of the directory to list.

        Returns:
            List[str]: A list of names of the entries in the directory.

        Raises:
            PermissionError: If the path is not in the allowed directories.
            FileNotFoundError: If the directory does not exist.
        """
        if not self._is_path_allowed(dir_path, check_extension=False):
            raise PermissionError(f"Access to path '{dir_path}' is denied.")
        
        if not self.directory_exists(dir_path):
            raise FileNotFoundError(f"Directory not found at path: '{dir_path}'")
            
        return os.listdir(dir_path)

    def directory_exists(self, dir_path: str) -> bool:
        """
        Checks if a directory exists at the given path.

        Args:
            dir_path (str): The path to check.

        Returns:
            bool: True if a directory exists, False otherwise.
        """
        if not self._is_path_allowed(dir_path, check_extension=False):
            return False
        return Path(dir_path).is_dir()

    @staticmethod
    def join_paths(base: str, *args: str) -> str:
        """
        Joins one or more path components intelligently.

        Args:
            base (str): The base path.
            *args (str): Additional path components to join.

        Returns:
            str: The joined path.
        """
        return os.path.join(base, *args)

    @staticmethod
    def get_parent_directory(path: str) -> str:
        """
        Gets the parent directory of a given path.

        Args:
            path (str): The path to process.

        Returns:
            str: The parent directory path.
        """
        return str(Path(path).parent)

    @staticmethod
    def get_filename(path: str) -> str:
        """
        Gets the filename from a path.

        Args:
            path (str): The path to process.

        Returns:
            str: The filename.
        """
        return Path(path).name


if __name__ == "__main__":
    print("--- FileSystemManager Demonstration ---")

    # --- Setup ---
    # Create a safe workspace for the demo
    workspace_dir = "demo_workspace"
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir)
    
    # Initialize the manager to only allow access within 'demo_workspace'
    # and only for .txt and .md files.
    fs_manager = FileSystemManager(
        allowed_paths=[workspace_dir],
        allowed_extensions=[".txt", ".md"]
    )
    print(f"Manager initialized. Allowed paths: {fs_manager.allowed_paths}")
    print(f"Allowed extensions: {fs_manager.allowed_extensions}\n")

    # --- Allowed Operations ---
    print("--- Testing Allowed Operations ---")
    try:
        # Create a subdirectory
        allowed_subdir = fs_manager.join_paths(workspace_dir, "reports")
        fs_manager.create_directory(allowed_subdir)
        print(f"Successfully created directory: {allowed_subdir}")

        # Write an allowed file type
        allowed_file = fs_manager.join_paths(allowed_subdir, "report.txt")
        fs_manager.write_file(allowed_file, "This is a test report.")
        print(f"Successfully wrote to file: {allowed_file}")

        # Read the file
        content = fs_manager.read_file(allowed_file)
        print(f"Successfully read content: '{content}'")

        # List directory contents
        contents = fs_manager.list_directory_contents(allowed_subdir)
        print(f"Directory contents of '{allowed_subdir}': {contents}")

        # Delete the file
        fs_manager.delete_file(allowed_file)
        print(f"Successfully deleted file: {allowed_file}")
        print(f"File exists after deletion: {fs_manager.file_exists(allowed_file)}")

    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"An unexpected error occurred during allowed operations: {e}")

    # --- Denied Operations ---
    print("\n--- Testing Denied Operations ---")
    try:
        # 1. Try to write outside the workspace
        disallowed_file_path = "sensitive_data.txt"
        print(f"Attempting to write to: {disallowed_file_path} (outside workspace)")
        fs_manager.write_file(disallowed_file_path, "secret")
    except PermissionError as e:
        print(f"SUCCESS: Operation correctly denied. Reason: {e}")

    try:
        # 2. Try to write a file with a disallowed extension
        disallowed_ext_path = fs_manager.join_paths(workspace_dir, "script.py")
        print(f"Attempting to write to: {disallowed_ext_path} (disallowed extension)")
        fs_manager.write_file(disallowed_ext_path, "print('hello')")
    except PermissionError as e:
        print(f"SUCCESS: Operation correctly denied. Reason: {e}")

    try:
        # 3. Try to read a system file
        system_file = "/etc/hosts" if os.name != 'nt' else 'C:\\Windows\\System32\\drivers\\etc\\hosts'
        print(f"Attempting to read system file: {system_file}")
        fs_manager.read_file(system_file)
    except PermissionError as e:
        print(f"SUCCESS: Operation correctly denied. Reason: {e}")
    except FileNotFoundError:
        print(f"INFO: System file '{system_file}' not found, skipping this test.")

    # --- Cleanup ---
    print("\n--- Cleaning up ---")
    shutil.rmtree(workspace_dir)
    print("Demo workspace deleted.")
