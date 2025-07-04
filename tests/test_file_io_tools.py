import unittest
import os
import shutil
from skyscope_sentinel.tools.file_io_tools import save_text_to_file_in_workspace, read_text_from_file_in_workspace, list_files_in_workspace_subdir
from skyscope_sentinel.config import Config

class TestFileIOTools(unittest.TestCase):

    def setUp(self):
        """Setup a temporary workspace for testing."""
        self.config = Config()
        self.base_workspace = self.config.AGENT_WORKSPACE
        self.test_workspace = os.path.join(self.base_workspace, "test_file_io_temp")

        # Override workspace for file_io_tools during test
        # This requires file_io_tools to be able to use a custom base path or for Config to be patchable.
        # For simplicity, we'll create files directly in a subdir of the actual AGENT_WORKSPACE
        # and ensure it's cleaned up.
        if os.path.exists(self.test_workspace):
            shutil.rmtree(self.test_workspace)
        os.makedirs(self.test_workspace, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary workspace."""
        if os.path.exists(self.test_workspace):
            shutil.rmtree(self.test_workspace)

    def test_save_and_read_text_file(self):
        """Test saving text to a file and reading it back."""
        test_subdir = "save_read_test"
        filename = "test_document.txt"
        content = "Hello, Skyscope Sentinel!\nThis is a test file.\nLine 3."

        # Test saving
        file_path = save_text_to_file_in_workspace(content, filename, subdir_path=os.path.join(self.test_workspace, test_subdir))

        self.assertIsNotNone(file_path, "save_text_to_file_in_workspace should return a file path.")
        self.assertTrue(os.path.exists(file_path), f"File should exist at {file_path}")

        # Construct expected path for verification (as save_text_to_file_in_workspace returns absolute)
        expected_path_check = os.path.join(self.test_workspace, test_subdir, filename)
        self.assertEqual(os.path.abspath(file_path), os.path.abspath(expected_path_check))

        # Test reading
        read_content = read_text_from_file_in_workspace(filename, subdir_path=os.path.join(self.test_workspace, test_subdir))
        self.assertEqual(content, read_content, "Read content should match saved content.")

    def test_read_non_existent_file(self):
        """Test reading a non-existent file returns an error message."""
        filename = "non_existent_file.txt"
        test_subdir = "non_existent_test"
        result = read_text_from_file_in_workspace(filename, subdir_path=os.path.join(self.test_workspace, test_subdir))
        self.assertTrue(result.startswith("Error: File not found"), "Should return error for non-existent file.")

    def test_save_file_no_subdir(self):
        """Test saving a file directly into the test_workspace (no additional subdir)."""
        filename = "root_test_document.txt"
        content = "This is a test in the root of test_workspace."

        file_path = save_text_to_file_in_workspace(content, filename, subdir_path=self.test_workspace) # Pass full test_workspace path
        self.assertTrue(os.path.exists(file_path))

        # Verify it's directly in self.test_workspace
        expected_path = os.path.join(self.test_workspace, filename)
        self.assertEqual(os.path.abspath(file_path), os.path.abspath(expected_path))

        with open(file_path, "r", encoding="utf-8") as f:
            saved_content = f.read()
        self.assertEqual(content, saved_content)

    def test_list_files_in_workspace_subdir(self):
        """Test listing files in a subdirectory of the workspace."""
        test_subdir_name = "list_test_subdir"
        test_subdir_path = os.path.join(self.test_workspace, test_subdir_name)
        os.makedirs(test_subdir_path, exist_ok=True)

        # Create some dummy files and a directory
        with open(os.path.join(test_subdir_path, "file1.txt"), "w") as f:
            f.write("content1")
        with open(os.path.join(test_subdir_path, "file2.md"), "w") as f:
            f.write("content2")
        os.makedirs(os.path.join(test_subdir_path, "nested_dir"), exist_ok=True)
        with open(os.path.join(test_subdir_path, "nested_dir", "file3.txt"), "w") as f:
            f.write("content3")

        # Test listing (relative to AGENT_WORKSPACE)
        # The list_files_in_workspace_subdir expects subdir_path relative to AGENT_WORKSPACE
        # So, we need to pass the relative path from base_workspace to test_subdir_name
        relative_subdir_for_tool = os.path.relpath(test_subdir_path, self.base_workspace)

        file_list_str = list_files_in_workspace_subdir(relative_subdir_for_tool)

        self.assertIn("file1.txt", file_list_str)
        self.assertIn("file2.md", file_list_str)
        self.assertIn("nested_dir/", file_list_str) # Directories should have a trailing slash
        self.assertNotIn("file3.txt", file_list_str) # Should not list recursively by default

    def test_list_files_non_existent_subdir(self):
        """Test listing files in a non-existent subdirectory."""
        non_existent_subdir = "this_does_not_exist"
        result = list_files_in_workspace_subdir(non_existent_subdir) # Relative to AGENT_WORKSPACE
        self.assertTrue(result.startswith("Error: Subdirectory not found"), "Should return error for non-existent subdir.")


if __name__ == '__main__':
    # This allows running the tests directly from this file
    # It requires skyscope_sentinel to be in PYTHONPATH or the CWD to be the project root.
    # For example, from project root: python -m tests.test_file_io_tools

    # Ensure skyscope_sentinel is in path for standalone run from tests/ directory
    import sys
    if os.path.join(os.getcwd(), '..') not in sys.path:
         sys.path.insert(0, os.path.join(os.getcwd(), '..'))

    unittest.main()
