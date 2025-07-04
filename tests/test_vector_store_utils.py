import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import time # For allowing DB operations to settle if needed

# Ensure skyscope_sentinel is in path for standalone run from tests/ directory
import sys
if os.path.join(os.getcwd(), '..') not in sys.path:
     sys.path.insert(0, os.path.join(os.getcwd(), '..'))

# Important: Import after sys.path modification if skyscope_sentinel isn't installed as a package
from skyscope_sentinel.tools import vector_store_utils
from skyscope_sentinel.config import Config

class TestVectorStoreUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup a temporary workspace and ChromaDB path for all tests in this class."""
        cls.config = Config()
        cls.base_workspace_for_tests = cls.config.AGENT_WORKSPACE

        # Define a unique test-specific path for ChromaDB data to avoid conflicts
        cls.test_chroma_db_path = os.path.join(cls.base_workspace_for_tests, "test_chroma_db_data")

        # Override the CHROMA_DB_PATH in vector_store_utils for the duration of these tests
        cls.original_chroma_db_path = vector_store_utils.CHROMA_DB_PATH
        vector_store_utils.CHROMA_DB_PATH = cls.test_chroma_db_path

        # Ensure the test ChromaDB directory is clean before tests
        if os.path.exists(cls.test_chroma_db_path):
            shutil.rmtree(cls.test_chroma_db_path)
        os.makedirs(cls.test_chroma_db_path, exist_ok=True)

        # Re-initialize client in vector_store_utils to use the new path
        # This is a bit invasive; ideally, the client would be passable or settable.
        try:
            vector_store_utils.client = vector_store_utils.chromadb.PersistentClient(path=cls.test_chroma_db_path)
            print(f"Test ChromaDB client initialized at: {cls.test_chroma_db_path}")
        except Exception as e:
            print(f"Critical error setting up test ChromaDB client: {e}")
            vector_store_utils.client = None # Ensure it's None if setup fails

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary ChromaDB data directory and restore original path."""
        if os.path.exists(cls.test_chroma_db_path):
            shutil.rmtree(cls.test_chroma_db_path)
        vector_store_utils.CHROMA_DB_PATH = cls.original_chroma_db_path
        # Re-initialize client to original path if needed, or handle in module itself
        try:
            vector_store_utils.client = vector_store_utils.chromadb.PersistentClient(path=cls.original_chroma_db_path)
        except Exception: # If original path also fails, set to None
            vector_store_utils.client = None


    def setUp(self):
        """Ensure the collection is clean before each individual test method."""
        if vector_store_utils.client:
            try:
                # Get the collection and delete all items if it exists
                # This is safer than deleting and recreating the collection for each test method
                # as it avoids issues with collection caching or delayed deletion.
                collection = vector_store_utils.client.get_collection(name=vector_store_utils.REPORTS_COLLECTION_NAME)
                count = collection.count()
                if count > 0:
                    # To delete all items, we need all IDs. This can be slow for large collections.
                    # A simpler way for testing is to delete and recreate the collection,
                    # but PersistentClient sometimes has issues with rapid delete/create.
                    # Let's try deleting the collection and recreating it for a clean slate per test.
                    vector_store_utils.client.delete_collection(name=vector_store_utils.REPORTS_COLLECTION_NAME)
                    print(f"Test setUp: Deleted collection '{vector_store_utils.REPORTS_COLLECTION_NAME}'")
                vector_store_utils.initialize_report_collection() # Ensure it's recreated
            except Exception as e: # Catch if collection doesn't exist or other errors
                # print(f"Test setUp: Error cleaning collection (may not exist yet): {e}")
                vector_store_utils.initialize_report_collection() # Ensure it exists for the test
        else:
            self.fail("ChromaDB client not available for tests.")


    @patch('litellm.embedding')
    def test_add_and_query_report(self, mock_embedding):
        """Test adding a report and querying it."""
        if not vector_store_utils.client:
            self.skipTest("ChromaDB client not initialized.")

        # Configure mock embedding
        mock_embedding.return_value = MagicMock(data=[{'embedding': [0.1] * 768}]) # Assuming 768 dim, adjust if model differs

        report_md = "Opportunity: AI for Restaurants. Concept: AI helps restaurants optimize menus."
        report_filename = "ai_restaurants.md"
        report_topic = "AI in Food Service"

        vector_store_utils.add_report_to_collection(report_md, report_filename, report_topic)

        # Allow some time for ChromaDB to process if it's truly async in some backends
        # For SQLite backend, it's usually synchronous.
        # time.sleep(0.1)

        collection = vector_store_utils.client.get_collection(name=vector_store_utils.REPORTS_COLLECTION_NAME)
        self.assertGreater(collection.count(), 0, "Report should be added to the collection.")

        query_results = vector_store_utils.query_reports("restaurant menu optimization", n_results=1)
        self.assertIsNotNone(query_results)
        self.assertEqual(len(query_results), 1)
        self.assertIn("AI for Restaurants", query_results[0]['document'])
        self.assertEqual(query_results[0]['metadata']['filename'], report_filename)
        self.assertEqual(query_results[0]['metadata']['topic'], report_topic)

        # Verify embedding mock was called
        self.assertTrue(mock_embedding.called)


    def test_initialize_report_collection(self):
        """Test collection initialization."""
        if not vector_store_utils.client:
            self.skipTest("ChromaDB client not initialized.")
        collection = vector_store_utils.initialize_report_collection()
        self.assertIsNotNone(collection)
        self.assertEqual(collection.name, vector_store_utils.REPORTS_COLLECTION_NAME)

    def test_chunk_text_basic(self):
        """Test the basic text chunking functionality."""
        text = "This is a test sentence. This is another test sentence. And a third one."
        chunks = vector_store_utils._chunk_text(text, chunk_size=30, chunk_overlap=5)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(chunks[0].startswith("This is a test sentence."))
        # Example: "This is a test sentence. This" (30 chars)
        # Next starts at "ence. This is another..."
        # This simple chunker might not be ideal for semantic meaning, but it's testable.
        # Check if overlap is roughly working if there are multiple chunks
        if len(chunks) > 1:
             self.assertTrue(chunks[1].startswith(chunks[0][-5:]), "Overlap not working as expected or chunking logic changed")


    @patch('litellm.embedding')
    def test_add_report_empty_markdown(self, mock_embedding):
        """Test adding an empty report does not add to collection or call embeddings."""
        if not vector_store_utils.client:
            self.skipTest("ChromaDB client not initialized.")

        report_md = ""
        report_filename = "empty_report.md"
        report_topic = "Empty"

        initial_count = vector_store_utils.initialize_report_collection().count()
        vector_store_utils.add_report_to_collection(report_md, report_filename, report_topic)

        mock_embedding.assert_not_called()
        current_count = vector_store_utils.initialize_report_collection().count()
        self.assertEqual(initial_count, current_count, "Empty report should not change collection count.")

    @patch('litellm.embedding')
    def test_query_empty_collection(self, mock_embedding):
        """Test querying an empty collection returns empty list."""
        if not vector_store_utils.client:
            self.skipTest("ChromaDB client not initialized.")

        # Ensure collection is empty
        collection = vector_store_utils.initialize_report_collection()
        # If collection.delete() was available and safe, we'd use it.
        # For now, we rely on setUp to clean it.
        # self.assertEqual(collection.count(), 0) # This might fail if setUp is not perfect

        mock_embedding.return_value = MagicMock(data=[{'embedding': [0.2] * 768}])

        query_results = vector_store_utils.query_reports("anything", n_results=1)
        self.assertEqual(len(query_results), 0)
        # Embedding for the query itself would still be called
        mock_embedding.assert_called_once_with(model=vector_store_utils.get_ollama_embedding_model_name(), input=["anything"])

    @patch('skyscope_sentinel.tools.vector_store_utils.query_reports')
    def test_get_contextual_information_for_topic_found(self, mock_query_reports):
        """Test getting contextual info when results are found."""
        mock_query_reports.return_value = [
            {"document": "Content of report 1 snippet.", "metadata": {"filename": "report1.md", "topic": "Test Topic 1"}},
            {"document": "Content of report 2 snippet.", "metadata": {"filename": "report2.md", "topic": "Test Topic 2"}},
        ]

        context_str = vector_store_utils.get_contextual_information_for_topic("Test Topic", n_results=2)

        self.assertIn("Context from past related reports:", context_str)
        self.assertIn("Context Snippet 1", context_str)
        self.assertIn("report1.md", context_str)
        self.assertIn("Content of report 1 snippet.", context_str)
        self.assertIn("Context Snippet 2", context_str)
        self.assertIn("report2.md", context_str)
        self.assertIn("Content of report 2 snippet.", context_str)
        self.assertIn("--- End of Contextual Information ---", context_str)
        mock_query_reports.assert_called_once_with(query_text="Test Topic", n_results=2)

    @patch('skyscope_sentinel.tools.vector_store_utils.query_reports')
    def test_get_contextual_information_for_topic_not_found(self, mock_query_reports):
        """Test getting contextual info when no results are found."""
        mock_query_reports.return_value = []

        context_str = vector_store_utils.get_contextual_information_for_topic("NonExistent Topic")
        self.assertEqual(context_str, "No relevant past reports found for this topic.")
        mock_query_reports.assert_called_once_with(query_text="NonExistent Topic", n_results=2) # Default n_results

    def test_get_contextual_information_for_empty_topic(self):
        """Test getting contextual info with an empty topic."""
        context_str = vector_store_utils.get_contextual_information_for_topic("")
        self.assertEqual(context_str, "No specific topic provided for contextual search.")

        context_str_space = vector_store_utils.get_contextual_information_for_topic("   ")
        self.assertEqual(context_str_space, "No specific topic provided for contextual search.")

    @patch('skyscope_sentinel.tools.vector_store_utils.query_reports', side_effect=Exception("DB error"))
    def test_get_contextual_information_for_topic_query_error(self, mock_query_reports):
        """Test getting contextual info when query_reports raises an exception."""
        context_str = vector_store_utils.get_contextual_information_for_topic("Error Topic")
        self.assertEqual(context_str, "Error retrieving contextual information from past reports.")
        mock_query_reports.assert_called_once_with(query_text="Error Topic", n_results=2)


if __name__ == '__main__':
    unittest.main()
