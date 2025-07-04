import chromadb
import os
import datetime
import litellm # For embeddings
from skyscope_sentinel.config import Config

# Get the path for the ChromaDB persistent store from config or use a default
config = Config()
CHROMA_DB_PATH = os.path.join(config.AGENT_WORKSPACE, "chroma_db")
REPORTS_COLLECTION_NAME = "opportunity_reports"

# Ensure the ChromaDB storage directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Initialize ChromaDB client (persistent)
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}. Vector store functionalities will be impaired.")
    client = None

def get_ollama_embedding_model_name():
    """Gets the Ollama model name configured for embeddings."""
    # Reuse the main Ollama model for embeddings for simplicity,
    # or specify a dedicated embedding model if available/configured.
    # LiteLLM expects model names like "ollama/mistral" or "mistral" if served by Ollama.
    # For embeddings, often a specific embedding model is better if available.
    # For now, using the configured chat model, assuming it can produce embeddings via litellm.
    # A common pattern is to use something like "nomic-embed-text" if pulled into Ollama.
    # Let's default to "ollama/nomic-embed-text" if main model is not embed-capable,
    # or allow override via config. For now, using main model.
    ollama_model = config.get_ollama_model_name() # e.g., "ollama/mistral"
    # if "nomic-embed-text" not in ollama_model and "embed" not in ollama_model:
        # print(f"Warning: Main model {ollama_model} might not be ideal for embeddings. Consider using a dedicated embedding model like 'nomic-embed-text'.")
    # LiteLLM handles the "ollama/" prefix for embeddings correctly.
    return ollama_model


def initialize_report_collection():
    """
    Initializes (creates if not exists) the ChromaDB collection for storing reports.
    Returns the collection object or None if client init failed.
    """
    if not client:
        print("ChromaDB client not initialized. Cannot get/create collection.")
        return None
    try:
        collection = client.get_or_create_collection(
            name=REPORTS_COLLECTION_NAME,
            # metadata={"hnsw:space": "cosine"} # Example: specify distance function if needed
        )
        print(f"ChromaDB collection '{REPORTS_COLLECTION_NAME}' initialized/retrieved from {CHROMA_DB_PATH}.")
        return collection
    except Exception as e:
        print(f"Error getting/creating ChromaDB collection '{REPORTS_COLLECTION_NAME}': {e}")
        return None

def _chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """Helper function to split text into manageable chunks."""
    # This is a very basic chunking strategy. More sophisticated methods exist.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
        if start < 0: # Ensure start doesn't go negative if overlap > remaining text
            start = 0
    # Remove last chunk if it's just whitespace or very small due to overlap
    if chunks and (not chunks[-1].strip() or len(chunks[-1]) < chunk_overlap / 2) and len(chunks)>1 :
        chunks.pop()
    return chunks


def add_report_to_collection(report_markdown: str, report_filename: str, topic: str = "Unknown Topic"):
    """
    Adds a report to the ChromaDB collection.
    The report is chunked, embeddings are generated, and then stored.

    Args:
        report_markdown (str): The content of the report in Markdown.
        report_filename (str): The filename of the report, used as an ID.
        topic (str): The main topic of the report for metadata.
    """
    collection = initialize_report_collection()
    if not collection:
        print("Failed to add report: collection not available.")
        return

    embedding_model = get_ollama_embedding_model_name()
    if not embedding_model:
        print("Failed to add report: Ollama embedding model not configured.")
        return

    chunks = _chunk_text(report_markdown)
    if not chunks:
        print(f"Report '{report_filename}' resulted in no chunks. Skipping.")
        return

    embeddings = []
    valid_chunks = []
    chunk_ids = []

    print(f"Generating embeddings for {len(chunks)} chunks from '{report_filename}' using model '{embedding_model}'...")
    for i, chunk_text in enumerate(chunks):
        try:
            response = litellm.embedding(model=embedding_model, input=[chunk_text])
            embeddings.append(response.data[0]['embedding'])
            valid_chunks.append(chunk_text)
            chunk_ids.append(f"{report_filename}_chunk_{i}")
        except Exception as e:
            print(f"Error generating embedding for chunk {i} of '{report_filename}': {e}")
            # Optionally skip this chunk or handle error differently
            continue

    if not valid_chunks: # If all chunks failed embedding
        print(f"No valid embeddings generated for report '{report_filename}'. Not adding to collection.")
        return

    print(f"Adding {len(valid_chunks)} chunks to ChromaDB collection '{REPORTS_COLLECTION_NAME}'...")
    try:
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=valid_chunks,
            metadatas=[{
                "filename": report_filename,
                "topic": topic,
                "chunk_index": j,
                "timestamp": datetime.datetime.now().isoformat()
            } for j in range(len(valid_chunks))]
        )
        print(f"Report '{report_filename}' successfully added to ChromaDB collection.")
    except Exception as e:
        print(f"Error adding report chunks to ChromaDB for '{report_filename}': {e}")


def query_reports(query_text: str, n_results: int = 3) -> list[dict]:
    """
    Queries the report collection for relevant documents based on the query text.

    Args:
        query_text (str): The text to search for.
        n_results (int): The number of results to return.

    Returns:
        list[dict]: A list of query results, each containing 'document' and 'metadata'.
                    Returns an empty list if an error occurs or no results are found.
    """
    collection = initialize_report_collection()
    if not collection:
        print("Failed to query reports: collection not available.")
        return []

    embedding_model = get_ollama_embedding_model_name()
    if not embedding_model:
        print("Failed to query reports: Ollama embedding model not configured.")
        return []

    try:
        print(f"Generating query embedding for: '{query_text[:50]}...' using model '{embedding_model}'")
        query_embedding_response = litellm.embedding(model=embedding_model, input=[query_text])
        query_embedding = query_embedding_response.data[0]['embedding']

        print(f"Querying collection '{REPORTS_COLLECTION_NAME}' with {n_results} results expected.")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        # Process results to be more directly usable
        # The 'results' object from ChromaDB query is a dict with keys like 'ids', 'documents', 'metadatas', 'distances'
        # Each of these keys holds a list of lists (one inner list per query embedding, we only have one)
        processed_results = []
        if results and results.get('documents') and results.get('metadatas'):
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results.get('distances', [[]])[0] # Handle if distances are not included

            for i in range(len(docs)):
                processed_results.append({
                    "document": docs[i],
                    "metadata": metadatas[i],
                    "distance": distances[i] if distances and i < len(distances) else None
                })

        print(f"Found {len(processed_results)} relevant document chunks.")
        return processed_results

    except Exception as e:
        print(f"Error querying reports: {e}")
        return []

if __name__ == "__main__":
    print("--- Testing Vector Store Utilities ---")

    # Ensure litellm is configured for Ollama (this should happen via global config or env vars)
    # For standalone test, you might need to set:
    # os.environ["OPENAI_API_KEY"] = "ollama" # Or any non-empty string if litellm defaults to OpenAI
    # os.environ["OPENAI_API_BASE"] = config.get_ollama_base_url() # if needed by older litellm
    # litellm.set_verbose=True # For debugging embedding calls

    if not client:
        print("Skipping vector store tests as ChromaDB client failed to initialize.")
    else:
        # 1. Initialize collection (should happen on import, but good to test directly)
        test_collection = initialize_report_collection()
        if test_collection:
            print(f"Test collection '{test_collection.name}' count: {test_collection.count()}")

            # 2. Add a dummy report
            dummy_report_md = """
# Amazing Opportunity X

## Concept
This is a fantastic concept for making money with AI. It involves robots and AI.

## Strategy
1. Build AI robots.
2. ???
3. Profit!
            """
            dummy_filename = "dummy_report_test.md"
            dummy_topic = "AI Robots for Profit"

            print(f"\nAttempting to add dummy report: {dummy_filename}")
            add_report_to_collection(dummy_report_md, dummy_filename, dummy_topic)

            # Verify count (can take a moment for Chroma to update fully)
            # For more reliable count after add, re-fetch collection or query
            # test_collection = client.get_collection(name=REPORTS_COLLECTION_NAME) # Re-fetch
            print(f"Collection count after attempting add: {test_collection.count()}") # May not reflect immediately for some backends

            # 3. Query the report
            print("\nAttempting to query for 'AI robots'")
            query_results = query_reports("AI robots strategy", n_results=2)

            if query_results:
                print("\nQuery Results:")
                for i, res in enumerate(query_results):
                    print(f"  Result {i+1}:")
                    print(f"    Distance: {res.get('distance')}")
                    print(f"    Filename: {res.get('metadata', {}).get('filename')}")
                    print(f"    Topic: {res.get('metadata', {}).get('topic')}")
                    print(f"    Chunk Index: {res.get('metadata', {}).get('chunk_index')}")
                    print(f"    Document Snippet: {res.get('document', '')[:100]}...")
                    print("-" * 20)
            else:
                print("No results found for 'AI robots' or query failed.")

            # Clean up: Delete the dummy report's chunks if possible (optional for this test)
            # This requires knowing the IDs. For a real cleanup, store IDs or query by metadata.
            # Example: test_collection.delete(ids=[f"{dummy_filename}_chunk_0", f"{dummy_filename}_chunk_1"])
            # print(f"\nAttempting to clean up dummy report entries for {dummy_filename}...")
            # ids_to_delete = [entry.id for entry in test_collection.get(where={"filename": dummy_filename})['ids']]
            # if ids_to_delete:
            #     test_collection.delete(ids=ids_to_delete)
            #     print(f"Cleaned up {len(ids_to_delete)} entries for {dummy_filename}.")
            # print(f"Collection count after cleanup: {test_collection.count()}")


        else:
            print("Failed to initialize test collection.")

    print("\n--- Vector Store Utilities Test Complete ---")
    print(f"ChromaDB data is persisted in: {CHROMA_DB_PATH}")
    print("To fully test embeddings, ensure your Ollama service is running and the model specified in config.py (for embeddings) is available.")
    print("The default embedding model used here is derived from your main Ollama model config.")


def get_contextual_information_for_topic(topic: str, n_results: int = 2) -> str:
    """
    Queries the vector store for past reports related to the given topic
    and formats the results into a string for prompt injection.

    Args:
        topic (str): The topic to search for contextual information.
        n_results (int): Number of results to fetch from the vector store.

    Returns:
        str: A formatted string containing contextual information, or an empty string if none found.
    """
    if not topic or not topic.strip():
        return "No specific topic provided for contextual search."

    print(f"[RAG Tool] Fetching contextual info for topic: '{topic}'")
    try:
        query_results = query_reports(query_text=topic, n_results=n_results)

        if not query_results:
            return "No relevant past reports found for this topic."

        context_str = "Context from past related reports:\n"
        for i, res in enumerate(query_results):
            context_str += f"\n--- Context Snippet {i+1} (from report: {res.get('metadata', {}).get('filename', 'N/A')}, topic: {res.get('metadata', {}).get('topic', 'N/A')}) ---\n"
            context_str += res.get('document', 'No content.')[:500] + "...\n" # Limit snippet length
        context_str += "\n--- End of Contextual Information ---\n"
        return context_str
    except Exception as e:
        print(f"[RAG Tool] Error fetching contextual information: {e}")
        return "Error retrieving contextual information from past reports."

# Note: The `litellm.embedding()` call will use the Ollama model specified in `config.py`
# if it's prefixed with "ollama/" or if LiteLLM is otherwise configured for Ollama.
# Ensure that model can generate embeddings or use a specific embedding model.
# `nomic-embed-text` is a common choice for Ollama embeddings.
# If `config.get_ollama_model_name()` returns e.g. "ollama/mistral", litellm will try to use mistral for embeddings.
# This might work but isn't always optimal. A dedicated embedding model is better.
# The `get_ollama_embedding_model_name` function can be enhanced to select a specific embedding model.
# For now, it uses the general model from config.
# The `litellm.embedding()` function may also require OPENAI_API_KEY to be set to "ollama"
# or some other non-empty string for certain LiteLLM versions when using custom providers like Ollama,
# if it defaults to thinking it's OpenAI. Setting it explicitly as `litellm.embedding(model="ollama/your-model", ...)` is usually robust.
# The current `get_ollama_embedding_model_name` should provide this "ollama/" prefix.
# The `client` initialization might fail if there are issues with file permissions for CHROMA_DB_PATH.
# The `_chunk_text` function is basic; consider libraries like `langchain.text_splitter` for more advanced chunking if needed.
