import streamlit as st
from typing import Any, Dict, List, Optional, TypedDict, Literal
from datetime import datetime
import uuid

# --- Type Definitions for State Objects ---

class Message(TypedDict):
    """Represents a single message in the chat history."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str

class KnowledgeItem(TypedDict):
    """Represents an item in the knowledge stack."""
    id: str
    title: str
    content: str
    source: Optional[str]
    timestamp: str

class UploadedFile(TypedDict):
    """Represents a file uploaded by the user."""
    id: str
    name: str
    type: str
    content: bytes
    timestamp: str

class ApiKeys(TypedDict):
    """Structure for storing API keys."""
    openai: str
    anthropic: str
    google: str
    huggingface: str

class ToolSettings(TypedDict):
    """Structure for storing tool enablement status."""
    web_search: bool
    deep_research: bool
    deep_thinking: bool
    browser_automation: bool
    quantum_computing: bool
    filesystem_access: bool

# --- Default State ---

DEFAULT_SYSTEM_PROMPT = """You are Skyscope Sentinel Intelligence, an advanced AI assistant with agentic capabilities.
You help users with a wide range of tasks including research, coding, analysis, and creative work.
You have access to various tools including web search, browser automation, and file operations.
You can leverage quantum computing concepts for complex problem-solving when appropriate.
Always be helpful, accurate, and ethical in your responses."""

DEFAULT_STATE = {
    "messages": [],
    "knowledge_stack": [],
    "file_uploads": [],
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "model_provider": "Local (Ollama)",
    "model_name": "llama3:latest",
    "api_keys": {
        "openai": "",
        "anthropic": "",
        "google": "",
        "huggingface": ""
    },
    "tools_enabled": {
        "web_search": False,
        "deep_research": False,
        "deep_thinking": False,
        "browser_automation": False,
        "quantum_computing": False,
        "filesystem_access": False
    },
    "browser_instance": None,
    "code_window_content": "",
    "quantum_mode": "Simulation",
}

# --- StateManager Class ---

class StateManager:
    """
    A class to encapsulate Streamlit session state management.

    This manager provides a structured and consistent API for interacting with
    st.session_state, making it easier to manage complex application states
    like chat history, user settings, and uploaded files.
    """

    def __init__(self, session_state: st.session_state):
        """
        Initializes the StateManager with a Streamlit session state object.

        Args:
            session_state (st.session_state): The session state object from Streamlit.
        """
        self.state = session_state

    def initialize_state(self) -> None:
        """
        Initializes the session state with default values if they don't exist.
        This should be called once at the beginning of the app's lifecycle.
        """
        for key, value in DEFAULT_STATE.items():
            if key not in self.state:
                self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the session state.

        Args:
            key (str): The key of the value to retrieve.
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The value from the session state or the default value.
        """
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Sets a value in the session state.

        Args:
            key (str): The key of the value to set.
            value (Any): The value to set.
        """
        self.state[key] = value

    # --- Chat History Management ---

    def add_message(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        """
        Adds a message to the chat history.

        Args:
            role (Literal["user", "assistant", "system"]): The role of the message sender.
            content (str): The content of the message.
        """
        message: Message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.state.messages.append(message)

    def get_messages(self) -> List[Message]:
        """
        Retrieves the entire chat history.

        Returns:
            List[Message]: A list of all messages in the chat history.
        """
        return self.state.messages

    def clear_messages(self) -> None:
        """Clears the chat history."""
        self.state.messages = []

    # --- Knowledge Stack Management ---

    def add_to_knowledge_stack(self, title: str, content: str, source: Optional[str] = None) -> None:
        """
        Adds an item to the knowledge stack.

        Args:
            title (str): The title of the knowledge item.
            content (str): The content of the item.
            source (Optional[str]): The source of the information (e.g., 'upload', 'web').
        """
        item: KnowledgeItem = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "source": source,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.state.knowledge_stack.append(item)

    def get_knowledge_stack(self) -> List[KnowledgeItem]:
        """
        Retrieves the entire knowledge stack.

        Returns:
            List[KnowledgeItem]: A list of all items in the knowledge stack.
        """
        return self.state.knowledge_stack

    def remove_from_knowledge_stack(self, item_id: str) -> bool:
        """
        Removes an item from the knowledge stack by its ID.

        Args:
            item_id (str): The unique ID of the item to remove.

        Returns:
            bool: True if an item was removed, False otherwise.
        """
        initial_len = len(self.state.knowledge_stack)
        self.state.knowledge_stack = [
            item for item in self.state.knowledge_stack if item['id'] != item_id
        ]
        return len(self.state.knowledge_stack) < initial_len

    def clear_knowledge_stack(self) -> None:
        """Clears the entire knowledge stack."""
        self.state.knowledge_stack = []

    # --- File Upload Management ---

    def add_uploaded_file(self, name: str, file_type: str, content: bytes) -> str:
        """
        Adds a record of an uploaded file to the state.

        Args:
            name (str): The name of the file.
            file_type (str): The MIME type of the file.
            content (bytes): The raw byte content of the file.
            
        Returns:
            str: The unique ID of the added file.
        """
        file_id = str(uuid.uuid4())
        uploaded_file: UploadedFile = {
            "id": file_id,
            "name": name,
            "type": file_type,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.state.file_uploads.append(uploaded_file)
        return file_id

    def get_uploaded_files(self) -> List[UploadedFile]:
        """
        Retrieves the list of all uploaded files.

        Returns:
            List[UploadedFile]: A list of all uploaded file records.
        """
        return self.state.file_uploads

    def remove_uploaded_file(self, file_id: str) -> bool:
        """
        Removes an uploaded file record from the state by its ID.

        Args:
            file_id (str): The unique ID of the file to remove.

        Returns:
            bool: True if a file was removed, False otherwise.
        """
        initial_len = len(self.state.file_uploads)
        self.state.file_uploads = [
            f for f in self.state.file_uploads if f['id'] != file_id
        ]
        return len(self.state.file_uploads) < initial_len

    # --- Settings Management ---

    def get_setting(self, key: str, default: Any = None) -> Any:
        """A convenience method to get top-level settings."""
        return self.get(key, default)

    def update_setting(self, key: str, value: Any) -> None:
        """A convenience method to update top-level settings."""
        self.set(key, value)

    def get_api_key(self, provider: str) -> str:
        """
        Gets an API key for a specific provider.

        Args:
            provider (str): The name of the provider (e.g., 'openai').

        Returns:
            str: The API key, or an empty string if not found.
        """
        return self.state.api_keys.get(provider, "")

    def set_api_key(self, provider: str, key: str) -> None:
        """
        Sets an API key for a specific provider.

        Args:
            provider (str): The name of the provider.
            key (str): The API key value.
        """
        self.state.api_keys[provider] = key

    def get_tool_enabled(self, tool_name: str) -> bool:
        """
        Checks if a specific tool is enabled.

        Args:
            tool_name (str): The name of the tool.

        Returns:
            bool: True if the tool is enabled, False otherwise.
        """
        return self.state.tools_enabled.get(tool_name, False)

    def set_tool_enabled(self, tool_name: str, status: bool) -> None:
        """
        Enables or disables a specific tool.

        Args:
            tool_name (str): The name of the tool.
            status (bool): The new enablement status.
        """
        if tool_name in self.state.tools_enabled:
            self.state.tools_enabled[tool_name] = status

# --- Global Accessor Function ---

_state_manager_instance = None

def get_state_manager() -> "StateManager":
    """
    Returns a singleton instance of the StateManager for the current session.
    
    This function ensures that the StateManager is initialized only once per
    session and provides a convenient way to access it from anywhere in the app.
    """
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager(st.session_state)
        _state_manager_instance.initialize_state()
    return _state_manager_instance
