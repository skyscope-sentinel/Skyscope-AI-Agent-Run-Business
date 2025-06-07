import sys
import os
import time

# --- Python Path Setup ---
# Ensure the 'skyscope_sentinel' directory is in the Python path
# to allow importing modules from it.
try:
    # Attempt direct import if script is run from project root or PYTHONPATH is set
    from skyscope_sentinel.agents.base_agent import BaseAgent
    from skyscope_sentinel.agents.ollama_agent import OllamaWorkerAgent
    from skyscope_sentinel.agents.messaging import AgentMessageQueue
    from skyscope_sentinel.ollama_integration import OllamaIntegration
except ImportError:
    # Fallback: Add project root to sys.path
    # This assumes 'scripts/' is one level down from the project root.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Retry imports after path adjustment
    try:
        from skyscope_sentinel.agents.base_agent import BaseAgent
        from skyscope_sentinel.agents.ollama_agent import OllamaWorkerAgent
        from skyscope_sentinel.agents.messaging import AgentMessageQueue
        from skyscope_sentinel.ollama_integration import OllamaIntegration
    except ImportError as e:
        print(f"Error: Could not import necessary modules after path adjustment: {e}")
        print(f"Current sys.path: {sys.path}")
        print("Please ensure the script is run from the project root or that Skyscope Sentinel is correctly installed.")
        sys.exit(1)

# --- Configuration ---
SUPERVISOR_AGENT_ID = "Supervisor01"
OLLAMA_WORKER_AGENT_ID = "OllamaWorker001"
# IMPORTANT: Ensure this model is available in your Ollama instance.
# You can pull it with `ollama pull qwen2:0.5b` (a small, relatively fast model).
OLLAMA_MODEL_NAME = "qwen2:0.5b"

# --- Main Demo Logic ---
def run_demo():
    print("--- Starting AI Agent Co-operation Demo ---")

    # 1. Initialize Core Services
    print("\n[Phase 1] Initializing core services...")
    message_queue = AgentMessageQueue()
    ollama_integration = OllamaIntegration()

    # Check Ollama version (and implicitly, if Ollama is accessible)
    ollama_version, error = ollama_integration.get_ollama_version_sync()
    if error:
        print(f"  Error connecting to Ollama or getting version: {error}")
        print("  Please ensure Ollama is running and accessible to proceed with the demo.")
        return
    print(f"  Ollama Integration initialized. Ollama version: {ollama_version}")
    print("  Agent Message Queue initialized.")

    # 2. Initialize Agents
    print("\n[Phase 2] Initializing Agents...")
    supervisor_agent = BaseAgent(agent_id=SUPERVISOR_AGENT_ID)
    ollama_worker_agent = OllamaWorkerAgent(
        agent_id=OLLAMA_WORKER_AGENT_ID,
        model_name=OLLAMA_MODEL_NAME,
        ollama_integration_instance=ollama_integration
    )
    print(f"  {supervisor_agent} initialized.")
    print(f"  {ollama_worker_agent} initialized (model: {OLLAMA_MODEL_NAME}).")

    # 3. Supervisor Agent sends a task to OllamaWorkerAgent via Message Queue
    print(f"\n[Phase 3] '{SUPERVISOR_AGENT_ID}' sending task to '{OLLAMA_WORKER_AGENT_ID}'...")
    task_prompt = "Write a short, optimistic news headline about the future of renewable energy."
    system_prompt_for_task = "You are a news editor creating concise and engaging headlines."
    task_content = {
        "task_type": "generate_text",
        "prompt": task_prompt,
        "system_prompt": system_prompt_for_task,
        "target_agent_type": "OllamaWorker" # For potential routing by supervisor
    }
    message_queue.send_message(
        sender_id=SUPERVISOR_AGENT_ID,
        recipient_id=OLLAMA_WORKER_AGENT_ID,
        content=task_content
    )
    print(f"  Message sent. Queue size: {message_queue.get_queue_size()}")

    # 4. OllamaWorkerAgent checks for messages and processes the task
    print(f"\n[Phase 4] '{OLLAMA_WORKER_AGENT_ID}' checking for messages and processing task...")
    worker_messages = message_queue.get_messages_for_agent(agent_id=OLLAMA_WORKER_AGENT_ID)

    if not worker_messages:
        print(f"  No messages found for {OLLAMA_WORKER_AGENT_ID}. Demo cannot proceed as expected.")
        return

    for msg in worker_messages:
        print(f"  '{OLLAMA_WORKER_AGENT_ID}' received message from '{msg['sender_id']}': {msg['content']}")
        ollama_worker_agent.receive_message(sender_id=msg['sender_id'], message_content=msg['content'])

        # Assuming the message content is the task description for Ollama agent
        if isinstance(msg['content'], dict) and msg['content'].get("task_type") == "generate_text":
            task_payload = {
                "prompt": msg['content'].get("prompt"),
                "system_prompt": msg['content'].get("system_prompt")
            }
            generated_text, error = ollama_worker_agent.process_task(task_payload)

            if error:
                error_message = f"  Task processing by '{OLLAMA_WORKER_AGENT_ID}' failed: {error}"
                print(error_message)
                # Send error message back to supervisor
                message_queue.send_message(
                    sender_id=OLLAMA_WORKER_AGENT_ID,
                    recipient_id=SUPERVISOR_AGENT_ID,
                    content={"status": "error", "detail": error_message, "original_task": task_content}
                )
            else:
                success_message = f"  Task processed by '{OLLAMA_WORKER_AGENT_ID}'. Generated text:"
                print(success_message)
                print(f"    \"\"\"\n    {generated_text}\n    \"\"\"")
                # Send result back to supervisor
                message_queue.send_message(
                    sender_id=OLLAMA_WORKER_AGENT_ID,
                    recipient_id=SUPERVISOR_AGENT_ID,
                    content={"status": "success", "result": generated_text, "original_task": task_content}
                )
        else:
            print(f"  '{OLLAMA_WORKER_AGENT_ID}' received non-task message or unknown task_type: {msg['content']}")

    print(f"  Message processing complete for '{OLLAMA_WORKER_AGENT_ID}'. Queue size: {message_queue.get_queue_size()}")

    # 5. Supervisor Agent checks for responses
    print(f"\n[Phase 5] '{SUPERVISOR_AGENT_ID}' checking for responses...")
    supervisor_responses = message_queue.get_messages_for_agent(agent_id=SUPERVISOR_AGENT_ID)

    if not supervisor_responses:
        print(f"  No responses found for {SUPERVISOR_AGENT_ID}.")

    for response in supervisor_responses:
        print(f"  '{SUPERVISOR_AGENT_ID}' received response from '{response['sender_id']}':")
        if response['content'].get('status') == 'success':
            print(f"    Task Succeeded! Result: '{response['content'].get('result')}'")
        elif response['content'].get('status') == 'error':
            print(f"    Task Failed! Detail: '{response['content'].get('detail')}'")
        else:
            print(f"    Unknown response format: {response['content']}")
        supervisor_agent.receive_message(sender_id=response['sender_id'], message_content=response['content'])

    print(f"  Response processing complete for '{SUPERVISOR_AGENT_ID}'. Queue size: {message_queue.get_queue_size()}")

    # Example of BaseAgent processing a generic task (not involving Ollama)
    print(f"\n[Phase 6] '{SUPERVISOR_AGENT_ID}' processing a local task...")
    supervisor_agent.process_task("Finalize daily report and send notifications.")
    print(f"  Supervisor status after local task: {supervisor_agent.get_status()}")


    print("\n--- AI Agent Co-operation Demo Finished ---")

if __name__ == "__main__":
    run_demo()
    # Add a small delay to ensure all print statements are flushed, especially if run in certain environments.
    time.sleep(0.1)
