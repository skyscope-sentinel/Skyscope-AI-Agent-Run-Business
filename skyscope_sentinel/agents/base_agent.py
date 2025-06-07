# skyscope_sentinel/agents/base_agent.py

class BaseAgent:
    """
    A base class for AI agents in the Skyscope Sentinel system.
    Provides fundamental functionalities like message handling, task processing, and status reporting.
    """

    def __init__(self, agent_id: str):
        """
        Initializes the BaseAgent.

        Args:
            agent_id (str): A unique identifier for the agent.
        """
        self.agent_id = agent_id
        self.status = "idle"  # Possible statuses: idle, message_received, processing, error, task_complete
        self.message_log = []  # Stores tuples of (sender_id, message_content)

        print(f"[BaseAgent {self.agent_id}] initialized and idle.")

    def receive_message(self, sender_id: str, message_content):
        """
        Handles an incoming message from another agent or system component.

        Args:
            sender_id (str): The ID of the message sender.
            message_content (str or dict): The content of the message.
        """
        print(f"[BaseAgent {self.agent_id}] received message from '{sender_id}': {message_content}")
        self.message_log.append((sender_id, message_content))
        self.status = "message_received"
        # In a real scenario, this might trigger further processing or task creation based on the message.

    def process_task(self, task_description):
        """
        Processes a given task. This is a placeholder for actual task execution logic.

        Args:
            task_description (str or dict): A description of the task to be performed.
        """
        print(f"[BaseAgent {self.agent_id}] received task: '{task_description}'. Processing...")
        self.status = "processing"

        # --- Placeholder for actual task logic ---
        # In a real agent, this method would contain the core logic for the agent's specialization.
        # For example, a ContentCreationAgent might call an LLM here,
        # a CodingAgent might write or execute code, etc.
        print(f"[BaseAgent {self.agent_id}] Task processing placeholder. No actual work done.")
        # --- End of placeholder ---

        # After processing, the agent might transition to 'task_complete' or back to 'idle'
        # For simplicity, we'll set it back to 'idle'.
        self.status = "idle"
        print(f"[BaseAgent {self.agent_id}] finished processing task, now idle.")


    def get_status(self) -> str:
        """
        Returns the current status of the agent.

        Returns:
            str: The current status (e.g., "idle", "processing").
        """
        return self.status

    def __str__(self):
        return f"BaseAgent(id='{self.agent_id}', status='{self.status}')"

    def __repr__(self):
        return f"BaseAgent(agent_id='{self.agent_id}')"

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("\n--- Testing BaseAgent ---")

    agent1 = BaseAgent(agent_id="Agent001-Test")
    print(f"Initial status of {agent1.agent_id}: {agent1.get_status()}")

    agent1.receive_message(sender_id="SystemControl", message_content="System online. Standby for tasks.")
    print(f"Status after message: {agent1.get_status()}")
    print(f"Message log for {agent1.agent_id}: {agent1.message_log}")

    agent1.process_task(task_description="Analyze market sentiment report for Q3.")
    print(f"Status after task processing: {agent1.get_status()}")

    agent2 = BaseAgent(agent_id="Agent002-Supervisor")
    agent2.receive_message(sender_id=agent1.agent_id, message_content={"report_summary_ref": "doc_xyz.pdf", "status": "analysis_pending"})
    print(f"Agent2 status: {agent2.get_status()}")

    print(f"\nString representation of agent1: {str(agent1)}")
    print(f"Official representation of agent1: {repr(agent1)}")
    print("--- End of BaseAgent Test ---")
