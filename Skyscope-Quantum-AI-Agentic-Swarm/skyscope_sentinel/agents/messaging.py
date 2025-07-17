# skyscope_sentinel/agents/messaging.py

import time # For potential timestamping in future, not used now

class AgentMessageQueue:
    """
    A simple in-memory message queue for inter-agent communication.
    Messages are dictionaries and are removed from the queue once retrieved by the recipient.
    """

    def __init__(self):
        """
        Initializes the AgentMessageQueue.
        """
        self.message_queue = []  # Stores message dictionaries
        print(f"[AgentMessageQueue] Initialized.")

    def send_message(self, sender_id: str, recipient_id: str, content):
        """
        Adds a message to the queue.

        Args:
            sender_id (str): The ID of the agent sending the message.
            recipient_id (str): The ID of the agent intended to receive the message.
            content (str or dict): The payload of the message.
        """
        message = {
            'message_id': f"msg_{int(time.time() * 1000)}_{len(self.message_queue)}", # Basic unique ID
            'sender_id': sender_id,
            'recipient_id': recipient_id,
            'content': content,
            'timestamp': time.time(),
            'status': 'unread'  # Status helps in managing message lifecycle if not removing immediately
        }
        self.message_queue.append(message)
        print(f"[AgentMessageQueue] Message sent from '{sender_id}' to '{recipient_id}': {content}")

    def get_messages_for_agent(self, agent_id: str) -> list:
        """
        Retrieves all unread messages for a specific agent and removes them from the queue.

        Args:
            agent_id (str): The ID of the agent whose messages are to be retrieved.

        Returns:
            list: A list of message dictionaries intended for the agent.
                  Returns an empty list if no new messages are found.
        """
        messages_for_agent = []
        remaining_messages = []

        for message in self.message_queue:
            if message['recipient_id'] == agent_id and message['status'] == 'unread':
                # For this basic queue, we can either mark as 'read' and filter later,
                # or simply give it to the agent and remove it from the active queue.
                # We'll go with removal for this implementation.
                messages_for_agent.append(message)
                # Optionally mark as 'read' if we weren't removing: message['status'] = 'read'
            else:
                remaining_messages.append(message)

        self.message_queue = remaining_messages # Replace queue with non-retrieved messages

        if messages_for_agent:
            print(f"[AgentMessageQueue] Retrieved {len(messages_for_agent)} message(s) for '{agent_id}'.")
        else:
            print(f"[AgentMessageQueue] No new messages for '{agent_id}'.")

        return messages_for_agent

    def get_all_messages(self) -> list:
        """
        Returns all messages currently in the queue. For debugging or inspection.
        Does not change message statuses or remove them.

        Returns:
            list: A list of all message dictionaries in the queue.
        """
        return self.message_queue

    def get_queue_size(self) -> int:
        """
        Returns the total number of messages currently in the queue.
        """
        return len(self.message_queue)

if __name__ == '__main__':
    print("\n--- Testing AgentMessageQueue ---")
    mq = AgentMessageQueue()

    # Test sending messages
    mq.send_message(sender_id="Agent001", recipient_id="Agent002", content="Hello Agent002, how are you?")
    mq.send_message(sender_id="System", recipient_id="Agent001", content={"task_id": "T123", "action": "process_data"})
    mq.send_message(sender_id="Agent003", recipient_id="Agent002", content="Meeting scheduled for 3 PM tomorrow.")
    mq.send_message(sender_id="Agent002", recipient_id="Agent001", content="I am fine, thanks! Ready for tasks.")

    print(f"Current queue size: {mq.get_queue_size()}")
    # print(f"All messages in queue: {mq.get_all_messages()}")

    # Test retrieving messages for Agent002
    messages_agent002 = mq.get_messages_for_agent(agent_id="Agent002")
    print(f"Messages for Agent002: {messages_agent002}")
    assert len(messages_agent002) == 2

    print(f"Queue size after Agent002 retrieval: {mq.get_queue_size()}")
    # print(f"Remaining messages in queue: {mq.get_all_messages()}")


    # Test retrieving messages for Agent001
    messages_agent001 = mq.get_messages_for_agent(agent_id="Agent001")
    print(f"Messages for Agent001: {messages_agent001}")
    assert len(messages_agent001) == 2 # One from System, one from Agent002

    print(f"Queue size after Agent001 retrieval: {mq.get_queue_size()}")
    assert mq.get_queue_size() == 0

    # Test retrieving messages for an agent with no messages
    messages_agent003 = mq.get_messages_for_agent(agent_id="Agent003")
    print(f"Messages for Agent003: {messages_agent003}")
    assert len(messages_agent003) == 0

    # Test sending another message after some retrievals
    mq.send_message(sender_id="System", recipient_id="Agent004", content="New critical alert for Agent004.")
    print(f"Queue size: {mq.get_queue_size()}")
    messages_agent004 = mq.get_messages_for_agent(agent_id="Agent004")
    print(f"Messages for Agent004: {messages_agent004}")
    assert len(messages_agent004) == 1
    assert mq.get_queue_size() == 0

    print("--- End of AgentMessageQueue Test ---")
