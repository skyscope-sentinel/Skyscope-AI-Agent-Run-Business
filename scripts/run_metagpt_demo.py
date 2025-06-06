import sys
import os
import time # For simulating delays or periodic checks
import json # For pretty printing PRD

# Add project root to Python path for sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from skyscope_sentinel.agents.base_agent import BaseAgent
from skyscope_sentinel.agents.metagpt_pm_agent import ProductManagerAgent
from skyscope_sentinel.agents.metagpt_engineer_agent import EngineerAgent
from skyscope_sentinel.agents.messaging import AgentMessageQueue
from skyscope_sentinel.ollama_integration import OllamaIntegration

# Configuration for models
# Ensure these models are available via `ollama list`
PM_MODEL = "qwen2:0.5b"  # Model for Product Manager (PRD generation)
ENG_MODEL = "qwen2:0.5b" # Model for Engineer (Code generation) - can be same or different

def check_models_availability(oi: OllamaIntegration, models_to_check: list[str]) -> bool:
    print("Checking Ollama model availability...")
    available_models_list, error = oi.list_models_sync()
    if error:
        print(f"ERROR: Could not list Ollama models: {error}")
        print("Please ensure Ollama is running and accessible.")
        return False

    available_model_names = []
    if available_models_list: # Ensure it's not None
        available_model_names = [m.get('name') for m in available_models_list if m.get('name')]

    all_models_found = True
    for model_name in models_to_check:
        if model_name not in available_model_names:
            print(f"ERROR: Required model '{model_name}' not found.")
            all_models_found = False

    if not all_models_found:
        print(f"Available models: {available_model_names}")
        print(f"Please ensure all required models ({', '.join(models_to_check)}) are pulled via Ollama (e.g., `ollama pull {models_to_check[0]}`).")
        return False

    print(f"All required models ({', '.join(models_to_check)}) are available.")
    return True

def main():
    print("--- Simplified MetaGPT-inspired Loop Demo (Ollama-based) ---")

    # 0. Setup Ollama Integration and Model Availability Check
    ollama_integration = OllamaIntegration()
    required_models = list(set([PM_MODEL, ENG_MODEL])) # Unique list of models needed
    if not check_models_availability(ollama_integration, required_models):
        return # Exit if models are not available

    # 1. Initialization
    print("\n1. Initializing components...")
    message_queue = AgentMessageQueue()

    user_agent = BaseAgent(agent_id="UserAgent_001")

    pm_agent = ProductManagerAgent(
        agent_id="PM_Agent_001",
        ollama_integration_instance=ollama_integration,
        model_name=PM_MODEL
    )

    engineer_agent = EngineerAgent(
        agent_id="Eng_Agent_001",
        ollama_integration_instance=ollama_integration,
        model_name=ENG_MODEL
    )

    print(f"Initialized agents: {user_agent.agent_id}, {pm_agent.agent_id}, {engineer_agent.agent_id}")

    # 2. User Initiates Task
    print("\n2. UserAgent sends initial requirement to ProductManagerAgent...")
    # user_requirement = "Create a command-line Python program that acts as a simple calculator. It should take two numbers and an operation (add, subtract, multiply, divide) from the user, perform the calculation, and print the result. Include basic error handling for division by zero."
    user_requirement = "Develop a Python script that takes a directory path as input using argparse and lists all .txt files in that directory. It should also count them and print the total."


    initial_message_content = {
        "type": "user_requirement",
        "requirement_text": user_requirement,
        "respond_to_agent_id": user_agent.agent_id
    }

    message_queue.send_message(
        sender_id=user_agent.agent_id,
        recipient_id=pm_agent.agent_id,
        content=initial_message_content
    )
    print(f"Message sent from {user_agent.agent_id} to {pm_agent.agent_id} with requirement: '{user_requirement[:70]}...'")

    # 3. Simulating Agent Workflow (Turn-Based)
    print("\n3. Simulating agent processing loop (up to 20 turns)...")

    max_turns = 20
    turns = 0
    final_product_received = False

    while turns < max_turns and not final_product_received:
        turns += 1
        print(f"\n--- Turn {turns} ---")
        action_taken_this_turn = False

        # ProductManagerAgent's Turn
        pm_messages = message_queue.get_messages_for_agent(pm_agent.agent_id)
        if pm_messages:
            action_taken_this_turn = True
            for msg in pm_messages:
                pm_agent.log(f"Received message from '{msg['sender_id']}': {str(msg['content'])[:100]}...")
                content = msg['content']
                if isinstance(content, dict) and content.get('type') == 'user_requirement':
                    original_user_id = content.get('respond_to_agent_id', user_agent.agent_id) # Get original sender

                    # PM generates PRD
                    prd_data = pm_agent.generate_prd(content['requirement_text'])
                    if prd_data:
                        # PM sends PRD to Engineer, including who the original requestor was
                        prd_message_to_engineer = {
                            "type": "prd_document",
                            "prd": prd_data,
                            "respond_to_agent_id": original_user_id
                        }
                        message_queue.send_message(
                            sender_id=pm_agent.agent_id,
                            recipient_id=engineer_agent.agent_id,
                            content=prd_message_to_engineer
                        )
                        pm_agent.log(f"Sent PRD to engineer '{engineer_agent.agent_id}' for user '{original_user_id}'.")
                    else:
                        pm_agent.log(f"Failed to generate PRD. Nothing sent to engineer '{engineer_agent.agent_id}'.")
                        # Optionally, send a failure message back to user_agent here
                        message_queue.send_message(
                            sender_id=pm_agent.agent_id,
                            recipient_id=original_user_id,
                            content={"type": "pm_error", "status": "failure", "error_message": "PM failed to generate PRD."}
                        )


        # EngineerAgent's Turn
        eng_messages = message_queue.get_messages_for_agent(engineer_agent.agent_id)
        if eng_messages:
            action_taken_this_turn = True
            for msg in eng_messages:
                engineer_agent.log(f"Received message from '{msg['sender_id']}': {str(msg['content'])[:100]}...")
                content = msg['content']
                if isinstance(content, dict) and content.get('type') == 'prd_document':
                    target_recipient_id = content.get('respond_to_agent_id', user_agent.agent_id) # Who to send code to

                    # Engineer generates code based on PRD
                    # The handle_prd_and_generate_code method will send the response
                    engineer_agent.handle_prd_and_generate_code(
                        prd_data=content['prd'],
                        message_queue=message_queue,
                        original_sender_id=target_recipient_id
                    )

        # UserAgent's Turn (to receive final product or errors)
        user_messages = message_queue.get_messages_for_agent(user_agent.agent_id)
        if user_messages:
            action_taken_this_turn = True
            for msg in user_messages:
                user_agent.receive_message(sender_id=msg['sender_id'], message_content=msg['content'])
                content = msg['content']
                if isinstance(content, dict):
                    if content.get('type') == 'code_generation_result':
                        print(f"  UserAgent received code generation result for project '{content.get('project_name')}':")
                        if content.get('status') == 'success':
                            print("    Status: SUCCESS")
                            code_snippet = content.get('generated_code_snippet', '')
                            print(f"    Generated Code Snippet (first 200 chars):\n      {code_snippet}")
                        else:
                            print("    Status: FAILURE")
                            print(f"      Error: {content.get('error_message', 'Unknown error')}")
                        final_product_received = True # Consider demo ended if final code/error is received
                    elif content.get('type') == 'pm_error':
                        print(f"  UserAgent received error from PM: {content.get('error_message')}")
                        final_product_received = True # Demo ends on PM error too

        if not action_taken_this_turn:
            print("  No agent actions this turn. Waiting for messages or processing...")

        if final_product_received:
            print("\nUserAgent received the final outcome. Demo loop will end.")
            break

        if turns >= max_turns: # Check if max_turns is reached
            print("Max turns reached.")
            break # Ensure loop terminates

        time.sleep(1.5) # Slightly increased sleep to clearly see turn progression and allow LLM time

    # 4. Final Status
    print("\n4. Demo Finished.")
    print(f"  ProductManagerAgent status: {pm_agent.get_status()}")
    print(f"  EngineerAgent status: {engineer_agent.get_status()}")
    print(f"  Message queue final size: {message_queue.get_queue_size()} (should be 0 if all processed)")
    if not final_product_received and turns >=max_turns:
        print("  WARNING: UserAgent did not receive the final code generation result within the turn limit.")

if __name__ == "__main__":
    main()
    time.sleep(0.1) # Ensure prints flush
