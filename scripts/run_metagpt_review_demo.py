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
from skyscope_sentinel.agents.metagpt_reviewer_agent import ReviewerAgent
from skyscope_sentinel.agents.messaging import AgentMessageQueue
from skyscope_sentinel.ollama_integration import OllamaIntegration

# Configuration for models
# Ensure these models are available via `ollama list`
PM_MODEL = "qwen2:0.5b"  # Model for Product Manager (PRD generation)
ENG_MODEL = "qwen2:0.5b" # Model for Engineer (Code generation) - can be same or different
REVIEWER_MODEL = "qwen2:0.5b" # Model for Reviewer

# Global context store for multi-turn operations
pending_review_orchestration_context = {}

def check_models_availability(oi: OllamaIntegration, models_to_check: list[str]) -> bool:
    print("Checking Ollama model availability...")
    available_models_list, error = oi.list_models_sync()
    if error:
        print(f"ERROR: Could not list Ollama models: {error}. Please ensure Ollama is running.")
        return False
    available_model_names = [m.get('name') for m in available_models_list if m.get('name')] if available_models_list else []
    all_models_found = True
    for model_name in models_to_check:
        if model_name not in available_model_names:
            print(f"ERROR: Required model '{model_name}' not found.")
            all_models_found = False
    if not all_models_found:
        print(f"Available models: {available_model_names}")
        print(f"Please ensure all required models ({', '.join(models_to_check)}) are pulled via Ollama.")
        return False
    print(f"All required models ({', '.join(models_to_check)}) are available.")
    return True

def main():
    print("--- MetaGPT Review Loop Demo (Ollama-based) - Enhanced Output ---")
    global pending_review_orchestration_context

    ollama_integration = OllamaIntegration()
    required_models = list(set([PM_MODEL, ENG_MODEL, REVIEWER_MODEL]))
    if not check_models_availability(ollama_integration, required_models):
        return

    message_queue = AgentMessageQueue()
    user_agent = BaseAgent(agent_id="UserAgent_001")
    pm_agent = ProductManagerAgent(agent_id="PM_Agent_001", ollama_integration_instance=ollama_integration, model_name=PM_MODEL)
    engineer_agent = EngineerAgent(agent_id="Eng_Agent_001", ollama_integration_instance=ollama_integration, model_name=ENG_MODEL)
    reviewer_agent = ReviewerAgent(agent_id="Reviewer_Agent_001", ollama_integration_instance=ollama_integration, model_name=REVIEWER_MODEL)

    print(f"Initialized agents: User, PM, Engineer, Reviewer")

    initial_user_requirement_for_summary = "Develop a Python script that functions as a basic timer. It should allow the user to start the timer, and it will count down from a specified number of seconds. When the timer reaches zero, it should print a 'Time's up!' message. Input for seconds should be via command-line argument using argparse."
    print(f"\nUser Requirement: '{initial_user_requirement_for_summary}'")

    initial_message_content = {
        "type": "user_requirement",
        "requirement_text": initial_user_requirement_for_summary,
        "respond_to_agent_id": user_agent.agent_id
    }
    message_queue.send_message(user_agent.agent_id, pm_agent.agent_id, initial_message_content)
    print(f"UserAgent sent requirement to PM_Agent.")

    max_turns = 30
    turns = 0
    final_product_received_by_user = False # Renamed for clarity

    summary_prd_was_revised = False
    summary_code_was_revised = False
    summary_final_code_snippet = "Not generated or not received."
    summary_final_status = "Unknown"

    while turns < max_turns and not final_product_received_by_user:
        turns += 1
        print(f"\n--- Turn {turns} ---")
        action_taken_this_turn = False

        # PM Agent Logic
        pm_msgs = message_queue.get_messages_for_agent(pm_agent.agent_id)
        if pm_msgs:
            action_taken_this_turn = True
            for msg in pm_msgs:
                pm_agent.log(f"Received: '{msg['content'].get('type', 'Unknown type')}' from {msg['sender_id']}")
                content = msg['content']
                if content.get('type') == 'user_requirement':
                    generated_prd = pm_agent.handle_requirement_and_send_prd(
                        user_requirement_text=content['requirement_text'],
                        message_queue=message_queue,
                        engineer_agent_id=engineer_agent.agent_id,
                        reviewer_agent_id=reviewer_agent.agent_id,
                        original_user_id=content['respond_to_agent_id']
                    )
                    if generated_prd:
                        pending_review_orchestration_context[pm_agent.agent_id] = {
                            'type': 'prd_review_pending',
                            'original_prd': generated_prd,
                            'user_requirement': content['requirement_text'],
                            'original_user_id': content['respond_to_agent_id'],
                            'engineer_agent_id': engineer_agent.agent_id
                        }
                        print(f"  DEMO: PM generated initial PRD (snippet): {str(generated_prd)[:150]}...")
                        pm_agent.log("Context stored for PRD review feedback.")
                elif content.get('type') == 'prd_review_feedback':
                    feedback = content['feedback']
                    print(f"  DEMO: PM received PRD Review Feedback - Approved: {feedback.get('approved')}, Suggestions: {feedback.get('suggestions')}")
                    if not feedback.get('approved') and feedback.get('suggestions'):
                        print(f"  DEMO: PM will attempt to regenerate PRD based on feedback.")

                    context = pending_review_orchestration_context.pop(pm_agent.agent_id, None)
                    if context and context['type'] == 'prd_review_pending':
                        pm_agent.handle_prd_review_feedback(
                            review_feedback_content=feedback,
                            original_prd_data=context['original_prd'],
                            user_requirement_text=context['user_requirement'],
                            message_queue=message_queue,
                            engineer_agent_id=context['engineer_agent_id'],
                            original_user_id=context['original_user_id']
                        )
                    else:
                        pm_agent.log(f"Error: Received PRD review feedback but no valid context found. Context: {context}")

        # Engineer Agent Logic
        eng_msgs = message_queue.get_messages_for_agent(engineer_agent.agent_id)
        if eng_msgs:
            action_taken_this_turn = True
            for msg in eng_msgs:
                engineer_agent.log(f"Received: '{msg['content'].get('type', 'Unknown type')}' from {msg['sender_id']}")
                content = msg['content']
                if content.get('type') == 'prd_document':
                    if content.get('review_was_conducted') and content.get('prd_was_revised'):
                        summary_prd_was_revised = True # Capture for final summary
                        print(f"  DEMO: Engineer received a REVISED PRD.")
                    else:
                        summary_prd_was_revised = False # Ensure it's reset/set if not revised
                        print(f"  DEMO: Engineer received original PRD (Review conducted: {content.get('review_was_conducted', False)}).")
                    print(f"  DEMO: Engineer received PRD (snippet): {str(content['prd'])[:150]}...")

                    generated_code = engineer_agent.handle_prd_and_generate_code(
                        prd_data=content['prd'],
                        message_queue=message_queue,
                        original_sender_id=content['respond_to_agent_id'],
                        reviewer_agent_id=reviewer_agent.agent_id
                    )
                    if generated_code:
                        pending_review_orchestration_context[engineer_agent.agent_id] = {
                            'type': 'code_review_pending',
                            'original_code': generated_code,
                            'prd_data': content['prd'],
                            'final_recipient_id': content['respond_to_agent_id']
                        }
                        print(f"  DEMO: Engineer generated initial code (snippet): {generated_code[:150]}...")
                        engineer_agent.log("Context stored for code review feedback.")
                elif content.get('type') == 'code_review_feedback':
                    feedback = content['feedback']
                    print(f"  DEMO: Engineer received Code Review Feedback - Approved: {feedback.get('approved')}, Suggestions: {feedback.get('suggestions')}")
                    if not feedback.get('approved') and feedback.get('suggestions'):
                        print(f"  DEMO: Engineer will attempt to regenerate code based on feedback.")

                    context = pending_review_orchestration_context.pop(engineer_agent.agent_id, None)
                    if context and context['type'] == 'code_review_pending':
                        engineer_agent.handle_code_review_feedback(
                            review_feedback_content=feedback,
                            original_code=context['original_code'],
                            prd_data=context['prd_data'],
                            message_queue=message_queue,
                            final_recipient_id=context['final_recipient_id']
                        )
                    else:
                        engineer_agent.log(f"Error: Received code review feedback but no valid context found. Context: {context}")

        # Reviewer Agent Logic
        rev_msgs = message_queue.get_messages_for_agent(reviewer_agent.agent_id)
        if rev_msgs:
            action_taken_this_turn = True
            for msg in rev_msgs:
                reviewer_agent.log(f"Received: '{msg['content'].get('type', 'Unknown type')}' from {msg['sender_id']}'")
                content = msg['content']
                feedback_payload = None
                if content.get('type') == 'prd_review_request':
                    reviewer_agent.log(f"Reviewing PRD: {str(content['prd_data'])[:100]}...")
                    feedback = reviewer_agent.review_prd(prd_data=content['prd_data'])
                    feedback_payload = {"type": "prd_review_feedback", "feedback": feedback}
                    reviewer_agent.log(f"PRD Review - Approved: {feedback.get('approved')}, Suggestions: {len(feedback.get('suggestions', []))}")
                elif content.get('type') == 'code_review_request':
                    reviewer_agent.log(f"Reviewing Code: {content.get('code_to_review', '')[:100]}...")
                    feedback = reviewer_agent.review_code(
                        code_string=content['code_to_review'],
                        language=content.get('language', 'python'),
                        prd_data=content.get('prd_data_context')
                    )
                    feedback_payload = {"type": "code_review_feedback", "feedback": feedback}
                    reviewer_agent.log(f"Code Review - Approved: {feedback.get('approved')}, Suggestions: {len(feedback.get('suggestions', []))}")

                if feedback_payload:
                    message_queue.send_message(
                        sender_id=reviewer_agent.agent_id,
                        recipient_id=content['respond_to_agent_id'],
                        content=feedback_payload
                    )
                    reviewer_agent.log(f"Sent {feedback_payload['type']} to {content['respond_to_agent_id']}")

        # User Agent Logic
        user_msgs = message_queue.get_messages_for_agent(user_agent.agent_id)
        if user_msgs:
            action_taken_this_turn = True
            for msg in user_msgs:
                user_agent.receive_message(msg['sender_id'], msg['content'])
                content = msg['content']
                msg_type = content.get('type', 'unknown')

                if msg_type == 'code_generation_result':
                    print(f"  UserAgent FINAL RESULT for project '{content.get('project_name')}':")
                    summary_final_status = content.get('status', 'Unknown')
                    print(f"    Status: {summary_final_status}")
                    print(f"    PRD Review Conducted & Potentially Revised (before code gen): Yes (Revised: {summary_prd_was_revised})")
                    print(f"    Code Review Conducted: {content.get('review_was_conducted')}")
                    summary_code_was_revised = content.get('code_was_revised', False)
                    print(f"    Code Subsequently Revised by Engineer: {summary_code_was_revised}")

                    if "success" in summary_final_status.lower() and content.get('full_code'):
                         summary_final_code_snippet = content.get('full_code')
                         print(f"    Full Code:\n--------------------\n{summary_final_code_snippet}\n--------------------")
                    else:
                         summary_final_code_snippet = f"Error: {content.get('error_message') or content.get('detail', 'N/A')}"
                         print(f"    Error/Detail: {summary_final_code_snippet}")
                    final_product_received_by_user = True
                    break
                elif msg_type in ['prd_generation_failed', 'code_generation_failed', 'pm_error', 'eng_error', 'error_pm_logic_critical', 'error_eng_logic_critical']:
                    error_detail = content.get('error') or content.get('detail')
                    print(f"  UserAgent received CRITICAL ERROR from {msg['sender_id']}: {error_detail}")
                    summary_final_status = f"Critical Error: {error_detail}"
                    final_product_received_by_user = True
                    break
            if final_product_received_by_user: break

        if not action_taken_this_turn and not final_product_received_by_user :
            print("  No agent actions this turn. Message queue might be empty or agents waiting for Ollama.")

        if final_product_received_by_user:
            print("Final product or critical error received by UserAgent. Halting demo.")
            break
        if turns >= max_turns:
            print("Max turns reached.")
            break

        time.sleep(1.5)

    print("\n--- Demo Run Summary ---")
    print(f"Initial User Requirement: '{initial_user_requirement_for_summary}'")
    print(f"PRD Was Revised After Review: {summary_prd_was_revised}")
    print(f"Code Was Revised After Review: {summary_code_was_revised}")
    print(f"Final Status Reported to User: {summary_final_status}")
    print(f"Final Code Snippet (or error, first 500 chars):\n--------------------\n{summary_final_code_snippet[:500]}...\n--------------------")

    print("\n--- Agent Final Statuses ---")
    print(f"PM Status: {pm_agent.get_status()}")
    print(f"Engineer Status: {engineer_agent.get_status()}")
    print(f"Reviewer Status: {reviewer_agent.get_status()}")
    print(f"Message queue final size: {message_queue.get_queue_size()} (should be 0 if all processed by end)")
    if not final_product_received_by_user and turns >=max_turns: # Corrected variable name here
        print("  WARNING: UserAgent did not receive the final product within the turn limit.")

if __name__ == "__main__":
    main()
    time.sleep(0.1) # Ensure prints flush
