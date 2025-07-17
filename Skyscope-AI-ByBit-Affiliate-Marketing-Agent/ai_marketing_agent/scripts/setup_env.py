import os
import platform

def setup_gemini_api_key():
    """
    Interactively prompts the user for their Gemini API Key and provides
    instructions for setting it as an environment variable.
    Optionally sets it for the current session if possible.
    """
    print("--- Gemini API Key Setup ---")
    print("This script will help you set up your Gemini API Key for use with the AI Marketing Agent.")
    print("The scripts are designed to read the API key from an environment variable.")
    print("\nIf you haven't already, you can obtain a Gemini API key from Google AI Studio.")

    api_key = input("\nPlease enter your Gemini API Key: ").strip()

    if not api_key:
        print("\nNo API key entered. Exiting setup.")
        return

    # Attempt to set for current session (might not affect parent shell or subsequent script runs directly)
    try:
        os.environ["GEMINI_API_KEY"] = api_key
        print("\nSuccessfully set GEMINI_API_KEY for the current Python script session.")
        print(f"Test: os.environ.get('GEMINI_API_KEY') is now: {os.environ.get('GEMINI_API_KEY')[:5]}... (partially hidden for security)")
    except Exception as e:
        print(f"Could not set environment variable for current session: {e}")


    print("\n--- Instructions for Persistent Setup (Recommended) ---")
    print("To make this API key available for all future terminal sessions, you need to add it to your shell's configuration file.")

    shell_config_file = ""
    current_shell = os.environ.get("SHELL", "").lower()
    export_command = f'export GEMINI_API_KEY="{api_key}"' # Default export command

    if "bash" in current_shell:
        shell_config_file = "~/.bashrc"
    elif "zsh" in current_shell:
        shell_config_file = "~/.zshrc"
    elif "fish" in current_shell:
        shell_config_file = "~/.config/fish/config.fish"
        export_command = f'set -gx GEMINI_API_KEY "{api_key}"' # Fish-specific command
    else:
        # For other shells, provide a generic placeholder
        shell_config_file = "your_shell_config_file (e.g., ~/.bashrc, ~/.zshrc, ~/.profile)"
        # The default export_command might work for many POSIX-compliant shells (like ksh, dash)
        # but users of other shells (e.g. csh, tcsh) would need different syntax (e.g. )

    print(f"\n1. Open your shell configuration file: {shell_config_file}")
    print(f"   You can usually do this with a command like: nano {shell_config_file} or code {shell_config_file} or vi {shell_config_file}")
    print("\n2. Add the following line to the end of the file:")
    print(f"   {export_command}")
    print("\n3. Save the file and exit the editor.")
    print(f"\n4. Apply the changes by running: source {shell_config_file}")
    print("   (Or, simply open a new terminal window/tab.)")
    print("\nAfter this, the AI Marketing Agent scripts should be able to access the Gemini API key.")
    print("------------------------------------")

if __name__ == "__main__":
    setup_gemini_api_key()
