import json
import base64
import string
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Data Conversion ---

def to_json(data: Union[Dict, List], pretty: bool = False) -> Optional[str]:
    """
    Converts a Python dictionary or list to a JSON string.

    Args:
        data (Union[Dict, List]): The Python object to convert.
        pretty (bool): If True, formats the JSON with indentation for readability.

    Returns:
        Optional[str]: The JSON string representation of the data, or None on error.
    """
    try:
        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)
    except TypeError as e:
        log_message(f"Error converting data to JSON: {e}", level="ERROR")
        return None

def from_json(json_string: str) -> Optional[Union[Dict, List]]:
    """
    Parses a JSON string into a Python dictionary or list.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        Optional[Union[Dict, List]]: The parsed Python object, or None on error.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON string: {e}", level="ERROR")
        return None

def to_base64(data: Union[str, bytes]) -> str:
    """
    Encodes a string or bytes object into a Base64 string.

    Args:
        data (Union[str, bytes]): The data to encode.

    Returns:
        str: The Base64 encoded string.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')

def from_base64(encoded_string: str) -> bytes:
    """
    Decodes a Base64 string into bytes.

    Args:
        encoded_string (str): The Base64 string to decode.

    Returns:
        bytes: The decoded bytes object.
    """
    return base64.b64decode(encoded_string)

# --- String Manipulation ---

def generate_random_string(length: int = 12) -> str:
    """
    Generates a random alphanumeric string of a given length.

    Args:
        length (int): The desired length of the string.

    Returns:
        str: The randomly generated string.
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a string to be a valid filename.

    Removes or replaces characters that are not allowed in filenames on
    common operating systems (Windows, macOS, Linux).

    Args:
        filename (str): The input string to sanitize.

    Returns:
        str: A sanitized, safe filename string.
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace whitespace with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Limit length to avoid issues with filesystem limits
    return sanitized[:250]

# --- Date and Time ---

def get_formatted_timestamp() -> str:
    """
    Gets the current timestamp in a standard, readable format.

    Returns:
        str: The formatted timestamp (e.g., "2025-07-07 15:30:00").
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Miscellaneous ---

def log_message(message: str, level: str = "INFO") -> None:
    """
    Prints a log message to the console with a timestamp and severity level.

    Args:
        message (str): The message to log.
        level (str): The severity level ("INFO", "WARNING", "ERROR").
    """
    timestamp = get_formatted_timestamp()
    print(f"[{timestamp}] [{level.upper()}] {message}", file=sys.stderr if level.upper() == "ERROR" else sys.stdout)

def run_shell_command(command: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """
    Runs a shell command and captures its output.

    Args:
        command (str): The command to execute.
        timeout (int): The timeout in seconds for the command.

    Returns:
        Tuple[bool, str, str]: A tuple containing:
            - success (bool): True if the command exited with code 0, False otherwise.
            - stdout (str): The standard output of the command.
            - stderr (str): The standard error of the command.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  # We check the returncode manually
        )
        success = result.returncode == 0
        if not success:
            log_message(f"Command failed with exit code {result.returncode}: {command}", level="WARNING")
        return success, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        log_message(f"Command timed out after {timeout} seconds: {command}", level="ERROR")
        return False, "", "Command timed out."
    except Exception as e:
        log_message(f"An exception occurred while running command '{command}': {e}", level="ERROR")
        return False, "", str(e)


if __name__ == "__main__":
    log_message("--- Running utils.py demonstration ---", level="INFO")

    # Data Conversion Demo
    log_message("--- Data Conversion Demo ---")
    my_data = {"name": "Skyscope", "version": 1.0, "features": ["AI", "Swarms"]}
    json_str = to_json(my_data, pretty=True)
    print(f"Pretty JSON:\n{json_str}")
    parsed_data = from_json(json_str)
    print(f"Parsed back to Python dict: {parsed_data}")

    b64_encoded = to_base64("Hello, Skyscope Sentinel!")
    print(f"Base64 Encoded: {b64_encoded}")
    b64_decoded = from_base64(b64_encoded)
    print(f"Base64 Decoded: {b64_decoded.decode('utf-8')}")
    print("-" * 20)

    # String Manipulation Demo
    log_message("--- String Manipulation Demo ---")
    random_str = generate_random_string(16)
    print(f"Generated Random String: {random_str}")
    unsafe_filename = "My Report / 2025? <final*>.txt"
    safe_filename = sanitize_filename(unsafe_filename)
    print(f"Sanitized Filename: '{unsafe_filename}' -> '{safe_filename}'")
    print("-" * 20)
    
    # Date and Time Demo
    log_message("--- Date and Time Demo ---")
    print(f"Current Formatted Timestamp: {get_formatted_timestamp()}")
    print("-" * 20)

    # Shell Command Demo
    log_message("--- Shell Command Demo ---")
    # Use 'dir' on Windows and 'ls -l' on other systems
    list_command = "dir" if sys.platform == "win32" else "ls -l"
    
    log_message(f"Running command: '{list_command}'")
    success, stdout, stderr = run_shell_command(list_command)
    if success:
        log_message("Command executed successfully.")
        print("Output:\n", stdout)
    else:
        log_message("Command failed.", level="ERROR")
        print("Error Output:\n", stderr)

    log_message("Running a command that fails...")
    success, stdout, stderr = run_shell_command("non_existent_command")
    if not success:
        log_message("Command correctly failed as expected.")
        print("Error Output:\n", stderr)
    print("-" * 20)
    
    log_message("--- Demonstration Complete ---", level="INFO")
