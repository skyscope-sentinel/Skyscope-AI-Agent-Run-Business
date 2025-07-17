import cv2
from pyzbar.pyzbar import decode
import os

# Attempt to import necessary libraries and provide helpful error messages if they are missing.
try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is not installed. Please install it by running: pip install opencv-python")
    exit(1)

try:
    from pyzbar.pyzbar import decode
except ImportError:
    print("ERROR: pyzbar is not installed. Please install it by running: pip install pyzbar")
    # pyzbar might also need its C libraries installed on the system, e.g., sudo apt-get install libzbar0
    print("Note: pyzbar might also require system libraries like libzbar0 (e.g., 'sudo apt-get install libzbar0' on Debian/Ubuntu).")
    exit(1)

def extract_qr_link_from_image(image_path, default_if_not_found=None):
    """
    Attempts to detect and decode a QR code from the given image file.

    Args:
        image_path (str): The path to the image file.
        default_if_not_found (any, optional): Value to return if no QR code is found
                                             or an error occurs. Defaults to None.

    Returns:
        str: The decoded QR code data (e.g., a URL) if found, otherwise default_if_not_found.
    """
    if not os.path.exists(image_path):
        print(f"QR Processor INFO: Image file not found at '{image_path}'.")
        return default_if_not_found

    try:
        # Load the image using OpenCV
        img = cv2.imread(image_path)

        if img is None:
            print(f"QR Processor ERROR: Could not read or decode image at '{image_path}'. File might be corrupted or not a supported image format.")
            return default_if_not_found

        # Decode QR codes
        decoded_objects = decode(img)

        if decoded_objects:
            # For simplicity, return the data from the first QR code found
            qr_data = decoded_objects[0].data.decode('utf-8')
            print(f"QR Processor INFO: Found QR code in '{image_path}'. Decoded data: {qr_data}")
            return qr_data
        else:
            print(f"QR Processor INFO: No QR code found in '{image_path}'.")
            return default_if_not_found
    except Exception as e:
        print(f"QR Processor ERROR: An exception occurred while processing image '{image_path}': {e}")
        return default_if_not_found

if __name__ == "__main__":
    print("--- Testing QR Code Processor ---")

    # Test images expected to be in the root directory relative to where the agent is run
    # Adjust paths if your execution context is different or move images to a specific 'assets' folder.
    # base_path = "../../" # Assuming scripts are run from ai_marketing_agent/scripts/
                        # and images are in the root of the repo.
                        # This might need adjustment depending on actual execution environment.
                        # For subtasks, it's safer to assume paths relative to repo root.

    # Let's list files in the assumed root to see if the images are there from the subtask's perspective
    # This is a debug step for path confirmation
    print(f"Checking for images in repo root (from subtask perspective)...")
    try:
        # Use a path relative to the repo root for `os.listdir`
        # The subtask environment usually has the repo root as the current working directory.
        repo_root_files = os.listdir(".") # List files in the current directory (expected to be repo root)
        print(f"Files in current directory (expected repo root): {repo_root_files}")
    except Exception as e:
        print(f"Error listing files in current directory: {e}")


    test_images_info = [
        {"path": "ByBit.png", "expected": "QR code link or default"},
        {"path": "ByBit2.png", "expected": "QR code link or default"},
        {"path": "ByBit3.png", "expected": "QR code link or default"},
        {"path": "non_existent_image.png", "expected": "Default value (None or custom)"},
        # Add a test for an image without a QR code if you have one,
        # otherwise, the existing images will serve this purpose if they lack QR codes.
    ]

    default_return = "NO_QR_CODE_FOUND"

    for image_info in test_images_info:
        image_file_path = image_info["path"] # Use direct path, assuming repo root
        print(f"\nProcessing image: '{image_file_path}' (Expected: {image_info['expected']})")

        # Check if file exists before processing, as the script itself does this.
        # This helps confirm paths from the subtask's execution context.
        if not os.path.exists(image_file_path):
             print(f"Test Main: Image '{image_file_path}' does not exist at this path from script's CWD.")
             # If files are in root, and script is in ai_marketing_agent/scripts, need to adjust path for __main__
             # However, the `extract_qr_link_from_image` function should be callable with direct paths relative to repo root.

        extracted_link = extract_qr_link_from_image(image_file_path, default_if_not_found=default_return)

        if extracted_link == default_return:
            print(f"Result for '{image_file_path}': Default value returned as expected or QR not found ('{default_return}').")
        elif "ERROR" in str(extracted_link): # Check if my function returned an error string by mistake
             print(f"Result for '{image_file_path}': Function seems to have returned an error message: {extracted_link}")
        else:
            print(f"Result for '{image_file_path}': Extracted Link -> {extracted_link}")

    print("\n--- QR Code Processor Test Finished ---")
