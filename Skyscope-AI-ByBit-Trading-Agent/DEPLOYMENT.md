# Deploying the Bybit Gemini Trading Bot

This guide provides instructions for deploying the Bybit Gemini Trading Bot on a Linux server (e.g., Ubuntu 20.04+).

## Prerequisites

*   **Linux Server:** A Linux server (Ubuntu 20.04 or later is recommended).
*   **Python:** Python 3.8 or higher installed. You can check with `python3 --version`.
*   **pip:** Python package installer. Usually comes with Python.
*   **git:** Version control system for cloning the repository. Install with `sudo apt update && sudo apt install git`.
*   **venv:** Python virtual environment module. Install with `sudo apt install python3-venv` if not already present.

## 1. Clone the Repository

First, clone the project repository to your server. Choose a suitable location, for example, your home directory or `/opt`.

```bash
git clone <your_repository_url> bybit_gemini_bot_project
cd bybit_gemini_bot_project
```
Replace `<your_repository_url>` with the actual URL of your Git repository.

## 2. Setup Python Virtual Environment

It's highly recommended to run the bot within a Python virtual environment to manage dependencies without interfering with system-wide packages.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```
You should see `(venv)` at the beginning of your shell prompt, indicating the virtual environment is active. To deactivate it later, simply type `deactivate`.

## 3. Configuration

The bot's configuration is managed through JSON files and environment variables.

### a. Configuration File

Copy the example configuration file and edit it with your settings:

```bash
cp configs/config.json.example configs/config.dev.json
nano configs/config.dev.json # Or use your preferred text editor
```

**Key fields to update in `configs/config.dev.json`:**

*   **Bybit API Keys:**
    *   `testnet_api_key_bybit`: Your Bybit Testnet API key.
    *   `testnet_api_secret_bybit`: Your Bybit Testnet API secret.
    *   `bybit_api_key` / `mainnet_api_key_bybit`: Your Bybit Mainnet API key (if using mainnet).
    *   `bybit_api_secret` / `mainnet_api_secret_bybit`: Your Bybit Mainnet API secret.
*   **Gemini API Key:**
    *   `gemini_api_key`: Your Google AI Studio (Gemini) API key.
*   **Operational Flags:**
    *   `use_bybit_testnet_connector`: Set to `true` to use Bybit Testnet, `false` for Mainnet. **Default to `true` for initial setup and testing.**
    *   `use_gemini_analyzer`: Set to `true` to enable Gemini analysis, `false` to disable (if you want to run without AI, e.g., for basic connectivity tests).
    *   `use_websockets`: Set to `true` to use WebSocket connections for real-time data, `false` to use REST API polling.
*   **Trading Parameters:**
    *   `trading_symbols`: A list of symbols to trade, e.g., `["BTCUSDT", "ETHUSDT"]`.
    *   `risk_per_trade_percentage`: Percentage of your available balance to risk per trade (e.g., `1.0` for 1%).
    *   `stop_loss_percentage`: Default stop-loss percentage if not provided by Gemini (e.g., `0.02` for 2%).
    *   `tp_fee_multiplier`: Multiplier for calculating take-profit based on fees (e.g., `3.0` for 3x fees).
    *   `taker_fee_rate`: The taker fee rate for your Bybit account (e.g., `0.00055` for 0.055%).

### b. API Key Security: Using Environment Variables (Recommended)

**IMPORTANT:** For security, especially on a server or if you version control your `config.dev.json` (which is generally not recommended for files with secrets), **do not hardcode live API keys directly into `config.dev.json`**.

The bot is configured (via `bybit_gemini_bot/config.py`) to first check for API keys in the JSON file. If a key is missing or contains a placeholder like "YOUR_...", it will then attempt to load the key from an environment variable.

**Using an `EnvironmentFile` with `systemd`:**

This is a secure way to provide environment variables to the bot when it runs as a service.

1.  **Create the Environment File:**
    Create a file, for example, at `/etc/trading_bot/environment_vars`.
    ```bash
    sudo mkdir -p /etc/trading_bot
    sudo nano /etc/trading_bot/environment_vars
    ```
2.  **Add API Keys to the File:**
    Add your API keys to this file, one per line:
    ```ini
    BYBIT_API_KEY="your_bybit_mainnet_api_key_here"
    BYBIT_API_SECRET="your_bybit_mainnet_api_secret_here"
    TESTNET_API_KEY_BYBIT="your_bybit_testnet_api_key_here"
    TESTNET_API_SECRET_BYBIT="your_bybit_testnet_api_secret_here"
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```
    **Note:** If you use these environment variables, you can leave the corresponding API key fields in `config.dev.json` as their placeholder values (e.g., "YOUR_BYBIT_API_KEY"). The bot will pick them up from the environment.

3.  **Secure the Environment File:**
    ```bash
    sudo chown your_username:your_groupname /etc/trading_bot/environment_vars # Change 'your_username' and 'your_groupname'
    sudo chmod 600 /etc/trading_bot/environment_vars
    ```
    This ensures only the specified user can read the file.

4.  **Reference in `systemd` Service:**
    The provided `trading_bot.service` file has a commented-out line:
    `# EnvironmentFile=/etc/trading_bot/environment_vars`
    Uncomment this line in your actual service file if you use this method.

## 4. Setup `systemd` Service

Using `systemd` allows the bot to run as a background service, start on boot, and restart on failure.

1.  **Place the Service File:**
    Copy the `trading_bot.service` file (generated in the previous step or provided with the project) to `/etc/systemd/system/`.
    ```bash
    sudo cp trading_bot.service /etc/systemd/system/trading_bot.service
    ```

2.  **Customize the Service File:**
    Edit the copied service file with your specific paths and user:
    ```bash
    sudo nano /etc/systemd/system/trading_bot.service
    ```
    **Key fields to customize:**
    *   `User`: The Linux user the bot will run as (e.g., `your_username`). **Do not use `root`**.
    *   `Group`: The group for the user (e.g., `your_groupname`).
    *   `WorkingDirectory`: The **absolute path** to the root of your project directory (e.g., `/home/your_username/bybit_gemini_bot_project` or `/opt/bybit_gemini_bot_project`).
    *   `ExecStart`: Ensure the paths to your virtual environment's `python` and `main.py` script are correct and absolute.
        *   Example: `ExecStart=/home/your_username/bybit_gemini_bot_project/venv/bin/python /home/your_username/bybit_gemini_bot_project/bybit_gemini_bot/main.py`
    *   `EnvironmentFile`: If you created an environment file for API keys, uncomment this line and ensure the path is correct.

3.  **Systemd Commands:**
    After creating and customizing the service file:
    ```bash
    # Reload the systemd daemon to recognize the new service
    sudo systemctl daemon-reload

    # Enable the service to start automatically on system boot
    sudo systemctl enable trading_bot.service

    # Start the service immediately
    sudo systemctl start trading_bot.service

    # Check the status of the service
    sudo systemctl status trading_bot.service
    ```
    Look for "active (running)" in the status. If there are errors, the status message will provide clues.

4.  **Viewing Logs:**
    Logs are sent to the systemd journal by default.
    ```bash
    # View live logs (follow mode)
    sudo journalctl -u trading_bot -f

    # View all logs for the service
    sudo journalctl -u trading_bot
    ```

## 5. Running the Bot

Once the service is started, the bot should be running. Monitor its activity using the `journalctl` commands above. The bot will log its actions, including initialization, API interactions, (test) trades, and any errors.

## 6. Updating the Bot

To update the bot with new code:

1.  **Navigate to the project directory:**
    ```bash
    cd /path/to/bybit_gemini_bot_project
    ```
2.  **Stop the service:**
    ```bash
    sudo systemctl stop trading_bot.service
    ```
3.  **Pull the latest code:**
    ```bash
    git pull origin main # Or your relevant branch
    ```
4.  **Re-activate virtual environment (if not already active for manual steps):**
    ```bash
    source venv/bin/activate
    ```
5.  **Install/update dependencies if `requirements.txt` changed:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Reload systemd daemon (if service file changed, unlikely for code updates):**
    ```bash
    # sudo systemctl daemon-reload # Only if trading_bot.service itself was modified
    ```
7.  **Restart the service:**
    ```bash
    sudo systemctl start trading_bot.service
    ```
8.  **Check status and logs:**
    ```bash
    sudo systemctl status trading_bot.service
    sudo journalctl -u trading_bot -f
    ```

This completes the deployment guide. Remember to prioritize security, especially for API keys and server access.
