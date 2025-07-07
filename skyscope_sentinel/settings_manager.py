from PySide6.QtCore import QSettings, QVariant
from cryptography.fernet import Fernet, InvalidToken
import os # For path joining if key is stored externally in future

# Define keys for settings that should be encrypted
# These should match the keys used in settings_page.py and config.py
SETTING_SERPER_API_KEY = "api_keys/serper_api_key"
SETTING_OPENAI_API_KEY = "api_keys/openai_api_key"
SETTING_E2B_API_KEY = "api_keys/e2b_api_key"
# Add other sensitive keys here as they are introduced, e.g., for financial info
SETTING_BTC_ADDRESS = "financials/btc_address"
SETTING_PAYPAL_EMAIL = "financials/paypal_email"
SETTING_PAYID = "financials/payid"
SETTING_BSB_NUMBER = "financials/bsb_number"
SETTING_ACCOUNT_NUMBER = "financials/account_number"


ENCRYPTED_SETTINGS_KEYS = {
    SETTING_SERPER_API_KEY,
    SETTING_OPENAI_API_KEY,
    SETTING_E2B_API_KEY,
    # Financial details that are sensitive if directly viewable, though public addresses are less so.
    # For consistency and future-proofing, encrypting them is a good practice.
    SETTING_BTC_ADDRESS, # While a public address, encrypting for privacy of user's config
    SETTING_PAYPAL_EMAIL,
    SETTING_PAYID,
    SETTING_BSB_NUMBER,
    SETTING_ACCOUNT_NUMBER,
}

class SettingsManager:
    # ##########################################################################
    # ## WARNING: HARDCODED KEY - FOR LOCAL DEVELOPMENT/TESTING ONLY!         ##
    # ## This key MUST be replaced with a secure key management solution      ##
    # ## (e.g., OS keychain, user-derived key, hardware token) before any     ##
    # ## deployment or use with real sensitive data.                          ##
    # ## Leaking this key would compromise all encrypted settings.            ##
    # ##########################################################################
    # This key was generated once using Fernet.generate_key() and then hardcoded.
    # In a real app, you would NOT store it like this.
    _ENCRYPTION_KEY = b'p9qHCyVODpP4PDJm6Qh20sZKA2Zt9gX7eU1mJzRe_hA='
    # ##########################################################################

    def __init__(self, organization_name="Skyscope", application_name="Sentinel"):
        self.settings = QSettings(organization_name, application_name)
        try:
            self.fernet = Fernet(self._ENCRYPTION_KEY)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Fernet for encryption: {e}. Settings will not be encrypted.")
            self.fernet = None # Disable encryption if key is bad or Fernet fails

    def _is_encrypted_key(self, key):
        return key in ENCRYPTED_SETTINGS_KEYS

    def save_setting(self, key, value):
        """Saves a setting. Encrypts if key is in ENCRYPTED_SETTINGS_KEYS."""

        if self._is_encrypted_key(key) and self.fernet:
            if value is None: # Avoid encrypting None
                print(f"Saving setting (encrypted - None): {key} = None")
                self.settings.setValue(key, None)
            elif isinstance(value, str):
                if not value: # Avoid encrypting empty string, store as is or as None
                    print(f"Saving setting (encrypted - empty string): {key} = ''")
                    self.settings.setValue(key, "") # Store empty string as is
                else:
                    try:
                        encrypted_value = self.fernet.encrypt(value.encode()).decode()
                        print(f"Saving setting (encrypted): {key} = [ENCRYPTED]")
                        self.settings.setValue(key, encrypted_value)
                    except Exception as e:
                        print(f"Error encrypting setting {key}: {e}. Saving as plain text as fallback.")
                        self.settings.setValue(key, value) # Fallback to plain text
            else: # Not a string, save as is (or handle type error)
                print(f"Warning: Value for encrypted key {key} is not a string. Saving as plain text.")
                self.settings.setValue(key, value)
        else:
            if self._is_encrypted_key(key) and not self.fernet:
                print(f"Warning: Fernet not initialized. Saving setting {key} as PLAIN TEXT.")
            # print(f"Saving setting: {key} = {value}") # Reduce console noise for non-encrypted
            self.settings.setValue(key, value)

        self.settings.sync()

    def load_setting(self, key, default_value=None):
        """Loads a setting. Decrypts if key is in ENCRYPTED_SETTINGS_KEYS."""
        raw_value = self.settings.value(key) # Load raw first

        if self._is_encrypted_key(key) and self.fernet:
            if raw_value is None or raw_value == "": # If None or empty string was stored
                # print(f"Loading setting (encrypted, was None/empty): {key} = {raw_value if raw_value is not None else default_value}")
                return raw_value if raw_value is not None else default_value

            if isinstance(raw_value, str):
                try:
                    decrypted_value = self.fernet.decrypt(raw_value.encode()).decode()
                    # print(f"Loading setting (decrypted): {key} = [DECRYPTED]")
                    return decrypted_value
                except InvalidToken:
                    print(f"Warning: Failed to decrypt setting {key} (InvalidToken - likely not encrypted or key mismatch). Returning raw or default.")
                    # Fallback: If it's an old plain text value, or if decryption fails, return it as is.
                    # This helps with migration if settings were previously unencrypted.
                    # Or, could return default_value to force re-entry. For now, return raw.
                    return raw_value
                except Exception as e:
                    print(f"Error decrypting setting {key}: {e}. Returning raw or default.")
                    return raw_value # Fallback
            else: # Should not happen if saved correctly as encrypted string
                print(f"Warning: Encrypted key {key} has non-string raw value. Returning as is or default.")
                return raw_value if raw_value is not None else default_value
        else: # Not an encrypted key or Fernet not available
            value = self.settings.value(key, default_value) # Use QSettings default handling
            # Type conversion logic from original implementation
            if default_value is not None and value is not None : # Ensure value is not None before type checks
                if isinstance(default_value, bool):
                    if isinstance(value, str):
                        if value.lower() == 'true': value = True
                        elif value.lower() == 'false': value = False
                        else: value = default_value
                    elif not isinstance(value, bool):
                        value = default_value
                elif isinstance(default_value, int):
                    try: value = int(value)
                    except (ValueError, TypeError): value = default_value
                elif isinstance(default_value, float):
                    try: value = float(value)
                    except (ValueError, TypeError): value = default_value
            # print(f"Loading setting: {key} (default: {default_value}) = {value}")
            return value

    def remove_setting(self, key):
        """Removes a setting."""
        self.settings.remove(key)
        self.settings.sync()

    def clear_all_settings(self):
        """Clears all application settings."""
        self.settings.clear()
        self.settings.sync()
        print("All application settings cleared.")

if __name__ == '__main__':
    # Example Usage
    settings_mgr = SettingsManager()

    # Test saving
    settings_mgr.save_setting("test/string_setting", "Hello World")
    settings_mgr.save_setting("test/int_setting", 123)
    settings_mgr.save_setting("test/bool_setting_true", True)
    settings_mgr.save_setting("test/bool_setting_false", False)

    # Test loading
    print(f"String: {settings_mgr.load_setting('test/string_setting', 'default_string')}")
    print(f"Int: {settings_mgr.load_setting('test/int_setting', 0)}")
    print(f"Bool True: {settings_mgr.load_setting('test/bool_setting_true', False)}")
    print(f"Bool False: {settings_mgr.load_setting('test/bool_setting_false', True)}")
    print(f"Non-existent: {settings_mgr.load_setting('test/non_existent', 'default_val')}")
    
    # Test boolean conversion from string (simulating how QSettings might store them)
    settings_mgr.save_setting("test/bool_from_string_true", "true")
    settings_mgr.save_setting("test/bool_from_string_false", "false")
    settings_mgr.save_setting("test/bool_from_string_invalid", "notabool")

    print(f"Bool from 'true' string: {settings_mgr.load_setting('test/bool_from_string_true', False)}")
    print(f"Bool from 'false' string: {settings_mgr.load_setting('test/bool_from_string_false', True)}")
    print(f"Bool from 'notabool' string: {settings_mgr.load_setting('test/bool_from_string_invalid', True)}") # Should be True (default)

    # Test clearing
    # settings_mgr.clear_all_settings()
    # print(f"After clear, String: {settings_mgr.load_setting('test/string_setting', 'default_string')}")
    print("Settings Manager test complete.")
