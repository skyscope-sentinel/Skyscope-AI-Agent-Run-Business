from PySide6.QtCore import QSettings, QVariant

class SettingsManager:
    def __init__(self, organization_name="Skyscope", application_name="Sentinel"):
        self.settings = QSettings(organization_name, application_name)

    def save_setting(self, key, value):
        """Saves a setting."""
        print(f"Saving setting: {key} = {value}")
        self.settings.setValue(key, value)
        self.settings.sync() # Ensure changes are written immediately

    def load_setting(self, key, default_value=None):
        """Loads a setting. Returns default_value if the key is not found."""
        value = self.settings.value(key, default_value)
        # QSettings might return strings for bools/ints on some platforms if not typed on save.
        # Try to infer type from default_value for robustness.
        if default_value is not None:
            if isinstance(default_value, bool):
                if isinstance(value, str):
                    if value.lower() == 'true': value = True
                    elif value.lower() == 'false': value = False
                    else: value = default_value # Fallback if string is not recognized bool
                elif not isinstance(value, bool): # If it's some other type, fallback
                    value = default_value
            elif isinstance(default_value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    value = default_value
            elif isinstance(default_value, float):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = default_value
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
