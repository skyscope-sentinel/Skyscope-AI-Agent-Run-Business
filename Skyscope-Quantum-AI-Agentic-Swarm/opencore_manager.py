import plistlib
import os
import uuid
from typing import Any, Dict, List, Optional, Union

# --- Utility Functions ---

def read_plist(file_path: str) -> Dict[str, Any]:
    """
    Reads a .plist file and returns its content as a dictionary.

    Args:
        file_path: The path to the .plist file.

    Returns:
        A dictionary containing the parsed data from the .plist file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other parsing errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    
    with open(file_path, 'rb') as fp:
        return plistlib.load(fp)

def write_plist(data: Dict[str, Any], file_path: str) -> None:
    """
    Writes a dictionary to a .plist file.

    Args:
        data: The dictionary to write.
        file_path: The path where the .plist file will be saved.
    """
    with open(file_path, 'wb') as fp:
        plistlib.dump(data, fp)

# --- Main Class ---

class OpenCoreManager:
    """
    A class to manage and manipulate OpenCore config.plist files.
    
    This manager provides methods to load, save, read, modify, and validate
    OpenCore configurations, simplifying the process of creating and
    maintaining Hackintosh EFIs.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the OpenCoreManager.

        Args:
            config_path (Optional[str]): The path to the config.plist file to load.
                                         If None, an empty configuration is created.
        """
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, file_path: str) -> None:
        """
        Loads and parses a config.plist file into the manager.

        Args:
            file_path (str): The path to the config.plist file.
        """
        self.config = read_plist(file_path)

    def save_config(self, file_path: str) -> None:
        """
        Saves the current configuration to a .plist file.

        Args:
            file_path (str): The path to save the config.plist file.
        """
        write_plist(self.config, file_path)

    def get_value(self, key_path: str) -> Any:
        """
        Gets a value from the configuration using a dot-separated path.

        Example:
            get_value('PlatformInfo.Generic.SystemProductName')

        Args:
            key_path (str): The dot-separated path to the key.

        Returns:
            The value at the specified path, or None if the path is not found.
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                if isinstance(value, list):
                    key = int(key)
                value = value[key]
            return value
        except (KeyError, TypeError, IndexError):
            return None

    def set_value(self, key_path: str, value: Any) -> bool:
        """
        Sets a value in the configuration using a dot-separated path.
        Creates nested dictionaries if they do not exist.

        Args:
            key_path (str): The dot-separated path to the key.
            value (Any): The value to set.
            
        Returns:
            True if the value was set successfully, False otherwise.
        """
        keys = key_path.split('.')
        current_level = self.config
        try:
            for i, key in enumerate(keys[:-1]):
                if isinstance(current_level, list):
                    key = int(key)
                    current_level = current_level[key]
                else:
                    if key not in current_level:
                        current_level[key] = {}
                    current_level = current_level[key]
            
            final_key = keys[-1]
            if isinstance(current_level, list):
                final_key = int(final_key)
                current_level[final_key] = value
            else:
                current_level[final_key] = value
            return True
        except (TypeError, IndexError):
            return False

    def _find_item_in_list(self, section_path: str, key: str, value: str) -> Optional[Dict[str, Any]]:
        """Helper to find an item in a list of dicts within the config."""
        item_list = self.get_value(section_path)
        if not isinstance(item_list, list):
            return None
        for item in item_list:
            if isinstance(item, dict) and item.get(key) == value:
                return item
        return None

    def _remove_item_from_list(self, section_path: str, key: str, value: str) -> bool:
        """Helper to remove an item from a list of dicts within the config."""
        item_list = self.get_value(section_path)
        if not isinstance(item_list, list):
            return False
        
        initial_len = len(item_list)
        new_list = [item for item in item_list if not (isinstance(item, dict) and item.get(key) == value)]
        
        if len(new_list) < initial_len:
            self.set_value(section_path, new_list)
            return True
        return False

    def add_kext(self, bundle_path: str, executable_path: str, plist_path: str, enabled: bool = True, comment: str = "") -> None:
        """
        Adds a new kext to the Kernel -> Add section.

        Args:
            bundle_path (str): The path to the kext bundle (e.g., "Lilu.kext").
            executable_path (str): The path to the executable inside the bundle.
            plist_path (str): The path to the Info.plist inside the bundle.
            enabled (bool): Whether the kext should be enabled.
            comment (str): An optional comment for the kext entry.
        """
        kext_entry = {
            "Arch": "Any",
            "BundlePath": bundle_path,
            "Comment": comment,
            "Enabled": enabled,
            "ExecutablePath": executable_path,
            "MaxKernel": "",
            "MinKernel": "",
            "PlistPath": plist_path,
        }
        kexts = self.get_value('Kernel.Add') or []
        # Avoid duplicates
        if not self._find_item_in_list('Kernel.Add', 'BundlePath', bundle_path):
            kexts.append(kext_entry)
            self.set_value('Kernel.Add', kexts)

    def remove_kext(self, bundle_path: str) -> bool:
        """
        Removes a kext from the Kernel -> Add section by its BundlePath.

        Args:
            bundle_path (str): The BundlePath of the kext to remove.
            
        Returns:
            True if the kext was removed, False otherwise.
        """
        return self._remove_item_from_list('Kernel.Add', 'BundlePath', bundle_path)

    def toggle_kext(self, bundle_path: str, enable: bool) -> bool:
        """
        Enables or disables a kext.

        Args:
            bundle_path (str): The BundlePath of the kext to toggle.
            enable (bool): True to enable, False to disable.
            
        Returns:
            True if the kext was found and toggled, False otherwise.
        """
        kext = self._find_item_in_list('Kernel.Add', 'BundlePath', bundle_path)
        if kext:
            kext['Enabled'] = enable
            return True
        return False

    def add_driver(self, path: str, enabled: bool = True, comment: str = "", arguments: str = "") -> None:
        """
        Adds a new UEFI driver to the UEFI -> Drivers section.

        Args:
            path (str): The path to the .efi driver file.
            enabled (bool): Whether the driver should be enabled.
            comment (str): An optional comment.
            arguments (str): Optional arguments for the driver.
        """
        driver_entry = {
            "Path": path,
            "Enabled": enabled,
            "Comment": comment,
            "Arguments": arguments,
        }
        drivers = self.get_value('UEFI.Drivers') or []
        if not self._find_item_in_list('UEFI.Drivers', 'Path', path):
            drivers.append(driver_entry)
            self.set_value('UEFI.Drivers', drivers)

    def remove_driver(self, path: str) -> bool:
        """
        Removes a UEFI driver by its path.

        Args:
            path (str): The path of the driver to remove.
            
        Returns:
            True if the driver was removed, False otherwise.
        """
        return self._remove_item_from_list('UEFI.Drivers', 'Path', path)

    def add_acpi_patch(self, patch: Dict[str, Any]) -> None:
        """
        Adds a new ACPI patch.

        Args:
            patch (Dict[str, Any]): A dictionary representing the patch.
        """
        patches = self.get_value('ACPI.Patch') or []
        patches.append(patch)
        self.set_value('ACPI.Patch', patches)

    def remove_acpi_patch(self, comment: str) -> bool:
        """
        Removes an ACPI patch by its comment.

        Args:
            comment (str): The unique comment of the patch to remove.
            
        Returns:
            True if the patch was removed, False otherwise.
        """
        return self._remove_item_from_list('ACPI.Patch', 'Comment', comment)

    def generate_smbios(self, model: str, count: int = 1) -> Dict[str, str]:
        """
        Generates new SMBIOS data for a given Mac model.
        
        NOTE: This is a placeholder. A real implementation would need a robust
        algorithm to generate valid serial numbers and MLB that pass Apple's checks.

        Args:
            model (str): The Mac model identifier (e.g., "MacPro7,1").
            count (int): The number of sets to generate (typically 1).

        Returns:
            A dictionary containing the generated SMBIOS data.
        """
        # This is a simplified generator. Real serials have complex structures.
        system_uuid = str(uuid.uuid4()).upper()
        serial_number = f"F5K{str(uuid.uuid4()).upper()[:9]}P7QM" # Example format
        mlb = f"F5K{str(uuid.uuid4()).upper()[:13]}C" # Example format
        
        smbios_data = {
            'SystemProductName': model,
            'SystemSerialNumber': serial_number,
            'SystemUUID': system_uuid,
            'MLB': mlb,
            'ROM': os.urandom(6) # Random 6-byte MAC address
        }
        
        # Apply to config
        self.set_value('PlatformInfo.Generic.SystemProductName', smbios_data['SystemProductName'])
        self.set_value('PlatformInfo.Generic.SystemSerialNumber', smbios_data['SystemSerialNumber'])
        self.set_value('PlatformInfo.Generic.SystemUUID', smbios_data['SystemUUID'])
        self.set_value('PlatformInfo.Generic.MLB', smbios_data['MLB'])
        self.set_value('PlatformInfo.Generic.ROM', smbios_data['ROM'])
        
        print(f"Generated SMBIOS for {model}: {smbios_data}")
        return smbios_data

    def apply_common_patches(self, cpu_family: str) -> None:
        """
        Applies a set of common patches based on hardware profile.
        
        NOTE: This is a placeholder to demonstrate the concept.
        
        Args:
            cpu_family (str): The CPU family (e.g., "Intel-Comet-Lake", "AMD-Ryzen").
        """
        print(f"Applying common patches for {cpu_family}...")
        if "AMD" in cpu_family:
            # Placeholder for AMD patches
            print("Applying AMD CPU patches...")
            # Example: self.add_kext(...) for AMD patches
        
        if "Intel" in cpu_family:
            print("Applying Intel CPU quirks...")
            # Example: self.set_value('Kernel.Quirks.AppleXcpmCfgLock', True)

    def validate_config(self) -> List[str]:
        """
        Performs a basic validation of the configuration.
        
        NOTE: This is a placeholder. A full implementation would be extensive,
        checking for common errors, conflicting kexts, and outdated settings.
        
        Returns:
            A list of strings, where each string is a warning or error message.
        """
        warnings = []
        print("Validating configuration...")
        
        # Example validation check
        if self.get_value('Kernel.Quirks.XhciPortLimit') and self.get_value('Misc.Debug.DisplayLevel') == 0:
            warnings.append("Warning: XhciPortLimit is enabled, but debug logging is off. USB mapping issues may be hard to diagnose.")
            
        if not self.get_value('PlatformInfo.Generic.SystemSerialNumber'):
            warnings.append("Error: SystemSerialNumber is missing. System may not boot.")

        if not warnings:
            print("Validation passed with no major warnings.")
        else:
            for warning in warnings:
                print(warning)
                
        return warnings

if __name__ == '__main__':
    # Example usage of the OpenCoreManager
    
    # Create a new manager instance
    oc_manager = OpenCoreManager()

    # Create a basic structure if starting from scratch
    oc_manager.config = {
        'ACPI': {'Add': [], 'Delete': [], 'Patch': [], 'Quirks': {}},
        'Booter': {'MmioWhitelist': [], 'Patch': [], 'Quirks': {}},
        'DeviceProperties': {'Add': {}, 'Delete': {}},
        'Kernel': {'Add': [], 'Block': [], 'Emulate': {}, 'Force': [], 'Patch': [], 'Quirks': {}},
        'Misc': {'BlessOverride': [], 'Boot': {}, 'Debug': {}, 'Entries': [], 'Security': {}, 'Tools': []},
        'NVRAM': {'Add': {}, 'Delete': {}, 'LegacySchema': {}},
        'PlatformInfo': {'Generic': {}},
        'UEFI': {'APFS': {}, 'AppleInput': {}, 'Audio': {}, 'ConnectDrivers': True, 'Drivers': [], 'Input': {}, 'Output': {}, 'ProtocolOverrides': {}, 'Quirks': {}}
    }
    
    # --- Modify the config ---
    print("1. Adding essential kexts...")
    oc_manager.add_kext("Lilu.kext", "Contents/MacOS/Lilu", "Contents/Info.plist")
    oc_manager.add_kext("VirtualSMC.kext", "Contents/MacOS/VirtualSMC", "Contents/Info.plist")
    oc_manager.add_kext("WhateverGreen.kext", "Contents/MacOS/WhateverGreen", "Contents/Info.plist")

    print("\n2. Adding essential UEFI drivers...")
    oc_manager.add_driver("OpenRuntime.efi")
    oc_manager.add_driver("HfsPlus.efi")
    
    print("\n3. Setting boot arguments...")
    oc_manager.set_value('NVRAM.Add.7C436110-AB2A-4BBB-A880-FE41995C9F82.boot-args', '-v debug=0x100 keepsyms=1')

    print("\n4. Generating SMBIOS data...")
    oc_manager.generate_smbios('iMacPro1,1')
    
    print("\n5. Disabling a kext...")
    oc_manager.toggle_kext('WhateverGreen.kext', False)

    # --- Validate and Save ---
    print("\n6. Validating the new configuration...")
    oc_manager.validate_config()

    print("\n7. Saving to 'config_generated.plist'...")
    oc_manager.save_config('config_generated.plist')
    print("Done. Check the 'config_generated.plist' file.")
    
    # --- Load and verify ---
    print("\n8. Loading the generated file to verify a value...")
    loader_test = OpenCoreManager('config_generated.plist')
    boot_args = loader_test.get_value('NVRAM.Add.7C436110-AB2A-4BBB-A880-FE41995C9F82.boot-args')
    print(f"Verified boot-args: {boot_args}")
    
    whatevergreen_status = loader_test._find_item_in_list('Kernel.Add', 'BundlePath', 'WhateverGreen.kext')
    print(f"Verified WhateverGreen status: {whatevergreen_status}")

