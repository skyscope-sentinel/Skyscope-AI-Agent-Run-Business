import os
import sys
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Any, Iterator

# --- Dependency Checks and Imports ---
# This setup ensures the module can be imported even if optional dependencies are missing,
# providing clear guidance to the user.

try:
    import lief
    LIEF_AVAILABLE = True
except ImportError:
    LIEF_AVAILABLE = False
    print("Warning: 'lief' library not found. Analysis and modification features will be disabled. Install with 'pip install lief'.")

try:
    import capstone
    CAPSTONE_AVAILABLE = True
except ImportError:
    CAPSTONE_AVAILABLE = False
    print("Warning: 'capstone' library not found. Disassembly feature will be disabled. Install with 'pip install capstone'.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class ReverseEngineer:
    """
    A class to encapsulate software reverse engineering functionalities.

    This class provides a high-level API to analyze, disassemble, decompile,
    and modify binary executable files (PE, ELF, Mach-O). It leverages powerful
    libraries like LIEF and Capstone and provides wrappers for external tools
    like Ghidra.
    """

    def __init__(self, file_path: str):
        """
        Initializes the ReverseEngineer with a path to a binary file.

        Args:
            file_path (str): The absolute path to the binary file to be analyzed.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            lief.bad_file: If the file is not a supported executable format.
            ImportError: If the required 'lief' library is not installed.
        """
        if not LIEF_AVAILABLE:
            raise ImportError("The 'lief' library is required to use the ReverseEngineer class.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' was not found.")

        self.file_path = file_path
        self.binary: Optional[lief.Binary] = lief.parse(self.file_path)

        if not self.binary:
            raise lief.bad_file(f"'{file_path}' is not a supported executable format or is corrupted.")

        logger.info(f"Successfully parsed '{os.path.basename(self.file_path)}' as a {self.binary.format.name} file.")

    def analyze(self) -> Dict[str, Any]:
        """
        Performs a comprehensive analysis of the loaded binary.

        Returns:
            Dict[str, Any]: A dictionary containing structured information about
                            the binary, including its format, architecture, entry point,
                            sections, and imported libraries/functions.
        """
        if not self.binary:
            return {"error": "Binary not loaded."}

        header = self.binary.header
        analysis_data = {
            "file_info": {
                "name": self.binary.name,
                "size": os.path.getsize(self.file_path),
                "format": self.binary.format.name,
                "architecture": self.binary.header.architecture.name,
                "is_pie": self.binary.is_pie,
                "has_nx": self.binary.has_nx,
            },
            "header": {
                "entry_point": hex(header.entrypoint),
                "virtual_size": header.virtual_size,
            },
            "sections": [
                {"name": s.name, "size": s.size, "virtual_address": hex(s.virtual_address), "offset": hex(s.offset)}
                for s in self.binary.sections
            ],
            "imports": [lib.name for lib in self.binary.libraries] if hasattr(self.binary, 'libraries') else [],
            "imported_functions": [f.name for f in self.binary.imported_functions] if hasattr(self.binary, 'imported_functions') else [],
            "exported_functions": [f.name for f in self.binary.exported_functions] if hasattr(self.binary, 'exported_functions') else [],
        }
        return analysis_data

    def disassemble(self, section_name: str = ".text") -> Iterator[Tuple[int, str, str]]:
        """
        Disassembles a specified section of the binary into assembly code.

        Args:
            section_name (str): The name of the section to disassemble (e.g., '.text').

        Yields:
            Iterator[Tuple[int, str, str]]: An iterator of tuples, where each tuple
                                            contains the address, mnemonic, and operands
                                            of a single instruction.

        Raises:
            ImportError: If the 'capstone' library is not installed.
            ValueError: If the specified section is not found in the binary.
        """
        if not CAPSTONE_AVAILABLE:
            raise ImportError("The 'capstone' library is required for disassembly.")
        if not self.binary:
            raise RuntimeError("Binary not loaded.")

        section = self.binary.get_section(section_name)
        if not section:
            raise ValueError(f"Section '{section_name}' not found in the binary.")

        arch_map = {
            lief.ARCHITECTURES.X86: (capstone.CS_ARCH_X86, capstone.CS_MODE_64 if self.binary.header.is_64 else capstone.CS_MODE_32),
            lief.ARCHITECTURES.ARM: (capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM),
            lief.ARCHITECTURES.ARM64: (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM),
        }

        cs_arch, cs_mode = arch_map.get(self.binary.header.architecture, (None, None))
        if not cs_arch:
            raise NotImplementedError(f"Disassembly for architecture '{self.binary.header.architecture.name}' is not supported.")

        md = capstone.Cs(cs_arch, cs_mode)
        code = bytes(section.content)
        base_address = section.virtual_address

        logger.info(f"Disassembling section '{section_name}' at base address {hex(base_address)}...")
        for i in md.disasm(code, base_address):
            yield (i.address, i.mnemonic, i.op_str)

    def decompile(self, tool: str = "ghidra", ghidra_project_path: str = "/workspace/ghidra_projects") -> str:
        """
        Decompiles the binary using an external tool like Ghidra.

        Note: This is a wrapper that constructs and executes a command. It assumes
        the specified tool is installed and accessible within the execution environment
        (e.g., the DevEnvironment container).

        Args:
            tool (str): The decompilation tool to use ('ghidra' is supported).
            ghidra_project_path (str): The path inside the environment to store Ghidra projects.

        Returns:
            str: The decompiled C-like code as a string.

        Raises:
            NotImplementedError: If the specified tool is not supported.
            RuntimeError: If the decompilation command fails.
        """
        if tool.lower() == 'ghidra':
            # This command assumes a headless Ghidra setup inside the DevEnvironment
            project_name = os.path.basename(self.file_path) + "_project"
            script_path = "/path/to/ghidra/support/analyzeHeadless" # This path must be configured in the env
            
            command = (
                f"{script_path} {os.path.join(ghidra_project_path, project_name)} -import {self.file_path} "
                f"-postScript DecompileToFile.java {tempfile.gettempdir()} -deleteProject"
            )
            
            logger.info(f"Constructed Ghidra decompilation command (for execution in dev env):\n`{command}`")
            
            # In a real scenario, this would be executed by the DevEnvironment manager
            # For demonstration, we'll simulate the output.
            # exit_code, stdout, stderr = dev_env.execute_command(command)
            # if exit_code != 0:
            #     raise RuntimeError(f"Ghidra decompilation failed: {stderr}")
            # return stdout
            
            return (
                "// --- Simulated Decompilation Output from Ghidra ---\n"
                "#include <stdio.h>\n\n"
                "int main(int argc, char **argv) {\n"
                "    puts(\"Hello, World! This is simulated decompiled code.\");\n"
                "    return 0;\n"
                "}\n"
            )
        else:
            raise NotImplementedError(f"Decompilation with tool '{tool}' is not currently supported.")

    def patch_bytes(self, address: int, new_bytes: bytes) -> None:
        """
        Patches the binary with new bytes at a specific virtual address.

        Args:
            address (int): The virtual address where the patch should be applied.
            new_bytes (bytes): The bytes to write at the specified address.
        
        Raises:
            RuntimeError: If the binary is not loaded.
        """
        if not self.binary:
            raise RuntimeError("Binary not loaded.")
        
        logger.info(f"Applying patch of {len(new_bytes)} bytes at address {hex(address)}.")
        self.binary.patch_address(address, list(new_bytes))

    def find_and_replace_string(self, old_string: str, new_string: str, section: str = ".rodata") -> bool:
        """
        Finds and replaces a string within a specific section of the binary.

        Note: The new string must be the same length as or shorter than the old
        string. If shorter, it will be padded with null bytes.

        Args:
            old_string (str): The string to search for.
            new_string (str): The string to replace it with.
            section (str): The section to search within (e.g., '.rodata', '.data').

        Returns:
            bool: True if the string was found and replaced, False otherwise.
        """
        if not self.binary:
            return False

        target_section = self.binary.get_section(section)
        if not target_section:
            logger.warning(f"Section '{section}' not found. Cannot perform string replacement.")
            return False

        old_bytes = old_string.encode('utf-8')
        new_bytes = new_string.encode('utf-8')

        if len(new_bytes) > len(old_bytes):
            logger.error("Replacement failed: New string cannot be longer than the old string.")
            return False

        # Pad the new string with null bytes to match the old one's length
        padded_new_bytes = new_bytes.ljust(len(old_bytes), b'\x00')

        content = bytearray(target_section.content)
        offset = content.find(old_bytes)

        if offset == -1:
            logger.warning(f"String '{old_string}' not found in section '{section}'.")
            return False

        address_to_patch = target_section.virtual_address + offset
        self.patch_bytes(address_to_patch, padded_new_bytes)
        logger.info(f"Replaced '{old_string}' with '{new_string}' at address {hex(address_to_patch)}.")
        return True

    def save(self, output_path: str) -> None:
        """
        Saves the (potentially modified) binary to a new file.

        Args:
            output_path (str): The path to save the new binary file.
        
        Raises:
            RuntimeError: If the binary is not loaded.
        """
        if not self.binary:
            raise RuntimeError("Binary not loaded.")
        
        logger.info(f"Building and saving modified binary to '{output_path}'...")
        self.binary.write(output_path)
        logger.info("Binary saved successfully.")


if __name__ == '__main__':
    # --- Demonstration ---
    # This block demonstrates the class's capabilities. It creates a dummy
    # executable file for analysis to ensure the script is runnable anywhere.

    # Create a simple dummy ELF file for demonstration if it doesn't exist
    DUMMY_FILE_PATH = "dummy_executable"
    if not os.path.exists(DUMMY_FILE_PATH):
        logger.info("Creating a dummy ELF executable for demonstration...")
        # A minimal 64-bit ELF executable that does nothing and exits.
        elf_bytes = bytes.fromhex(
            "7f454c4602010100000000000000000002003e0001000000400040000000000"
            "400000000000000000000000000000000000000000000000000000040003800"
            "010000000000000001000000050000000000000000000000000040000000000"
            "000004000000000000074000000000000074000000000000000000000000000"
            "000000000000000000000000000000000000000000000002000000000000000"
            "b83c000000bf010000000f05"
        )
        with open(DUMMY_FILE_PATH, "wb") as f:
            f.write(elf_bytes)
        # Make it executable on Unix-like systems
        if sys.platform != "win32":
            os.chmod(DUMMY_FILE_PATH, 0o755)

    try:
        # 1. Initialize the reverse engineering tool
        logger.info("\n--- Initializing ReverseEngineer ---")
        rev_eng = ReverseEngineer(DUMMY_FILE_PATH)

        # 2. Analyze the binary
        logger.info("\n--- Analyzing Binary ---")
        analysis = rev_eng.analyze()
        print(json.dumps(analysis, indent=2))

        # 3. Disassemble the .text section
        logger.info("\n--- Disassembling .text Section ---")
        try:
            for addr, mnemonic, op_str in rev_eng.disassemble():
                print(f"0x{addr:x}:\t{mnemonic}\t{op_str}")
        except (ImportError, ValueError, NotImplementedError) as e:
            logger.error(f"Disassembly failed: {e}")

        # 4. Decompile (simulated)
        logger.info("\n--- Decompiling (Simulated) ---")
        try:
            decompiled_code = rev_eng.decompile()
            print(decompiled_code)
        except Exception as e:
            logger.error(f"Decompilation failed: {e}")

        # 5. Patch the binary and save
        logger.info("\n--- Patching Binary ---")
        # This patch changes 'mov edi, 1' to 'mov edi, 2' in our dummy binary
        rev_eng.patch_bytes(address=0x400079, new_bytes=b'\x02')
        MODIFIED_FILE_PATH = "dummy_executable_modified"
        rev_eng.save(MODIFIED_FILE_PATH)
        
        # Verify the patch
        logger.info(f"Verifying patch in '{MODIFIED_FILE_PATH}'...")
        rev_eng_modified = ReverseEngineer(MODIFIED_FILE_PATH)
        patched_section = rev_eng_modified.binary.get_section(".text")
        # The patch is at offset 0x79 in the file, which is 0x79 in the section content
        if patched_section.content[0x79] == 0x02:
             logger.info("Verification successful: Patch confirmed.")
        else:
             logger.error("Verification failed: Patch not found in modified file.")

    except Exception as e:
        logger.error(f"An error occurred during the main demonstration: {e}")

