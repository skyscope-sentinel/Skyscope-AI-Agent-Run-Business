# Skyscope Sentinel - Core Knowledge Base

This document serves as a foundational knowledge base for the Skyscope Sentinel AI. It contains essential information on various technologies required for development, automation, and system administration tasks.

---

## 1. Shell Scripting

### 1.1. Bash (Bourne-Again Shell)

**Overview:**
Bash is the default command-line shell for most Linux distributions and macOS. It's a powerful tool for automating tasks, managing files, and interacting with the operating system.

**Key Concepts:**
- **Shebang:** The `#!/bin/bash` line at the start of a script specifies the interpreter.
- **Variables:** Declared without a `$` (`VAR="value"`) and accessed with a `$` (`echo $VAR`).
- **Conditionals:** `if`, `else`, `elif` statements. Use `[[ ... ]]` for modern, robust tests.
  ```bash
  if [[ "$VAR" == "value" ]]; then
    echo "Match"
  fi
  ```
- **Loops:** `for`, `while`, `until`.
  ```bash
  for i in {1..5}; do
    echo "Number: $i"
  done
  ```
- **Functions:** Reusable blocks of code.
  ```bash
  my_function() {
    echo "Hello, $1"
  }
  my_function "World"
  ```
- **Pipes (`|`):** Send the output of one command as the input to another.
- **Redirection (`>`, `>>`, `<`):** Redirect standard output, standard error, and standard input.

**Common Commands:**
- `ls`: List directory contents.
- `cd`: Change directory.
- `pwd`: Print working directory.
- `cp`: Copy files or directories.
- `mv`: Move or rename files or directories.
- `rm`: Remove files or directories.
- `grep`: Search for patterns in text.
- `find`: Search for files and directories.
- `chmod`: Change file permissions.
- `chown`: Change file ownership.
- `curl`, `wget`: Download files from the internet.

**Best Practices:**
- Always quote variables (`"$VAR"`) to prevent word splitting and globbing issues.
- Use `set -euo pipefail` at the beginning of scripts to make them more robust (exit on error, exit on unset variables, fail on pipe errors).
- Use descriptive variable and function names.
- Check for command success or failure using the exit code (`$?`).

### 1.2. PowerShell

**Overview:**
PowerShell is a cross-platform task automation solution from Microsoft, consisting of a command-line shell, a scripting language, and a configuration management framework. It is built on the .NET Framework and is object-oriented.

**Key Concepts:**
- **Cmdlets:** Lightweight commands that are the core of PowerShell. They follow a `Verb-Noun` naming convention (e.g., `Get-Process`).
- **Objects:** PowerShell commands return objects, not just text. This allows for powerful and structured data manipulation.
- **Pipeline (`|`):** Passes objects from one cmdlet to the next. You can access properties of these objects directly.
- **Variables:** Start with a `$` (e.g., `$myVar = "value"`).
- **Providers:** Allow access to different data stores (like the filesystem or registry) as if they were drives.

**Common Commands:**
- `Get-ChildItem` (alias `ls`, `dir`): List items in a location.
- `Set-Location` (alias `cd`, `sl`): Change the current working location.
- `Get-Content` (alias `cat`, `gc`): Get the content of a file.
- `Set-Content` (alias `sc`): Write content to a file.
- `Copy-Item`, `Move-Item`, `Remove-Item`: File operations.
- `Get-Process`, `Stop-Process`: Manage running processes.
- `Get-Service`, `Start-Service`, `Stop-Service`: Manage services.
- `Invoke-WebRequest`, `Invoke-RestMethod`: Make web requests.

**Best Practices:**
- Embrace the pipeline. Instead of parsing text, pass objects and filter/select properties (`| Where-Object`, `| Select-Object`).
- Use the full cmdlet names in scripts for clarity (`Get-ChildItem` instead of `ls`).
- Use approved verbs for custom functions to maintain consistency.
- Handle errors using `try`/`catch`/`finally` blocks.

---

## 2. Package Managers

### 2.1. pip

**Overview:** The standard package installer for Python. It allows you to install and manage software packages written in Python.

**Common Commands:**
- `pip install <package>`: Install a package.
- `pip install -r requirements.txt`: Install packages from a requirements file.
- `pip uninstall <package>`: Remove a package.
- `pip list`: List installed packages.
- `pip freeze > requirements.txt`: Generate a list of installed packages and their versions.
- `pip install --upgrade <package>`: Upgrade a package.

**Best Practices:**
- Always use `pip` within a virtual environment (`venv`) to avoid conflicts between project dependencies.
- Use `requirements.txt` files to document and manage project dependencies.
- Regularly update packages to get security patches and new features.

### 2.2. npm (Node Package Manager)

**Overview:** The default package manager for the JavaScript runtime environment Node.js.

**Common Commands:**
- `npm install <package>`: Install a package locally.
- `npm install -g <package>`: Install a package globally.
- `npm uninstall <package>`: Remove a package.
- `npm init`: Initialize a project, creating a `package.json` file.
- `npm run <script>`: Run a script defined in `package.json`.
- `npm ci`: Install dependencies cleanly from `package-lock.json`.

**Best Practices:**
- Use `package.json` to manage project dependencies and scripts.
- Commit the `package-lock.json` file to ensure reproducible builds across different environments.
- Use `npm ci` in automated/CI environments for faster, more reliable installs.

### 2.3. uv

**Overview:** An extremely fast Python package installer and resolver, written in Rust. It's designed as a drop-in replacement for `pip` and `pip-tools`.

**Common Commands:**
- `uv pip install <package>`: Installs a package, often much faster than pip.
- `uv venv`: Creates a virtual environment.
- `uv pip sync requirements.txt`: Synchronizes the environment with a requirements file, removing packages that are not listed.
- `uv pip compile requirements.in -o requirements.txt`: Compiles dependencies.

**Best Practices:**
- Use `uv` for its significant speed advantage in CI/CD pipelines and local development.
- Leverage `uv pip sync` to maintain a clean and predictable environment.

### 2.4. Homebrew

**Overview:** A free and open-source software package management system that simplifies the installation of software on Apple's macOS, as well as Linux.

**Common Commands:**
- `brew install <formula>`: Install a software package (formula).
- `brew install --cask <cask>`: Install a macOS application (cask).
- `brew uninstall <formula>`: Remove a package.
- `brew update`: Fetch the newest version of Homebrew and all formulae.
- `brew upgrade`: Upgrade all outdated packages.
- `brew list`: List installed formulae.
- `brew doctor`: Check for potential issues with your Homebrew setup.

**Best Practices:**
- Run `brew update` regularly before installing new packages.
- Run `brew doctor` if you encounter issues.
- Use `brew services` to manage background services installed via Homebrew.

---

## 3. Programming Languages

### 3.1. Python

**Overview:** A high-level, interpreted programming language known for its readability, simplicity, and extensive standard library. It's widely used in web development, data science, AI, and automation.

**Key Concepts:**
- **Dynamic Typing:** Variable types are determined at runtime.
- **Indentation:** Uses whitespace to define code blocks, which enforces a readable style.
- **Data Structures:** Built-in lists, tuples, dictionaries, and sets.
- **Object-Oriented:** Everything is an object. Supports classes, inheritance, and polymorphism.
- **Modules and Packages:** A modular architecture that allows for code organization and reuse.
- **Comprehensions:** Concise syntax for creating lists, dictionaries, and sets (e.g., `[x*x for x in range(10)]`).
- **Generators:** Functions that yield values one at a time, allowing for memory-efficient iteration over large datasets.

**Best Practices:**
- Follow the **PEP 8** style guide for code formatting and conventions.
- Use virtual environments for every project.
- Write clear, concise docstrings for modules, classes, and functions.
- Use type hints for better code clarity and static analysis.
- Write unit tests to ensure code correctness.

---

## 4. Operating Systems

### 4.1. macOS

**Overview:** Apple's Unix-based operating system. It features a graphical user interface (Aqua) built on top of a POSIX-compliant core (Darwin).

**Key Concepts:**
- **File System (APFS):** The Apple File System is optimized for flash/SSD storage, featuring strong encryption, space sharing, and snapshots.
- **Directory Structure:**
  - `/System`: Core OS files. Should not be modified.
  - `/Library`: System-wide applications and resources available to all users.
  - `~/Library`: User-specific settings, caches, and application support files.
  - `/Applications`: Location for user-installed applications.
- **Application Bundles (`.app`):** Appear as single files but are actually directories containing the app's executable, resources, and metadata.
- **Property Lists (`.plist`):** XML or binary files used for storing configuration data. Crucial for system and app configuration, including OpenCore.
- **Command-Line Tools:**
  - `diskutil`: Manage disks and partitions.
  - `pmset`: Manage power settings.
  - `sw_vers`: Show macOS version information.
  - `defaults`: Read and write `.plist` configuration files.
  - `system_profiler`: Get detailed hardware and software information.

### 4.2. OS Development & Hackintosh Concepts

**Overview:** This section covers concepts relevant to modifying or building operating system components, with a focus on terms used in the Hackintosh community for running macOS on non-Apple hardware.

**Key Concepts:**
- **Bootloader:** Software that loads the operating system kernel into memory.
  - **UEFI (Unified Extensible Firmware Interface):** The modern standard for firmware that replaces the legacy BIOS.
  - **OpenCore:** A sophisticated bootloader used to inject data and patch the system in memory to make macOS believe it's running on genuine Apple hardware. It is the current standard for modern Hackintosh systems.
- **Kernel:** The core of the operating system that manages the CPU, memory, and peripherals. The macOS kernel is named **XNU**.
- **Kexts (Kernel Extensions):** Similar to drivers in other operating systems. They are bundles of code that extend the functionality of the XNU kernel. For Hackintoshes, kexts are used to add support for non-Apple hardware.
  - **Lilu.kext:** A patcher that provides a platform for other kexts to modify the system. A prerequisite for many other kexts.
  - **VirtualSMC.kext:** Emulates the System Management Controller found on real Macs, which is essential for macOS to boot.
  - **WhateverGreen.kext:** A kext for patching graphics-related issues (DRM, board-id, etc.).
  - **AppleALC.kext:** A kext for enabling native Apple HD Audio on non-Apple hardware.
- **ACPI (Advanced Configuration and Power Interface):** A standard for handling power management and hardware configuration.
  - **DSDT (Differentiated System Description Table):** A primary ACPI table that describes the base system's devices. Hackintosh setups often require patching the DSDT (or using supplementary **SSDT** files) to fix hardware incompatibilities.
- **SMBIOS (System Management BIOS):** Defines data structures and access methods that allow a user to interact with the BIOS. In a Hackintosh context, OpenCore spoofs the SMBIOS data to make the system appear as a specific Mac model (e.g., `iMacPro1,1`, `MacPro7,1`). This affects which features are enabled and how power management behaves.
- **config.plist:** The central configuration file for the OpenCore bootloader. It's a highly structured XML file that tells OpenCore which kexts to load, which patches to apply, what SMBIOS to use, and how to handle every aspect of the boot process.

