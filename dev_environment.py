import os
import sys
import tarfile
import tempfile
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

# Attempt to import the Docker library, providing guidance if it's missing.
try:
    import docker
    from docker.client import DockerClient
    from docker.models.containers import Container
    from docker.errors import DockerException, NotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    # Define dummy classes to allow the script to be imported without errors
    class DockerClient: pass
    class Container: pass
    class DockerException(Exception): pass
    class NotFound(DockerException): pass
    class APIError(DockerException): pass

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class DevEnvironment:
    """
    Manages a secure, isolated development environment using Docker.

    This class provides a programmatic interface to create, manage, and interact
    with a sandboxed container, suitable for tasks like code execution, reverse
    engineering, and OS development. It handles setup, command execution,
    file transfers, and resource cleanup.
    """

    def __init__(
        self,
        image_name: str = "ubuntu:22.04",
        container_name: str = "skyscope-dev-env",
        workspace_dir: Optional[str] = None,
        storage_gb: int = 200,
    ):
        """
        Initializes the development environment manager.

        Args:
            image_name (str): The Docker image to use for the environment.
                              Defaults to a standard Ubuntu image.
            container_name (str): The name for the Docker container.
            workspace_dir (Optional[str]): The local directory to mount as the
                                           container's workspace. If None, a
                                           Docker volume is used instead.
            storage_gb (int): The desired storage size in GB for the workspace
                              volume. Note: This is a conceptual parameter, as
                              Docker volumes share the host's storage pool.
        """
        if not DOCKER_AVAILABLE:
            raise ImportError("The 'docker' library is not installed. Please install it with 'pip install docker'.")

        self.image_name = image_name
        self.container_name = container_name
        self.workspace_dir = workspace_dir
        self.storage_gb = storage_gb  # Stored for metadata purposes
        self.client: DockerClient = self._get_docker_client()
        self.container: Optional[Container] = self._find_container()

    def _get_docker_client(self) -> DockerClient:
        """
        Establishes a connection with the Docker daemon.

        Returns:
            DockerClient: An instance of the Docker client.

        Raises:
            DockerException: If the Docker daemon is not running or accessible.
        """
        try:
            client = docker.from_env()
            client.ping()
            logger.info("Successfully connected to Docker daemon.")
            return client
        except DockerException:
            logger.error("Failed to connect to Docker daemon. Is it running?")
            raise

    def _find_container(self) -> Optional[Container]:
        """
        Finds the container by name if it already exists.

        Returns:
            Optional[Container]: The container object if found, otherwise None.
        """
        try:
            return self.client.containers.get(self.container_name)
        except NotFound:
            return None

    def _ensure_image(self) -> None:
        """
        Ensures the specified Docker image is available locally, pulling if necessary.
        """
        try:
            self.client.images.get(self.image_name)
            logger.info(f"Docker image '{self.image_name}' found locally.")
        except docker.errors.ImageNotFound:
            logger.info(f"Image '{self.image_name}' not found. Pulling from registry...")
            try:
                self.client.images.pull(self.image_name)
                logger.info(f"Successfully pulled image '{self.image_name}'.")
            except APIError as e:
                logger.error(f"Failed to pull image '{self.image_name}': {e}")
                raise

    def setup(self, force_recreate: bool = False) -> None:
        """
        Sets up and starts the sandboxed development environment.

        This method ensures the Docker image is present, creates a persistent
        workspace, and starts the container.

        Args:
            force_recreate (bool): If True, any existing container with the same
                                   name will be removed and a new one created.
        """
        if self.is_running() and not force_recreate:
            logger.info(f"Container '{self.container_name}' is already running.")
            return

        if self.container and force_recreate:
            logger.info(f"Force recreate: Removing existing container '{self.container_name}'.")
            self.cleanup()

        self._ensure_image()

        mounts = []
        if self.workspace_dir:
            # Use a bind mount from the host filesystem
            host_path = os.path.abspath(self.workspace_dir)
            os.makedirs(host_path, exist_ok=True)
            mounts.append(docker.types.Mount(target="/workspace", source=host_path, type="bind"))
            logger.info(f"Using bind mount: '{host_path}' -> '/workspace'")
        else:
            # Use a named Docker volume for persistence
            volume_name = f"{self.container_name}-workspace"
            try:
                self.client.volumes.get(volume_name)
                logger.info(f"Using existing Docker volume: '{volume_name}'")
            except NotFound:
                logger.info(f"Creating new Docker volume: '{volume_name}'")
                self.client.volumes.create(volume_name)
            mounts.append(docker.types.Mount(target="/workspace", source=volume_name, type="volume"))

        try:
            logger.info(f"Creating and starting container '{self.container_name}'...")
            self.container = self.client.containers.run(
                self.image_name,
                name=self.container_name,
                detach=True,
                tty=True,  # Keep stdin open
                mounts=mounts,
                command="tail -f /dev/null"  # Keep container alive
            )
            logger.info(f"Container '{self.container_name}' started successfully (ID: {self.container.short_id}).")
        except APIError as e:
            logger.error(f"Failed to start container: {e}")
            raise

    def is_running(self) -> bool:
        """
        Checks if the development environment container is currently running.

        Returns:
            bool: True if the container is running, False otherwise.
        """
        if not self.container:
            return False
        try:
            self.container.reload()
            return self.container.status == 'running'
        except NotFound:
            self.container = None
            return False

    def execute_command(self, command: str, timeout: int = 300) -> Tuple[int, str, str]:
        """
        Executes a shell command inside the running container.

        Args:
            command (str): The command to execute.
            timeout (int): The timeout in seconds for the command execution.

        Returns:
            Tuple[int, str, str]: A tuple containing:
                - exit_code (int): The exit code of the command.
                - stdout (str): The standard output from the command.
                - stderr (str): The standard error from the command.
        
        Raises:
            RuntimeError: If the container is not running.
        """
        if not self.is_running() or not self.container:
            raise RuntimeError("Environment is not running. Call setup() first.")
        
        logger.info(f"Executing command: `{command}`")
        exit_code, (stdout, stderr) = self.container.exec_run(
            command,
            tty=True,
            workdir="/workspace",
            demux=True  # Demultiplex stdout and stderr
        )
        
        stdout_str = stdout.decode('utf-8', errors='ignore') if stdout else ""
        stderr_str = stderr.decode('utf-8', errors='ignore') if stderr else ""

        if exit_code == 0:
            logger.info(f"Command finished successfully with exit code {exit_code}.")
        else:
            logger.warning(f"Command finished with exit code {exit_code}.")
            if stderr_str:
                logger.warning(f"Stderr: {stderr_str.strip()}")

        return exit_code, stdout_str, stderr_str

    def install_packages(self, packages: List[str], manager: str = 'apt') -> bool:
        """
        Installs system or language packages inside the environment.

        Args:
            packages (List[str]): A list of package names to install.
            manager (str): The package manager to use ('apt', 'pip', 'npm').

        Returns:
            bool: True if the installation was successful, False otherwise.
        """
        package_str = " ".join(packages)
        if manager == 'apt':
            command = f"apt-get update && apt-get install -y {package_str}"
        elif manager == 'pip':
            command = f"pip install --upgrade pip && pip install {package_str}"
        elif manager == 'npm':
            command = f"npm install -g {package_str}"
        else:
            logger.error(f"Unsupported package manager: {manager}")
            return False

        logger.info(f"Installing packages using {manager}: {package_str}")
        exit_code, _, stderr = self.execute_command(command)
        
        if exit_code != 0:
            logger.error(f"Package installation failed. Error: {stderr}")
            return False
        
        logger.info("Packages installed successfully.")
        return True

    def copy_to(self, src_path: str, dest_path: str) -> None:
        """
        Copies a file or directory from the host to the container.

        Args:
            src_path (str): The source path on the host.
            dest_path (str): The destination path inside the container.
        
        Raises:
            RuntimeError: If the container is not running.
            FileNotFoundError: If the source path does not exist.
        """
        if not self.is_running() or not self.container:
            raise RuntimeError("Environment is not running.")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source path '{src_path}' does not exist on host.")

        # Create a tar archive in memory
        pw_tar = BytesIO()
        with tarfile.open(fileobj=pw_tar, mode='w') as tar:
            tar.add(src_path, arcname=os.path.basename(src_path))
        pw_tar.seek(0)

        # The destination path should be the directory where the content will be placed
        self.container.put_archive(path=dest_path, data=pw_tar)
        logger.info(f"Copied '{src_path}' from host to container at '{dest_path}'.")

    def copy_from(self, src_path: str, dest_path: str) -> None:
        """
        Copies a file or directory from the container to the host.

        Args:
            src_path (str): The source path inside the container.
            dest_path (str): The destination path on the host.
        
        Raises:
            RuntimeError: If the container is not running.
        """
        if not self.is_running() or not self.container:
            raise RuntimeError("Environment is not running.")

        bits, _ = self.container.get_archive(src_path)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_tar:
            for chunk in bits:
                tmp_tar.write(chunk)
            tmp_tar_path = tmp_tar.name

        with tarfile.open(tmp_tar_path) as tar:
            tar.extractall(path=dest_path)
        
        os.remove(tmp_tar_path)
        logger.info(f"Copied '{src_path}' from container to host at '{dest_path}'.")

    def get_status(self) -> Dict[str, Any]:
        """
        Retrieves the current status and resource usage of the environment.

        Returns:
            Dict[str, Any]: A dictionary containing status information.
        """
        if not self.is_running() or not self.container:
            return {"status": "stopped"}

        stats = self.container.stats(stream=False)
        
        # Calculate CPU percentage
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
        system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
        num_cpus = stats['cpu_stats']['online_cpus']
        cpu_percent = (cpu_delta / system_cpu_delta) * num_cpus * 100.0 if system_cpu_delta > 0 else 0

        return {
            "id": self.container.short_id,
            "name": self.container.name,
            "status": self.container.status,
            "image": self.image_name,
            "memory_usage_mb": stats['memory_stats'].get('usage', 0) / (1024 * 1024),
            "cpu_percent": cpu_percent,
        }

    def cleanup(self, remove_volume: bool = False) -> None:
        """
        Stops and removes the container and optionally its associated volume.

        Args:
            remove_volume (bool): If True, also removes the workspace volume.
                                  Use with caution as this deletes all data.
        """
        if not self.container:
            logger.info("No container found to clean up.")
            return

        try:
            logger.info(f"Stopping container '{self.container_name}'...")
            self.container.stop()
            logger.info(f"Removing container '{self.container_name}'...")
            self.container.remove()
            self.container = None
            logger.info("Container cleaned up successfully.")

            if remove_volume and not self.workspace_dir:
                volume_name = f"{self.container_name}-workspace"
                try:
                    volume = self.client.volumes.get(volume_name)
                    logger.warning(f"Removing volume '{volume_name}'...")
                    volume.remove(force=True)
                    logger.info("Volume removed successfully.")
                except NotFound:
                    logger.info(f"Volume '{volume_name}' not found, skipping removal.")
        except NotFound:
            logger.warning(f"Container '{self.container_name}' was already removed.")
            self.container = None
        except APIError as e:
            logger.error(f"Error during cleanup: {e}")
            raise

if __name__ == '__main__':
    logger.info("--- DevEnvironment Demonstration ---")
    
    # Initialize the environment manager
    dev_env = DevEnvironment(container_name="skyscope-demo-env")
    
    try:
        # 1. Setup the environment
        dev_env.setup(force_recreate=True)
        
        # 2. Check if it's running
        if dev_env.is_running():
            logger.info("Environment is up and running.")
        else:
            logger.error("Environment setup failed.")
            sys.exit(1)
            
        # 3. Execute some basic commands
        logger.info("\n--- Executing basic commands ---")
        exit_code, out, err = dev_env.execute_command("whoami")
        logger.info(f"whoami -> Output: {out.strip()}")
        
        exit_code, out, err = dev_env.execute_command("ls -la /workspace")
        logger.info(f"ls -la /workspace ->\n{out.strip()}")
        
        # 4. Install a package
        logger.info("\n--- Installing a package (cowsay) ---")
        if dev_env.install_packages(["cowsay"]):
            exit_code, out, err = dev_env.execute_command("cowsay 'Hello from the sandbox!'")
            logger.info(f"cowsay output:\n{out}")
        
        # 5. File Management
        logger.info("\n--- Testing file management ---")
        # Create a dummy file on the host
        with open("host_file.txt", "w") as f:
            f.write("This file was created on the host.")
        
        # Copy file to container
        dev_env.copy_to("host_file.txt", "/workspace/")
        exit_code, out, _ = dev_env.execute_command("ls -l /workspace/host_file.txt")
        logger.info(f"File in container:\n{out.strip()}")
        
        # Copy file from container
        os.makedirs("from_container", exist_ok=True)
        dev_env.copy_from("/etc/hostname", "./from_container/")
        with open("./from_container/hostname", "r") as f:
            logger.info(f"Content of file copied from container (/etc/hostname): {f.read().strip()}")
            
        # Clean up local files
        os.remove("host_file.txt")
        os.remove("./from_container/hostname")
        os.rmdir("from_container")

        # 6. Get final status
        logger.info("\n--- Getting final status ---")
        status = dev_env.get_status()
        logger.info(f"Final Status: {status}")

    except Exception as e:
        logger.error(f"An error occurred during the demonstration: {e}")
    finally:
        # 7. Clean up the environment
        logger.info("\n--- Cleaning up the environment ---")
        dev_env.cleanup(remove_volume=True)

