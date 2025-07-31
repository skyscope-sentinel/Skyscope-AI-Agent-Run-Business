import os
import sys
import json
import time
import logging
import shutil
import sqlite3
import subprocess
import threading
import tempfile
import platform
import re
import uuid
import socket
import hashlib
import datetime
import tarfile
import psutil
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import docker
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, Boolean, DateTime, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
import pandas as pd
import schedule
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/database_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("database_manager")

# Constants
DATA_DIR = Path("data")
CONFIG_DIR = Path("config")
BACKUP_DIR = Path("backups/databases")
DB_CONFIG_PATH = CONFIG_DIR / "database_config.json"
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_MYSQL_PORT = 3306
DEFAULT_POSTGRES_IMAGE = "postgres:14-alpine"
DEFAULT_MYSQL_IMAGE = "mysql:8-debian"
DEFAULT_POSTGRES_MEMORY = "256m"
DEFAULT_MYSQL_MEMORY = "256m"
MAX_BACKUP_AGE_DAYS = 30
MAX_BACKUP_COUNT = 10

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

class DatabaseType(Enum):
    """Enumeration of supported database types."""
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"

class ContainerStatus(Enum):
    """Enumeration of container statuses."""
    RUNNING = "running"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    CREATED = "created"
    EXITED = "exited"
    PAUSED = "paused"
    DEAD = "dead"
    NOT_FOUND = "not_found"

@dataclass
class DatabaseConfig:
    """Configuration for a database."""
    name: str
    db_type: DatabaseType
    host: str = "localhost"
    port: int = -1  # Will be set based on db_type if not specified
    username: str = ""
    password: str = ""
    database: str = ""
    container_name: str = ""
    image: str = ""
    memory_limit: str = ""
    cpu_limit: float = 0.5
    volume_path: str = ""
    auto_start: bool = True
    auto_backup: bool = True
    backup_schedule: str = "0 3 * * *"  # 3 AM daily in cron format
    max_connections: int = 20
    connection_timeout: int = 30
    query_timeout: int = 60
    enabled: bool = True
    
    def __post_init__(self):
        """Set defaults based on database type."""
        if not self.port or self.port == -1:
            if self.db_type == DatabaseType.POSTGRES:
                self.port = DEFAULT_POSTGRES_PORT
            elif self.db_type == DatabaseType.MYSQL:
                self.port = DEFAULT_MYSQL_PORT
        
        if not self.image:
            if self.db_type == DatabaseType.POSTGRES:
                self.image = DEFAULT_POSTGRES_IMAGE
            elif self.db_type == DatabaseType.MYSQL:
                self.image = DEFAULT_MYSQL_IMAGE
        
        if not self.memory_limit:
            if self.db_type == DatabaseType.POSTGRES:
                self.memory_limit = DEFAULT_POSTGRES_MEMORY
            elif self.db_type == DatabaseType.MYSQL:
                self.memory_limit = DEFAULT_MYSQL_MEMORY
        
        if not self.container_name:
            safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '', self.name).lower()
            self.container_name = f"skyscope_db_{safe_name}"
        
        if not self.volume_path:
            self.volume_path = str(DATA_DIR / "db_volumes" / self.name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "db_type": self.db_type.value,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "database": self.database,
            "container_name": self.container_name,
            "image": self.image,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "volume_path": self.volume_path,
            "auto_start": self.auto_start,
            "auto_backup": self.auto_backup,
            "backup_schedule": self.backup_schedule,
            "max_connections": self.max_connections,
            "connection_timeout": self.connection_timeout,
            "query_timeout": self.query_timeout,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseConfig':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            db_type=DatabaseType(data["db_type"]),
            host=data.get("host", "localhost"),
            port=data.get("port", -1),
            username=data.get("username", ""),
            password=data.get("password", ""),
            database=data.get("database", ""),
            container_name=data.get("container_name", ""),
            image=data.get("image", ""),
            memory_limit=data.get("memory_limit", ""),
            cpu_limit=data.get("cpu_limit", 0.5),
            volume_path=data.get("volume_path", ""),
            auto_start=data.get("auto_start", True),
            auto_backup=data.get("auto_backup", True),
            backup_schedule=data.get("backup_schedule", "0 3 * * *"),
            max_connections=data.get("max_connections", 20),
            connection_timeout=data.get("connection_timeout", 30),
            query_timeout=data.get("query_timeout", 60),
            enabled=data.get("enabled", True)
        )
    
    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        if self.db_type == DatabaseType.POSTGRES:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.MYSQL:
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.SQLITE:
            db_path = os.path.join(self.volume_path, f"{self.database}.db")
            return f"sqlite:///{db_path}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def get_docker_env_vars(self) -> Dict[str, str]:
        """Get environment variables for Docker container."""
        if self.db_type == DatabaseType.POSTGRES:
            return {
                "POSTGRES_USER": self.username,
                "POSTGRES_PASSWORD": self.password,
                "POSTGRES_DB": self.database,
                "PGDATA": "/var/lib/postgresql/data"
            }
        elif self.db_type == DatabaseType.MYSQL:
            return {
                "MYSQL_ROOT_PASSWORD": self.password,
                "MYSQL_USER": self.username,
                "MYSQL_PASSWORD": self.password,
                "MYSQL_DATABASE": self.database,
                "MYSQL_DATADIR": "/var/lib/mysql"
            }
        else:
            return {}
    
    def get_docker_ports(self) -> Dict[str, int]:
        """Get port mappings for Docker container."""
        if self.db_type == DatabaseType.POSTGRES:
            return {f"{DEFAULT_POSTGRES_PORT}/tcp": self.port}
        elif self.db_type == DatabaseType.MYSQL:
            return {f"{DEFAULT_MYSQL_PORT}/tcp": self.port}
        else:
            return {}
    
    def get_docker_volumes(self) -> Dict[str, Dict[str, str]]:
        """Get volume mappings for Docker container."""
        volume_path = Path(self.volume_path)
        volume_path.mkdir(parents=True, exist_ok=True)
        
        if self.db_type == DatabaseType.POSTGRES:
            return {
                str(volume_path): {
                    "bind": "/var/lib/postgresql/data",
                    "mode": "rw"
                }
            }
        elif self.db_type == DatabaseType.MYSQL:
            return {
                str(volume_path): {
                    "bind": "/var/lib/mysql",
                    "mode": "rw"
                }
            }
        else:
            return {}

class DockerManager:
    """Manager for Docker containers."""
    
    def __init__(self):
        self.client = None
        self.available = False
        self.init_docker_client()
    
    def init_docker_client(self) -> bool:
        """Initialize Docker client."""
        try:
            self.client = docker.from_env()
            self.client.ping()
            self.available = True
            logger.info("Docker client initialized successfully")
            return True
        except Exception as e:
            self.available = False
            logger.error(f"Failed to initialize Docker client: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Docker is available."""
        if not self.available:
            self.init_docker_client()
        return self.available
    
    def get_container_status(self, container_name: str) -> ContainerStatus:
        """Get status of a container."""
        if not self.is_available():
            return ContainerStatus.NOT_FOUND
        
        try:
            containers = self.client.containers.list(all=True, filters={"name": container_name})
            if not containers:
                return ContainerStatus.NOT_FOUND
            
            container = containers[0]
            status = container.status.lower()
            
            if status == "running":
                return ContainerStatus.RUNNING
            elif status == "exited":
                return ContainerStatus.EXITED
            elif status == "paused":
                return ContainerStatus.PAUSED
            elif status == "restarting":
                return ContainerStatus.RESTARTING
            elif status == "created":
                return ContainerStatus.CREATED
            elif status == "dead":
                return ContainerStatus.DEAD
            else:
                return ContainerStatus.STOPPED
        except Exception as e:
            logger.error(f"Error getting container status for {container_name}: {e}")
            return ContainerStatus.NOT_FOUND
    
    def create_container(self, config: DatabaseConfig) -> bool:
        """Create a container for a database."""
        if not self.is_available():
            logger.error("Docker is not available")
            return False
        
        container_name = config.container_name
        
        # Check if container already exists
        status = self.get_container_status(container_name)
        if status != ContainerStatus.NOT_FOUND:
            logger.info(f"Container {container_name} already exists with status {status}")
            return True
        
        try:
            logger.info(f"Creating container {container_name} from image {config.image}")
            
            # Create container
            self.client.containers.create(
                image=config.image,
                name=container_name,
                environment=config.get_docker_env_vars(),
                ports=config.get_docker_ports(),
                volumes=config.get_docker_volumes(),
                restart_policy={"Name": "unless-stopped"},
                mem_limit=config.memory_limit,
                cpu_quota=int(config.cpu_limit * 100000),
                cpu_period=100000,
                detach=True
            )
            
            logger.info(f"Container {container_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating container {container_name}: {e}")
            return False
    
    def start_container(self, container_name: str) -> bool:
        """Start a container."""
        if not self.is_available():
            logger.error("Docker is not available")
            return False
        
        try:
            status = self.get_container_status(container_name)
            
            if status == ContainerStatus.NOT_FOUND:
                logger.error(f"Container {container_name} not found")
                return False
            
            if status == ContainerStatus.RUNNING:
                logger.info(f"Container {container_name} is already running")
                return True
            
            logger.info(f"Starting container {container_name}")
            container = self.client.containers.get(container_name)
            container.start()
            
            # Wait for container to be fully started
            start_time = time.time()
            while time.time() - start_time < 60:  # 60 second timeout
                status = self.get_container_status(container_name)
                if status == ContainerStatus.RUNNING:
                    logger.info(f"Container {container_name} started successfully")
                    return True
                time.sleep(1)
            
            logger.warning(f"Container {container_name} started but status is {status}")
            return status == ContainerStatus.RUNNING
        except Exception as e:
            logger.error(f"Error starting container {container_name}: {e}")
            return False
    
    def stop_container(self, container_name: str) -> bool:
        """Stop a container."""
        if not self.is_available():
            logger.error("Docker is not available")
            return False
        
        try:
            status = self.get_container_status(container_name)
            
            if status == ContainerStatus.NOT_FOUND:
                logger.error(f"Container {container_name} not found")
                return False
            
            if status != ContainerStatus.RUNNING:
                logger.info(f"Container {container_name} is not running (status: {status})")
                return True
            
            logger.info(f"Stopping container {container_name}")
            container = self.client.containers.get(container_name)
            container.stop(timeout=30)
            
            # Wait for container to be fully stopped
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                status = self.get_container_status(container_name)
                if status != ContainerStatus.RUNNING:
                    logger.info(f"Container {container_name} stopped successfully")
                    return True
                time.sleep(1)
            
            logger.warning(f"Failed to stop container {container_name}, forcing...")
            container.kill()
            return True
        except Exception as e:
            logger.error(f"Error stopping container {container_name}: {e}")
            return False
    
    def remove_container(self, container_name: str, force: bool = False) -> bool:
        """Remove a container."""
        if not self.is_available():
            logger.error("Docker is not available")
            return False
        
        try:
            status = self.get_container_status(container_name)
            
            if status == ContainerStatus.NOT_FOUND:
                logger.info(f"Container {container_name} not found, nothing to remove")
                return True
            
            if status == ContainerStatus.RUNNING and not force:
                logger.warning(f"Container {container_name} is running, stop it first or use force=True")
                return False
            
            logger.info(f"Removing container {container_name}")
            container = self.client.containers.get(container_name)
            container.remove(force=force)
            
            logger.info(f"Container {container_name} removed successfully")
            return True
        except Exception as e:
            logger.error(f"Error removing container {container_name}: {e}")
            return False
    
    def get_container_logs(self, container_name: str, lines: int = 100) -> str:
        """Get logs from a container."""
        if not self.is_available():
            logger.error("Docker is not available")
            return "Docker is not available"
        
        try:
            status = self.get_container_status(container_name)
            
            if status == ContainerStatus.NOT_FOUND:
                return f"Container {container_name} not found"
            
            container = self.client.containers.get(container_name)
            logs = container.logs(tail=lines).decode('utf-8')
            return logs
        except Exception as e:
            logger.error(f"Error getting logs for container {container_name}: {e}")
            return f"Error getting logs: {e}"
    
    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """Get resource usage statistics for a container."""
        if not self.is_available():
            logger.error("Docker is not available")
            return {}
        
        try:
            status = self.get_container_status(container_name)
            
            if status != ContainerStatus.RUNNING:
                return {"status": status.value, "error": "Container is not running"}
            
            container = self.client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
            cpu_usage = 0
            if system_delta > 0 and cpu_delta > 0:
                cpu_usage = (cpu_delta / system_delta) * 100.0
            
            # Calculate memory usage
            memory_usage = stats["memory_stats"].get("usage", 0)
            memory_limit = stats["memory_stats"].get("limit", 1)
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                "status": status.value,
                "cpu_usage": round(cpu_usage, 2),
                "memory_usage": memory_usage,
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "memory_limit": memory_limit,
                "memory_limit_mb": round(memory_limit / (1024 * 1024), 2),
                "memory_percent": round(memory_percent, 2),
                "network_rx": stats.get("networks", {}).get("eth0", {}).get("rx_bytes", 0),
                "network_tx": stats.get("networks", {}).get("eth0", {}).get("tx_bytes", 0)
            }
        except Exception as e:
            logger.error(f"Error getting stats for container {container_name}: {e}")
            return {"status": "error", "error": str(e)}

class BackupManager:
    """Manager for database backups."""
    
    def __init__(self, docker_manager: DockerManager):
        self.docker_manager = docker_manager
        self.backup_dir = BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, config: DatabaseConfig) -> Optional[str]:
        """Create a backup of a database."""
        db_type = config.db_type
        backup_name = f"{config.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        logger.info(f"Creating backup of {config.name} to {backup_path}")
        
        try:
            if db_type == DatabaseType.SQLITE:
                return self._backup_sqlite(config, backup_path)
            elif db_type == DatabaseType.POSTGRES:
                return self._backup_postgres(config, backup_path)
            elif db_type == DatabaseType.MYSQL:
                return self._backup_mysql(config, backup_path)
            else:
                logger.error(f"Unsupported database type for backup: {db_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating backup for {config.name}: {e}")
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def _backup_sqlite(self, config: DatabaseConfig, backup_path: Path) -> Optional[str]:
        """Create a backup of a SQLite database."""
        db_path = Path(config.volume_path) / f"{config.database}.db"
        
        if not db_path.exists():
            logger.error(f"SQLite database file not found: {db_path}")
            return None
        
        try:
            # Create a temporary directory for the backup
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / db_path.name
                
                # Copy the database file to the temporary directory
                shutil.copy2(db_path, temp_path)
                
                # Create a tarball of the temporary directory
                with tarfile.open(backup_path, "w:gz") as tar:
                    tar.add(temp_path, arcname=db_path.name)
            
            logger.info(f"SQLite backup created successfully: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Error creating SQLite backup: {e}")
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def _backup_postgres(self, config: DatabaseConfig, backup_path: Path) -> Optional[str]:
        """Create a backup of a PostgreSQL database."""
        if not self.docker_manager.is_available():
            logger.error("Docker is not available for PostgreSQL backup")
            return None
        
        container_status = self.docker_manager.get_container_status(config.container_name)
        if container_status != ContainerStatus.RUNNING:
            logger.error(f"PostgreSQL container {config.container_name} is not running (status: {container_status})")
            return None
        
        try:
            # Create a temporary directory for the backup
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / f"{config.database}.sql"
                
                # Run pg_dump inside the container
                container = self.docker_manager.client.containers.get(config.container_name)
                pg_dump_cmd = f"pg_dump -U {config.username} -d {config.database} -f /tmp/backup.sql"
                exec_result = container.exec_run(pg_dump_cmd)
                
                if exec_result.exit_code != 0:
                    logger.error(f"pg_dump failed: {exec_result.output.decode('utf-8')}")
                    return None
                
                # Copy the backup file from the container
                bits, stat = container.get_archive("/tmp/backup.sql")
                with open(temp_path, 'wb') as f:
                    for chunk in bits:
                        f.write(chunk)
                
                # Create a tarball of the backup file
                with tarfile.open(backup_path, "w:gz") as tar:
                    tar.add(temp_path, arcname=f"{config.database}.sql")
            
            logger.info(f"PostgreSQL backup created successfully: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Error creating PostgreSQL backup: {e}")
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def _backup_mysql(self, config: DatabaseConfig, backup_path: Path) -> Optional[str]:
        """Create a backup of a MySQL database."""
        if not self.docker_manager.is_available():
            logger.error("Docker is not available for MySQL backup")
            return None
        
        container_status = self.docker_manager.get_container_status(config.container_name)
        if container_status != ContainerStatus.RUNNING:
            logger.error(f"MySQL container {config.container_name} is not running (status: {container_status})")
            return None
        
        try:
            # Create a temporary directory for the backup
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / f"{config.database}.sql"
                
                # Run mysqldump inside the container
                container = self.docker_manager.client.containers.get(config.container_name)
                mysqldump_cmd = f"mysqldump -u {config.username} -p{config.password} {config.database} > /tmp/backup.sql"
                exec_result = container.exec_run(["sh", "-c", mysqldump_cmd])
                
                if exec_result.exit_code != 0:
                    logger.error(f"mysqldump failed: {exec_result.output.decode('utf-8')}")
                    return None
                
                # Copy the backup file from the container
                bits, stat = container.get_archive("/tmp/backup.sql")
                with open(temp_path, 'wb') as f:
                    for chunk in bits:
                        f.write(chunk)
                
                # Create a tarball of the backup file
                with tarfile.open(backup_path, "w:gz") as tar:
                    tar.add(temp_path, arcname=f"{config.database}.sql")
            
            logger.info(f"MySQL backup created successfully: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Error creating MySQL backup: {e}")
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def restore_backup(self, config: DatabaseConfig, backup_path: str) -> bool:
        """Restore a database from a backup."""
        db_type = config.db_type
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        logger.info(f"Restoring {config.name} from backup {backup_path}")
        
        try:
            if db_type == DatabaseType.SQLITE:
                return self._restore_sqlite(config, backup_path)
            elif db_type == DatabaseType.POSTGRES:
                return self._restore_postgres(config, backup_path)
            elif db_type == DatabaseType.MYSQL:
                return self._restore_mysql(config, backup_path)
            else:
                logger.error(f"Unsupported database type for restore: {db_type}")
                return False
        except Exception as e:
            logger.error(f"Error restoring backup for {config.name}: {e}")
            return False
    
    def _restore_sqlite(self, config: DatabaseConfig, backup_path: Path) -> bool:
        """Restore a SQLite database from a backup."""
        db_path = Path(config.volume_path) / f"{config.database}.db"
        
        try:
            # Create a temporary directory for the restore
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the backup
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall(path=temp_dir)
                
                # Find the database file in the extracted files
                extracted_files = list(Path(temp_dir).glob("*.db"))
                if not extracted_files:
                    logger.error(f"No database file found in backup: {backup_path}")
                    return False
                
                # Ensure the target directory exists
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the database file to the target location
                shutil.copy2(extracted_files[0], db_path)
            
            logger.info(f"SQLite database restored successfully: {db_path}")
            return True
        except Exception as e:
            logger.error(f"Error restoring SQLite database: {e}")
            return False
    
    def _restore_postgres(self, config: DatabaseConfig, backup_path: Path) -> bool:
        """Restore a PostgreSQL database from a backup."""
        if not self.docker_manager.is_available():
            logger.error("Docker is not available for PostgreSQL restore")
            return False
        
        container_status = self.docker_manager.get_container_status(config.container_name)
        if container_status != ContainerStatus.RUNNING:
            logger.error(f"PostgreSQL container {config.container_name} is not running (status: {container_status})")
            return False
        
        try:
            # Create a temporary directory for the restore
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the backup
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall(path=temp_dir)
                
                # Find the SQL file in the extracted files
                extracted_files = list(Path(temp_dir).glob("*.sql"))
                if not extracted_files:
                    logger.error(f"No SQL file found in backup: {backup_path}")
                    return False
                
                # Copy the SQL file to the container
                container = self.docker_manager.client.containers.get(config.container_name)
                with open(extracted_files[0], 'rb') as f:
                    container.put_archive("/tmp", f.read())
                
                # Drop and recreate the database
                drop_cmd = f"dropdb -U {config.username} --if-exists {config.database}"
                create_cmd = f"createdb -U {config.username} {config.database}"
                restore_cmd = f"psql -U {config.username} -d {config.database} -f /tmp/{extracted_files[0].name}"
                
                # Execute commands
                exec_result = container.exec_run(drop_cmd)
                if exec_result.exit_code != 0:
                    logger.warning(f"dropdb warning: {exec_result.output.decode('utf-8')}")
                
                exec_result = container.exec_run(create_cmd)
                if exec_result.exit_code != 0:
                    logger.error(f"createdb failed: {exec_result.output.decode('utf-8')}")
                    return False
                
                exec_result = container.exec_run(restore_cmd)
                if exec_result.exit_code != 0:
                    logger.error(f"psql restore failed: {exec_result.output.decode('utf-8')}")
                    return False
            
            logger.info(f"PostgreSQL database restored successfully: {config.database}")
            return True
        except Exception as e:
            logger.error(f"Error restoring PostgreSQL database: {e}")
            return False
    
    def _restore_mysql(self, config: DatabaseConfig, backup_path: Path) -> bool:
        """Restore a MySQL database from a backup."""
        if not self.docker_manager.is_available():
            logger.error("Docker is not available for MySQL restore")
            return False
        
        container_status = self.docker_manager.get_container_status(config.container_name)
        if container_status != ContainerStatus.RUNNING:
            logger.error(f"MySQL container {config.container_name} is not running (status: {container_status})")
            return False
        
        try:
            # Create a temporary directory for the restore
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the backup
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall(path=temp_dir)
                
                # Find the SQL file in the extracted files
                extracted_files = list(Path(temp_dir).glob("*.sql"))
                if not extracted_files:
                    logger.error(f"No SQL file found in backup: {backup_path}")
                    return False
                
                # Copy the SQL file to the container
                container = self.docker_manager.client.containers.get(config.container_name)
                with open(extracted_files[0], 'rb') as f:
                    container.put_archive("/tmp", f.read())
                
                # Drop and recreate the database
                drop_cmd = f"mysql -u {config.username} -p{config.password} -e 'DROP DATABASE IF EXISTS {config.database}'"
                create_cmd = f"mysql -u {config.username} -p{config.password} -e 'CREATE DATABASE {config.database}'"
                restore_cmd = f"mysql -u {config.username} -p{config.password} {config.database} < /tmp/{extracted_files[0].name}"
                
                # Execute commands
                exec_result = container.exec_run(["sh", "-c", drop_cmd])
                if exec_result.exit_code != 0:
                    logger.warning(f"MySQL drop warning: {exec_result.output.decode('utf-8')}")
                
                exec_result = container.exec_run(["sh", "-c", create_cmd])
                if exec_result.exit_code != 0:
                    logger.error(f"MySQL create failed: {exec_result.output.decode('utf-8')}")
                    return False
                
                exec_result = container.exec_run(["sh", "-c", restore_cmd])
                if exec_result.exit_code != 0:
                    logger.error(f"MySQL restore failed: {exec_result.output.decode('utf-8')}")
                    return False
            
            logger.info(f"MySQL database restored successfully: {config.database}")
            return True
        except Exception as e:
            logger.error(f"Error restoring MySQL database: {e}")
            return False
    
    def list_backups(self, db_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        try:
            # Get all backup files
            backup_files = list(self.backup_dir.glob("*.tar.gz"))
            
            for backup_file in backup_files:
                # Parse backup name
                name_parts = backup_file.stem.split('_')
                if len(name_parts) < 3:
                    continue
                
                backup_db_name = name_parts[0]
                
                # Filter by database name if specified
                if db_name and backup_db_name != db_name:
                    continue
                
                # Get file stats
                stats = backup_file.stat()
                
                backups.append({
                    "name": backup_file.stem,
                    "path": str(backup_file),
                    "db_name": backup_db_name,
                    "timestamp": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "size": stats.st_size,
                    "size_mb": round(stats.st_size / (1024 * 1024), 2)
                })
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return backups
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def cleanup_old_backups(self, db_name: Optional[str] = None) -> int:
        """Clean up old backups."""
        try:
            backups = self.list_backups(db_name)
            now = datetime.datetime.now()
            deleted_count = 0
            
            # Group backups by database name
            backups_by_db = {}
            for backup in backups:
                db = backup["db_name"]
                if db not in backups_by_db:
                    backups_by_db[db] = []
                backups_by_db[db].append(backup)
            
            # Process each database's backups
            for db, db_backups in backups_by_db.items():
                # Keep the most recent MAX_BACKUP_COUNT backups
                if len(db_backups) > MAX_BACKUP_COUNT:
                    for backup in db_backups[MAX_BACKUP_COUNT:]:
                        # Check if backup is older than MAX_BACKUP_AGE_DAYS
                        backup_time = datetime.datetime.fromisoformat(backup["timestamp"])
                        age_days = (now - backup_time).days
                        
                        if age_days > MAX_BACKUP_AGE_DAYS:
                            backup_path = Path(backup["path"])
                            if backup_path.exists():
                                backup_path.unlink()
                                deleted_count += 1
                                logger.info(f"Deleted old backup: {backup_path}")
            
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return 0

class DatabaseConnection:
    """Connection to a database."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.Session = None
        self.metadata = MetaData()
        self.Base = declarative_base()
    
    def connect(self) -> bool:
        """Connect to the database."""
        try:
            connection_string = self.config.get_connection_string()
            
            # Create engine with connection pool settings
            self.engine = create_engine(
                connection_string,
                pool_size=min(self.config.max_connections, 5),  # Start with a smaller pool
                max_overflow=self.config.max_connections - 5,   # Allow growth up to max_connections
                pool_timeout=self.config.connection_timeout,
                pool_recycle=300,  # Recycle connections every 5 minutes
                connect_args={"connect_timeout": self.config.connection_timeout} if self.config.db_type != DatabaseType.SQLITE else {}
            )
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Connected to {self.config.db_type.value} database: {self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to {self.config.db_type.value} database {self.config.name}: {e}")
            self.engine = None
            self.Session = None
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None
            logger.info(f"Disconnected from database: {self.config.name}")
    
    @contextmanager
    def session(self):
        """Get a session for the database."""
        if not self.Session:
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results as dictionaries."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Error executing query on {self.config.name}: {e}")
            raise
    
    def execute_non_query(self, query: str, params: Dict[str, Any] = None) -> int:
        """Execute a non-query statement and return affected rows."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return result.rowcount
        except Exception as e:
            logger.error(f"Error executing non-query on {self.config.name}: {e}")
            raise
    
    def get_table_names(self) -> List[str]:
        """Get list of table names in the database."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error getting table names for {self.config.name}: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema for a table."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            inspector = inspect(self.engine)
            return inspector.get_columns(table_name)
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name} in {self.config.name}: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        try:
            return table_name in self.get_table_names()
        except Exception:
            return False
    
    def create_table(self, table_name: str, columns: Dict[str, Any]) -> bool:
        """Create a table with the specified columns."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            # Map column types to SQLAlchemy types
            column_map = {
                "integer": Integer,
                "string": String,
                "float": Float,
                "boolean": Boolean,
                "datetime": DateTime
            }
            
            # Create table definition
            table_columns = []
            for name, col_def in columns.items():
                col_type = col_def.get("type", "string").lower()
                nullable = col_def.get("nullable", True)
                primary_key = col_def.get("primary_key", False)
                
                # Get SQLAlchemy type
                sa_type = column_map.get(col_type, String)
                
                # Add length for string columns if specified
                if col_type == "string" and "length" in col_def:
                    sa_type = sa_type(col_def["length"])
                
                # Create column
                table_columns.append(Column(name, sa_type, nullable=nullable, primary_key=primary_key))
            
            # Create table
            table = Table(table_name, self.metadata, *table_columns)
            table.create(self.engine)
            
            logger.info(f"Created table {table_name} in {self.config.name}")
            return True
        except Exception as e:
            logger.error(f"Error creating table {table_name} in {self.config.name}: {e}")
            return False
    
    def drop_table(self, table_name: str) -> bool:
        """Drop a table."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            # Drop table
            self.metadata.reflect(bind=self.engine)
            if table_name in self.metadata.tables:
                self.metadata.tables[table_name].drop(self.engine)
                logger.info(f"Dropped table {table_name} in {self.config.name}")
                return True
            else:
                logger.warning(f"Table {table_name} not found in {self.config.name}")
                return False
        except Exception as e:
            logger.error(f"Error dropping table {table_name} in {self.config.name}: {e}")
            return False
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> int:
        """Insert data into a table."""
        if not data:
            return 0
        
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            # Reflect table structure
            self.metadata.reflect(bind=self.engine, only=[table_name])
            if table_name not in self.metadata.tables:
                raise ValueError(f"Table {table_name} not found in {self.config.name}")
            
            table = self.metadata.tables[table_name]
            
            # Insert data
            with self.engine.connect() as conn:
                result = conn.execute(table.insert(), data)
                return result.rowcount
        except Exception as e:
            logger.error(f"Error inserting data into {table_name} in {self.config.name}: {e}")
            raise
    
    def update_data(self, table_name: str, data: Dict[str, Any], condition: str, params: Dict[str, Any] = None) -> int:
        """Update data in a table."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            # Build update query
            set_clauses = ", ".join([f"{k} = :{k}" for k in data.keys()])
            query = f"UPDATE {table_name} SET {set_clauses} WHERE {condition}"
            
            # Combine data and params
            all_params = {**data, **(params or {})}
            
            # Execute update
            with self.engine.connect() as conn:
                result = conn.execute(text(query), all_params)
                return result.rowcount
        except Exception as e:
            logger.error(f"Error updating data in {table_name} in {self.config.name}: {e}")
            raise
    
    def delete_data(self, table_name: str, condition: str, params: Dict[str, Any] = None) -> int:
        """Delete data from a table."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            # Build delete query
            query = f"DELETE FROM {table_name} WHERE {condition}"
            
            # Execute delete
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return result.rowcount
        except Exception as e:
            logger.error(f"Error deleting data from {table_name} in {self.config.name}: {e}")
            raise
    
    def query_to_dataframe(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query and return results as a pandas DataFrame."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            return pd.read_sql(text(query), self.engine, params=params or {})
        except Exception as e:
            logger.error(f"Error executing query to DataFrame on {self.config.name}: {e}")
            raise
    
    def dataframe_to_table(self, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> int:
        """Write a pandas DataFrame to a table."""
        if not self.is_connected():
            if not self.connect():
                raise Exception(f"Failed to connect to database: {self.config.name}")
        
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            return len(df)
        except Exception as e:
            logger.error(f"Error writing DataFrame to table {table_name} in {self.config.name}: {e}")
            raise

class DatabaseManager:
    """Manager for databases."""
    
    def __init__(self):
        self.docker_manager = DockerManager()
        self.backup_manager = BackupManager(self.docker_manager)
        self.configs: Dict[str, DatabaseConfig] = {}
        self.connections: Dict[str, DatabaseConnection] = {}
        self.scheduler = schedule.Scheduler()
        self._scheduler_thread = None
        self._stop_scheduler = threading.Event()
        
        # Load configurations
        self.load_configs()
        
        # Start scheduler
        self._start_scheduler()
    
    def load_configs(self) -> None:
        """Load database configurations."""
        if not DB_CONFIG_PATH.exists():
            logger.info(f"Database config file not found: {DB_CONFIG_PATH}")
            self._create_default_config()
        
        try:
            with open(DB_CONFIG_PATH, 'r') as f:
                config_data = json.load(f)
            
            self.configs = {}
            for name, data in config_data.items():
                self.configs[name] = DatabaseConfig.from_dict(data)
            
            logger.info(f"Loaded {len(self.configs)} database configurations")
        except Exception as e:
            logger.error(f"Error loading database configurations: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration."""
        # Create SQLite config
        sqlite_config = DatabaseConfig(
            name="default_sqlite",
            db_type=DatabaseType.SQLITE,
            database="skyscope_data",
            enabled=True
        )
        
        # Create PostgreSQL config
        postgres_config = DatabaseConfig(
            name="default_postgres",
            db_type=DatabaseType.POSTGRES,
            host="localhost",
            port=5432,
            username="skyscope",
            password=self._generate_password(),
            database="skyscope_data",
            enabled=False
        )
        
        # Create MySQL config
        mysql_config = DatabaseConfig(
            name="default_mysql",
            db_type=DatabaseType.MYSQL,
            host="localhost",
            port=3306,
            username="skyscope",
            password=self._generate_password(),
            database="skyscope_data",
            enabled=False
        )
        
        # Add configs
        self.configs = {
            sqlite_config.name: sqlite_config,
            postgres_config.name: postgres_config,
            mysql_config.name: mysql_config
        }
        
        # Save configs
        self.save_configs()
    
    def _generate_password(self, length: int = 16) -> str:
        """Generate a random password."""
        import random
        import string
        chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
        return ''.join(random.choice(chars) for _ in range(length))
    
    def save_configs(self) -> None:
        """Save database configurations."""
        try:
            config_data = {name: config.to_dict() for name, config in self.configs.items()}
            
            with open(DB_CONFIG_PATH, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved {len(self.configs)} database configurations")
        except Exception as e:
            logger.error(f"Error saving database configurations: {e}")
    
    def add_config(self, config: DatabaseConfig) -> bool:
        """Add a database configuration."""
        if config.name in self.configs:
            logger.warning(f"Database configuration already exists: {config.name}")
            return False
        
        self.configs[config.name] = config
        self.save_configs()
        
        # Schedule backups if enabled
        if config.auto_backup:
            self._schedule_backup(config)
        
        logger.info(f"Added database configuration: {config.name}")
        return True
    
    def update_config(self, config: DatabaseConfig) -> bool:
        """Update a database configuration."""
        if config.name not in self.configs:
            logger.warning(f"Database configuration not found: {config.name}")
            return False
        
        # Check if connection exists and needs to be updated
        if config.name in self.connections:
            # Close existing connection
            self.connections[config.name].disconnect()
            del self.connections[config.name]
        
        # Update config
        self.configs[config.name] = config
        self.save_configs()
        
        # Update backup schedule
        self._update_backup_schedule(config)
        
        logger.info(f"Updated database configuration: {config.name}")
        return True
    
    def remove_config(self, name: str) -> bool:
        """Remove a database configuration."""
        if name not in self.configs:
            logger.warning(f"Database configuration not found: {name}")
            return False
        
        # Check if connection exists
        if name in self.connections:
            # Close connection
            self.connections[name].disconnect()
            del self.connections[name]
        
        # Remove config
        del self.configs[name]
        self.save_configs()
        
        # Remove from scheduler
        self._remove_from_scheduler(name)
        
        logger.info(f"Removed database configuration: {name}")
        return True
    
    def get_config(self, name: str) -> Optional[DatabaseConfig]:
        """Get a database configuration."""
        return self.configs.get(name)
    
    def get_configs(self) -> Dict[str, DatabaseConfig]:
        """Get all database configurations."""
        return self.configs.copy()
    
    def get_connection(self, name: str) -> Optional[DatabaseConnection]:
        """Get a database connection."""
        # Check if connection exists
        if name in self.connections:
            # Check if connection is valid
            if self.connections[name].is_connected():
                return self.connections[name]
            else:
                # Reconnect
                self.connections[name].connect()
                return self.connections[name]
        
        # Check if config exists
        if name not in self.configs:
            logger.warning(f"Database configuration not found: {name}")
            return None
        
        # Create connection
        config = self.configs[name]
        
        # Check if database needs to be started
        if config.db_type in [DatabaseType.POSTGRES, DatabaseType.MYSQL] and config.auto_start:
            self._ensure_container_running(config)
        
        # Create connection
        connection = DatabaseConnection(config)
        if connection.connect():
            self.connections[name] = connection
            return connection
        else:
            return None
    
    def _ensure_container_running(self, config: DatabaseConfig) -> bool:
        """Ensure a database container is running."""
        if config.db_type == DatabaseType.SQLITE:
            return True
        
        if not self.docker_manager.is_available():
            logger.error("Docker is not available")
            return False
        
        # Check if container exists
        status = self.docker_manager.get_container_status(config.container_name)
        
        if status == ContainerStatus.NOT_FOUND:
            # Create container
            if not self.docker_manager.create_container(config):
                logger.error(f"Failed to create container for {config.name}")
                return False
        
        if status != ContainerStatus.RUNNING:
            # Start container
            if not self.docker_manager.start_container(config.container_name):
                logger.error(f"Failed to start container for {config.name}")
                return False
        
        return True
    
    def start_database(self, name: str) -> bool:
        """Start a database."""
        if name not in self.configs:
            logger.warning(f"Database configuration not found: {name}")
            return False
        
        config = self.configs[name]
        
        if config.db_type == DatabaseType.SQLITE:
            logger.info(f"SQLite database {name} does not need to be started")
            return True
        
        return self._ensure_container_running(config)
    
    def stop_database(self, name: str) -> bool:
        """Stop a database."""
        if name not in self.configs:
            logger.warning(f"Database configuration not found: {name}")
            return False
        
        config = self.configs[name]
        
        if config.db_type == DatabaseType.SQLITE:
            logger.info(f"SQLite database {name} does not need to be stopped")
            return True
        
        # Close connection if exists
        if name in self.connections:
            self.connections[name].disconnect()
            del self.connections[name]
        
        # Stop container
        if not self.docker_manager.stop_container(config.container_name):
            logger.error(f"Failed to stop container for {name}")
            return False
        
        return True
    
    def get_database_status(self, name: str) -> Dict[str, Any]:
        """Get status of a database."""
        if name not in self.configs:
            return {"error": f"Database configuration not found: {name}"}
        
        config = self.configs[name]
        
        if config.db_type == DatabaseType.SQLITE:
            # Check if database file exists
            db_path = Path(config.volume_path) / f"{config.database}.db"
            exists = db_path.exists()
            
            return {
                "name": config.name,
                "type": config.db_type.value,
                "status": "available" if exists else "not_created",
                "path": str(db_path),
                "size": db_path.stat().st_size if exists else 0,
                "size_mb": round(db_path.stat().st_size / (1024 * 1024), 2) if exists else 0,
                "connected": name in self.connections and self.connections[name].is_connected()
            }
        else:
            # Get container status
            container_status = self.docker_manager.get_container_status(config.container_name)
            
            status_info = {
                "name": config.name,
                "type": config.db_type.value,
                "status": container_status.value,
                "container": config.container_name,
                "host": config.host,
                "port": config.port,
                "connected": name in self.connections and self.connections[name].is_connected()
            }
            
            # Add resource usage if running
            if container_status == ContainerStatus.RUNNING:
                stats = self.docker_manager.get_container_stats(config.container_name)
                status_info.update({
                    "cpu_usage": stats.get("cpu_usage", 0),
                    "memory_usage_mb": stats.get("memory_usage_mb", 0),
                    "memory_limit_mb": stats.get("memory_limit_mb", 0),
                    "memory_percent": stats.get("memory_percent", 0)
                })
            
            return status_info
    
    def get_database_logs(self, name: str, lines: int = 100) -> str:
        """Get logs for a database."""
        if name not in self.configs:
            return f"Database configuration not found: {name}"
        
        config = self.configs[name]
        
        if config.db_type == DatabaseType.SQLITE:
            return "SQLite does not produce logs"
        
        return self.docker_manager.get_container_logs(config.container_name, lines)
    
    def create_backup(self, name: str) -> Optional[str]:
        """Create a backup of a database."""
        if name not in self.configs:
            logger.warning(f"Database configuration not found: {name}")
            return None
        
        config = self.configs[name]
        return self.backup_manager.create_backup(config)
    
    def restore_backup(self, name: str, backup_path: str) -> bool:
        """Restore a database from a backup."""
        if name not in self.configs:
            logger.warning(f"Database configuration not found: {name}")
            return False
        
        config = self.configs[name]
        
        # Close connection if exists
        if name in self.connections:
            self.connections[name].disconnect()
            del self.connections[name]
        
        return self.backup_manager.restore_backup(config, backup_path)
    
    def list_backups(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups."""
        return self.backup_manager.list_backups(name)
    
    def cleanup_old_backups(self, name: Optional[str] = None) -> int:
        """Clean up old backups."""
        return self.backup_manager.cleanup_old_backups(name)
    
    def _schedule_backup(self, config: DatabaseConfig) -> None:
        """Schedule a backup for a database."""
        if not config.auto_backup:
            return
        
        # Parse cron schedule
        schedule_parts = config.backup_schedule.split()
        if len(schedule_parts) != 5:
            logger.warning(f"Invalid backup schedule for {config.name}: {config.backup_schedule}")
            return
        
        minute, hour, day, month, day_of_week = schedule_parts
        
        # Schedule backup
        job = None
        
        if minute != "*" and hour != "*" and day == "*" and month == "*" and day_of_week == "*":
            # Daily at specific time
            job = self.scheduler.every().day.at(f"{hour.zfill(2)}:{minute.zfill(2)}")
        elif minute != "*" and hour != "*" and day != "*" and month == "*" and day_of_week == "*":
            # Monthly at specific day and time
            job = self.scheduler.every().month.at(f"{day}-{hour.zfill(2)}:{minute.zfill(2)}")
        elif minute != "*" and hour != "*" and day == "*" and month == "*" and day_of_week != "*":
            # Weekly at specific day and time
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            day_idx = int(day_of_week) % 7
            day_name = days[day_idx]
            job_func = getattr(self.scheduler.every(), day_name)
            job = job_func.at(f"{hour.zfill(2)}:{minute.zfill(2)}")
        else:
            # Default to daily at 3 AM
            job = self.scheduler.every().day.at("03:00")
        
        if job:
            job.do(self._backup_job, config.name)
            logger.info(f"Scheduled backup for {config.name}: {config.backup_schedule}")
    
    def _update_backup_schedule(self, config: DatabaseConfig) -> None:
        """Update backup schedule for a database."""
        # Remove existing schedule
        self._remove_from_scheduler(config.name)
        
        # Add new schedule if enabled
        if config.auto_backup:
            self._schedule_backup(config)
    
    def _remove_from_scheduler(self, name: str) -> None:
        """Remove a database from the scheduler."""
        jobs_to_remove = []
        for job in self.scheduler.jobs:
            if hasattr(job, "job_func") and hasattr(job.job_func, "__name__") and job.job_func.__name__ == "_backup_job":
                if len(job.job_func.args) > 0 and job.job_func.args[0] == name:
                    jobs_to_remove.append(job)
        
        for job in jobs_to_remove:
            self.scheduler.cancel_job(job)
    
    def _backup_job(self, name: str) -> None:
        """Job function for scheduled backups."""
        logger.info(f"Running scheduled backup for {name}")
        try:
            backup_path = self.create_backup(name)
            if backup_path:
                logger.info(f"Scheduled backup created successfully: {backup_path}")
                
                # Clean up old backups
                deleted = self.cleanup_old_backups(name)
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old backups for {name}")
            else:
                logger.error(f"Scheduled backup failed for {name}")
        except Exception as e:
            logger.error(f"Error in scheduled backup for {name}: {e}")
    
    def _start_scheduler(self) -> None:
        """Start the scheduler thread."""
        if self._scheduler_thread is not None:
            return
        
        # Schedule backups for all enabled databases
        for config in self.configs.values():
            if config.enabled and config.auto_backup:
                self._schedule_backup(config)
        
        # Start scheduler thread
        self._stop_scheduler.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        
        logger.info("Scheduler started")
    
    def _scheduler_loop(self) -> None:
        """Scheduler thread loop."""
        while not self._stop_scheduler.is_set():
            self.scheduler.run_pending()
            time.sleep(1)
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        if self._scheduler_thread is None:
            return
        
        self._stop_scheduler.set()
        self._scheduler_thread.join(timeout=5)
        self._scheduler_thread = None
        
        logger.info("Scheduler stopped")
    
    def start_all_databases(self) -> Dict[str, bool]:
        """Start all enabled databases."""
        results = {}
        
        for name, config in self.configs.items():
            if config.enabled and config.auto_start:
                results[name] = self.start_database(name)
        
        return results
    
    def stop_all_databases(self) -> Dict[str, bool]:
        """Stop all databases."""
        results = {}
        
        # Close all connections
        for name in list(self.connections.keys()):
            self.connections[name].disconnect()
            del self.connections[name]
        
        # Stop all databases
        for name in self.configs.keys():
            results[name] = self.stop_database(name)
        
        return results
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all databases."""
        return {name: self.get_database_status(name) for name in self.configs.keys()}
    
    def execute_query(self, name: str, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a query on a database."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.execute_query(query, params)
    
    def execute_non_query(self, name: str, query: str, params: Dict[str, Any] = None) -> int:
        """Execute a non-query statement on a database."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.execute_non_query(query, params)
    
    def query_to_dataframe(self, name: str, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query and return results as a pandas DataFrame."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.query_to_dataframe(query, params)
    
    def dataframe_to_table(self, name: str, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> int:
        """Write a pandas DataFrame to a table."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.dataframe_to_table(df, table_name, if_exists)
    
    def get_table_names(self, name: str) -> List[str]:
        """Get list of table names in a database."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.get_table_names()
    
    def get_table_schema(self, name: str, table_name: str) -> List[Dict[str, Any]]:
        """Get schema for a table."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.get_table_schema(table_name)
    
    def table_exists(self, name: str, table_name: str) -> bool:
        """Check if a table exists."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.table_exists(table_name)
    
    def create_table(self, name: str, table_name: str, columns: Dict[str, Any]) -> bool:
        """Create a table with the specified columns."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.create_table(table_name, columns)
    
    def drop_table(self, name: str, table_name: str) -> bool:
        """Drop a table."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.drop_table(table_name)
    
    def insert_data(self, name: str, table_name: str, data: List[Dict[str, Any]]) -> int:
        """Insert data into a table."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.insert_data(table_name, data)
    
    def update_data(self, name: str, table_name: str, data: Dict[str, Any], condition: str, params: Dict[str, Any] = None) -> int:
        """Update data in a table."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.update_data(table_name, data, condition, params)
    
    def delete_data(self, name: str, table_name: str, condition: str, params: Dict[str, Any] = None) -> int:
        """Delete data from a table."""
        connection = self.get_connection(name)
        if not connection:
            raise ValueError(f"Failed to get connection to database: {name}")
        
        return connection.delete_data(table_name, condition, params)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Stop scheduler
        self.stop_scheduler()
        
        # Close all connections
        for name in list(self.connections.keys()):
            self.connections[name].disconnect()
        
        self.connections = {}

# Initialize database manager as a singleton
_database_manager = None

def get_database_manager() -> DatabaseManager:
    """Get the database manager singleton."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager

# Example usage
if __name__ == "__main__":
    # Initialize the database manager
    db_manager = get_database_manager()
    
    # Start all databases
    print("Starting all databases...")
    results = db_manager.start_all_databases()
    for name, success in results.items():
        print(f"  {name}: {'Started' if success else 'Failed'}")
    
    # Get status of all databases
    print("\nDatabase statuses:")
    statuses = db_manager.get_all_statuses()
    for name, status in statuses.items():
        print(f"  {name}: {status['status']}")
    
    # Example query on SQLite database
    try:
        print("\nCreating example table...")
        db_manager.create_table("default_sqlite", "example", {
            "id": {"type": "integer", "primary_key": True},
            "name": {"type": "string", "length": 100},
            "value": {"type": "float"}
        })
        
        print("Inserting data...")
        db_manager.insert_data("default_sqlite", "example", [
            {"name": "Item 1", "value": 10.5},
            {"name": "Item 2", "value": 20.75},
            {"name": "Item 3", "value": 30.25}
        ])
        
        print("Querying data...")
        results = db_manager.execute_query("default_sqlite", "SELECT * FROM example")
        for row in results:
            print(f"  {row['id']}: {row['name']} = {row['value']}")
    except Exception as e:
        print(f"Error in example: {e}")
    
    # Clean up
    print("\nCleaning up...")
    db_manager.cleanup()
