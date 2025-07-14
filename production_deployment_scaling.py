import os
import sys
import json
import time
import yaml
import uuid
import shutil
import logging
import asyncio
import aiohttp
import requests
import subprocess
import threading
import traceback
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import ipaddress
import socket
import datetime
import base64
import hashlib
import hmac
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import internal modules
try:
    from performance_monitor import PerformanceMonitor
    from database_manager import DatabaseManager
    from enhanced_security_compliance import SecurityManager, ComplianceManager
except ImportError:
    print("Warning: Some internal modules could not be imported. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("production_deployment_scaling")

# Constants
CONFIG_DIR = Path("config")
DEPLOYMENT_DIR = Path("deployment")
TEMPLATES_DIR = Path("templates")
TERRAFORM_DIR = Path("terraform")
KUBERNETES_DIR = Path("kubernetes")
DOCKER_DIR = Path("docker")
LOGS_DIR = Path("logs")
METRICS_DIR = Path("metrics")
SECRETS_DIR = Path("secrets")

# Ensure directories exist
for directory in [CONFIG_DIR, DEPLOYMENT_DIR, TEMPLATES_DIR, TERRAFORM_DIR, 
                 KUBERNETES_DIR, DOCKER_DIR, LOGS_DIR, METRICS_DIR, SECRETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default configuration path
DEFAULT_CONFIG_PATH = CONFIG_DIR / "deployment_config.json"

class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITAL_OCEAN = "digital_ocean"
    LINODE = "linode"
    VULTR = "vultr"
    LOCAL = "local"
    CUSTOM = "custom"

class ContainerOrchestrator(Enum):
    """Container orchestration platforms."""
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    ECS = "ecs"
    NOMAD = "nomad"
    CUSTOM = "custom"

class LoadBalancerType(Enum):
    """Load balancer types."""
    NGINX = "nginx"
    HAPROXY = "haproxy"
    TRAEFIK = "traefik"
    ENVOY = "envoy"
    CLOUD_NATIVE = "cloud_native"
    CUSTOM = "custom"

class DeploymentStrategy(Enum):
    """Deployment strategies."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    CUSTOM = "custom"

class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_COUNT = "request_count"
    CUSTOM_METRIC = "custom_metric"

class MonitoringSystem(Enum):
    """Monitoring systems."""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    CLOUDWATCH = "cloudwatch"
    CUSTOM = "custom"

class LoggingSystem(Enum):
    """Logging systems."""
    ELK = "elk"
    LOKI = "loki"
    CLOUDWATCH = "cloudwatch"
    STACKDRIVER = "stackdriver"
    CUSTOM = "custom"

class CICDSystem(Enum):
    """CI/CD systems."""
    JENKINS = "jenkins"
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    CIRCLE_CI = "circle_ci"
    AZURE_DEVOPS = "azure_devops"
    CUSTOM = "custom"

class InfrastructureProvider(Enum):
    """Infrastructure as Code providers."""
    TERRAFORM = "terraform"
    CLOUDFORMATION = "cloudformation"
    ARM_TEMPLATES = "arm_templates"
    PULUMI = "pulumi"
    CUSTOM = "custom"

class ResourceType(Enum):
    """Resource types."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    CUSTOM = "custom"

class NetworkType(Enum):
    """Network types."""
    VPC = "vpc"
    VNET = "vnet"
    SUBNET = "subnet"
    SECURITY_GROUP = "security_group"
    LOAD_BALANCER = "load_balancer"
    CUSTOM = "custom"

class DatabaseType(Enum):
    """Database types."""
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    DYNAMODB = "dynamodb"
    COSMOSDB = "cosmosdb"
    CUSTOM = "custom"

class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    CUSTOM = "custom"

class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

@dataclass
class ResourceRequirements:
    """Resource requirements for deployment."""
    cpu: str = "1"
    memory: str = "1Gi"
    storage: str = "10Gi"
    gpu: str = "0"
    replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 5
    custom_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "storage": self.storage,
            "gpu": self.gpu,
            "replicas": self.replicas,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "custom_requirements": self.custom_requirements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceRequirements':
        """Create from dictionary."""
        return cls(
            cpu=data.get("cpu", "1"),
            memory=data.get("memory", "1Gi"),
            storage=data.get("storage", "10Gi"),
            gpu=data.get("gpu", "0"),
            replicas=data.get("replicas", 1),
            min_replicas=data.get("min_replicas", 1),
            max_replicas=data.get("max_replicas", 5),
            custom_requirements=data.get("custom_requirements", {})
        )

@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    enabled: bool = True
    policy: ScalingPolicy = ScalingPolicy.CPU_UTILIZATION
    target_value: float = 70.0  # e.g., 70% CPU utilization
    min_replicas: int = 1
    max_replicas: int = 10
    cooldown_period: int = 300  # seconds
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "policy": self.policy.value,
            "target_value": self.target_value,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "cooldown_period": self.cooldown_period,
            "custom_metrics": self.custom_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScalingConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            policy=ScalingPolicy(data.get("policy", ScalingPolicy.CPU_UTILIZATION.value)),
            target_value=data.get("target_value", 70.0),
            min_replicas=data.get("min_replicas", 1),
            max_replicas=data.get("max_replicas", 10),
            cooldown_period=data.get("cooldown_period", 300),
            custom_metrics=data.get("custom_metrics", {})
        )

@dataclass
class NetworkConfig:
    """Network configuration."""
    vpc_id: Optional[str] = None
    subnet_ids: List[str] = field(default_factory=list)
    security_group_ids: List[str] = field(default_factory=list)
    load_balancer_type: LoadBalancerType = LoadBalancerType.CLOUD_NATIVE
    public_ip: bool = True
    domain_name: Optional[str] = None
    use_https: bool = True
    custom_network_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vpc_id": self.vpc_id,
            "subnet_ids": self.subnet_ids,
            "security_group_ids": self.security_group_ids,
            "load_balancer_type": self.load_balancer_type.value,
            "public_ip": self.public_ip,
            "domain_name": self.domain_name,
            "use_https": self.use_https,
            "custom_network_config": self.custom_network_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkConfig':
        """Create from dictionary."""
        return cls(
            vpc_id=data.get("vpc_id"),
            subnet_ids=data.get("subnet_ids", []),
            security_group_ids=data.get("security_group_ids", []),
            load_balancer_type=LoadBalancerType(data.get("load_balancer_type", LoadBalancerType.CLOUD_NATIVE.value)),
            public_ip=data.get("public_ip", True),
            domain_name=data.get("domain_name"),
            use_https=data.get("use_https", True),
            custom_network_config=data.get("custom_network_config", {})
        )

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    system: MonitoringSystem = MonitoringSystem.PROMETHEUS
    logging_system: LoggingSystem = LoggingSystem.ELK
    metrics_retention_days: int = 30
    logs_retention_days: int = 90
    alert_endpoints: Dict[str, str] = field(default_factory=dict)
    dashboards: List[str] = field(default_factory=list)
    custom_monitoring_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system": self.system.value,
            "logging_system": self.logging_system.value,
            "metrics_retention_days": self.metrics_retention_days,
            "logs_retention_days": self.logs_retention_days,
            "alert_endpoints": self.alert_endpoints,
            "dashboards": self.dashboards,
            "custom_monitoring_config": self.custom_monitoring_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """Create from dictionary."""
        return cls(
            system=MonitoringSystem(data.get("system", MonitoringSystem.PROMETHEUS.value)),
            logging_system=LoggingSystem(data.get("logging_system", LoggingSystem.ELK.value)),
            metrics_retention_days=data.get("metrics_retention_days", 30),
            logs_retention_days=data.get("logs_retention_days", 90),
            alert_endpoints=data.get("alert_endpoints", {}),
            dashboards=data.get("dashboards", []),
            custom_monitoring_config=data.get("custom_monitoring_config", {})
        )

@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: DatabaseType = DatabaseType.POSTGRES
    version: str = "14"
    size: str = "db.t3.medium"  # AWS RDS instance type or equivalent
    storage_gb: int = 20
    multi_az: bool = True
    backup_retention_days: int = 7
    encryption_enabled: bool = True
    connection_string_secret: Optional[str] = None
    custom_db_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "version": self.version,
            "size": self.size,
            "storage_gb": self.storage_gb,
            "multi_az": self.multi_az,
            "backup_retention_days": self.backup_retention_days,
            "encryption_enabled": self.encryption_enabled,
            "connection_string_secret": self.connection_string_secret,
            "custom_db_config": self.custom_db_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseConfig':
        """Create from dictionary."""
        return cls(
            type=DatabaseType(data.get("type", DatabaseType.POSTGRES.value)),
            version=data.get("version", "14"),
            size=data.get("size", "db.t3.medium"),
            storage_gb=data.get("storage_gb", 20),
            multi_az=data.get("multi_az", True),
            backup_retention_days=data.get("backup_retention_days", 7),
            encryption_enabled=data.get("encryption_enabled", True),
            connection_string_secret=data.get("connection_string_secret"),
            custom_db_config=data.get("custom_db_config", {})
        )

@dataclass
class CICDConfig:
    """CI/CD configuration."""
    system: CICDSystem = CICDSystem.GITHUB_ACTIONS
    repository_url: str = ""
    branch: str = "main"
    build_steps: List[str] = field(default_factory=list)
    test_steps: List[str] = field(default_factory=list)
    deploy_steps: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    custom_cicd_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system": self.system.value,
            "repository_url": self.repository_url,
            "branch": self.branch,
            "build_steps": self.build_steps,
            "test_steps": self.test_steps,
            "deploy_steps": self.deploy_steps,
            "environment_variables": self.environment_variables,
            "secrets": self.secrets,
            "triggers": self.triggers,
            "custom_cicd_config": self.custom_cicd_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CICDConfig':
        """Create from dictionary."""
        return cls(
            system=CICDSystem(data.get("system", CICDSystem.GITHUB_ACTIONS.value)),
            repository_url=data.get("repository_url", ""),
            branch=data.get("branch", "main"),
            build_steps=data.get("build_steps", []),
            test_steps=data.get("test_steps", []),
            deploy_steps=data.get("deploy_steps", []),
            environment_variables=data.get("environment_variables", {}),
            secrets=data.get("secrets", []),
            triggers=data.get("triggers", []),
            custom_cicd_config=data.get("custom_cicd_config", {})
        )

@dataclass
class DeploymentConfig:
    """Main deployment configuration."""
    name: str
    environment: DeploymentEnvironment
    cloud_provider: CloudProvider
    region: str
    orchestrator: ContainerOrchestrator
    deployment_strategy: DeploymentStrategy
    resource_requirements: ResourceRequirements
    scaling_config: ScalingConfig
    network_config: NetworkConfig
    monitoring_config: MonitoringConfig
    database_config: Optional[DatabaseConfig] = None
    cicd_config: Optional[CICDConfig] = None
    infrastructure_provider: InfrastructureProvider = InfrastructureProvider.TERRAFORM
    high_availability: bool = True
    disaster_recovery: bool = True
    backup_enabled: bool = True
    encryption_enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "environment": self.environment.value,
            "cloud_provider": self.cloud_provider.value,
            "region": self.region,
            "orchestrator": self.orchestrator.value,
            "deployment_strategy": self.deployment_strategy.value,
            "resource_requirements": self.resource_requirements.to_dict(),
            "scaling_config": self.scaling_config.to_dict(),
            "network_config": self.network_config.to_dict(),
            "monitoring_config": self.monitoring_config.to_dict(),
            "infrastructure_provider": self.infrastructure_provider.value,
            "high_availability": self.high_availability,
            "disaster_recovery": self.disaster_recovery,
            "backup_enabled": self.backup_enabled,
            "encryption_enabled": self.encryption_enabled,
            "tags": self.tags,
            "custom_config": self.custom_config
        }
        
        if self.database_config:
            result["database_config"] = self.database_config.to_dict()
        
        if self.cicd_config:
            result["cicd_config"] = self.cicd_config.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentConfig':
        """Create from dictionary."""
        config = cls(
            name=data["name"],
            environment=DeploymentEnvironment(data["environment"]),
            cloud_provider=CloudProvider(data["cloud_provider"]),
            region=data["region"],
            orchestrator=ContainerOrchestrator(data["orchestrator"]),
            deployment_strategy=DeploymentStrategy(data["deployment_strategy"]),
            resource_requirements=ResourceRequirements.from_dict(data["resource_requirements"]),
            scaling_config=ScalingConfig.from_dict(data["scaling_config"]),
            network_config=NetworkConfig.from_dict(data["network_config"]),
            monitoring_config=MonitoringConfig.from_dict(data["monitoring_config"]),
            infrastructure_provider=InfrastructureProvider(data.get("infrastructure_provider", InfrastructureProvider.TERRAFORM.value)),
            high_availability=data.get("high_availability", True),
            disaster_recovery=data.get("disaster_recovery", True),
            backup_enabled=data.get("backup_enabled", True),
            encryption_enabled=data.get("encryption_enabled", True),
            tags=data.get("tags", {}),
            custom_config=data.get("custom_config", {})
        )
        
        if "database_config" in data:
            config.database_config = DatabaseConfig.from_dict(data["database_config"])
        
        if "cicd_config" in data:
            config.cicd_config = CICDConfig.from_dict(data["cicd_config"])
        
        return config
    
    def save(self, filepath: Path = None) -> Path:
        """Save configuration to file."""
        if filepath is None:
            filepath = CONFIG_DIR / f"deployment_{self.name}_{self.environment.value}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Deployment configuration saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving deployment configuration: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: Path) -> 'DeploymentConfig':
        """Load configuration from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Deployment configuration loaded from {filepath}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading deployment configuration: {e}")
            raise

@dataclass
class DeploymentRecord:
    """Record of a deployment."""
    id: str
    config_name: str
    environment: DeploymentEnvironment
    status: DeploymentStatus
    start_time: int
    end_time: Optional[int] = None
    version: str = "1.0.0"
    commit_hash: Optional[str] = None
    deployed_by: str = "system"
    resources_created: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "config_name": self.config_name,
            "environment": self.environment.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "version": self.version,
            "commit_hash": self.commit_hash,
            "deployed_by": self.deployed_by,
            "resources_created": self.resources_created,
            "logs": self.logs,
            "metrics": self.metrics,
            "rollback_to": self.rollback_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentRecord':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            config_name=data["config_name"],
            environment=DeploymentEnvironment(data["environment"]),
            status=DeploymentStatus(data["status"]),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            version=data.get("version", "1.0.0"),
            commit_hash=data.get("commit_hash"),
            deployed_by=data.get("deployed_by", "system"),
            resources_created=data.get("resources_created", {}),
            logs=data.get("logs", []),
            metrics=data.get("metrics", {}),
            rollback_to=data.get("rollback_to")
        )
    
    def add_log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.datetime.now().isoformat()
        self.logs.append(f"{timestamp}: {message}")
    
    def update_status(self, status: DeploymentStatus) -> None:
        """Update deployment status."""
        self.status = status
        if status in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]:
            self.end_time = int(time.time())
    
    def save(self) -> None:
        """Save deployment record to file."""
        filepath = DEPLOYMENT_DIR / f"deployment_{self.id}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Deployment record saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving deployment record: {e}")

class DockerManager:
    """Manager for Docker containers and images."""
    
    def __init__(self):
        """Initialize the Docker manager."""
        self.docker_executable = self._find_docker_executable()
        if not self.docker_executable:
            logger.warning("Docker executable not found. Docker operations will fail.")
    
    def _find_docker_executable(self) -> Optional[str]:
        """Find the Docker executable."""
        try:
            # Check if docker is in PATH
            result = subprocess.run(["which", "docker"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Check common locations
            common_locations = [
                "/usr/bin/docker",
                "/usr/local/bin/docker",
                "C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe"
            ]
            
            for location in common_locations:
                if os.path.isfile(location):
                    return location
            
            return None
        except Exception as e:
            logger.error(f"Error finding Docker executable: {e}")
            return None
    
    def build_image(self, dockerfile_path: Path, tag: str, build_args: Dict[str, str] = None) -> bool:
        """Build a Docker image."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot build image.")
            return False
        
        try:
            cmd = [self.docker_executable, "build", "-t", tag, "-f", str(dockerfile_path), str(dockerfile_path.parent)]
            
            if build_args:
                for key, value in build_args.items():
                    cmd.extend(["--build-arg", f"{key}={value}"])
            
            logger.info(f"Building Docker image: {tag}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error building Docker image: {process.stderr}")
                return False
            
            logger.info(f"Docker image built successfully: {tag}")
            return True
        except Exception as e:
            logger.error(f"Error building Docker image: {e}")
            return False
    
    def push_image(self, tag: str, registry: str = None) -> bool:
        """Push a Docker image to a registry."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot push image.")
            return False
        
        try:
            # If registry is provided, tag the image for the registry
            if registry:
                registry_tag = f"{registry}/{tag}"
                subprocess.run([self.docker_executable, "tag", tag, registry_tag], check=True)
                tag = registry_tag
            
            logger.info(f"Pushing Docker image: {tag}")
            process = subprocess.run([self.docker_executable, "push", tag], capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error pushing Docker image: {process.stderr}")
                return False
            
            logger.info(f"Docker image pushed successfully: {tag}")
            return True
        except Exception as e:
            logger.error(f"Error pushing Docker image: {e}")
            return False
    
    def run_container(self, image: str, name: str = None, ports: Dict[int, int] = None, 
                     volumes: Dict[str, str] = None, env_vars: Dict[str, str] = None,
                     network: str = None, command: str = None) -> Optional[str]:
        """Run a Docker container."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot run container.")
            return None
        
        try:
            cmd = [self.docker_executable, "run", "-d"]
            
            if name:
                cmd.extend(["--name", name])
            
            if ports:
                for host_port, container_port in ports.items():
                    cmd.extend(["-p", f"{host_port}:{container_port}"])
            
            if volumes:
                for host_path, container_path in volumes.items():
                    cmd.extend(["-v", f"{host_path}:{container_path}"])
            
            if env_vars:
                for key, value in env_vars.items():
                    cmd.extend(["-e", f"{key}={value}"])
            
            if network:
                cmd.extend(["--network", network])
            
            cmd.append(image)
            
            if command:
                cmd.extend(command.split())
            
            logger.info(f"Running Docker container: {image}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error running Docker container: {process.stderr}")
                return None
            
            container_id = process.stdout.strip()
            logger.info(f"Docker container started: {container_id}")
            return container_id
        except Exception as e:
            logger.error(f"Error running Docker container: {e}")
            return None
    
    def stop_container(self, container_id: str) -> bool:
        """Stop a Docker container."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot stop container.")
            return False
        
        try:
            logger.info(f"Stopping Docker container: {container_id}")
            process = subprocess.run([self.docker_executable, "stop", container_id], capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error stopping Docker container: {process.stderr}")
                return False
            
            logger.info(f"Docker container stopped: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Error stopping Docker container: {e}")
            return False
    
    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove a Docker container."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot remove container.")
            return False
        
        try:
            cmd = [self.docker_executable, "rm"]
            if force:
                cmd.append("-f")
            cmd.append(container_id)
            
            logger.info(f"Removing Docker container: {container_id}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error removing Docker container: {process.stderr}")
                return False
            
            logger.info(f"Docker container removed: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing Docker container: {e}")
            return False
    
    def create_network(self, name: str, driver: str = "bridge") -> bool:
        """Create a Docker network."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot create network.")
            return False
        
        try:
            logger.info(f"Creating Docker network: {name}")
            process = subprocess.run(
                [self.docker_executable, "network", "create", "--driver", driver, name],
                capture_output=True, text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error creating Docker network: {process.stderr}")
                return False
            
            logger.info(f"Docker network created: {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Docker network: {e}")
            return False
    
    def create_volume(self, name: str) -> bool:
        """Create a Docker volume."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot create volume.")
            return False
        
        try:
            logger.info(f"Creating Docker volume: {name}")
            process = subprocess.run(
                [self.docker_executable, "volume", "create", name],
                capture_output=True, text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error creating Docker volume: {process.stderr}")
                return False
            
            logger.info(f"Docker volume created: {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Docker volume: {e}")
            return False
    
    def compose_up(self, compose_file: Path, project_name: str = None) -> bool:
        """Run docker-compose up."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot run docker-compose.")
            return False
        
        try:
            cmd = [self.docker_executable, "compose", "-f", str(compose_file)]
            
            if project_name:
                cmd.extend(["-p", project_name])
            
            cmd.extend(["up", "-d"])
            
            logger.info(f"Running docker-compose up: {compose_file}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error running docker-compose up: {process.stderr}")
                return False
            
            logger.info(f"docker-compose up completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error running docker-compose up: {e}")
            return False
    
    def compose_down(self, compose_file: Path, project_name: str = None, volumes: bool = False) -> bool:
        """Run docker-compose down."""
        if not self.docker_executable:
            logger.error("Docker executable not found. Cannot run docker-compose.")
            return False
        
        try:
            cmd = [self.docker_executable, "compose", "-f", str(compose_file)]
            
            if project_name:
                cmd.extend(["-p", project_name])
            
            cmd.append("down")
            
            if volumes:
                cmd.append("-v")
            
            logger.info(f"Running docker-compose down: {compose_file}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error running docker-compose down: {process.stderr}")
                return False
            
            logger.info(f"docker-compose down completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error running docker-compose down: {e}")
            return False
    
    def generate_dockerfile(self, base_image: str, app_dir: Path, requirements_file: Path = None,
                           expose_ports: List[int] = None, env_vars: Dict[str, str] = None,
                           cmd: str = None) -> Path:
        """Generate a Dockerfile."""
        try:
            dockerfile_path = app_dir / "Dockerfile"
            
            dockerfile_content = [
                f"FROM {base_image}",
                "",
                "WORKDIR /app",
                ""
            ]
            
            # Copy requirements file if provided
            if requirements_file and requirements_file.exists():
                rel_path = requirements_file.relative_to(app_dir) if app_dir in requirements_file.parents else requirements_file.name
                dockerfile_content.extend([
                    f"COPY {rel_path} .",
                    "RUN pip install --no-cache-dir -r requirements.txt",
                    ""
                ])
            
            # Copy application files
            dockerfile_content.extend([
                "COPY . .",
                ""
            ])
            
            # Set environment variables
            if env_vars:
                for key, value in env_vars.items():
                    dockerfile_content.append(f"ENV {key}={value}")
                dockerfile_content.append("")
            
            # Expose ports
            if expose_ports:
                for port in expose_ports:
                    dockerfile_content.append(f"EXPOSE {port}")
                dockerfile_content.append("")
            
            # Set command
            if cmd:
                dockerfile_content.append(f"CMD {cmd}")
            
            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write("\n".join(dockerfile_content))
            
            logger.info(f"Dockerfile generated: {dockerfile_path}")
            return dockerfile_path
        except Exception as e:
            logger.error(f"Error generating Dockerfile: {e}")
            raise

class KubernetesManager:
    """Manager for Kubernetes resources."""
    
    def __init__(self):
        """Initialize the Kubernetes manager."""
        self.kubectl_executable = self._find_kubectl_executable()
        if not self.kubectl_executable:
            logger.warning("kubectl executable not found. Kubernetes operations will fail.")
    
    def _find_kubectl_executable(self) -> Optional[str]:
        """Find the kubectl executable."""
        try:
            # Check if kubectl is in PATH
            result = subprocess.run(["which", "kubectl"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Check common locations
            common_locations = [
                "/usr/bin/kubectl",
                "/usr/local/bin/kubectl",
                os.path.expanduser("~/.local/bin/kubectl"),
                os.path.expanduser("~/bin/kubectl")
            ]
            
            for location in common_locations:
                if os.path.isfile(location):
                    return location
            
            return None
        except Exception as e:
            logger.error(f"Error finding kubectl executable: {e}")
            return None
    
    def apply_manifest(self, manifest_path: Path) -> bool:
        """Apply a Kubernetes manifest."""
        if not self.kubectl_executable:
            logger.error("kubectl executable not found. Cannot apply manifest.")
            return False
        
        try:
            logger.info(f"Applying Kubernetes manifest: {manifest_path}")
            process = subprocess.run(
                [self.kubectl_executable, "apply", "-f", str(manifest_path)],
                capture_output=True, text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error applying Kubernetes manifest: {process.stderr}")
                return False
            
            logger.info(f"Kubernetes manifest applied successfully")
            return True
        except Exception as e:
            logger.error(f"Error applying Kubernetes manifest: {e}")
            return False
    
    def delete_resource(self, resource_type: str, resource_name: str, namespace: str = "default") -> bool:
        """Delete a Kubernetes resource."""
        if not self.kubectl_executable:
            logger.error("kubectl executable not found. Cannot delete resource.")
            return False
        
        try:
            logger.info(f"Deleting Kubernetes resource: {resource_type}/{resource_name} in namespace {namespace}")
            process = subprocess.run(
                [self.kubectl_executable, "delete", resource_type, resource_name, "-n", namespace],
                capture_output=True, text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error deleting Kubernetes resource: {process.stderr}")
                return False
            
            logger.info(f"Kubernetes resource deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting Kubernetes resource: {e}")
            return False
    
    def get_resource(self, resource_type: str, resource_name: str = None, namespace: str = "default",
                    output_format: str = "json") -> Optional[str]:
        """Get a Kubernetes resource."""
        if not self.kubectl_executable:
            logger.error("kubectl executable not found. Cannot get resource.")
            return None
        
        try:
            cmd = [self.kubectl_executable, "get", resource_type]
            
            if resource_name:
                cmd.append(resource_name)
            
            cmd.extend(["-n", namespace, "-o", output_format])
            
            logger.info(f"Getting Kubernetes resource: {resource_type}/{resource_name} in namespace {namespace}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error getting Kubernetes resource: {process.stderr}")
                return None
            
            return process.stdout
        except Exception as e:
            logger.error(f"Error getting Kubernetes resource: {e}")
            return None
    
    def create_namespace(self, namespace: str) -> bool:
        """Create a Kubernetes namespace."""
        if not self.kubectl_executable:
            logger.error("kubectl executable not found. Cannot create namespace.")
            return False
        
        try:
            logger.info(f"Creating Kubernetes namespace: {namespace}")
            process = subprocess.run(
                [self.kubectl_executable, "create", "namespace", namespace],
                capture_output=True, text=True
            )
            
            if process.returncode != 0:
                # Check if namespace already exists
                if "already exists" in process.stderr:
                    logger.info(f"Kubernetes namespace {namespace} already exists")
                    return True
                
                logger.error(f"Error creating Kubernetes namespace: {process.stderr}")
                return False
            
            logger.info(f"Kubernetes namespace created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating Kubernetes namespace: {e}")
            return False
    
    def create_secret(self, name: str, data: Dict[str, str], namespace: str = "default", 
                     secret_type: str = "generic") -> bool:
        """Create a Kubernetes secret."""
        if not self.kubectl_executable:
            logger.error("kubectl executable not found. Cannot create secret.")
            return False
        
        try:
            cmd = [self.kubectl_executable, "create", "secret", secret_type, name, "-n", namespace]
            
            for key, value in data.items():
                cmd.extend(["--from-literal", f"{key}={value}"])
            
            logger.info(f"Creating Kubernetes secret: {name} in namespace {namespace}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                # Check if secret already exists
                if "already exists" in process.stderr:
                    logger.info(f"Kubernetes secret {name} already exists, updating...")
                    return self.update_secret(name, data, namespace, secret_type)
                
                logger.error(f"Error creating Kubernetes secret: {process.stderr}")
                return False
            
            logger.info(f"Kubernetes secret created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating Kubernetes secret: {e}")
            return False
    
    def update_secret(self, name: str, data: Dict[str, str], namespace: str = "default",
                     secret_type: str = "generic") -> bool:
        """Update a Kubernetes secret."""
        if not self.kubectl_executable:
            logger.error("kubectl executable not found. Cannot update secret.")
            return False
        
        try:
            # Delete existing secret
            delete_process = subprocess.run(
                [self.kubectl_executable, "delete", "secret", name, "-n", namespace],
                capture_output=True, text=True
            )
            
            if delete_process.returncode != 0:
                logger.error(f"Error deleting existing secret: {delete_process.stderr}")
                return False
            
            # Create new secret
            return self.create_secret(name, data, namespace, secret_type)
        except Exception as e:
            logger.error(f"Error updating Kubernetes secret: {e}")
            return False
    
    def apply_deployment(self, name: str, image: str, replicas: int = 1, namespace: str = "default",
                        ports: List[Dict[str, Any]] = None, env_vars: Dict[str, str] = None,
                        resources: Dict[str, Dict[str, str]] = None, labels: Dict[str, str] = None) -> bool:
        """Apply a Kubernetes deployment."""
        try:
            # Create deployment manifest
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                    "labels": labels or {"app": name}
                },
                "spec": {
                    "replicas": replicas,
                    "selector": {
                        "matchLabels": labels or {"app": name}
                    },
                    "template": {
                        "metadata": {
                            "labels": labels or {"app": name}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": name,
                                    "image": image,
                                    "imagePullPolicy": "Always",
                                    "ports": ports or [{"containerPort": 80}]
                                }
                            ]
                        }
                    }
                }
            }
            
            # Add environment variables if provided
            if env_vars:
                container = deployment["spec"]["template"]["spec"]["containers"][0]
                container["env"] = [{"name": key, "value": value} for key, value in env_vars.items()]
            
            # Add resource requirements if provided
            if resources:
                container = deployment["spec"]["template"]["spec"]["containers"][0]
                container["resources"] = resources
            
            # Write deployment manifest to file
            manifest_path = KUBERNETES_DIR / f"{name}_deployment.yaml"
            with open(manifest_path, 'w') as f:
                yaml.dump(deployment, f)
            
            # Apply deployment manifest
            return self.apply_manifest(manifest_path)
        except Exception as e:
            logger.error(f"Error applying Kubernetes deployment: {e}")
            return False
    
    def apply_service(self, name: str, port: int, target_port: int, namespace: str = "default",
                     service_type: str = "ClusterIP", selector: Dict[str, str] = None) -> bool:
        """Apply a Kubernetes service."""
        try:
            # Create service manifest
            service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": name,
                    "namespace": namespace
                },
                "spec": {
                    "selector": selector or {"app": name},
                    "ports": [
                        {
                            "port": port,
                            "targetPort": target_port
                        }
                    ],
                    "type": service_type
                }
            }
            
            # Write service manifest to file
            manifest_path = KUBERNETES_DIR / f"{name}_service.yaml"
            with open(manifest_path, 'w') as f:
                yaml.dump(service, f)
            
            # Apply service manifest
            return self.apply_manifest(manifest_path)
        except Exception as e:
            logger.error(f"Error applying Kubernetes service: {e}")
            return False
    
    def apply_ingress(self, name: str, host: str, service_name: str, service_port: int, 
                     namespace: str = "default", tls: bool = False, tls_secret: str = None) -> bool:
        """Apply a Kubernetes ingress."""
        try:
            # Create ingress manifest
            ingress = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                    "annotations": {
                        "kubernetes.io/ingress.class": "nginx"
                    }
                },
                "spec": {
                    "rules": [
                        {
                            "host": host,
                            "http": {
                                "paths": [
                                    {
                                        "path": "/",
                                        "pathType": "Prefix",
                                        "backend": {
                                            "service": {
                                                "name": service_name,
                                                "port": {
                                                    "number": service_port
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
            
            # Add TLS configuration if enabled
            if tls:
                ingress["spec"]["tls"] = [
                    {
                        "hosts": [host],
                        "secretName": tls_secret or f"{name}-tls"
                    }
                ]
            
            # Write ingress manifest to file
            manifest_path = KUBERNETES_DIR / f"{name}_ingress.yaml"
            with open(manifest_path, 'w') as f:
                yaml.dump(ingress, f)
            
            # Apply ingress manifest
            return self.apply_manifest(manifest_path)
        except Exception as e:
            logger.error(f"Error applying Kubernetes ingress: {e}")
            return False
    
    def apply_horizontal_pod_autoscaler(self, name: str, deployment_name: str, min_replicas: int,
                                       max_replicas: int, cpu_utilization: int = 70,
                                       namespace: str = "default") -> bool:
        """Apply a Kubernetes Horizontal Pod Autoscaler."""
        try:
            # Create HPA manifest
            hpa = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": name,
                    "namespace": namespace
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": deployment_name
                    },
                    "minReplicas": min_replicas,
                    "maxReplicas": max_replicas,
                    "metrics": [
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "cpu",
                                "target": {
                                    "type": "Utilization",
                                    "averageUtilization": cpu_utilization
                                }
                            }
                        }
                    ]
                }
            }
            
            # Write HPA manifest to file
            manifest_path = KUBERNETES_DIR / f"{name}_hpa.yaml"
            with open(manifest_path, 'w') as f:
                yaml.dump(hpa, f)
            
            # Apply HPA manifest
            return self.apply_manifest(manifest_path)
        except Exception as e:
            logger.error(f"Error applying Kubernetes HPA: {e}")
            return False
    
    def get_pod_logs(self, pod_name: str, namespace: str = "default", container: str = None,
                    tail: int = None) -> Optional[str]:
        """Get logs from a Kubernetes pod."""
        if not self.kubectl_executable:
            logger.error("kubectl executable not found. Cannot get pod logs.")
            return None
        
        try:
            cmd = [self.kubectl_executable, "logs", pod_name, "-n", namespace]
            
            if container:
                cmd.extend(["-c", container])
            
            if tail:
                cmd.extend(["--tail", str(tail)])
            
            logger.info(f"Getting logs from pod: {pod_name} in namespace {namespace}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error getting pod logs: {process.stderr}")
                return None
            
            return process.stdout
        except Exception as e:
            logger.error(f"Error getting pod logs: {e}")
            return None

class TerraformManager:
    """Manager for Terraform infrastructure."""
    
    def __init__(self):
        """Initialize the Terraform manager."""
        self.terraform_executable = self._find_terraform_executable()
        if not self.terraform_executable:
            logger.warning("Terraform executable not found. Terraform operations will fail.")
    
    def _find_terraform_executable(self) -> Optional[str]:
        """Find the Terraform executable."""
        try:
            # Check if terraform is in PATH
            result = subprocess.run(["which", "terraform"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Check common locations
            common_locations = [
                "/usr/bin/terraform",
                "/usr/local/bin/terraform",
                os.path.expanduser("~/.local/bin/terraform"),
                os.path.expanduser("~/bin/terraform")
            ]
            
            for location in common_locations:
                if os.path.isfile(location):
                    return location
            
            return None
        except Exception as e:
            logger.error(f"Error finding Terraform executable: {e}")
            return None
    
    def init(self, working_dir: Path) -> bool:
        """Initialize a Terraform configuration."""
        if not self.terraform_executable:
            logger.error("Terraform executable not found. Cannot initialize.")
            return False
        
        try:
            logger.info(f"Initializing Terraform in {working_dir}")
            process = subprocess.run(
                [self.terraform_executable, "init"],
                cwd=str(working_dir),
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error initializing Terraform: {process.stderr}")
                return False
            
            logger.info("Terraform initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Terraform: {e}")
            return False
    
    def plan(self, working_dir: Path, var_file: Path = None, out_file: Path = None) -> Optional[str]:
        """Create a Terraform execution plan."""
        if not self.terraform_executable:
            logger.error("Terraform executable not found. Cannot create plan.")
            return None
        
        try:
            cmd = [self.terraform_executable, "plan"]
            
            if var_file:
                cmd.extend(["-var-file", str(var_file)])
            
            if out_file:
                cmd.extend(["-out", str(out_file)])
            
            logger.info(f"Creating Terraform plan in {working_dir}")
            process = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error creating Terraform plan: {process.stderr}")
                return None
            
            logger.info("Terraform plan created successfully")
            return process.stdout
        except Exception as e:
            logger.error(f"Error creating Terraform plan: {e}")
            return None
    
    def apply(self, working_dir: Path, var_file: Path = None, plan_file: Path = None,
             auto_approve: bool = False) -> Optional[str]:
        """Apply a Terraform configuration."""
        if not self.terraform_executable:
            logger.error("Terraform executable not found. Cannot apply configuration.")
            return None
        
        try:
            cmd = [self.terraform_executable, "apply"]
            
            if auto_approve:
                cmd.append("-auto-approve")
            
            if var_file:
                cmd.extend(["-var-file", str(var_file)])
            
            if plan_file:
                cmd.append(str(plan_file))
            
            logger.info(f"Applying Terraform configuration in {working_dir}")
            process = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error applying Terraform configuration: {process.stderr}")
                return None
            
            logger.info("Terraform configuration applied successfully")
            return process.stdout
        except Exception as e:
            logger.error(f"Error applying Terraform configuration: {e}")
            return None
    
    def destroy(self, working_dir: Path, var_file: Path = None, auto_approve: bool = False) -> bool:
        """Destroy Terraform-managed infrastructure."""
        if not self.terraform_executable:
            logger.error("Terraform executable not found. Cannot destroy infrastructure.")
            return False
        
        try:
            cmd = [self.terraform_executable, "destroy"]
            
            if auto_approve:
                cmd.append("-auto-approve")
            
            if var_file:
                cmd.extend(["-var-file", str(var_file)])
            
            logger.info(f"Destroying Terraform-managed infrastructure in {working_dir}")
            process = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error destroying Terraform-managed infrastructure: {process.stderr}")
                return False
            
            logger.info("Terraform-managed infrastructure destroyed successfully")
            return True
        except Exception as e:
            logger.error(f"Error destroying Terraform-managed infrastructure: {e}")
            return False
    
    def output(self, working_dir: Path, output_name: str = None, json_format: bool = True) -> Optional[str]:
        """Get Terraform outputs."""
        if not self.terraform_executable:
            logger.error("Terraform executable not found. Cannot get outputs.")
            return None
        
        try:
            cmd = [self.terraform_executable, "output"]
            
            if json_format:
                cmd.append("-json")
            
            if output_name:
                cmd.append(output_name)
            
            logger.info(f"Getting Terraform outputs in {working_dir}")
            process = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error getting Terraform outputs: {process.stderr}")
                return None
            
            return process.stdout
        except Exception as e:
            logger.error(f"Error getting Terraform outputs: {e}")
            return None
    
    def generate_aws_vpc_config(self, name: str, cidr_block: str = "10.0.0.0/16",
                               public_subnets: List[str] = None,
                               private_subnets: List[str] = None) -> Path:
        """Generate Terraform configuration for an AWS VPC."""
        try:
            # Create directory for Terraform configuration
            tf_dir = TERRAFORM_DIR / f"{name}_vpc"
            tf_dir.mkdir(parents=True, exist_ok=True)
            
            # Set default subnets if not provided
            if not public_subnets:
                public_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
            
            if not private_subnets:
                private_subnets = ["10.0.3.0/24", "10.0.4.0/24"]
            
            # Create main.tf
            main_tf = tf_dir / "main.tf"
            with open(main_tf, 'w') as f:
                f.write(f"""
provider "aws" {{
  region = var.region
}}

module "vpc" {{
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "{name}"
  cidr = "{cidr_block}"

  azs             = var.availability_zones
  private_subnets = {json.dumps(private_subnets)}
  public_subnets  = {json.dumps(public_subnets)}

  enable_nat_gateway = true
  single_nat_gateway = var.single_nat_gateway
  enable_vpn_gateway = false

  tags = {{
    Terraform   = "true"
    Environment = var.environment
    Project     = "{name}"
  }}
}}

resource "aws_security_group" "allow_web" {{
  name        = "{name}-web-sg"
  description = "Allow web traffic"
  vpc_id      = module.vpc.vpc_id

  ingress {{
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  ingress {{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  tags = {{
    Name        = "{name}-web-sg"
    Terraform   = "true"
    Environment = var.environment
    Project     = "{name}"
  }}
}}
""")
            
            # Create variables.tf
            variables_tf = tf_dir / "variables.tf"
            with open(variables_tf, 'w') as f:
                f.write("""
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "availability_zones" {
  description = "AWS availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b"]
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "single_nat_gateway" {
  description = "Use a single NAT gateway for all private subnets"
  type        = bool
  default     = true
}
""")
            
            # Create outputs.tf
            outputs_tf = tf_dir / "outputs.tf"
            with open(outputs_tf, 'w') as f:
                f.write("""
output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "nat_public_ips" {
  description = "List of public Elastic IPs created for AWS NAT Gateway"
  value       = module.vpc.nat_public_ips
}

output "web_security_group_id" {
  description = "ID of the web security group"
  value       = aws_security_group.allow_web.id
}
""")
            
            logger.info(f"AWS VPC Terraform configuration generated in {tf_dir}")
            return tf_dir
        except Exception as e:
            logger.error(f"Error generating AWS VPC Terraform configuration: {e}")
            raise
    
    def generate_aws_eks_config(self, name: str, vpc_id: str, subnet_ids: List[str],
                               node_instance_type: str = "t3.medium",
                               node_count: int = 2) -> Path:
        """Generate Terraform configuration for an AWS EKS cluster."""
        try:
            # Create directory for Terraform configuration
            tf_dir = TERRAFORM_DIR / f"{name}_eks"
            tf_dir.mkdir(parents=True, exist_ok=True)
            
            # Create main.tf
            main_tf = tf_dir / "main.tf"
            with open(main_tf, 'w') as f:
                f.write(f"""
provider "aws" {{
  region = var.region
}}

module "eks" {{
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 18.0"

  cluster_name    = "{name}"
  cluster_version = "1.24"

  vpc_id     = "{vpc_id}"
  subnet_ids = {json.dumps(subnet_ids)}

  eks_managed_node_groups = {{
    default = {{
      name = "{name}-node-group"

      instance_types = ["{node_instance_type}"]

      min_size     = {node_count}
      max_size     = {node_count * 2}
      desired_size = {node_count}
    }}
  }}

  tags = {{
    Terraform   = "true"
    Environment = var.environment
    Project     = "{name}"
  }}
}}

# Configure kubectl
resource "null_resource" "configure_kubectl" {{
  provisioner "local-exec" {{
    command = "aws eks update-kubeconfig --name {name} --region ${{var.region}}"
  }}

  depends_on = [module.eks]
}}
""")
            
            # Create variables.tf
            variables_tf = tf_dir / "variables.tf"
            with open(variables_tf, 'w') as f:
                f.write("""
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}
""")
            
            # Create outputs.tf
            outputs_tf = tf_dir / "outputs.tf"
            with open(outputs_tf, 'w') as f:
                f.write("""
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}
""")
            
            logger.info(f"AWS EKS Terraform configuration generated in {tf_dir}")
            return tf_dir
        except Exception as e:
            logger.error(f"Error generating AWS EKS Terraform configuration: {e}")
            raise
    
    def generate_aws_rds_config(self, name: str, vpc_id: str, subnet_ids: List[str],
                               db_instance_class: str = "db.t3.medium",
                               engine: str = "postgres",
                               engine_version: str = "14",
                               allocated_storage: int = 20) -> Path:
        """Generate Terraform configuration for an AWS RDS instance."""
        try:
            # Create directory for Terraform configuration
            tf_dir = TERRAFORM_DIR / f"{name}_rds"
            tf_dir.mkdir(parents=True, exist_ok=True)
            
            # Create main.tf
            main_tf = tf_dir / "main.tf"
            with open(main_tf, 'w') as f:
                f.write(f"""
provider "aws" {{
  region = var.region
}}

resource "aws_db_subnet_group" "{name}" {{
  name       = "{name}-subnet-group"
  subnet_ids = {json.dumps(subnet_ids)}

  tags = {{
    Name        = "{name}-subnet-group"
    Terraform   = "true"
    Environment = var.environment
    Project     = "{name}"
  }}
}}

resource "aws_security_group" "{name}_db_sg" {{
  name        = "{name}-db-sg"
  description = "Allow database traffic"
  vpc_id      = "{vpc_id}"

  ingress {{
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # Adjust this to match your VPC CIDR
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  tags = {{
    Name        = "{name}-db-sg"
    Terraform   = "true"
    Environment = var.environment
    Project     = "{name}"
  }}
}}

resource "aws_db_instance" "{name}" {{
  identifier             = "{name}"
  allocated_storage      = {allocated_storage}
  engine                 = "{engine}"
  engine_version         = "{engine_version}"
  instance_class         = "{db_instance_class}"
  db_name                = "{name.replace('-', '_')}"
  username               = "dbadmin"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.{name}.name
  vpc_security_group_ids = [aws_security_group.{name}_db_sg.id]
  skip_final_snapshot    = true
  multi_az               = var.multi_az
  storage_encrypted      = true
  backup_retention_period = 7

  tags = {{
    Name        = "{name}"
    Terraform   = "true"
    Environment = var.environment
    Project     = "{name}"
  }}
}}
""")
            
            # Create variables.tf
            variables_tf = tf_dir / "variables.tf"
            with open(variables_tf, 'w') as f:
                f.write("""
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "db_password" {
  description = "Password for the database"
  type        = string
  sensitive   = true
}

variable "multi_az" {
  description = "Enable multi-AZ deployment"
  type        = bool
  default     = true
}
""")
            
            # Create outputs.tf
            outputs_tf = tf_dir / "outputs.tf"
            with open(outputs_tf, 'w') as f:
                f.write(f"""
output "db_instance_address" {{
  description = "The address of the RDS instance"
  value       = aws_db_instance.{name}.address
}}

output "db_instance_endpoint" {{
  description = "The connection endpoint"
  value       = aws_db_instance.{name}.endpoint
}}

output "db_instance_name" {{
  description = "The database name"
  value       = aws_db_instance.{name}.db_name
}}

output "db_instance_username" {{
  description = "The master username for the database"
  value       = aws_db_instance.{name}.username
  sensitive   = true
}}

output "db_instance_port" {{
  description = "The database port"
  value       = aws_db_instance.{name}.port
}}
""")
            
            logger.info(f"AWS RDS Terraform configuration generated in {tf_dir}")
            return tf_dir
        except Exception as e:
            logger.error(f"Error generating AWS RDS Terraform configuration: {e}")
            raise

class CloudManager:
    """Manager for cloud resources."""
    
    def __init__(self):
        """Initialize the cloud manager."""
        self.terraform_manager = TerraformManager()
        self.credentials = {}
    
    def set_credentials(self, provider: CloudProvider, credentials: Dict[str, str]) -> None:
        """Set cloud provider credentials."""
        self.credentials[provider] = credentials
        
        # Set environment variables based on provider
        if provider == CloudProvider.AWS:
            os.environ["AWS_ACCESS_KEY_ID"] = credentials.get("access_key", "")
            os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.get("secret_key", "")
            os.environ["AWS_DEFAULT_REGION"] = credentials.get("region", "us-west-2")
        elif provider == CloudProvider.AZURE:
            os.environ["ARM_CLIENT_ID"] = credentials.get("client_id", "")
            os.environ["ARM_CLIENT_SECRET"] = credentials.get("client_secret", "")
            os.environ["ARM_SUBSCRIPTION_ID"] = credentials.get("subscription_id", "")
            os.environ["ARM_TENANT_ID"] = credentials.get("tenant_id", "")
        elif provider == CloudProvider.GCP:
            # For GCP, we need to write the credentials to a file
            if "credentials_json" in credentials:
                creds_file = SECRETS_DIR / "gcp_credentials.json"
                with open(creds_file, 'w') as f:
                    f.write(credentials["credentials_json"])
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_file)
            
            os.environ["GOOGLE_PROJECT"] = credentials.get("project_id", "")
    
    def deploy_infrastructure(self, config: DeploymentConfig) -> Tuple[bool, Dict[str, Any]]:
        """Deploy infrastructure based on configuration."""
        try:
            # Set credentials for the cloud provider
            if config.cloud_provider in self.credentials:
                self.set_credentials(config.cloud_provider, self.credentials[config.cloud_provider])
            
            # Create deployment record
            deployment_id = str(uuid.uuid4())
            record = DeploymentRecord(
                id=deployment_id,
                config_name=config.name,
                environment=config.environment,
                status=DeploymentStatus.IN_PROGRESS,
                start_time=int(time.time())
            )
            record.add_log(f"Starting infrastructure deployment for {config.name} in {config.environment.value}")
            
            # Create infrastructure based on provider
            outputs = {}
            if config.cloud_provider == CloudProvider.AWS:
                success, aws_outputs = self._deploy_aws_infrastructure(config, record)
                if not success:
                    record.update_status(DeploymentStatus.FAILED)
                    record.add_log("AWS infrastructure deployment failed")
                    record.save()
                    return False, {}
                
                outputs = aws_outputs
            elif config.cloud_provider == CloudProvider.AZURE:
                success, azure_outputs = self._deploy_azure_infrastructure(config, record)
                if not success:
                    record.update_status(DeploymentStatus.FAILED)
                    record.add_log("Azure infrastructure deployment failed")
                    record.save()
                    return False, {}
                
                outputs = azure_outputs
            elif config.cloud_provider == CloudProvider.GCP:
                success, gcp_outputs = self._deploy_gcp_infrastructure(config, record)
                if not success:
                    record.update_status(DeploymentStatus.FAILED)
                    record.add_log("GCP infrastructure deployment failed")
                    record.save()
                    return False, {}
                
                outputs = gcp_outputs
            else:
                record.update_status(DeploymentStatus.FAILED)
                record.add_log(f"Unsupported cloud provider: {config.cloud_provider.value}")
                record.save()
                return False, {}
            
            # Update deployment record
            record.update_status(DeploymentStatus.COMPLETED)
            record.add_log("Infrastructure deployment completed successfully")
            record.resources_created = outputs
            record.save()
            
            return True, outputs
        except Exception as e:
            logger.error(f"Error deploying infrastructure: {e}")
            logger.error(traceback.format_exc())
            
            # Update deployment record
            record = DeploymentRecord(
                id=str(uuid.uuid4()),
                config_name=config.name,
                environment=config.environment,
                status=DeploymentStatus.FAILED,
                start_time=int(time.time()),
                end_time=int(time.time())
            )
            record.add_log(f"Infrastructure deployment failed: {str
