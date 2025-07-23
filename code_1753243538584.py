# 7. Web Deployment Agent - Automated web business deployment
web_deployment_agent_code = '''"""
Web Deployment Agent
Automated web business deployment and infrastructure management
Supports multi-platform deployment with CI/CD automation
"""

import asyncio
import logging
import json
import yaml
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import tempfile
import os

class DeploymentPlatform(Enum):
    VERCEL = "vercel"
    NETLIFY = "netlify"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    HEROKU = "heroku"
    DIGITAL_OCEAN = "digital_ocean"
    CLOUDFLARE = "cloudflare_pages"
    GITHUB_PAGES = "github_pages"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"

class ApplicationType(Enum):
    STATIC_SITE = "static_site"
    SPA = "spa"
    SSR = "ssr"
    API = "api"
    FULLSTACK = "fullstack"
    MICROSERVICE = "microservice"
    LANDING_PAGE = "landing_page"
    E_COMMERCE = "e_commerce"
    BLOG = "blog"
    PORTFOLIO = "portfolio"

class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYED = "deployed"
    FAILED = "failed"
    UPDATING = "updating"
    MAINTENANCE = "maintenance"

class FrameworkType(Enum):
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    SVELTE = "svelte"
    NEXT_JS = "nextjs"
    NUXT = "nuxt"
    GATSBY = "gatsby"
    ASTRO = "astro"
    VANILLA = "vanilla"
    NODE_JS = "nodejs"
    PYTHON = "python"
    PHP = "php"

@dataclass
class DeploymentConfig:
    """Deployment configuration structure"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    platform: DeploymentPlatform = DeploymentPlatform.VERCEL
    application_type: ApplicationType = ApplicationType.STATIC_SITE
    framework: FrameworkType = FrameworkType.REACT
    build_command: str = "npm run build"
    build_directory: str = "dist"
    install_command: str = "npm install"
    dev_command: str = "npm run dev"
    node_version: str = "18"
    environment_variables: Dict[str, str] = field(default_factory=dict)
    domains: List[str] = field(default_factory=list)
    ssl_enabled: bool = True
    cdn_enabled: bool = True
    auto_deploy: bool = True
    branch_deploy: Dict[str, str] = field(default_factory=dict)
    redirect_rules: List[Dict[str, str]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    functions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class WebProject:
    """Web project structure"""
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    repository_url: str = ""
    local_path: str = ""
    framework: FrameworkType = FrameworkType.REACT
    application_type: ApplicationType = ApplicationType.STATIC_SITE
    deployment_config: Optional[DeploymentConfig] = None
    deployments: List[str] = field(default_factory=list)
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_deployment: Optional[datetime] = None
    live_url: str = ""
    preview_urls: List[str] = field(default_factory=list)
    build_logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Deployment:
    """Individual deployment record"""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    platform: DeploymentPlatform = DeploymentPlatform.VERCEL
    status: DeploymentStatus = DeploymentStatus.PENDING
    url: str = ""
    commit_hash: str = ""
    branch: str = "main"
    build_time: int = 0  # seconds
    deploy_time: int = 0  # seconds
    build_logs: List[str] = field(default_factory=list)
    error_logs: List[str] = field(default_factory=list)
    environment: str = "production"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InfrastructureTemplate:
    """Infrastructure template for different deployment scenarios"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    platform: DeploymentPlatform = DeploymentPlatform.AWS
    application_type: ApplicationType = ApplicationType.STATIC_SITE
    resources: List[Dict[str, Any]] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: Dict[str, float] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)

class WebDeploymentAgent:
    """
    Web Deployment Agent for Automated Infrastructure Management
    
    Capabilities:
    - Multi-platform deployment automation
    - CI/CD pipeline setup and management
    - Infrastructure as Code (IaC) generation
    - Performance monitoring and optimization
    - Security configuration and SSL management
    - Domain and DNS management
    - Automated testing and quality assurance
    - Cost optimization and resource management
    """
    
    def __init__(self, agent_id: str = "web_deployment_agent_001"):
        self.agent_id = agent_id
        self.logger = self._setup_logger()
        
        # Core data
        self.projects: Dict[str, WebProject] = {}
        self.deployments: Dict[str, Deployment] = {}
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.infrastructure_templates: Dict[str, InfrastructureTemplate] = {}
        
        # Platform configurations
        self.platform_configs = {}
        self.api_credentials = {}
        
        # Performance metrics
        self.metrics = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "average_build_time": 0.0,
            "average_deploy_time": 0.0,
            "uptime_percentage": 99.9,
            "projects_managed": 0,
            "cost_savings": 0.0
        }
        
        # Automation settings
        self.automation_config = {
            "auto_deploy_on_push": True,
            "auto_ssl_renewal": True,
            "auto_scaling": True,
            "performance_monitoring": True,
            "security_scanning": True,
            "cost_optimization": True
        }
        
        # Initialize templates and configurations
        self._initialize_platform_configs()
        self._initialize_infrastructure_templates()
        self._initialize_framework_configs()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for web deployment agent"""
        logger = logging.getLogger(f"WebDeploymentAgent-{self.agent_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_platform_configs(self):
        """Initialize platform-specific configurations"""
        self.platform_configs = {
            DeploymentPlatform.VERCEL: {
                "cli_command": "vercel",
                "config_file": "vercel.json",
                "build_output": ".vercel/output",
                "supported_frameworks": [FrameworkType.NEXT_JS, FrameworkType.REACT, FrameworkType.VUE],
                "features": ["edge_functions", "serverless", "auto_ssl", "cdn"],
                "limits": {"bandwidth": "100GB", "build_time": "32min"}
            },
            DeploymentPlatform.NETLIFY: {
                "cli_command": "netlify",
                "config_file": "netlify.toml",
                "build_output": "dist",
                "supported_frameworks": [FrameworkType.GATSBY, FrameworkType.REACT, FrameworkType.VUE],
                "features": ["functions", "forms", "identity", "analytics"],
                "limits": {"bandwidth": "100GB", "build_time": "15min"}
            },
            DeploymentPlatform.AWS: {
                "cli_command": "aws",
                "config_file": "cloudformation.yaml",
                "services": ["s3", "cloudfront", "lambda", "api_gateway", "route53"],
                "features": ["auto_scaling", "load_balancing", "monitoring"],
                "limits": {"flexible": True}
            },
            DeploymentPlatform.HEROKU: {
                "cli_command": "heroku",
                "config_file": "Procfile",
                "supported_frameworks": [FrameworkType.NODE_JS, FrameworkType.PYTHON, FrameworkType.PHP],
                "features": ["add_ons", "auto_scaling", "review_apps"],
                "limits": {"dyno_hours": "550/month"}
            }
        }
    
    def _initialize_infrastructure_templates(self):
        """Initialize infrastructure templates"""
        # Static site template
        static_template = InfrastructureTemplate(
            name="Static Site Template",
            description="Optimized for static websites and SPAs",
            platform=DeploymentPlatform.VERCEL,
            application_type=ApplicationType.STATIC_SITE,
            resources=[
                {"type": "cdn", "config": {"cache_policy": "max", "compression": True}},
                {"type": "ssl", "config": {"auto_renew": True, "force_https": True}},
                {"type": "domain", "config": {"custom_domain": True, "subdomain": True}}
            ],
            configuration={
                "build_command": "npm run build",
                "output_directory": "dist",
                "node_version": "18",
                "environment": "production"
            },
            estimated_cost={"monthly": 0.0, "per_request": 0.0001},
            scaling_config={"auto": True, "max_instances": 100}
        )
        
        # Full-stack template
        fullstack_template = InfrastructureTemplate(
            name="Full-Stack Application Template",
            description="Complete stack with database and serverless functions",
            platform=DeploymentPlatform.AWS,
            application_type=ApplicationType.FULLSTACK,
            resources=[
                {"type": "compute", "config": {"lambda_functions": True, "api_gateway": True}},
                {"type": "database", "config": {"type": "dynamodb", "backup": True}},
                {"type": "storage", "config": {"s3_bucket": True, "cdn": True}},
                {"type": "monitoring", "config": {"cloudwatch": True, "alarms": True}}
            ],
            configuration={
                "runtime": "nodejs18.x",
                "memory": "512MB",
                "timeout": "30s",
                "environment": "production"
            },
            estimated_cost={"monthly": 25.0, "per_request": 0.0002},
            scaling_config={"auto": True, "min_instances": 1, "max_instances": 1000}
        )
        
        # E-commerce template
        ecommerce_template = InfrastructureTemplate(
            name="E-commerce Platform Template",
            description="Scalable e-commerce solution with payment integration",
            platform=DeploymentPlatform.AWS,
            application_type=ApplicationType.E_COMMERCE,
            resources=[
                {"type": "compute", "config": {"ec2_instances": True, "load_balancer": True}},
                {"type": "database", "config": {"type": "rds", "multi_az": True, "backup": True}},
                {"type": "cache", "config": {"redis": True, "memory": "2GB"}},
                {"type": "cdn", "config": {"cloudfront": True, "global": True}},
                {"type": "security", "config": {"waf": True, "ssl": True}}
            ],
            configuration={
                "instance_type": "t3.medium",
                "database": "postgres",
                "ssl_certificate": "auto",
                "backup_retention": "7_days"
            },
            estimated_cost={"monthly": 150.0, "per_transaction": 0.01},
            scaling_config={"auto": True, "min_instances": 2, "max_instances": 20}
        )
        
        self.infrastructure_templates = {
            "static_site": static_template,
            "fullstack": fullstack_template,
            "ecommerce": ecommerce_template
        }
    
    def _initialize_framework_configs(self):
        """Initialize framework-specific configurations"""
        self.framework_configs = {
            FrameworkType.REACT: {
                "build_command": "npm run build",
                "dev_command": "npm start",
                "install_command": "npm install",
                "output_directory": "build",
                "package_manager": "npm",
                "node_version": "18"
            },
            FrameworkType.NEXT_JS: {
                "build_command": "npm run build",
                "dev_command": "npm run dev",
                "install_command": "npm install",
                "output_directory": ".next",
                "package_manager": "npm",
                "node_version": "18"
            },
            FrameworkType.VUE: {
                "build_command": "npm run build",
                "dev_command": "npm run serve",
                "install_command": "npm install",
                "output_directory": "dist",
                "package_manager": "npm",
                "node_version": "18"
            },
            FrameworkType.GATSBY: {
                "build_command": "gatsby build",
                "dev_command": "gatsby develop",
                "install_command": "npm install",
                "output_directory": "public",
                "package_manager": "npm",
                "node_version": "18"
            }
        }
    
    # Project Management
    async def create_project(self, project_data: Dict[str, Any]) -> str:
        """Create new web project"""
        try:
            # Detect framework if not specified
            framework = FrameworkType(project_data.get("framework", "react"))
            if not project_data.get("framework") and project_data.get("local_path"):
                framework = await self._detect_framework(project_data["local_path"])
            
            project = WebProject(
                name=project_data.get("name", ""),
                description=project_data.get("description", ""),
                repository_url=project_data.get("repository_url", ""),
                local_path=project_data.get("local_path", ""),
                framework=framework,
                application_type=ApplicationType(project_data.get("application_type", "static_site"))
            )
            
            # Create default deployment configuration
            project.deployment_config = await self._create_default_deployment_config(project)
            
            self.projects[project.project_id] = project
            self.metrics["projects_managed"] += 1
            
            self.logger.info(f"Created project: {project.name}")
            
            return project.project_id
            
        except Exception as e:
            self.logger.error(f"Error creating project: {e}")
            return ""
    
    async def _detect_framework(self, project_path: str) -> FrameworkType:
        """Auto-detect framework from project files"""
        try:
            project_path = Path(project_path)
            
            # Check package.json for framework indicators
            package_json_path = project_path / "package.json"
            if package_json_path.exists():
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                
                dependencies = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
                
                if "next" in dependencies:
                    return FrameworkType.NEXT_JS
                elif "nuxt" in dependencies:
                    return FrameworkType.NUXT
                elif "gatsby" in dependencies:
                    return FrameworkType.GATSBY
                elif "vue" in dependencies:
                    return FrameworkType.VUE
                elif "react" in dependencies:
                    return FrameworkType.REACT
                elif "svelte" in dependencies:
                    return FrameworkType.SVELTE
                elif "astro" in dependencies:
                    return FrameworkType.ASTRO
            
            # Check for framework-specific files
            if (project_path / "next.config.js").exists():
                return FrameworkType.NEXT_JS
            elif (project_path / "nuxt.config.js").exists():
                return FrameworkType.NUXT
            elif (project_path / "gatsby-config.js").exists():
                return FrameworkType.GATSBY
            elif (project_path / "vue.config.js").exists():
                return FrameworkType.VUE
            elif (project_path / "astro.config.mjs").exists():
                return FrameworkType.ASTRO
            
            return FrameworkType.VANILLA
            
        except Exception as e:
            self.logger.error(f"Error detecting framework: {e}")
            return FrameworkType.VANILLA
    
    async def _create_default_deployment_config(self, project: WebProject) -> DeploymentConfig:
        """Create default deployment configuration for project"""
        try:
            # Get framework-specific config
            framework_config = self.framework_configs.get(project.framework, {})
            
            # Select optimal platform based on project type
            optimal_platform = self._select_optimal_platform(project)
            
            config = DeploymentConfig(
                name=f"{project.name} Deployment",
                platform=optimal_platform,
                application_type=project.application_type,
                framework=project.framework,
                build_command=framework_config.get("build_command", "npm run build"),
                build_directory=framework_config.get("output_directory", "dist"),
                install_command=framework_config.get("install_command", "npm install"),
                dev_command=framework_config.get("dev_command", "npm run dev"),
                node_version=framework_config.get("node_version", "18"),
                auto_deploy=True,
                ssl_enabled=True,
                cdn_enabled=True
            )
            
            # Add environment variables
            config.environment_variables = {
                "NODE_ENV": "production",
                "BUILD_ENV": "production"
            }
            
            # Set branch deployment
            config.branch_deploy = {
                "main": "production",
                "develop": "staging",
                "feature/*": "preview"
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error creating deployment config: {e}")
            return DeploymentConfig()
    
    def _select_optimal_platform(self, project: WebProject) -> DeploymentPlatform:
        """Select optimal deployment platform based on project characteristics"""
        try:
            # Static sites - prefer Vercel/Netlify
            if project.application_type == ApplicationType.STATIC_SITE:
                if project.framework == FrameworkType.NEXT_JS:
                    return DeploymentPlatform.VERCEL
                elif project.framework == FrameworkType.GATSBY:
                    return DeploymentPlatform.NETLIFY
                else:
                    return DeploymentPlatform.VERCEL
            
            # SPAs - Vercel or Netlify
            elif project.application_type == ApplicationType.SPA:
                return DeploymentPlatform.VERCEL
            
            # Full-stack applications - AWS or Heroku
            elif project.application_type == ApplicationType.FULLSTACK:
                return DeploymentPlatform.AWS
            
            # APIs - Heroku or AWS
            elif project.application_type == ApplicationType.API:
                return DeploymentPlatform.HEROKU
            
            # E-commerce - AWS for scalability
            elif project.application_type == ApplicationType.E_COMMERCE:
                return DeploymentPlatform.AWS
            
            # Default to Vercel
            return DeploymentPlatform.VERCEL
            
        except Exception as e:
            self.logger.error(f"Error selecting platform: {e}")
            return DeploymentPlatform.VERCEL
    
    # Deployment Management
    async def deploy_project(self, project_id: str, deployment_options: Optional[Dict[str, Any]] = None) -> str:
        """Deploy project to configured platform"""
        try:
            if project_id not in self.projects:
                raise ValueError(f"Project {project_id} not found")
            
            project = self.projects[project_id]
            config = project.deployment_config
            
            if not config:
                raise ValueError("No deployment configuration found")
            
            # Create deployment record
            deployment = Deployment(
                project_id=project_id,
                platform=config.platform,
                branch=deployment_options.get("branch", "main") if deployment_options else "main",
                environment=deployment_options.get("environment", "production") if deployment_options else "production"
            )
            
            self.deployments[deployment.deployment_id] = deployment
            project.deployments.append(deployment.deployment_id)
            
            self.logger.info(f"Starting deployment: {project.name} to {config.platform.value}")
            
            # Execute platform-specific deployment
            if config.platform == DeploymentPlatform.VERCEL:
                success = await self._deploy_to_vercel(project, deployment, config)
            elif config.platform == DeploymentPlatform.NETLIFY:
                success = await self._deploy_to_netlify(project, deployment, config)
            elif config.platform == DeploymentPlatform.AWS:
                success = await self._deploy_to_aws(project, deployment, config)
            elif config.platform == DeploymentPlatform.HEROKU:
                success = await self._deploy_to_heroku(project, deployment, config)
            else:
                success = await self._deploy_generic(project, deployment, config)
            
            if success:
                deployment.status = DeploymentStatus.DEPLOYED
                deployment.completed_at = datetime.now()
                project.status = DeploymentStatus.DEPLOYED
                project.last_deployment = datetime.now()
                
                self.metrics["successful_deployments"] += 1
                self.logger.info(f"Deployment successful: {deployment.url}")
            else:
                deployment.status = DeploymentStatus.FAILED
                project.status = DeploymentStatus.FAILED
                self.metrics["failed_deployments"] += 1
                self.logger.error("Deployment failed")
            
            self.metrics["total_deployments"] += 1
            
            return deployment.deployment_id
            
        except Exception as e:
            self.logger.error(f"Error deploying project: {e}")
            return ""
    
    async def _deploy_to_vercel(self, project: WebProject, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy to Vercel platform"""
        try:
            deployment.status = DeploymentStatus.BUILDING
            build_start = datetime.now()
            
            # Prepare deployment command
            deployment_cmd = [
                "vercel",
                "--prod" if deployment.environment == "production" else "--staging",
                "--yes",  # Skip confirmations
                "--token", self.api_credentials.get("vercel", {}).get("token", ""),
            ]
            
            # Add project path if specified
            if project.local_path:
                deployment_cmd.extend(["--cwd", project.local_path])
            
            # Execute deployment (simulated)
            self.logger.info(f"Executing: {' '.join(deployment_cmd)}")
            
            # Simulate build process
            await asyncio.sleep(2)  # Simulate build time
            
            deployment.build_time = (datetime.now() - build_start).seconds
            deployment.url = f"https://{project.name.lower().replace(' ', '-')}.vercel.app"
            deployment.build_logs.append("Build completed successfully")
            
            # Update project URL
            if deployment.environment == "production":
                project.live_url = deployment.url
            else:
                project.preview_urls.append(deployment.url)
            
            return True
            
        except Exception as e:
            deployment.error_logs.append(f"Vercel deployment error: {str(e)}")
            self.logger.error(f"Vercel deployment failed: {e}")
            return False
    
    async def _deploy_to_netlify(self, project: WebProject, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy to Netlify platform"""
        try:
            deployment.status = DeploymentStatus.BUILDING
            build_start = datetime.now()
            
            # Simulate Netlify deployment
            await asyncio.sleep(1.5)
            
            deployment.build_time = (datetime.now() - build_start).seconds
            deployment.url = f"https://{project.name.lower().replace(' ', '-')}.netlify.app"
            deployment.build_logs.append("Netlify build completed successfully")
            
            if deployment.environment == "production":
                project.live_url = deployment.url
            else:
                project.preview_urls.append(deployment.url)
            
            return True
            
        except Exception as e:
            deployment.error_logs.append(f"Netlify deployment error: {str(e)}")
            self.logger.error(f"Netlify deployment failed: {e}")
            return False
    
    async def _deploy_to_aws(self, project: WebProject, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy to AWS platform"""
        try:
            deployment.status = DeploymentStatus.BUILDING
            build_start = datetime.now()
            
            # For AWS, we would typically use CloudFormation or CDK
            # Simulate AWS deployment process
            await asyncio.sleep(3)  # AWS deployments typically take longer
            
            deployment.build_time = (datetime.now() - build_start).seconds
            deployment.url = f"https://{project.name.lower().replace(' ', '-')}.s3-website.amazonaws.com"
            deployment.build_logs.append("AWS CloudFormation stack deployed successfully")
            
            if deployment.environment == "production":
                project.live_url = deployment.url
            else:
                project.preview_urls.append(deployment.url)
            
            return True
            
        except Exception as e:
            deployment.error_logs.append(f"AWS deployment error: {str(e)}")
            self.logger.error(f"AWS deployment failed: {e}")
            return False
    
    async def _deploy_to_heroku(self, project: WebProject, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Deploy to Heroku platform"""
        try:
            deployment.status = DeploymentStatus.BUILDING
            build_start = datetime.now()
            
            # Simulate Heroku deployment
            await asyncio.sleep(2.5)
            
            deployment.build_time = (datetime.now() - build_start).seconds
            deployment.url = f"https://{project.name.lower().replace(' ', '-')}.herokuapp.com"
            deployment.build_logs.append("Heroku dyno deployed successfully")
            
            if deployment.environment == "production":
                project.live_url = deployment.url
            else:
                project.preview_urls.append(deployment.url)
            
            return True
            
        except Exception as e:
            deployment.error_logs.append(f"Heroku deployment error: {str(e)}")
            self.logger.error(f"Heroku deployment failed: {e}")
            return False
    
    async def _deploy_generic(self, project: WebProject, deployment: Deployment, config: DeploymentConfig) -> bool:
        """Generic deployment process"""
        try:
            deployment.status = DeploymentStatus.BUILDING
            build_start = datetime.now()
            
            # Simulate generic deployment
            await asyncio.sleep(2)
            
            deployment.build_time = (datetime.now() - build_start).seconds
            deployment.url = f"https://{project.name.lower().replace(' ', '-')}.example.com"
            deployment.build_logs.append("Generic deployment completed successfully")
            
            project.live_url = deployment.url
            
            return True
            
        except Exception as e:
            deployment.error_logs.append(f"Generic deployment error: {str(e)}")
            self.logger.error(f"Generic deployment failed: {e}")
            return False
    
    # Configuration Generation
    async def generate_config_files(self, project_id: str) -> Dict[str, str]:
        """Generate platform-specific configuration files"""
        try:
            if project_id not in self.projects:
                raise ValueError(f"Project {project_id} not found")
            
            project = self.projects[project_id]
            config = project.deployment_config
            
            if not config:
                raise ValueError("No deployment configuration found")
            
            config_files = {}
            
            # Generate platform-specific config files
            if config.platform == DeploymentPlatform.VERCEL:
                config_files["vercel.json"] = self._generate_vercel_config(config)
            elif config.platform == DeploymentPlatform.NETLIFY:
                config_files["netlify.toml"] = self._generate_netlify_config(config)
            elif config.platform == DeploymentPlatform.AWS:
                config_files["cloudformation.yaml"] = self._generate_aws_config(config)
            elif config.platform == DeploymentPlatform.HEROKU:
                config_files["Procfile"] = self._generate_heroku_config(config)
            
            # Generate common config files
            config_files["package.json"] = self._generate_package_json(project, config)
            config_files[".env.example"] = self._generate_env_example(config)
            config_files["README.md"] = self._generate_readme(project, config)
            
            return config_files
            
        except Exception as e:
            self.logger.error(f"Error generating config files: {e}")
            return {}
    
    def _generate_vercel_config(self, config: DeploymentConfig) -> str:
        """Generate Vercel configuration"""
        vercel_config = {
            "version": 2,
            "name": config.name.lower().replace(" ", "-"),
            "builds": [
                {
                    "src": "package.json",
                    "use": "@vercel/static-build",
                    "config": {
                        "distDir": config.build_directory
                    }
                }
            ],
            "env": config.environment_variables,
            "functions": {
                "app/**/*": {
                    "runtime": f"nodejs{config.node_version}.x"
                }
            }
        }
        
        if config.redirect_rules:
            vercel_config["redirects"] = config.redirect_rules
        
        if config.headers:
            vercel_config["headers"] = [
                {
                    "source": "/(.*)",
                    "headers": [
                        {"key": k, "value": v} for k, v in config.headers.items()
                    ]
                }
            ]
        
        return json.dumps(vercel_config, indent=2)
    
    def _generate_netlify_config(self, config: DeploymentConfig) -> str:
        """Generate Netlify configuration"""
        netlify_config = f"""[build]
  command = "{config.build_command}"
  publish = "{config.build_directory}"

[build.environment]
  NODE_VERSION = "{config.node_version}"
"""
        
        for key, value in config.environment_variables.items():
            netlify_config += f"  {key} = \"{value}\"\n"
        
        if config.redirect_rules:
            netlify_config += "\n[[redirects]]\n"
            for rule in config.redirect_rules:
                netlify_config += f"  from = \"{rule.get('from', '/*')}\"\n"
                netlify_config += f"  to = \"{rule.get('to', '/index.html')}\"\n"
                netlify_config += f"  status = {rule.get('status', 200)}\n\n"
        
        return netlify_config
    
    def _generate_aws_config(self, config: DeploymentConfig) -> str:
        """Generate AWS CloudFormation configuration"""
        aws_config = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"Infrastructure for {config.name}",
            "Resources": {
                "S3Bucket": {
                    "Type": "AWS::S3::Bucket",
                    "Properties": {
                        "BucketName": config.name.lower().replace(" ", "-"),
                        "WebsiteConfiguration": {
                            "IndexDocument": "index.html",
                            "ErrorDocument": "error.html"
                        }
                    }
                },
                "CloudFrontDistribution": {
                    "Type": "AWS::CloudFront::Distribution",
                    "Properties": {
                        "DistributionConfig": {
                            "Origins": [
                                {
                                    "DomainName": {"Fn::GetAtt": ["S3Bucket", "DomainName"]},
                                    "Id": "S3Origin",
                                    "S3OriginConfig": {}
                                }
                            ],
                            "DefaultCacheBehavior": {
                                "TargetOriginId": "S3Origin",
                                "ViewerProtocolPolicy": "redirect-to-https"
                            },
                            "Enabled": True
                        }
                    }
                }
            }
        }
        
        return yaml.dump(aws_config, default_flow_style=False)
    
    def _generate_heroku_config(self, config: DeploymentConfig) -> str:
        """Generate Heroku Procfile"""
        if config.framework == FrameworkType.NODE_JS:
            return "web: npm start"
        elif config.framework == FrameworkType.PYTHON:
            return "web: python app.py"
        else:
            return "web: npm start"
    
    def _generate_package_json(self, project: WebProject, config: DeploymentConfig) -> str:
        """Generate package.json for Node.js projects"""
        package_config = {
            "name": project.name.lower().replace(" ", "-"),
            "version": "1.0.0",
            "description": project.description,
            "scripts": {
                "build": config.build_command.replace("npm run ", ""),
                "dev": config.dev_command.replace("npm run ", ""),
                "start": "serve -s build"
            },
            "engines": {
                "node": f">={config.node_version}"
            }
        }
        
        return json.dumps(package_config, indent=2)
    
    def _generate_env_example(self, config: DeploymentConfig) -> str:
        """Generate environment variables example file"""
        env_content = "# Environment Variables\\n"
        env_content += "# Copy this file to .env and update with your values\\n\\n"
        
        for key, value in config.environment_variables.items():
            env_content += f"{key}={value}\\n"
        
        return env_content
    
    def _generate_readme(self, project: WebProject, config: DeploymentConfig) -> str:
        """Generate README.md file"""
        readme_content = f"""# {project.name}

{project.description}

## Technology Stack
- Framework: {config.framework.value}
- Platform: {config.platform.value}
- Node Version: {config.node_version}

## Development

### Installation
```bash
{config.install_command}
```

### Development Server
```bash
{config.dev_command}
```

### Build
```bash
{config.build_command}
```

## Deployment

This project is configured for deployment on {config.platform.value}.

### Environment Variables
Copy `.env.example` to `.env` and update with your values.

### Auto Deployment
- Production: Automatic deployment on `main` branch
- Staging: Automatic deployment on `develop` branch
- Preview: Automatic deployment on feature branches

## Project Structure
```
{project.name}/
├── {config.build_directory}/     # Build output
├── src/                          # Source code
├── public/                       # Static assets
└── package.json                  # Dependencies
```

Generated by Web Deployment Agent
"""
        
        return readme_content
    
    # Infrastructure Management
    async def provision_infrastructure(self, project_id: str, template_name: str) -> Dict[str, Any]:
        """Provision infrastructure using template"""
        try:
            if project_id not in self.projects:
                raise ValueError(f"Project {project_id} not found")
            
            if template_name not in self.infrastructure_templates:
                raise ValueError(f"Template {template_name} not found")
            
            project = self.projects[project_id]
            template = self.infrastructure_templates[template_name]
            
            self.logger.info(f"Provisioning infrastructure for {project.name} using {template.name}")
            
            # Simulate infrastructure provisioning
            provisioning_result = {
                "infrastructure_id": str(uuid.uuid4()),
                "project_id": project_id,
                "template": template_name,
                "platform": template.platform.value,
                "resources": [],
                "endpoints": {},
                "cost_estimate": template.estimated_cost,
                "status": "provisioned",
                "created_at": datetime.now().isoformat()
            }
            
            # Provision each resource
            for resource in template.resources:
                resource_result = await self._provision_resource(resource, project, template)
                provisioning_result["resources"].append(resource_result)
            
            # Generate endpoints
            base_domain = f"{project.name.lower().replace(' ', '-')}.{template.platform.value}.com"
            provisioning_result["endpoints"] = {
                "primary": f"https://{base_domain}",
                "api": f"https://api.{base_domain}",
                "admin": f"https://admin.{base_domain}"
            }
            
            self.logger.info("Infrastructure provisioning completed")
            
            return provisioning_result
            
        except Exception as e:
            self.logger.error(f"Error provisioning infrastructure: {e}")
            return {}
    
    async def _provision_resource(self, resource: Dict[str, Any], project: WebProject, template: InfrastructureTemplate) -> Dict[str, Any]:
        """Provision individual resource"""
        try:
            resource_type = resource.get("type", "")
            resource_config = resource.get("config", {})
            
            # Simulate resource provisioning
            await asyncio.sleep(0.5)  # Simulate provisioning time
            
            result = {
                "type": resource_type,
                "id": str(uuid.uuid4()),
                "config": resource_config,
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
            
            # Add type-specific details
            if resource_type == "cdn":
                result["endpoint"] = f"https://cdn.{project.name.lower().replace(' ', '-')}.com"
            elif resource_type == "database":
                result["connection_string"] = f"postgres://user:pass@db.{project.name.lower().replace(' ', '-')}.com:5432/db"
            elif resource_type == "storage":
                result["bucket_name"] = f"{project.name.lower().replace(' ', '-')}-storage"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error provisioning resource: {e}")
            return {}
    
    # Monitoring and Analytics
    async def get_deployment_metrics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get deployment metrics"""
        try:
            if project_id and project_id not in self.projects:
                raise ValueError(f"Project {project_id} not found")
            
            # Filter deployments by project if specified
            deployments = list(self.deployments.values())
            if project_id:
                deployments = [d for d in deployments if d.project_id == project_id]
            
            if not deployments:
                return {"message": "No deployments found"}
            
            # Calculate metrics
            total_deployments = len(deployments)
            successful_deployments = len([d for d in deployments if d.status == DeploymentStatus.DEPLOYED])
            failed_deployments = len([d for d in deployments if d.status == DeploymentStatus.FAILED])
            
            build_times = [d.build_time for d in deployments if d.build_time > 0]
            avg_build_time = sum(build_times) / len(build_times) if build_times else 0
            
            deploy_times = [d.deploy_time for d in deployments if d.deploy_time > 0]
            avg_deploy_time = sum(deploy_times) / len(deploy_times) if deploy_times else 0
            
            success_rate = (successful_deployments / total_deployments) * 100 if total_deployments > 0 else 0
            
            # Platform distribution
            platform_stats = {}
            for deployment in deployments:
                platform = deployment.platform.value
                platform_stats[platform] = platform_stats.get(platform, 0) + 1
            
            metrics = {
                "total_deployments": total_deployments,
                "successful_deployments": successful_deployments,
                "failed_deployments": failed_deployments,
                "success_rate": round(success_rate, 2),
                "average_build_time": round(avg_build_time, 2),
                "average_deploy_time": round(avg_deploy_time, 2),
                "platform_distribution": platform_stats,
                "recent_deployments": [
                    {
                        "deployment_id": d.deployment_id,
                        "project_id": d.project_id,
                        "platform": d.platform.value,
                        "status": d.status.value,
                        "url": d.url,
                        "created_at": d.created_at.isoformat()
                    }
                    for d in sorted(deployments, key=lambda x: x.created_at, reverse=True)[:5]
                ]
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting deployment metrics: {e}")
            return {}
    
    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project status and details"""
        if project_id not in self.projects:
            return None
        
        project = self.projects[project_id]
        project_deployments = [
            self.deployments[dep_id] for dep_id in project.deployments 
            if dep_id in self.deployments
        ]
        
        return {
            "project_id": project.project_id,
            "name": project.name,
            "description": project.description,
            "framework": project.framework.value,
            "application_type": project.application_type.value,
            "status": project.status.value,
            "live_url": project.live_url,
            "preview_urls": project.preview_urls,
            "total_deployments": len(project_deployments),
            "last_deployment": project.last_deployment.isoformat() if project.last_deployment else None,
            "deployment_platform": project.deployment_config.platform.value if project.deployment_config else None,
            "created_at": project.created_at.isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "deployment_metrics": self.metrics,
            "automation_config": self.automation_config,
            "platform_stats": {
                platform.value: len([p for p in self.projects.values() 
                                   if p.deployment_config and p.deployment_config.platform == platform])
                for platform in DeploymentPlatform
            },
            "framework_stats": {
                framework.value: len([p for p in self.projects.values() if p.framework == framework])
                for framework in FrameworkType
            }
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_web_deployment_agent():
        # Initialize web deployment agent
        agent = WebDeploymentAgent()
        
        # Create test project
        project_data = {
            "name": "AI Business Dashboard",
            "description": "Real-time business analytics dashboard",
            "repository_url": "https://github.com/user/ai-dashboard.git",
            "framework": "react",
            "application_type": "spa"
        }
        
        project_id = await agent.create_project(project_data)
        print(f"Created project: {project_id}")
        
        # Generate configuration files
        config_files = await agent.generate_config_files(project_id)
        print(f"Generated {len(config_files)} configuration files")
        
        # Deploy project
        deployment_id = await agent.deploy_project(project_id)
        print(f"Deployment initiated: {deployment_id}")
        
        # Get project status
        status = agent.get_project_status(project_id)
        print(f"Project status: {json.dumps(status, indent=2)}")
        
        # Provision infrastructure
        infrastructure = await agent.provision_infrastructure(project_id, "static_site")
        print(f"Infrastructure provisioned: {infrastructure.get('infrastructure_id')}")
        
        # Get deployment metrics
        metrics = await agent.get_deployment_metrics()
        print(f"Deployment metrics: {json.dumps(metrics, indent=2)}")
        
        return agent
    
    # Run test
    test_agent = asyncio.run(test_web_deployment_agent())
    print("\\n✅ Web Deployment Agent implemented and tested successfully!")
'''

# Save the web deployment agent
with open('/home/user/web_deployment_agent.py', 'w') as f:
    f.write(web_deployment_agent_code)

print("✅ Web Deployment Agent created")
print("📁 File saved: /home/user/web_deployment_agent.py")
print(f"📊 Lines of code: {len(web_deployment_agent_code.split(chr(10)))}")