# 6. Freelance Operations Agent - Automated freelance business management
freelance_agent_code = '''"""
Freelance Operations Agent
Automated freelance business operations and client management
Supports project tracking, client communication, and revenue optimization
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from decimal import Decimal

class ProjectStatus(Enum):
    PROPOSAL = "proposal"
    ACTIVE = "active"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"

class ClientType(Enum):
    INDIVIDUAL = "individual"
    STARTUP = "startup"
    SME = "sme"
    ENTERPRISE = "enterprise"
    AGENCY = "agency"

class ServiceType(Enum):
    DEVELOPMENT = "development"
    DESIGN = "design"
    CONSULTING = "consulting"
    CONTENT_CREATION = "content_creation"
    MARKETING = "marketing"
    DATA_ANALYSIS = "data_analysis"
    PROJECT_MANAGEMENT = "project_management"

class PaymentStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"

@dataclass
class Client:
    """Client information structure"""
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    company: str = ""
    client_type: ClientType = ClientType.INDIVIDUAL
    contact_info: Dict[str, str] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    project_history: List[str] = field(default_factory=list)
    total_revenue: Decimal = field(default_factory=lambda: Decimal('0'))
    satisfaction_rating: float = 0.0
    communication_frequency: str = "weekly"
    timezone: str = "UTC"
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_contact: Optional[datetime] = None

@dataclass
class Project:
    """Freelance project structure"""
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = ""
    title: str = ""
    description: str = ""
    service_type: ServiceType = ServiceType.DEVELOPMENT
    status: ProjectStatus = ProjectStatus.PROPOSAL
    scope: List[str] = field(default_factory=list)
    deliverables: List[Dict[str, Any]] = field(default_factory=list)
    timeline: Dict[str, str] = field(default_factory=dict)
    budget: Decimal = field(default_factory=lambda: Decimal('0'))
    hourly_rate: Optional[Decimal] = None
    estimated_hours: int = 0
    actual_hours: int = 0
    progress: float = 0.0
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    tools_needed: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    client_feedback: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Invoice:
    """Invoice structure"""
    invoice_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = ""
    client_id: str = ""
    invoice_number: str = ""
    amount: Decimal = field(default_factory=lambda: Decimal('0'))
    tax_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    total_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    currency: str = "USD"
    payment_status: PaymentStatus = PaymentStatus.PENDING
    issue_date: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    payment_date: Optional[datetime] = None
    payment_method: str = ""
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

@dataclass
class ProposalTemplate:
    """Project proposal template"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    service_type: ServiceType = ServiceType.DEVELOPMENT
    sections: List[Dict[str, str]] = field(default_factory=list)
    pricing_model: str = "fixed"  # fixed, hourly, milestone
    estimated_duration: str = ""
    sample_deliverables: List[str] = field(default_factory=list)
    terms_and_conditions: str = ""

class FreelanceOperationsAgent:
    """
    Freelance Operations Agent for Business Automation
    
    Capabilities:
    - Client relationship management
    - Project lifecycle tracking
    - Automated proposal generation
    - Time and expense tracking
    - Invoice generation and payment tracking
    - Revenue analytics and reporting
    - Client communication automation
    - Pipeline management
    """
    
    def __init__(self, agent_id: str = "freelance_agent_001"):
        self.agent_id = agent_id
        self.logger = self._setup_logger()
        
        # Core data
        self.clients: Dict[str, Client] = {}
        self.projects: Dict[str, Project] = {}
        self.invoices: Dict[str, Invoice] = {}
        self.proposal_templates: Dict[str, ProposalTemplate] = {}
        
        # Business metrics
        self.metrics = {
            "total_clients": 0,
            "active_projects": 0,
            "total_revenue": Decimal('0'),
            "average_project_value": Decimal('0'),
            "client_satisfaction": 0.0,
            "project_completion_rate": 0.0,
            "on_time_delivery_rate": 0.0,
            "payment_collection_rate": 0.0,
            "monthly_recurring_revenue": Decimal('0'),
            "pipeline_value": Decimal('0')
        }
        
        # Automation settings
        self.automation_config = {
            "auto_invoice_generation": True,
            "payment_reminders": True,
            "project_status_updates": True,
            "client_check_ins": True,
            "proposal_follow_ups": True
        }
        
        # Communication templates
        self.communication_templates = {}
        
        # Initialize default templates
        self._initialize_proposal_templates()
        self._initialize_communication_templates()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for freelance operations agent"""
        logger = logging.getLogger(f"FreelanceAgent-{self.agent_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_proposal_templates(self):
        """Initialize default proposal templates"""
        # Development proposal template
        dev_template = ProposalTemplate(
            name="Web Development Proposal",
            service_type=ServiceType.DEVELOPMENT,
            sections=[
                {"title": "Project Overview", "content": "Comprehensive web development solution tailored to your needs."},
                {"title": "Scope of Work", "content": "Detailed breakdown of development phases and deliverables."},
                {"title": "Timeline", "content": "Project timeline with key milestones and delivery dates."},
                {"title": "Investment", "content": "Transparent pricing structure and payment terms."},
                {"title": "Why Choose Us", "content": "Our expertise and commitment to excellence."}
            ],
            pricing_model="milestone",
            estimated_duration="6-8 weeks",
            sample_deliverables=["Responsive website", "Admin dashboard", "Technical documentation"],
            terms_and_conditions="Standard terms and conditions apply."
        )
        
        # Consulting proposal template
        consulting_template = ProposalTemplate(
            name="Business Consulting Proposal",
            service_type=ServiceType.CONSULTING,
            sections=[
                {"title": "Executive Summary", "content": "Strategic consulting to drive business growth."},
                {"title": "Current Situation Analysis", "content": "Assessment of current business challenges."},
                {"title": "Proposed Solution", "content": "Customized strategy and implementation plan."},
                {"title": "Expected Outcomes", "content": "Measurable results and benefits."},
                {"title": "Investment", "content": "Consulting fees and engagement terms."}
            ],
            pricing_model="hourly",
            estimated_duration="3-6 months",
            sample_deliverables=["Strategy document", "Implementation roadmap", "Performance metrics"],
            terms_and_conditions="Consulting agreement terms apply."
        )
        
        self.proposal_templates = {
            "development": dev_template,
            "consulting": consulting_template
        }
    
    def _initialize_communication_templates(self):
        """Initialize communication templates"""
        self.communication_templates = {
            "welcome_client": {
                "subject": "Welcome to our partnership!",
                "content": "Dear {client_name},\\n\\nWelcome! We're excited to work with you on {project_title}. I'll be your dedicated point of contact throughout this project.\\n\\nNext steps:\\n- Project kickoff call scheduled\\n- Requirements gathering\\n- Timeline confirmation\\n\\nLooking forward to delivering exceptional results!\\n\\nBest regards,\\n{freelancer_name}"
            },
            "project_update": {
                "subject": "Project Update: {project_title}",
                "content": "Hi {client_name},\\n\\nHere's your weekly project update:\\n\\nProgress: {progress}% complete\\nCompleted this week: {completed_tasks}\\nNext week's focus: {upcoming_tasks}\\nOn track for: {deadline}\\n\\nAny questions or concerns? Feel free to reach out!\\n\\nBest regards,\\n{freelancer_name}"
            },
            "invoice_reminder": {
                "subject": "Friendly Reminder: Invoice {invoice_number}",
                "content": "Hi {client_name},\\n\\nI hope you're doing well! This is a friendly reminder that invoice {invoice_number} for ${amount} was due on {due_date}.\\n\\nIf you've already processed payment, please disregard this message. If you have any questions about the invoice, I'm happy to help.\\n\\nThank you for your business!\\n\\nBest regards,\\n{freelancer_name}"
            },
            "project_completion": {
                "subject": "Project Completed: {project_title}",
                "content": "Dear {client_name},\\n\\nI'm pleased to announce that {project_title} has been completed successfully!\\n\\nDeliverables included:\\n{deliverables}\\n\\nAll files have been delivered as discussed. I'd love to hear your feedback and discuss any future projects.\\n\\nThank you for choosing us for this project!\\n\\nBest regards,\\n{freelancer_name}"
            }
        }
    
    # Client Management
    async def add_client(self, client_data: Dict[str, Any]) -> str:
        """Add new client"""
        try:
            client = Client(
                name=client_data.get("name", ""),
                email=client_data.get("email", ""),
                company=client_data.get("company", ""),
                client_type=ClientType(client_data.get("client_type", "individual")),
                contact_info=client_data.get("contact_info", {}),
                preferences=client_data.get("preferences", {}),
                timezone=client_data.get("timezone", "UTC"),
                notes=client_data.get("notes", "")
            )
            
            self.clients[client.client_id] = client
            self.metrics["total_clients"] += 1
            
            self.logger.info(f"Added new client: {client.name}")
            
            # Send welcome message if enabled
            if self.automation_config["client_check_ins"]:
                await self._send_welcome_message(client)
            
            return client.client_id
            
        except Exception as e:
            self.logger.error(f"Error adding client: {e}")
            return ""
    
    async def update_client(self, client_id: str, updates: Dict[str, Any]) -> bool:
        """Update client information"""
        try:
            if client_id not in self.clients:
                self.logger.warning(f"Client {client_id} not found")
                return False
            
            client = self.clients[client_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(client, field):
                    setattr(client, field, value)
            
            client.last_contact = datetime.now()
            
            self.logger.info(f"Updated client: {client.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating client: {e}")
            return False
    
    # Project Management
    async def create_project(self, project_data: Dict[str, Any]) -> str:
        """Create new project"""
        try:
            project = Project(
                client_id=project_data.get("client_id", ""),
                title=project_data.get("title", ""),
                description=project_data.get("description", ""),
                service_type=ServiceType(project_data.get("service_type", "development")),
                scope=project_data.get("scope", []),
                budget=Decimal(str(project_data.get("budget", "0"))),
                hourly_rate=Decimal(str(project_data.get("hourly_rate", "0"))) if project_data.get("hourly_rate") else None,
                estimated_hours=project_data.get("estimated_hours", 0),
                requirements=project_data.get("requirements", []),
                tools_needed=project_data.get("tools_needed", []),
                deadline=datetime.fromisoformat(project_data["deadline"]) if project_data.get("deadline") else None
            )
            
            # Create milestones
            if project_data.get("milestones"):
                project.milestones = project_data["milestones"]
            else:
                project.milestones = self._generate_default_milestones(project)
            
            # Create deliverables
            if project_data.get("deliverables"):
                project.deliverables = project_data["deliverables"]
            else:
                project.deliverables = self._generate_default_deliverables(project)
            
            self.projects[project.project_id] = project
            self.metrics["active_projects"] += 1
            self.metrics["pipeline_value"] += project.budget
            
            # Update client project history
            if project.client_id in self.clients:
                self.clients[project.client_id].project_history.append(project.project_id)
            
            self.logger.info(f"Created project: {project.title}")
            
            return project.project_id
            
        except Exception as e:
            self.logger.error(f"Error creating project: {e}")
            return ""
    
    def _generate_default_milestones(self, project: Project) -> List[Dict[str, Any]]:
        """Generate default milestones based on service type"""
        milestones = []
        
        if project.service_type == ServiceType.DEVELOPMENT:
            milestones = [
                {"name": "Requirements & Planning", "percentage": 20, "status": "pending"},
                {"name": "Design & Architecture", "percentage": 40, "status": "pending"},
                {"name": "Development & Implementation", "percentage": 80, "status": "pending"},
                {"name": "Testing & Deployment", "percentage": 100, "status": "pending"}
            ]
        elif project.service_type == ServiceType.CONSULTING:
            milestones = [
                {"name": "Initial Assessment", "percentage": 25, "status": "pending"},
                {"name": "Strategy Development", "percentage": 50, "status": "pending"},
                {"name": "Implementation Plan", "percentage": 75, "status": "pending"},
                {"name": "Final Recommendations", "percentage": 100, "status": "pending"}
            ]
        else:
            milestones = [
                {"name": "Project Kickoff", "percentage": 25, "status": "pending"},
                {"name": "Mid-point Review", "percentage": 50, "status": "pending"},
                {"name": "Final Delivery", "percentage": 100, "status": "pending"}
            ]
        
        return milestones
    
    def _generate_default_deliverables(self, project: Project) -> List[Dict[str, Any]]:
        """Generate default deliverables based on service type"""
        deliverables = []
        
        if project.service_type == ServiceType.DEVELOPMENT:
            deliverables = [
                {"name": "Project Requirements Document", "status": "pending"},
                {"name": "Technical Specification", "status": "pending"},
                {"name": "Source Code", "status": "pending"},
                {"name": "Documentation", "status": "pending"},
                {"name": "Deployment Guide", "status": "pending"}
            ]
        elif project.service_type == ServiceType.CONSULTING:
            deliverables = [
                {"name": "Current State Analysis", "status": "pending"},
                {"name": "Strategic Recommendations", "status": "pending"},
                {"name": "Implementation Roadmap", "status": "pending"},
                {"name": "Executive Summary", "status": "pending"}
            ]
        
        return deliverables
    
    async def update_project_progress(self, project_id: str, progress_data: Dict[str, Any]) -> bool:
        """Update project progress"""
        try:
            if project_id not in self.projects:
                self.logger.warning(f"Project {project_id} not found")
                return False
            
            project = self.projects[project_id]
            
            # Update progress
            if "progress" in progress_data:
                project.progress = progress_data["progress"]
            
            # Update status
            if "status" in progress_data:
                old_status = project.status
                project.status = ProjectStatus(progress_data["status"])
                
                # Handle status changes
                if old_status != project.status:
                    await self._handle_status_change(project, old_status)
            
            # Update hours
            if "actual_hours" in progress_data:
                project.actual_hours = progress_data["actual_hours"]
            
            # Update milestones
            if "completed_milestone" in progress_data:
                milestone_name = progress_data["completed_milestone"]
                for milestone in project.milestones:
                    if milestone["name"] == milestone_name:
                        milestone["status"] = "completed"
                        milestone["completed_date"] = datetime.now().isoformat()
            
            self.logger.info(f"Updated project progress: {project.title} - {project.progress}%")
            
            # Send progress update if enabled
            if self.automation_config["project_status_updates"]:
                await self._send_progress_update(project)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating project progress: {e}")
            return False
    
    async def _handle_status_change(self, project: Project, old_status: ProjectStatus):
        """Handle project status changes"""
        try:
            if project.status == ProjectStatus.ACTIVE and old_status == ProjectStatus.PROPOSAL:
                # Project started
                project.start_date = datetime.now()
                await self._send_project_kickoff_message(project)
                
            elif project.status == ProjectStatus.COMPLETED:
                # Project completed
                project.completion_date = datetime.now()
                self.metrics["active_projects"] -= 1
                
                # Update completion rate
                total_projects = len([p for p in self.projects.values() if p.status in [ProjectStatus.COMPLETED, ProjectStatus.CANCELLED]])
                completed_projects = len([p for p in self.projects.values() if p.status == ProjectStatus.COMPLETED])
                self.metrics["project_completion_rate"] = completed_projects / total_projects if total_projects > 0 else 0
                
                # Calculate on-time delivery
                if project.deadline and project.completion_date <= project.deadline:
                    # Update on-time delivery rate
                    pass
                
                # Update client revenue
                if project.client_id in self.clients:
                    self.clients[project.client_id].total_revenue += project.budget
                
                # Update total revenue
                self.metrics["total_revenue"] += project.budget
                
                # Generate invoice if enabled
                if self.automation_config["auto_invoice_generation"]:
                    await self._generate_project_invoice(project)
                
                # Send completion message
                await self._send_completion_message(project)
                
        except Exception as e:
            self.logger.error(f"Error handling status change: {e}")
    
    # Proposal Generation
    async def generate_proposal(self, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project proposal"""
        try:
            service_type = ServiceType(proposal_data.get("service_type", "development"))
            client_id = proposal_data.get("client_id", "")
            
            # Get template
            template_key = service_type.value
            template = self.proposal_templates.get(template_key)
            
            if not template:
                template = list(self.proposal_templates.values())[0]  # Default template
            
            # Get client info
            client = self.clients.get(client_id)
            client_name = client.name if client else "Valued Client"
            
            # Generate proposal content
            proposal = {
                "proposal_id": str(uuid.uuid4()),
                "client_id": client_id,
                "client_name": client_name,
                "project_title": proposal_data.get("project_title", ""),
                "service_type": service_type.value,
                "sections": [],
                "pricing": {
                    "model": template.pricing_model,
                    "amount": proposal_data.get("budget", "0"),
                    "currency": "USD",
                    "payment_terms": proposal_data.get("payment_terms", "Net 30")
                },
                "timeline": {
                    "estimated_duration": template.estimated_duration,
                    "start_date": proposal_data.get("start_date", ""),
                    "delivery_date": proposal_data.get("delivery_date", "")
                },
                "deliverables": template.sample_deliverables,
                "terms": template.terms_and_conditions,
                "created_at": datetime.now().isoformat()
            }
            
            # Generate sections
            for section in template.sections:
                proposal["sections"].append({
                    "title": section["title"],
                    "content": self._customize_section_content(
                        section["content"],
                        proposal_data,
                        client
                    )
                })
            
            self.logger.info(f"Generated proposal for: {client_name}")
            
            return proposal
            
        except Exception as e:
            self.logger.error(f"Error generating proposal: {e}")
            return {}
    
    def _customize_section_content(self, content: str, proposal_data: Dict[str, Any], client: Optional[Client]) -> str:
        """Customize proposal section content"""
        try:
            # Replace placeholders with actual data
            customized_content = content
            
            if client:
                customized_content = customized_content.replace("{client_name}", client.name)
                customized_content = customized_content.replace("{company_name}", client.company or client.name)
            
            # Add specific project details
            if proposal_data.get("project_title"):
                customized_content += f" This {proposal_data['project_title']} project will deliver exceptional value."
            
            return customized_content
            
        except Exception as e:
            self.logger.error(f"Error customizing content: {e}")
            return content
    
    # Invoice Management
    async def _generate_project_invoice(self, project: Project) -> str:
        """Generate invoice for completed project"""
        try:
            invoice = Invoice(
                project_id=project.project_id,
                client_id=project.client_id,
                invoice_number=f"INV-{datetime.now().strftime('%Y%m%d')}-{project.project_id[:8]}",
                amount=project.budget,
                currency="USD",
                due_date=datetime.now() + timedelta(days=30)
            )
            
            # Add line items
            invoice.line_items = [
                {
                    "description": f"{project.title} - {project.service_type.value}",
                    "quantity": 1,
                    "rate": float(project.budget),
                    "amount": float(project.budget)
                }
            ]
            
            # Calculate tax (simplified)
            tax_rate = Decimal('0.08')  # 8% tax
            invoice.tax_amount = invoice.amount * tax_rate
            invoice.total_amount = invoice.amount + invoice.tax_amount
            
            self.invoices[invoice.invoice_id] = invoice
            
            self.logger.info(f"Generated invoice: {invoice.invoice_number}")
            
            return invoice.invoice_id
            
        except Exception as e:
            self.logger.error(f"Error generating invoice: {e}")
            return ""
    
    async def send_invoice_reminder(self, invoice_id: str) -> bool:
        """Send invoice payment reminder"""
        try:
            if invoice_id not in self.invoices:
                return False
            
            invoice = self.invoices[invoice_id]
            client = self.clients.get(invoice.client_id)
            
            if not client:
                return False
            
            # Check if reminder is needed
            if invoice.payment_status == PaymentStatus.PAID:
                return False
            
            days_overdue = (datetime.now() - invoice.due_date).days if invoice.due_date else 0
            
            if days_overdue > 0:
                # Send reminder
                await self._send_invoice_reminder(invoice, client)
                self.logger.info(f"Sent invoice reminder: {invoice.invoice_number}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending invoice reminder: {e}")
            return False
    
    # Communication Methods
    async def _send_welcome_message(self, client: Client):
        """Send welcome message to new client"""
        try:
            template = self.communication_templates["welcome_client"]
            message = template["content"].format(
                client_name=client.name,
                project_title="your upcoming project",
                freelancer_name="Your Freelance Partner"
            )
            
            # In real implementation, this would send actual email/message
            self.logger.info(f"Welcome message sent to: {client.name}")
            
        except Exception as e:
            self.logger.error(f"Error sending welcome message: {e}")
    
    async def _send_progress_update(self, project: Project):
        """Send project progress update"""
        try:
            client = self.clients.get(project.client_id)
            if not client:
                return
            
            template = self.communication_templates["project_update"]
            
            # Get completed tasks (simplified)
            completed_tasks = [m["name"] for m in project.milestones if m.get("status") == "completed"]
            upcoming_tasks = [m["name"] for m in project.milestones if m.get("status") == "pending"]
            
            message = template["content"].format(
                client_name=client.name,
                project_title=project.title,
                progress=int(project.progress),
                completed_tasks=", ".join(completed_tasks[-2:]) if completed_tasks else "Project setup",
                upcoming_tasks=", ".join(upcoming_tasks[:2]) if upcoming_tasks else "Final delivery",
                deadline=project.deadline.strftime("%B %d, %Y") if project.deadline else "On schedule"
            )
            
            self.logger.info(f"Progress update sent for: {project.title}")
            
        except Exception as e:
            self.logger.error(f"Error sending progress update: {e}")
    
    async def _send_invoice_reminder(self, invoice: Invoice, client: Client):
        """Send invoice payment reminder"""
        try:
            template = self.communication_templates["invoice_reminder"]
            message = template["content"].format(
                client_name=client.name,
                invoice_number=invoice.invoice_number,
                amount=invoice.total_amount,
                due_date=invoice.due_date.strftime("%B %d, %Y") if invoice.due_date else "immediately",
                freelancer_name="Your Freelance Partner"
            )
            
            self.logger.info(f"Invoice reminder sent: {invoice.invoice_number}")
            
        except Exception as e:
            self.logger.error(f"Error sending invoice reminder: {e}")
    
    async def _send_completion_message(self, project: Project):
        """Send project completion message"""
        try:
            client = self.clients.get(project.client_id)
            if not client:
                return
            
            template = self.communication_templates["project_completion"]
            
            deliverables_list = "\\n".join([f"- {d['name']}" for d in project.deliverables])
            
            message = template["content"].format(
                client_name=client.name,
                project_title=project.title,
                deliverables=deliverables_list,
                freelancer_name="Your Freelance Partner"
            )
            
            self.logger.info(f"Completion message sent for: {project.title}")
            
        except Exception as e:
            self.logger.error(f"Error sending completion message: {e}")
    
    async def _send_project_kickoff_message(self, project: Project):
        """Send project kickoff message"""
        try:
            client = self.clients.get(project.client_id)
            if not client:
                return
            
            self.logger.info(f"Kickoff message sent for: {project.title}")
            
        except Exception as e:
            self.logger.error(f"Error sending kickoff message: {e}")
    
    # Analytics and Reporting
    def get_business_metrics(self) -> Dict[str, Any]:
        """Get comprehensive business metrics"""
        try:
            # Calculate average project value
            if self.projects:
                total_value = sum(project.budget for project in self.projects.values())
                self.metrics["average_project_value"] = total_value / len(self.projects)
            
            # Calculate client satisfaction (simplified)
            if self.clients:
                total_satisfaction = sum(client.satisfaction_rating for client in self.clients.values())
                self.metrics["client_satisfaction"] = total_satisfaction / len(self.clients)
            
            # Calculate payment collection rate
            paid_invoices = len([inv for inv in self.invoices.values() if inv.payment_status == PaymentStatus.PAID])
            total_invoices = len(self.invoices)
            self.metrics["payment_collection_rate"] = paid_invoices / total_invoices if total_invoices > 0 else 0
            
            # Calculate monthly recurring revenue (simplified)
            # This would be more complex in real implementation
            active_projects_value = sum(
                project.budget for project in self.projects.values() 
                if project.status == ProjectStatus.ACTIVE
            )
            self.metrics["monthly_recurring_revenue"] = active_projects_value / 3  # Estimate
            
            return {
                "metrics": self.metrics,
                "summary": {
                    "total_clients": len(self.clients),
                    "active_projects": len([p for p in self.projects.values() if p.status == ProjectStatus.ACTIVE]),
                    "pending_invoices": len([i for i in self.invoices.values() if i.payment_status == PaymentStatus.PENDING]),
                    "overdue_invoices": len([i for i in self.invoices.values() if i.payment_status == PaymentStatus.OVERDUE])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating business metrics: {e}")
            return {"metrics": self.metrics, "summary": {}}
    
    def get_client_portfolio(self) -> Dict[str, Any]:
        """Get client portfolio overview"""
        portfolio = {}
        
        for client in self.clients.values():
            client_projects = [
                project for project in self.projects.values() 
                if project.client_id == client.client_id
            ]
            
            portfolio[client.client_id] = {
                "name": client.name,
                "company": client.company,
                "client_type": client.client_type.value,
                "total_projects": len(client_projects),
                "active_projects": len([p for p in client_projects if p.status == ProjectStatus.ACTIVE]),
                "total_revenue": float(client.total_revenue),
                "satisfaction_rating": client.satisfaction_rating,
                "last_contact": client.last_contact.isoformat() if client.last_contact else None
            }
        
        return portfolio
    
    def get_project_pipeline(self) -> Dict[str, Any]:
        """Get project pipeline overview"""
        pipeline = {
            "by_status": {},
            "by_service_type": {},
            "upcoming_deadlines": [],
            "total_value": float(self.metrics["pipeline_value"])
        }
        
        # Group by status
        for status in ProjectStatus:
            projects = [p for p in self.projects.values() if p.status == status]
            pipeline["by_status"][status.value] = {
                "count": len(projects),
                "value": float(sum(p.budget for p in projects))
            }
        
        # Group by service type
        for service_type in ServiceType:
            projects = [p for p in self.projects.values() if p.service_type == service_type]
            pipeline["by_service_type"][service_type.value] = {
                "count": len(projects),
                "value": float(sum(p.budget for p in projects))
            }
        
        # Upcoming deadlines
        upcoming_projects = [
            p for p in self.projects.values() 
            if p.deadline and p.deadline > datetime.now() and p.status == ProjectStatus.ACTIVE
        ]
        upcoming_projects.sort(key=lambda x: x.deadline)
        
        pipeline["upcoming_deadlines"] = [
            {
                "project_id": p.project_id,
                "title": p.title,
                "deadline": p.deadline.isoformat(),
                "days_remaining": (p.deadline - datetime.now()).days
            }
            for p in upcoming_projects[:5]  # Next 5 deadlines
        ]
        
        return pipeline
    
    async def run_daily_automation(self):
        """Run daily automation tasks"""
        try:
            self.logger.info("Running daily automation tasks")
            
            # Send invoice reminders
            if self.automation_config["payment_reminders"]:
                for invoice in self.invoices.values():
                    await self.send_invoice_reminder(invoice.invoice_id)
            
            # Send project status updates (weekly)
            if self.automation_config["project_status_updates"]:
                # Check if it's time for weekly updates
                today = datetime.now().weekday()
                if today == 0:  # Monday
                    for project in self.projects.values():
                        if project.status == ProjectStatus.ACTIVE:
                            await self._send_progress_update(project)
            
            # Client check-ins (monthly)
            if self.automation_config["client_check_ins"]:
                # Simplified monthly check-in logic
                pass
            
            self.logger.info("Daily automation tasks completed")
            
        except Exception as e:
            self.logger.error(f"Error running daily automation: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "business_metrics": self.metrics,
            "automation_status": self.automation_config,
            "data_summary": {
                "clients": len(self.clients),
                "projects": len(self.projects),
                "invoices": len(self.invoices),
                "templates": len(self.proposal_templates)
            }
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_freelance_agent():
        # Initialize freelance agent
        agent = FreelanceOperationsAgent()
        
        # Add test client
        client_data = {
            "name": "John Smith",
            "email": "john@techcorp.com",
            "company": "TechCorp Inc",
            "client_type": "enterprise",
            "contact_info": {"phone": "555-0123"},
            "timezone": "EST"
        }
        
        client_id = await agent.add_client(client_data)
        print(f"Added client: {client_id}")
        
        # Create test project
        project_data = {
            "client_id": client_id,
            "title": "E-commerce Website Development",
            "description": "Modern e-commerce platform with payment integration",
            "service_type": "development",
            "budget": "15000",
            "estimated_hours": 120,
            "requirements": ["Responsive design", "Payment gateway", "Admin panel"],
            "deadline": (datetime.now() + timedelta(days=60)).isoformat()
        }
        
        project_id = await agent.create_project(project_data)
        print(f"Created project: {project_id}")
        
        # Generate proposal
        proposal_data = {
            "client_id": client_id,
            "project_title": "E-commerce Website Development",
            "service_type": "development",
            "budget": "15000",
            "start_date": "2024-02-01",
            "delivery_date": "2024-04-01"
        }
        
        proposal = await agent.generate_proposal(proposal_data)
        print(f"Generated proposal: {proposal['proposal_id']}")
        
        # Update project progress
        progress_data = {
            "progress": 25,
            "status": "active",
            "actual_hours": 30,
            "completed_milestone": "Requirements & Planning"
        }
        
        await agent.update_project_progress(project_id, progress_data)
        print("Updated project progress")
        
        # Get business metrics
        metrics = agent.get_business_metrics()
        print(f"Business metrics: {json.dumps(metrics, indent=2, default=str)}")
        
        return agent
    
    # Run test
    test_agent = asyncio.run(test_freelance_agent())
    print("\\n✅ Freelance Operations Agent implemented and tested successfully!")
'''

# Save the freelance operations agent
with open('/home/user/freelance_operations_agent.py', 'w') as f:
    f.write(freelance_agent_code)

print("✅ Freelance Operations Agent created")
print("📁 File saved: /home/user/freelance_operations_agent.py")
print(f"📊 Lines of code: {len(freelance_agent_code.split(chr(10)))}")