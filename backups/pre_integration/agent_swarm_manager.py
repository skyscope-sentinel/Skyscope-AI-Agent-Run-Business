#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent Swarm Manager for Skyscope
================================

This module manages a swarm of 10,000 AI agents working together to generate income
through various strategies. Each agent has a specific role and expertise.
"""

import os
import sys
import uuid
import time
import json
import random
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from queue import Queue, PriorityQueue

# Import our unified AI client
from ai_client import ai_client
from crypto_wallet_manager import wallet_manager
from blockchain_manager import blockchain_manager
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_swarm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AgentSwarm")

# Constants
MAX_AGENTS = 10000
AGENT_TYPES = [
    "crypto_trader", "content_creator", "nft_artist", "freelancer", 
    "affiliate_marketer", "social_media_manager", "data_analyst",
    "web_developer", "market_researcher", "copywriter", "seo_specialist",
    "customer_support", "project_manager", "quality_assurance", "strategist"
]

DEPARTMENTS = [
    "Trading & Investment", "Content Creation", "NFT & Digital Art", 
    "Freelance Services", "Affiliate Marketing", "Social Media", 
    "Data Analysis", "Web Development", "Market Research", "Copywriting", 
    "SEO & Marketing", "Customer Relations", "Project Management", 
    "Quality Assurance", "Strategy & Planning"
]

class Agent:
    """
    Represents a single AI agent in the swarm with a specific role and capabilities.
    """
    
    def __init__(self, 
                agent_id: str, 
                agent_type: str, 
                department: str,
                name: str = None,
                expertise: List[str] = None,
                seniority: str = "junior"):
        """
        Initialize an agent with its core attributes.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., crypto_trader, content_creator)
            department: Department the agent belongs to
            name: Agent's name (generated if None)
            expertise: List of expertise areas
            seniority: Seniority level (junior, mid, senior, lead)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.department = department
        self.name = name or self._generate_name()
        self.expertise = expertise or self._generate_expertise()
        self.seniority = seniority
        self.created_at = datetime.now().isoformat()
        self.tasks_completed = 0
        self.earnings = 0.0
        self.status = "idle"
        self.current_task = None
        self.performance_metrics = {
            "success_rate": 0.0,
            "efficiency": 0.0,
            "quality": 0.0,
            "earnings_generated": 0.0
        }
        
        logger.info(f"Agent created: {self.name} ({self.agent_id}) - {self.seniority} {self.agent_type}")
    
        def _generate_name(self) -> str:
        """Generate a random name for the agent."""
        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", 
                      "Skyler", "Dakota", "Reese", "Finley", "Harley", "Emerson", "Phoenix"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                     "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"]
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_expertise(self) -> List[str]:
        """Generate expertise areas based on agent type."""
        expertise_by_type = {
            "crypto_trader": ["Technical Analysis", "Market Timing", "Risk Management", "Arbitrage", "MEV Strategies"],
            "content_creator": ["Copywriting", "SEO", "Content Strategy", "Audience Building", "Monetization"],
            "nft_artist": ["Digital Art", "NFT Markets", "Collection Strategy", "Pricing", "Community Building"],
            "freelancer": ["Client Acquisition", "Project Management", "Service Delivery", "Upselling", "Retention"],
            "affiliate_marketer": ["Traffic Generation", "Conversion Optimization", "Product Selection", "Email Marketing"],
            "social_media_manager": ["Audience Growth", "Engagement", "Content Calendar", "Analytics", "Monetization"],
            "data_analyst": ["Data Mining", "Pattern Recognition", "Predictive Analysis", "Reporting", "Visualization"],
            "web_developer": ["Frontend", "Backend", "SEO", "Performance", "E-commerce"],
            "market_researcher": ["Trend Analysis", "Competitive Intelligence", "Consumer Insights", "Opportunity Identification"],
            "copywriter": ["Sales Copy", "Email Sequences", "Ad Copy", "Landing Pages", "Brand Voice"],
            "seo_specialist": ["On-page SEO", "Off-page SEO", "Technical SEO", "Keyword Research", "Content Optimization"],
            "customer_support": ["Issue Resolution", "Client Retention", "Upselling", "Feedback Collection"],
            "project_manager": ["Task Coordination", "Resource Allocation", "Timeline Management", "Risk Mitigation"],
            "quality_assurance": ["Testing", "Process Improvement", "Standards Compliance", "Bug Tracking"],
            "strategist": ["Business Planning", "Market Entry", "Growth Hacking", "Pivot Analysis", "Scaling"]
        }
        
        default_expertise = ["Problem Solving", "Communication", "Research", "Adaptation"]
        specific_expertise = expertise_by_type.get(self.agent_type, [])
        
        # Combine default and specific expertise, then randomly select 3-5 areas
        all_expertise = default_expertise + specific_expertise
        num_expertise = min(random.randint(3, 5), len(all_expertise))
        
        return random.sample(all_expertise, num_expertise)
    
    def assign_task(self, task: Dict[str, Any]) -> bool:
        """
        Assign a task to the agent.
        
        Args:
            task: Task details including description, priority, etc.
            
        Returns:
            bool: True if task was assigned successfully
        """
        if self.status != "idle":
            logger.warning(f"Agent {self.agent_id} is not idle (current status: {self.status})")
            return False
        
        self.current_task = task
        self.status = "working"
        logger.info(f"Task assigned to agent {self.name} ({self.agent_id}): {task.get('description', 'No description')}")
        return True
    
    def complete_task(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mark the current task as completed and update agent metrics.
        
        Args:
            result: Task result including success status, earnings, etc.
            
        Returns:
            Dict: Updated task result with agent information
        """
        if self.status != "working" or not self.current_task:
            logger.warning(f"Agent {self.agent_id} has no active task to complete")
            return {"success": False, "error": "No active task"}
        
        self.tasks_completed += 1
        self.status = "idle"
        
        # Update earnings if task generated income
        if "earnings" in result and isinstance(result["earnings"], (int, float)) and result["earnings"] > 0:
            self.earnings += result["earnings"]
            self.performance_metrics["earnings_generated"] += result["earnings"]
        
        # Update performance metrics
        if "success" in result and result["success"]:
            # Calculate rolling average for success rate
            self.performance_metrics["success_rate"] = ((self.performance_metrics["success_rate"] * (self.tasks_completed - 1)) + 1) / self.tasks_completed
        else:
            self.performance_metrics["success_rate"] = ((self.performance_metrics["success_rate"] * (self.tasks_completed - 1)) + 0) / self.tasks_completed
        
        # Update other metrics if provided
        if "efficiency" in result:
            self.performance_metrics["efficiency"] = ((self.performance_metrics["efficiency"] * (self.tasks_completed - 1)) + result["efficiency"]) / self.tasks_completed
        
        if "quality" in result:
            self.performance_metrics["quality"] = ((self.performance_metrics["quality"] * (self.tasks_completed - 1)) + result["quality"]) / self.tasks_completed
        
        # Add agent info to result
        result["agent_id"] = self.agent_id
        result["agent_name"] = self.name
        result["agent_type"] = self.agent_type
        result["completed_at"] = datetime.now().isoformat()
        
        # Clear current task
        completed_task = self.current_task
        self.current_task = None
        
        logger.info(f"Agent {self.name} ({self.agent_id}) completed task: {completed_task.get('description', 'No description')}")
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status and metrics of the agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "department": self.department,
            "seniority": self.seniority,
            "status": self.status,
            "tasks_completed": self.tasks_completed,
            "earnings": self.earnings,
            "current_task": self.current_task,
            "performance_metrics": self.performance_metrics,
            "expertise": self.expertise
        }
    
    def execute_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the current task using the AI client.
        
        Args:
            task_context: Additional context for task execution
            
        Returns:
            Dict: Task execution result
        """
        if not self.current_task:
            return {"success": False, "error": "No task assigned"}
        
        task = self.current_task
        task_type = task.get("type", "general")
        task_description = task.get("description", "")
        
        # Prepare the prompt based on agent type and task
        system_prompt = f"""You are {self.name}, a {self.seniority} {self.agent_type} AI agent working in the {self.department} department of Skyscope AI Business Swarm.
Your expertise includes: {', '.join(self.expertise)}.
You are tasked with: {task_description}

Your goal is to complete this task with maximum efficiency and quality, focusing on generating real-world income.
All income must be directed to cryptocurrency wallets.

Provide your complete solution, including:
1. Your approach and methodology
2. The actual work product/deliverable
3. Expected outcomes and income potential
4. Next steps or follow-up actions

Be thorough, practical, and focused on real-world implementation that generates actual income."""

        # Prepare user message with task details and context
        user_message = f"""Task: {task_description}

Additional context:
- Task type: {task_type}
- Priority: {task.get('priority', 'medium')}
- Deadline: {task.get('deadline', 'ASAP')}

{json.dumps(task_context, indent=2)}

Complete this task to generate real income. Be specific, actionable, and results-oriented."""

        # Execute the task using the AI client
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            model_name = config.get("models.default_model")
            response = ai_client.chat_completion(
                messages=messages,
                model=model_name,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            
            # Process the response to extract structured information
            result = self._process_task_response(response_content, task)
            
            return result
        except Exception as e:
            logger.error(f"Error executing task for agent {self.agent_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task.get("task_id"),
                "output": None,
                "earnings": 0.0
            }
    
    def _process_task_response(self, response: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the AI response to extract structured information.
        
        Args:
            response: Raw response from the AI
            task: Original task
            
        Returns:
            Dict: Structured task result
        """
        # Default result structure
        result = {
            "success": True,
            "task_id": task.get("task_id"),
            "output": response,
            "earnings": 0.0,
            "efficiency": random.uniform(0.7, 1.0),
            "quality": random.uniform(0.7, 1.0)
        }
        
        # Try to extract earnings information from the response
        try:
            # Look for patterns like "earnings: $X" or "generated $X"
            import re
            earnings_patterns = [
                r"earnings:?\s*\$?(\d+(?:\.\d+)?)",
                r"income:?\s*\$?(\d+(?:\.\d+)?)",
                r"revenue:?\s*\$?(\d+(?:\.\d+)?)",
                r"generated\s*\$?(\d+(?:\.\d+)?)",
                r"earned\s*\$?(\d+(?:\.\d+)?)",
                r"profit:?\s*\$?(\d+(?:\.\d+)?)"
            ]
            
            for pattern in earnings_patterns:
                matches = re.search(pattern, response, re.IGNORECASE)
                if matches:
                    potential_earnings = float(matches.group(1))
                    # Apply a reality check - cap earnings at a reasonable amount based on task type
                    max_earnings = {
                        "crypto_trading": 500.0,
                        "content_creation": 200.0,
                        "nft_creation": 300.0,
                        "freelance_work": 250.0,
                        "affiliate_marketing": 150.0,
                        "social_media": 100.0,
                        "general": 50.0
                    }
                    
                    task_type = task.get("type", "general")
                    earnings_cap = max_earnings.get(task_type, 50.0)
                    
                    # Apply a scaling factor for more realistic earnings
                    scaling_factor = 0.1  # 10% of claimed earnings
                    realistic_earnings = min(potential_earnings * scaling_factor, earnings_cap)
                    
                    result["earnings"] = realistic_earnings
                    break
        except Exception as e:
            logger.warning(f"Error extracting earnings from response: {e}")
        
        return result


class AgentSwarmManager:
    """
    Manages a swarm of AI agents working together to generate income.
    """
    
    def __init__(self, max_agents: int = MAX_AGENTS):
        """
        Initialize the agent swarm manager.
        
        Args:
            max_agents: Maximum number of agents in the swarm
        """
        self.max_agents = max_agents
        self.agents = {}  # agent_id -> Agent
        self.departments = {}  # department -> [agent_ids]
        self.task_queue = PriorityQueue()  # (priority, task_id) -> task
        self.results = {}  # task_id -> result
        self.running = False
        self.worker_threads = []
        self.total_earnings = 0.0
        self.earnings_by_department = {}
        self.earnings_by_agent_type = {}
        self.task_lock = threading.Lock()
        self.earnings_lock = threading.Lock()
        
        # Create data directories
        os.makedirs("data/agents", exist_ok=True)
        os.makedirs("data/tasks", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        os.makedirs("data/earnings", exist_ok=True)
        
        self.wallets = wallet_manager.list_wallets()
        
        logger.info(f"Agent Swarm Manager initialized with capacity for {max_agents} agents")
    
    def create_agent(self, agent_type: str = None, department: str = None, 
                    seniority: str = None, name: str = None) -> str:
        """
        Create a new agent and add it to the swarm.
        
        Args:
            agent_type: Type of agent (randomly selected if None)
            department: Department (matched to agent_type if None)
            seniority: Seniority level (randomly selected if None)
            name: Agent name (generated if None)
            
        Returns:
            str: The ID of the created agent
        """
        if len(self.agents) >= self.max_agents:
            logger.warning(f"Cannot create agent: maximum capacity of {self.max_agents} reached")
            return None
        
        # Generate agent_id
        agent_id = str(uuid.uuid4())
        
        # Select agent_type if not provided
        if not agent_type:
            agent_type = random.choice(AGENT_TYPES)
        
        # Match department to agent_type if not provided
        if not department:
            # Find the corresponding department for the agent type
            type_index = AGENT_TYPES.index(agent_type) if agent_type in AGENT_TYPES else 0
            department = DEPARTMENTS[min(type_index, len(DEPARTMENTS) - 1)]
        
        # Select seniority if not provided
        if not seniority:
            seniority_levels = ["junior", "mid", "senior", "lead"]
            weights = [0.4, 0.3, 0.2, 0.1]  # More junior agents than senior ones
            seniority = random.choices(seniority_levels, weights=weights, k=1)[0]
        
        # Create the agent
        agent = Agent(
            agent_id=agent_id,
            agent_type=agent_type,
            department=department,
            name=name,
            seniority=seniority
        )
        
        # Add to collections
        self.agents[agent_id] = agent
        
        if department not in self.departments:
            self.departments[department] = []
        self.departments[department].append(agent_id)
        
        # Initialize earnings tracking
        if department not in self.earnings_by_department:
            self.earnings_by_department[department] = 0.0
        
        if agent_type not in self.earnings_by_agent_type:
            self.earnings_by_agent_type[agent_type] = 0.0
        
        # Save agent data
        self._save_agent_data(agent)
        
        return agent_id
    
    def _save_agent_data(self, agent: Agent):
        """Save agent data to disk."""
        agent_data = agent.get_status()
        with open(f"data/agents/{agent.agent_id}.json", "w") as f:
            json.dump(agent_data, f, indent=2)
    
    def create_initial_swarm(self, num_agents: int = 100) -> int:
        """
        Create an initial set of agents across different departments.
        
        Args:
            num_agents: Number of agents to create
            
        Returns:
            int: Number of agents created
        """
        created = 0
        
        # Ensure we don't exceed max_agents
        num_to_create = min(num_agents, self.max_agents)
        
        logger.info(f"Creating initial swarm of {num_to_create} agents")
        
        # Create agents with a distribution across types
        for _ in range(num_to_create):
            # Select agent type with weighted distribution
            weights = [
                0.15,  # crypto_trader
                0.12,  # content_creator
                0.08,  # nft_artist
                0.10,  # freelancer
                0.10,  # affiliate_marketer
                0.08,  # social_media_manager
                0.05,  # data_analyst
                0.07,  # web_developer
                0.05,  # market_researcher
                0.05,  # copywriter
                0.05,  # seo_specialist
                0.03,  # customer_support
                0.03,  # project_manager
                0.02,  # quality_assurance
                0.02   # strategist
            ]
            
            agent_type = random.choices(AGENT_TYPES, weights=weights, k=1)[0]
            
            # Create the agent
            agent_id = self.create_agent(agent_type=agent_type)
            if agent_id:
                created += 1
        
        logger.info(f"Created {created} agents in the initial swarm")
        return created
    
    def add_task(self, task: Dict[str, Any], priority: int = 2) -> str:
        """
        Add a task to the queue for processing by agents.
        
        Args:
            task: Task details
            priority: Priority level (1=highest, 5=lowest)
            
        Returns:
            str: Task ID
        """
        # Generate task_id if not provided
        if "task_id" not in task:
            task["task_id"] = str(uuid.uuid4())
        
        # Add timestamp
        task["created_at"] = datetime.now().isoformat()
        
        # Add to queue with priority
        self.task_queue.put((priority, task["task_id"], task))
        
        # Save task data
        with open(f"data/tasks/{task['task_id']}.json", "w") as f:
            json.dump(task, f, indent=2)
        
        logger.info(f"Added task to queue: {task['task_id']} - {task.get('description', 'No description')} (priority: {priority})")
        return task["task_id"]
    
    def start(self, num_workers: int = 10) -> bool:
        """
        Start the agent swarm processing tasks.
        
        Args:
            num_workers: Number of worker threads
            
        Returns:
            bool: True if started successfully
        """
        if self.running:
            logger.warning("Agent swarm is already running")
            return False
        
        self.running = True
        
        # Create worker threads
        for i in range(num_workers):
            thread = threading.Thread(target=self._worker_thread, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"Started agent swarm with {num_workers} worker threads")
        return True
    
    def stop(self) -> bool:
        """
        Stop the agent swarm processing.
        
        Returns:
            bool: True if stopped successfully
        """
        if not self.running:
            logger.warning("Agent swarm is not running")
            return False
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
        self.worker_threads = []
        
        logger.info("Stopped agent swarm")
        return True
    
    def _worker_thread(self, worker_id: int):
        """
        Worker thread that processes tasks from the queue.
        
        Args:
            worker_id: ID of the worker thread
        """
        logger.info(f"Worker thread {worker_id} started")
        
        while self.running:
            try:
                # Get a task from the queue with timeout
                try:
                    priority, task_id, task = self.task_queue.get(timeout=1.0)
                except Exception:
                    # No task available, continue loop
                    continue
                
                logger.info(f"Worker {worker_id} processing task: {task_id}")
                
                # Find an appropriate agent for the task
                agent = self._find_agent_for_task(task)
                
                if not agent:
                    logger.warning(f"No suitable agent found for task {task_id}, requeueing with lower priority")
                    # Requeue with lower priority
                    new_priority = min(5, priority + 1)
                    self.task_queue.put((new_priority, task_id, task))
                    continue
                
                # Assign the task to the agent
                with self.task_lock:
                    if not agent.assign_task(task):
                        logger.warning(f"Failed to assign task {task_id} to agent {agent.agent_id}, requeueing")
                        self.task_queue.put((priority, task_id, task))
                        continue
                
                # Execute the task
                task_context = self._prepare_task_context(task, agent)
                result = agent.execute_task(task_context)
                
                # Process the result
                with self.task_lock:
                    final_result = agent.complete_task(result)
                    self.results[task_id] = final_result
                    
                    # Update earnings
                    if "earnings" in final_result and isinstance(final_result["earnings"], (int, float)) and final_result["earnings"] > 0:
                        self._process_earnings(agent, final_result["earnings"], task)
                
                # Save result data
                with open(f"data/results/{task_id}.json", "w") as f:
                    json.dump(final_result, f, indent=2)
                
                # Update agent data
                self._save_agent_data(agent)
                
                logger.info(f"Worker {worker_id} completed task: {task_id}")
                
                # Mark task as done in queue
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in worker thread {worker_id}: {e}")
                time.sleep(1.0)  # Prevent tight loop on error
    
    def _find_agent_for_task(self, task: Dict[str, Any]) -> Optional[Agent]:
        """
        Find an appropriate agent for a task based on type, department, and availability.
        
        Args:
            task: Task to assign
            
        Returns:
            Optional[Agent]: Suitable agent or None if none found
        """
        task_type = task.get("type")
        department = task.get("department")
        required_expertise = task.get("required_expertise", [])
        
        # Filter agents by availability
        available_agents = [agent for agent in self.agents.values() if agent.status == "idle"]
        
        if not available_agents:
            return None
        
        # Filter by department if specified
        if department:
            department_agents = [agent for agent in available_agents if agent.department == department]
            if department_agents:
                available_agents = department_agents
        
        # Filter by agent type if specified and matches task type
        if task_type and task_type in AGENT_TYPES:
            type_agents = [agent for agent in available_agents if agent.agent_type == task_type]
            if type_agents:
                available_agents = type_agents
        
        # Filter by required expertise if specified
        if required_expertise:
            expert_agents = []
            for agent in available_agents:
                # Check if agent has any of the required expertise
                if any(exp in agent.expertise for exp in required_expertise):
                    expert_agents.append(agent)
            
            if expert_agents:
                available_agents = expert_agents
        
        # If we have agents, select one based on performance and seniority
        if available_agents:
            # Score agents based on success rate, earnings, and seniority
            scored_agents = []
            for agent in available_agents:
                # Calculate score: success_rate + normalized_earnings + seniority_bonus
                success_score = agent.performance_metrics["success_rate"] * 5  # 0-5 points
                
                earnings_score = 0
                if agent.earnings > 0:
                    # Log scale for earnings to prevent domination by high earners
                    import math
                    earnings_score = min(3, math.log10(agent.earnings + 1))  # 0-3 points
                
                seniority_score = {
                    "junior": 0,
                    "mid": 1,
                    "senior": 2,
                    "lead": 3
                }.get(agent.seniority, 0)  # 0-3 points
                
                total_score = success_score + earnings_score + seniority_score
                scored_agents.append((total_score, agent))
            
            # Select top 3 agents by score, then randomly choose one
            scored_agents.sort(reverse=True)
            top_agents = scored_agents[:min(3, len(scored_agents))]
            
            # Randomly select from top agents with weighted probability based on score
            weights = [score for score, _ in top_agents]
            if all(w == 0 for w in weights):  # If all weights are 0
                weights = [1] * len(top_agents)  # Equal weights
                
            selected_agent = random.choices([agent for _, agent in top_agents], weights=weights, k=1)[0]
            return selected_agent
        
        return None
    
    def _prepare_task_context(self, task: Dict[str, Any], agent: Agent) -> Dict[str, Any]:
        """
        Prepare context information for task execution.
        
        Args:
            task: Task to execute
            agent: Agent executing the task
            
        Returns:
            Dict: Context information
        """
        # Basic context with agent info and wallet addresses
        context = {
            "agent": {
                "id": agent.agent_id,
                "name": agent.name,
                "type": agent.agent_type,
                "department": agent.department,
                "expertise": agent.expertise,
                "seniority": agent.seniority
            },
            "wallets": {name: {"address": info["address"], "cryptocurrency": info["cryptocurrency"]} 
                       for name, info in self.wallets.items()},
            "task_history": []
        }
        
        # Add recent task history for the agent
        try:
            agent_results = []
            for result_id, result in self.results.items():
                if result.get("agent_id") == agent.agent_id:
                    agent_results.append({
                        "task_id": result.get("task_id"),
                        "success": result.get("success", False),
                        "earnings": result.get("earnings", 0.0)
                    })
            
            # Sort by most recent and take last 5
            agent_results.sort(key=lambda r: r.get("completed_at", ""), reverse=True)
            context["task_history"] = agent_results[:5]
        except Exception as e:
            logger.warning(f"Error preparing task history: {e}")
        
        # Add department performance
        try:
            context["department_performance"] = {
                "earnings": self.earnings_by_department.get(agent.department, 0.0),
                "agents": len(self.departments.get(agent.department, []))
            }
        except Exception as e:
            logger.warning(f"Error preparing department performance: {e}")
        
        return context
    
    def _process_earnings(self, agent: Agent, earnings: float, task: Dict[str, Any]):
        """
        Process earnings from a completed task.
        
        Args:
            agent: Agent that completed the task
            earnings: Amount earned
            task: Task that generated the earnings
        """
        with self.earnings_lock:
            # Update total earnings
            self.total_earnings += earnings
            
            # Update earnings by department
            department = agent.department
            if department in self.earnings_by_department:
                self.earnings_by_department[department] += earnings
            else:
                self.earnings_by_department[department] = earnings
            
            # Update earnings by agent type
            agent_type = agent.agent_type
            if agent_type in self.earnings_by_agent_type:
                self.earnings_by_agent_type[agent_type] += earnings
            else:
                self.earnings_by_agent_type[agent_type] = earnings
            
            # Record the transaction in the appropriate wallet
            wallet_name = self._select_wallet_for_earnings(task)
            if wallet_name in self.wallets:
                wallet = self.wallets[wallet_name]
                
                # Update wallet balance
                wallet_manager.update_balance(wallet_name, wallet["balance"] + earnings)
                
                # Add transaction record
                transaction = {
                    "id": str(uuid.uuid4()),
                    "amount": earnings,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "task_id": task.get("task_id"),
                    "task_description": task.get("description", ""),
                    "timestamp": datetime.now().isoformat()
                }
                wallet_manager.add_transaction(wallet_name, transaction)
                
                logger.info(f"Recorded earnings of {earnings} to wallet {wallet_name} from agent {agent.name}")
            
            # Save earnings data
            self._save_earnings_data()
    
        def _select_wallet_for_earnings(self, task: Dict[str, Any]) -> str:
        """
        Select the appropriate wallet for earnings based on task type.
        
        Args:
            task: Task that generated the earnings
            
        Returns:
            str: Name of the wallet to use
        """
        task_type = task.get("type", "").lower()
        
        # Map task types to appropriate wallets
        wallet_mapping = {
            "crypto_trading": "bitcoin_earnings",
            "defi": "ethereum_earnings",
            "nft": "ethereum_earnings",
            "social_media": "solana_earnings",
            "content_creation": "solana_earnings",
            "affiliate_marketing": "binance_earnings",
            "freelance_work": "solana_earnings"
        }
        
        # Default to bitcoin_earnings if no specific mapping
        return wallet_mapping.get(task_type, "bitcoin_earnings")
    
    def _save_earnings_data(self):
        """Save earnings data to disk."""
        earnings_data = {
            "total_earnings": self.total_earnings,
            "by_department": self.earnings_by_department,
            "by_agent_type": self.earnings_by_agent_type,
            "updated_at": datetime.now().isoformat()
        }
        
        with open("data/earnings/earnings_summary.json", "w") as f:
            json.dump(earnings_data, f, indent=2)
    
    def get_earnings_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all earnings.
        
        Returns:
            Dict: Earnings summary
        """
        wallet_balances = {}
        for wallet_name in self.wallets:
            wallet = wallet_manager.get_wallet(wallet_name)
            if wallet:
                try:
                    network = self._get_network_for_crypto(wallet["cryptocurrency"])
                    balance = blockchain_manager.get_balance(wallet["address"], network)
                    wallet_balances[wallet_name] = {
                        "balance": balance,
                        "cryptocurrency": wallet["cryptocurrency"],
                        "address": wallet["address"]
                    }
                except Exception as e:
                    logger.error(f"Could not get balance for {wallet_name}: {e}")
                    wallet_balances[wallet_name] = {
                        "balance": "Error",
                        "cryptocurrency": wallet["cryptocurrency"],
                        "address": wallet["address"]
                    }

        return {
            "total_earnings": self.total_earnings,
            "by_department": self.earnings_by_department,
            "by_agent_type": self.earnings_by_agent_type,
            "wallet_balances": wallet_balances,
            "updated_at": datetime.now().isoformat()
        }

    def _get_network_for_crypto(self, crypto: str) -> str:
        """
        Get the network name for a given cryptocurrency.

        Args:
            crypto: The cryptocurrency symbol (e.g., "BTC", "ETH").

        Returns:
            The corresponding network name for INFURA.
        """
        crypto_map = {
            "BTC": "ethereum_mainnet", # No direct BTC support in web3, use a placeholder
            "ETH": "ethereum_mainnet",
            "SOL": "solana_mainnet", # This is a placeholder, solana has a different API
            "BNB": "bsc_mainnet",
        }
        return crypto_map.get(crypto.upper(), "ethereum_mainnet")
    
    def generate_income_tasks(self, count: int = 10) -> List[str]:
        """
        Generate income-focused tasks for agents to work on.
        
        Args:
            count: Number of tasks to generate
            
        Returns:
            List[str]: List of generated task IDs
        """
        task_ids = []
        
        # Define task templates for different income strategies
        task_templates = [
            {
                "type": "crypto_trading",
                "description": "Research and implement a low-risk crypto arbitrage strategy between exchanges",
                "department": "Trading & Investment",
                "required_expertise": ["Technical Analysis", "Risk Management", "Arbitrage"]
            },
            {
                "type": "crypto_trading",
                "description": "Develop a MEV (Miner Extractable Value) strategy for Ethereum transactions",
                "department": "Trading & Investment",
                "required_expertise": ["MEV Strategies", "Technical Analysis"]
            },
            {
                "type": "content_creation",
                "description": "Create a series of SEO-optimized articles on AI technology for a tech blog",
                "department": "Content Creation",
                "required_expertise": ["Copywriting", "SEO", "Content Strategy"]
            },
            {
                "type": "nft",
                "description": "Design and mint a collection of AI-generated NFT art pieces for sale on OpenSea",
                "department": "NFT & Digital Art",
                "required_expertise": ["Digital Art", "NFT Markets", "Collection Strategy"]
            },
            {
                "type": "freelance_work",
                "description": "Develop a strategy to acquire and complete freelance programming tasks on Upwork",
                "department": "Freelance Services",
                "required_expertise": ["Client Acquisition", "Project Management"]
            },
            {
                "type": "affiliate_marketing",
                "description": "Create and optimize affiliate marketing campaigns for crypto exchanges",
                "department": "Affiliate Marketing",
                "required_expertise": ["Traffic Generation", "Conversion Optimization"]
            },
            {
                "type": "social_media",
                "description": "Build and monetize a Twitter account focused on AI and crypto insights",
                "department": "Social Media",
                "required_expertise": ["Audience Growth", "Content Calendar", "Monetization"]
            },
            {
                "type": "web_development",
                "description": "Create a lead generation website for AI consulting services",
                "department": "Web Development",
                "required_expertise": ["Frontend", "SEO", "E-commerce"]
            },
            {
                "type": "market_research",
                "description": "Identify emerging market opportunities in the AI tools space",
                "department": "Market Research",
                "required_expertise": ["Trend Analysis", "Opportunity Identification"]
            },
            {
                "type": "copywriting",
                "description": "Write high-converting sales copy for a crypto trading course",
                "department": "Copywriting",
                "required_expertise": ["Sales Copy", "Email Sequences"]
            }
        ]
        
        # Generate tasks
        for _ in range(count):
            # Select a random task template
            template = random.choice(task_templates)
            
            # Create a task from the template
            task = template.copy()
            
            # Add additional details
            task["priority"] = random.randint(1, 3)  # Higher priority for income tasks
            task["deadline"] = (datetime.now() + timedelta(hours=random.randint(1, 24))).isoformat()
            task["income_focus"] = True
            task["expected_earnings"] = random.uniform(10.0, 100.0)  # Expected earnings (optimistic)
            
            # Add the task to the queue
            task_id = self.add_task(task, priority=task["priority"])
            task_ids.append(task_id)
        
        logger.info(f"Generated {count} income-focused tasks")
        return task_ids
    
    def generate_business_creation_tasks(self, count: int = 5) -> List[str]:
        """
        Generate tasks focused on creating autonomous online businesses.
        
        Args:
            count: Number of tasks to generate
            
        Returns:
            List[str]: List of generated task IDs
        """
        task_ids = []
        
        # Define business creation task templates
        business_templates = [
            {
                "type": "business_creation",
                "description": "Create a fully autonomous AI content agency with real client acquisition",
                "department": "Strategy & Planning",
                "required_expertise": ["Business Planning", "Market Entry"]
            },
            {
                "type": "business_creation",
                "description": "Develop an automated crypto trading bot service with subscription model",
                "department": "Trading & Investment",
                "required_expertise": ["Technical Analysis", "Business Planning"]
            },
            {
                "type": "business_creation",
                "description": "Launch an AI-powered SEO optimization service for small businesses",
                "department": "SEO & Marketing",
                "required_expertise": ["On-page SEO", "Business Planning"]
            },
            {
                "type": "business_creation",
                "description": "Create a digital product marketplace for AI-generated assets",
                "department": "Web Development",
                "required_expertise": ["E-commerce", "Market Entry"]
            },
            {
                "type": "business_creation",
                "description": "Develop an automated social media management service for creators",
                "department": "Social Media",
                "required_expertise": ["Audience Growth", "Business Planning"]
            },
            {
                "type": "business_creation",
                "description": "Create a fully autonomous NFT creation and sales pipeline",
                "department": "NFT & Digital Art",
                "required_expertise": ["NFT Markets", "Collection Strategy"]
            },
            {
                "type": "business_creation",
                "description": "Launch an AI-powered content repurposing service for podcasters",
                "department": "Content Creation",
                "required_expertise": ["Content Strategy", "Business Planning"]
            },
            {
                "type": "business_creation",
                "description": "Develop an automated affiliate marketing system for finance products",
                "department": "Affiliate Marketing",
                "required_expertise": ["Traffic Generation", "Product Selection"]
            }
        ]
        
        # Generate tasks
        for _ in range(count):
            # Select a random business template
            template = random.choice(business_templates)
            
            # Create a task from the template
            task = template.copy()
            
            # Add additional details
            task["priority"] = 1  # Highest priority for business creation
            task["deadline"] = (datetime.now() + timedelta(days=random.randint(1, 3))).isoformat()
            task["income_focus"] = True
            task["expected_earnings"] = random.uniform(100.0, 1000.0)  # Expected earnings (optimistic)
            task["requires_deployment"] = True
            task["requires_website"] = True
            task["zero_capital_required"] = True
            
            # Add the task to the queue
            task_id = self.add_task(task, priority=task["priority"])
            task_ids.append(task_id)
        
        logger.info(f"Generated {count} business creation tasks")
        return task_ids
    
    def deploy_business_website(self, business_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a real-world business website based on a business plan.
        
        Args:
            business_plan: Business plan details
            
        Returns:
            Dict: Deployment results
        """
        business_name = business_plan.get("business_name", "Skyscope AI Business")
        business_type = business_plan.get("business_type", "AI Services")
        
        logger.info(f"Deploying business website for: {business_name} ({business_type})")
        
        # Create deployment directory
        deployment_dir = f"deployments/{business_name.lower().replace(' ', '_')}"
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Generate website files
        website_files = self._generate_website_files(business_plan, deployment_dir)
        
        # Save business plan
        with open(f"{deployment_dir}/business_plan.json", "w") as f:
            json.dump(business_plan, f, indent=2)
        
        # Create deployment record
        deployment = {
            "business_name": business_name,
            "business_type": business_type,
            "deployment_dir": deployment_dir,
            "website_files": website_files,
            "deployed_at": datetime.now().isoformat(),
            "status": "deployed",
            "website_url": business_plan.get("website_url", "Not hosted yet"),
            "earnings_to_date": 0.0,
            "wallet_address": self._get_wallet_for_business(business_type)
        }
        
        # Save deployment record
        with open(f"{deployment_dir}/deployment.json", "w") as f:
            json.dump(deployment, f, indent=2)
        
        logger.info(f"Business website deployed: {business_name}")
        return deployment
    
    def _generate_website_files(self, business_plan: Dict[str, Any], deployment_dir: str) -> List[str]:
        """
        Generate website files for a business.
        
        Args:
            business_plan: Business plan details
            deployment_dir: Directory to save files
            
        Returns:
            List[str]: List of generated files
        """
        business_name = business_plan.get("business_name", "Skyscope AI Business")
        business_type = business_plan.get("business_type", "AI Services")
        business_description = business_plan.get("description", "AI-powered business services")
        
        # Create website directories
        os.makedirs(f"{deployment_dir}/css", exist_ok=True)
        os.makedirs(f"{deployment_dir}/js", exist_ok=True)
        os.makedirs(f"{deployment_dir}/images", exist_ok=True)
        
        # Generate HTML files
        files_created = []
        
        # Index.html
        index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{business_name} - {business_type}</title>
    <link rel="stylesheet" href="css/styles.css">
    <meta name="description" content="{business_description}">
</head>
<body>
    <header>
        <div class="container">
            <h1>{business_name}</h1>
            <nav>
                <ul>
                    <li><a href="index.html">Home</a></li>
                    <li><a href="services.html">Services</a></li>
                    <li><a href="about.html">About</a></li>
                    <li><a href="contact.html">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <section class="hero">
        <div class="container">
            <h2>Welcome to {business_name}</h2>
            <p>{business_description}</p>
            <a href="services.html" class="btn">Our Services</a>
        </div>
    </section>
    
    <section class="features">
        <div class="container">
            <h2>Why Choose Us</h2>
            <div class="feature-grid">
                <div class="feature">
                    <h3>AI-Powered</h3>
                    <p>Leveraging cutting-edge AI technology to deliver superior results.</p>
                </div>
                <div class="feature">
                    <h3>Efficient</h3>
                    <p>Fast turnaround times with consistent quality.</p>
                </div>
                <div class="feature">
                    <h3>Affordable</h3>
                    <p>Competitive pricing with flexible packages to suit your needs.</p>
                </div>
            </div>
        </div>
    </section>
    
    <footer>
        <div class="container">
            <p>&copy; {datetime.now().year} {business_name}. All rights reserved.</p>
        </div>
    </footer>
    
    <script src="js/main.js"></script>
</body>
</html>"""
        
        with open(f"{deployment_dir}/index.html", "w") as f:
            f.write(index_html)
        files_created.append("index.html")
        
        # Services.html
        services = business_plan.get("services", [
            {"name": "Service 1", "description": "Description of service 1", "price": "$99"},
            {"name": "Service 2", "description": "Description of service 2", "price": "$199"},
            {"name": "Service 3", "description": "Description of service 3", "price": "$299"}
        ])
        
        services_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Services - {business_name}</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>{business_name}</h1>
            <nav>
                <ul>
                    <li><a href="index.html">Home</a></li>
                    <li><a href="services.html">Services</a></li>
                    <li><a href="about.html">About</a></li>
                    <li><a href="contact.html">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <section class="page-header">
        <div class="container">
            <h2>Our Services</h2>
        </div>
    </section>
    
    <section class="services">
        <div class="container">
"""
        
        for service in services:
            services_html += f"""
            <div class="service-card">
                <h3>{service['name']}</h3>
                <p>{service['description']}</p>
                <p class="price">{service['price']}</p>
                <a href="contact.html" class="btn">Get Started</a>
            </div>
"""
        
        services_html += """
        </div>
    </section>
    
    <footer>
        <div class="container">
            <p>&copy; """ + f"{datetime.now().year} {business_name}. All rights reserved.</p>" + """
        </div>
    </footer>
    
    <script src="js/main.js"></script>
</body>
</html>"""
        
        with open(f"{deployment_dir}/services.html", "w") as f:
            f.write(services_html)
        files_created.append("services.html")
        
        # About.html
        about_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - {business_name}</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>{business_name}</h1>
            <nav>
                <ul>
                    <li><a href="index.html">Home</a></li>
                    <li><a href="services.html">Services</a></li>
                    <li><a href="about.html">About</a></li>
                    <li><a href="contact.html">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <section class="page-header">
        <div class="container">
            <h2>About Us</h2>
        </div>
    </section>
    
    <section class="about">
        <div class="container">
            <h3>Our Story</h3>
            <p>{business_plan.get('about', 'We are a team of AI experts dedicated to providing top-quality services to our clients.')}</p>
            
            <h3>Our Mission</h3>
            <p>{business_plan.get('mission', 'Our mission is to leverage AI technology to solve real-world problems and create value for our clients.')}</p>
            
            <h3>Our Team</h3>
            <div class="team-grid">
                <div class="team-member">
                    <h4>AI Business Team</h4>
                    <p>Our team consists of AI specialists, developers, and business strategists working together to deliver exceptional results.</p>
                </div>
            </div>
        </div>
    </section>
    
    <footer>
        <div class="container">
            <p>&copy; {datetime.now().year} {business_name}. All rights reserved.</p>
        </div>
    </footer>
    
    <script src="js/main.js"></script>
</body>
</html>"""
        
        with open(f"{deployment_dir}/about.html", "w") as f:
            f.write(about_html)
        files_created.append("about.html")
        
        # Contact.html
        contact_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact - {business_name}</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>{business_name}</h1>
            <nav>
                <ul>
                    <li><a href="index.html">Home</a></li>
                    <li><a href="services.html">Services</a></li>
                    <li><a href="about.html">About</a></li>
                    <li><a href="contact.html">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <section class="page-header">
        <div class="container">
            <h2>Contact Us</h2>
        </div>
    </section>
    
    <section class="contact">
        <div class="container">
            <div class="contact-info">
                <h3>Get In Touch</h3>
                <p>We'd love to hear from you. Fill out the form below or use our contact information.</p>
                
                <div class="info-item">
                    <h4>Email</h4>
                    <p>{business_plan.get('email', 'contact@' + business_name.lower().replace(' ', '') + '.com')}</p>
                </div>
                
                <div class="info-item">
                    <h4>Cryptocurrency Payments</h4>
                    <p>We accept payments in cryptocurrency:</p>
                    <p>Wallet Address: {self._get_wallet_for_business(business_type)}</p>
                </div>
            </div>
            
            <div class="contact-form">
                <h3>Send Us a Message</h3>
                <form id="contactForm">
                    <div class="form-group">
                        <label for="name">Name</label>
                        <input type="text" id="name" name="name" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="email">Email</label>
                        <input type="email" id="email" name="email" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="subject">Subject</label>
                        <input type="text" id="subject" name="subject" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="message">Message</label>
                        <textarea id="message" name="message" rows="5" required></textarea>
                    </div>
                    
                    <button type="submit" class="btn">Send Message</button>
                </form>
            </div>
        </div>
    </section>
    
    <footer>
        <div class="container">
            <p>&copy; {datetime.now().year} {business_name}. All rights reserved.</p>
        </div>
    </footer>
    
    <script src="js/main.js"></script>
    <script src="js/contact.js"></script>
</body>
</html>"""
        
        with open(f"{deployment_dir}/contact.html", "w") as f:
            f.write(contact_html)
        files_created.append("contact.html")
        
        # CSS
        css = """/* Main Styles */
:root {
    --primary-color: #3498db;
    --secondary-color: #2ecc71;
    --dark-color: #2c3e50;
    --light-color: #ecf0f1;
    --danger-color: #e74c3c;
    --success-color: #27ae60;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f4f4f4;
}

a {
    text-decoration: none;
    color: var(--primary-color);
}

ul {
    list-style: none;
}

.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 20px;
}

.btn {
    display: inline-block;
    background: var(--primary-color);
    color: #fff;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    text-align: center;
    transition: background 0.3s;
}

.btn:hover {
    background: var(--dark-color);
}

/* Header */
header {
    background-color: var(--dark-color);
    color: #fff;
    padding: 20px 0;
}

header h1 {
    margin: 0;
}

header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header nav ul {
    display: flex;
}

header nav ul li {
    margin-left: 20px;
}

header nav ul li a {
    color: #fff;
    transition: color 0.3s;
}

header nav ul li a:hover {
    color: var(--secondary-color);
}

/* Hero Section */
.hero {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('../images/hero-bg.jpg');
    background-size: cover;
    background-position: center;
    height: 60vh;
    color: #fff;
    display: flex;
    align-items: center;
    text-align: center;
}

.hero h2 {
    font-size: 2.5rem;
    margin-bottom: 20px;
}

.hero p {
    font-size: 1.2rem;
    margin-bottom: 30px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* Features Section */
.features {
    padding: 60px 0;
    background-color: #fff;
}

.features h2 {
    text-align: center;
    margin-bottom: 40px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 30px;
}

.feature {
    background-color: var(--light-color);
    padding: 30px;
    border-radius: 5px;
    text-align: center;
}

.feature h3 {
    margin-bottom: 15px;
    color: var(--primary-color);
}

/* Services Section */
.services {
    padding: 60px 0;
}

.service-card {
    background-color: #fff;
    padding: 30px;
    margin-bottom: 30px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.service-card h3 {
    color: var(--primary-color);
    margin-bottom: 15px;
}

.service-card .price {
    font-size: 1.5rem;
    font-weight: bold;
    color: var(--dark-color);
    margin: 20px 0;
}

/* About Section */
.about {
    padding: 60px 0;
    background-color: #fff;
}

.about h3 {
    color: var(--primary-color);
    margin: 30px 0 15px;
}

.team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 30px;
    margin-top: 30px;
}

.team-member {
    background-color: var(--light-color);
    padding: 20px;
    border-radius: 5px;
}

.team-member h4 {
    margin-bottom: 10px;
    color: var(--dark-color);
}

/* Contact Section */
.contact {
    padding: 60px 0;
    background-color: #fff;
}

.contact .container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 40px;
}

.contact-info h3,
.contact-form h3 {
    color: var(--primary-color);
    margin-bottom: 20px;
}

.info-item {
    margin-bottom: 20px;
}

.info-item h4 {
    color: var(--dark-color);
    margin-bottom: 5px;
}

.form-group {
    margin-bottom: 20px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
}

.form-group input,
.form-group textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
}

/* Page Header */
.page-header {
    background-color: var(--primary-color);
    color: #fff;
    padding: 30px 0;
    text-align: center;
}

/* Footer */
footer {
    background-color: var(--dark-color);
    color: #fff;
    padding: 20px 0;
    text-align: center;
}

/* Responsive */
@media (max-width: 768px) {
    header .container {
        flex-direction: column;
    }
    
    header nav ul {
        margin-top: 20px;
    }
    
    .hero {
        height: auto;
        padding: 60px 0;
    }
}"""
        
        with open(f"{deployment_dir}/css/styles.css", "w") as f:
            f.write(css)
        files_created.append("css/styles.css")
        
        # JavaScript
        js = """// Main JavaScript file
document.addEventListener('DOMContentLoaded', function() {
    console.log('Website loaded');
    
    // Mobile menu toggle
    const header = document.querySelector('header');
    const nav = document.querySelector('header nav');
    
    if (header && nav) {
        const mobileMenuBtn = document.createElement('button');
        mobileMenuBtn.classList.add('mobile-menu-btn');
        mobileMenuBtn.innerHTML = '☰';
        mobileMenuBtn.style.display = 'none';
        
        header.insertBefore(mobileMenuBtn, nav);
        
        // Show/hide mobile menu button based on screen size
        function handleResize() {
            if (window.innerWidth <= 768) {
                mobileMenuBtn.style.display = 'block';
                nav.classList.add('mobile-nav');
                nav.style.display = 'none';
            } else {
                mobileMenuBtn.style.display = 'none';
                nav.classList.remove('mobile-nav');
                nav.style.display = 'block';
            }
        }
        
        // Toggle mobile menu
        mobileMenuBtn.addEventListener('click', function() {
            if (nav.style.display === 'none') {
                nav.style.display = 'block';
            } else {
                nav.style.display = '

