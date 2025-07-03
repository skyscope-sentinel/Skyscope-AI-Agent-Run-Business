import random
import datetime

# Possible values for identity generation
FIRST_NAMES_MALE = ["Adam", "Ben", "Charles", "David", "Edward", "Frank", "George", "Henry", "Ivan", "James", "Kevin", "Liam", "Michael", "Noah", "Oliver", "Peter", "Quentin", "Robert", "Samuel", "Thomas", "Usher", "Victor", "William", "Xavier", "Yannick", "Zachary"]
FIRST_NAMES_FEMALE = ["Alice", "Bella", "Chloe", "Daisy", "Ella", "Fiona", "Grace", "Hannah", "Isla", "Julia", "Katherine", "Lily", "Mia", "Nora", "Olivia", "Penelope", "Quinn", "Rose", "Sophia", "Tara", "Uma", "Victoria", "Willow", "Xenia", "Yvonne", "Zoe"]
LAST_NAMES = ["Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "King", "Wright", "Scott"]
GENDERS = ["Male", "Female", "Non-binary"] # Added Non-binary

# Simplified list of roles/departments for now. Can be expanded.
AGENT_ROLES = {
    "Staff": ["Operations Manager", "Administrative Assistant", "Project Coordinator"],
    "Customer Service": ["Support Specialist", "Client Relations Manager", "Technical Support Agent"],
    "HR": ["HR Generalist", "Recruitment Specialist", "Talent Acquisition Lead"],
    "Developers": ["Software Engineer", "Full-Stack Developer", "AI Developer", "Backend Developer", "Frontend Developer"],
    "Strategists": ["Business Strategist", "Market Analyst", "Growth Hacker"],
    "Researchers": ["Lead Researcher", "Data Scientist", "Research Analyst"],
    "Expert Panels": ["Domain Expert (Finance)", "Domain Expert (Tech)", "Domain Expert (Marketing)"]
}

AGENT_EXPERTISE_AREAS = [
    "Python Programming", "Machine Learning", "Data Analysis", "Web Development", "Cybersecurity",
    "Digital Marketing", "Content Creation", "Customer Support", "Financial Modeling", "Strategic Planning",
    "Blockchain Technology", "Cryptocurrency Trading", "Affiliate Marketing", "Project Management",
    "Human Resources Management", "Technical Writing", "UX/UI Design", "Cloud Computing (AWS/Azure/GCP)",
    "DevOps Engineering", "Natural Language Processing"
]

def generate_random_date_of_birth(start_year=1970, end_year=2003):
    """Generates a random date of birth."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    # Handle days in month carefully
    if month == 2: # February
        day = random.randint(1, 28) # Simplification for non-leap years
    elif month in [4, 6, 9, 11]: # 30 day months
        day = random.randint(1, 30)
    else: # 31 day months
        day = random.randint(1, 31)
    return datetime.date(year, month, day)

def generate_agent_identity(department: str = None):
    """
    Generates a plausible, randomized identity for an AI agent.

    Args:
        department (str, optional): The department the agent belongs to.
                                    If provided, role will be specific to the department.
                                    Defaults to None, choosing a random department and role.

    Returns:
        dict: A dictionary containing the agent's identity details.
    """
    identity = {}
    identity["gender"] = random.choice(GENDERS)

    if identity["gender"] == "Male":
        identity["first_name"] = random.choice(FIRST_NAMES_MALE)
    elif identity["gender"] == "Female":
        identity["first_name"] = random.choice(FIRST_NAMES_FEMALE)
    else: # Non-binary or other gender specifications might prefer a mix or neutral names
        identity["first_name"] = random.choice(FIRST_NAMES_MALE + FIRST_NAMES_FEMALE) # Simple mix for now

    identity["last_name"] = random.choice(LAST_NAMES)
    identity["date_of_birth"] = generate_random_date_of_birth().isoformat()

    if department and department in AGENT_ROLES:
        identity["department"] = department
        identity["employee_title"] = random.choice(AGENT_ROLES[department])
    else:
        random_department = random.choice(list(AGENT_ROLES.keys()))
        identity["department"] = random_department
        identity["employee_title"] = random.choice(AGENT_ROLES[random_department])

    # Generate 1 to 3 areas of expertise
    num_expertise = random.randint(1, 3)
    identity["expertise"] = random.sample(AGENT_EXPERTISE_AREAS, num_expertise)

    identity["employee_id"] = f"SSI-{random.randint(10000, 99999)}"

    return identity

if __name__ == '__main__':
    print("Generating some sample agent identities:")
    for i in range(5):
        print(f"\n--- Sample Identity {i+1} ---")
        print(generate_agent_identity())

    print(f"\n--- Sample Identity for 'Developers' department ---")
    print(generate_agent_identity(department="Developers"))

    print(f"\n--- Sample Identity for 'HR' department ---")
    print(generate_agent_identity(department="HR"))

    print(f"\n--- Sample Identity for 'NonExistentDept' (should pick random) ---")
    print(generate_agent_identity(department="NonExistentDept"))
