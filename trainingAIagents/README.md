##  AI Agent Behavioral Tool Learning 

* a type of Closed-Loop Agent Learning 
* AI Agents and Training

## Current Jupyters

* 3 works
* 4 same as 3 but started adding PPO


```


# ================================
# CREWAI + TRAINED TOOL SELECTION
# ================================

from crewai import Agent
from langchain.tools import Tool

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# 1. DEFINE TOOLS
# -------------------------------
def calculator_tool(x: str):
    try:
        return str(eval(x))
    except:
        return "Error in calculation"

def search_tool(x: str):
    return f"Searching for: {x}"

# -------------------------------
# 2. LOGGING WRAPPER
# -------------------------------
logs = []

def logging_tool(tool_name, func):
    def wrapper(x):
        logs.append((x, tool_name))
        return func(x)
    return wrapper

tools = [
    Tool(
        name="Calculator",
        func=logging_tool("Calculator", calculator_tool),
        description="Use for math operations"
    ),
    Tool(
        name="Search",
        func=logging_tool("Search", search_tool),
        description="Use for general knowledge"
    )
]

# -------------------------------
# 3. CREWAI AGENT (FOR DATA COLLECTION)
# -------------------------------
agent = Agent(
    role="Assistant",
    goal="Solve problems using the correct tool",
    backstory="You choose tools wisely.",
    tools=tools,
    verbose=False
)

# -------------------------------
# 4. GENERATE DATA (RUN AGENT)
# -------------------------------
queries = [
    "2 + 2",
    "10 * 5",
    "100 / 4",
    "Who is Einstein?",
    "capital of France",
    "latest news",
    "3 * 9",
    "square root of 16"
]

print("\n--- Collecting data from agent ---\n")

for q in queries:
    try:
        agent.execute_task(q)
    except:
        pass

print("Collected logs:")
print(logs)

# -------------------------------
# 5. TRAIN MODEL (QUERY -> TOOL)
# -------------------------------
texts = [x[0] for x in logs]
labels = [x[1] for x in logs]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

print("\n--- Model trained ---\n")

# -------------------------------
# 6. TRAINED POLICY (PREDICT TOOL)
# -------------------------------
def predict_tool(query):
    X = vectorizer.transform([query])
    return model.predict(X)[0]

# -------------------------------
# 7. FINAL AGENT (USES TRAINED POLICY)
# -------------------------------
def run_agent(query):
    chosen_tool = predict_tool(query)

    print(f"\nQuery: {query}")
    print(f"Chosen Tool (trained): {chosen_tool}")

    if chosen_tool == "Calculator":
        result = calculator_tool(query)
    else:
        result = search_tool(query)

    print(f"Result: {result}")
    return result

# -------------------------------
# 8. TEST
# -------------------------------
print("\n--- Testing trained agent ---")

run_agent("5 * 6")
run_agent("Who invented electricity?")
run_agent("12 + 45")
run_agent("population of Japan")



```

some RL

```


# ==========================================
# CREWAI + PPO (FROM SCRATCH, MINIMAL)
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import random

from crewai import Agent
from langchain.tools import Tool

# -------------------------------
# 1. TOOLS
# -------------------------------
def calculator_tool(x):
    try:
        return str(eval(x))
    except:
        return "error"

def search_tool(x):
    return f"searching: {x}"

tools = [
    Tool(name="Calculator", func=calculator_tool, description="Math"),
    Tool(name="Search", func=search_tool, description="General")
]

# -------------------------------
# 2. CREWAI AGENT (used for execution only)
# -------------------------------
agent = Agent(
    role="Assistant",
    goal="Use tools correctly",
    backstory="You are efficient.",
    tools=tools,
    verbose=False
)

# -------------------------------
# 3. STATE ENCODING (VERY SIMPLE)
# -------------------------------
def encode(query):
    # 1D feature: contains digit or not
    return torch.tensor([1.0 if any(c.isdigit() for c in query) else 0.0])

# -------------------------------
# 4. POLICY NETWORK
# -------------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 2)   # 2 actions
        )

    def forward(self, x):
        return self.net(x)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# -------------------------------
# 5. PPO HYPERPARAMS
# -------------------------------
gamma = 0.99
eps_clip = 0.2

# -------------------------------
# 6. DATA
# -------------------------------
queries = [
    "2 + 2",
    "10 * 5",
    "100 / 4",
    "Who is Einstein?",
    "capital of France",
    "latest news",
    "3 * 9",
    "square root of 16"
]

def correct_action(query):
    return 0 if any(c.isdigit() for c in query) else 1
    # 0 = Calculator, 1 = Search

# -------------------------------
# 7. PPO TRAINING LOOP
# -------------------------------
print("\n--- TRAINING PPO ---\n")

for epoch in range(200):

    states = []
    actions = []
    rewards = []
    old_log_probs = []

    # collect batch
    for _ in range(16):
        q = random.choice(queries)
        s = encode(q)

        logits = policy(s)
        probs = torch.softmax(logits, dim=0)

        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        log_prob = dist.log_prob(a)

        # reward
        r = 1.0 if a.item() == correct_action(q) else -1.0

        states.append(s)
        actions.append(a)
        rewards.append(r)
        old_log_probs.append(log_prob.detach())

    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)

    # compute returns (simple, no baseline)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # PPO update
    logits = policy(states)
    probs = torch.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)

    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * returns
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * returns

    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training done.")

# -------------------------------
# 8. TRAINED AGENT (USES POLICY)
# -------------------------------
def run_agent(query):
    s = encode(query)
    logits = policy(s)
    action = torch.argmax(logits).item()

    print(f"\nQuery: {query}")
    print(f"Chosen action: {['Calculator','Search'][action]}")

    if action == 0:
        result = calculator_tool(query)
    else:
        result = search_tool(query)

    print(f"Result: {result}")

# -------------------------------
# 9. TEST
# -------------------------------
print("\n--- TESTING ---")

run_agent("5 * 6")
run_agent("Who invented electricity?")
run_agent("12 + 45")
run_agent("population of Japan")


```


another

```

# ==========================================
# CREWAI + PPO (FROM SCRATCH, MINIMAL)
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import random

from crewai import Agent
from langchain.tools import Tool

# -------------------------------
# 1. TOOLS
# -------------------------------
def calculator_tool(x):
    try:
        return str(eval(x))
    except:
        return "error"

def search_tool(x):
    return f"searching: {x}"

tools = [
    Tool(name="Calculator", func=calculator_tool, description="Math"),
    Tool(name="Search", func=search_tool, description="General")
]

# -------------------------------
# 2. CREWAI AGENT (used for execution only)
# -------------------------------
agent = Agent(
    role="Assistant",
    goal="Use tools correctly",
    backstory="You are efficient.",
    tools=tools,
    verbose=False
)

# -------------------------------
# 3. STATE ENCODING (VERY SIMPLE)
# -------------------------------
def encode(query):
    # 1D feature: contains digit or not
    return torch.tensor([1.0 if any(c.isdigit() for c in query) else 0.0])

# -------------------------------
# 4. POLICY NETWORK
# -------------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 2)   # 2 actions
        )

    def forward(self, x):
        return self.net(x)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# -------------------------------
# 5. PPO HYPERPARAMS
# -------------------------------
gamma = 0.99
eps_clip = 0.2

# -------------------------------
# 6. DATA
# -------------------------------
queries = [
    "2 + 2",
    "10 * 5",
    "100 / 4",
    "Who is Einstein?",
    "capital of France",
    "latest news",
    "3 * 9",
    "square root of 16"
]

def correct_action(query):
    return 0 if any(c.isdigit() for c in query) else 1
    # 0 = Calculator, 1 = Search

# -------------------------------
# 7. PPO TRAINING LOOP
# -------------------------------
print("\n--- TRAINING PPO ---\n")

for epoch in range(200):

    states = []
    actions = []
    rewards = []
    old_log_probs = []

    # collect batch
    for _ in range(16):
        q = random.choice(queries)
        s = encode(q)

        logits = policy(s)
        probs = torch.softmax(logits, dim=0)

        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        log_prob = dist.log_prob(a)

        # reward
        r = 1.0 if a.item() == correct_action(q) else -1.0

        states.append(s)
        actions.append(a)
        rewards.append(r)
        old_log_probs.append(log_prob.detach())

    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)

    # compute returns (simple, no baseline)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # PPO update
    logits = policy(states)
    probs = torch.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)

    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * returns
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * returns

    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training done.")

# -------------------------------
# 8. TRAINED AGENT (USES POLICY)
# -------------------------------
def run_agent(query):
    s = encode(query)
    logits = policy(s)
    action = torch.argmax(logits).item()

    print(f"\nQuery: {query}")
    print(f"Chosen action: {['Calculator','Search'][action]}")

    if action == 0:
        result = calculator_tool(query)
    else:
        result = search_tool(query)

    print(f"Result: {result}")

# -------------------------------
# 9. TEST
# -------------------------------
print("\n--- TESTING ---")

run_agent("5 * 6")
run_agent("Who invented electricity?")
run_agent("12 + 45")
run_agent("population of Japan")


```



and this


```


# ==========================================
# CREWAI + PPO + LLM REWARD + MULTI-STEP
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import random

from crewai import Agent
from langchain.tools import Tool

# -------------------------------
# 1. TOOLS
# -------------------------------
def calculator_tool(x):
    try:
        return str(eval(x))
    except:
        return "error"

def search_tool(x):
    return f"searching: {x}"

tools = [
    Tool(name="Calculator", func=calculator_tool, description="Math"),
    Tool(name="Search", func=search_tool, description="General")
]

# -------------------------------
# 2. CREWAI AGENT
# -------------------------------
agent = Agent(
    role="Assistant",
    goal="Use tools correctly",
    backstory="Efficient agent",
    tools=tools,
    verbose=False
)

# -------------------------------
# 3. SIMPLE STATE ENCODING
# -------------------------------
def encode(query, step):
    has_digit = 1.0 if any(c.isdigit() for c in query) else 0.0
    return torch.tensor([has_digit, float(step)])

# -------------------------------
# 4. POLICY NETWORK
# -------------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# -------------------------------
# 5. PPO PARAMS
# -------------------------------
gamma = 0.95
eps_clip = 0.2

# -------------------------------
# 6. DATA
# -------------------------------
queries = [
    "2 + 2",
    "10 * 5",
    "Who is Einstein?",
    "capital of France",
    "3 * 9",
    "latest news"
]

# -------------------------------
# 7. LLM REWARD (SIMULATED)
# Replace this with real LLM call if desired
# -------------------------------
def llm_reward(query, result):
    # simple proxy for teaching:
    if any(c.isdigit() for c in query):
        return 1.0 if result != "error" else -1.0
    else:
        return 1.0 if "searching" in result else -1.0

# -------------------------------
# 8. RUN TRAJECTORY (2 steps)
# -------------------------------
def run_episode(query):

    states = []
    actions = []
    log_probs = []
    rewards = []

    current_input = query

    for step in range(2):

        s = encode(current_input, step)
        logits = policy(s)
        probs = torch.softmax(logits, dim=0)

        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        log_prob = dist.log_prob(a)

        # execute action
        if a.item() == 0:
            result = calculator_tool(current_input)
        else:
            result = search_tool(current_input)

        states.append(s)
        actions.append(a)
        log_probs.append(log_prob)

        current_input = result  # next state depends on output

    # final reward from LLM
    final_reward = llm_reward(query, current_input)

    rewards = [0.0, final_reward]  # reward only at end

    return states, actions, log_probs, rewards

# -------------------------------
# 9. TRAIN PPO
# -------------------------------
print("\n--- TRAINING RLHF PPO ---\n")

for epoch in range(200):

    batch_states = []
    batch_actions = []
    batch_old_log_probs = []
    batch_returns = []

    for _ in range(16):

        q = random.choice(queries)

        states, actions, log_probs, rewards = run_episode(q)

        # compute returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        for i in range(len(states)):
            batch_states.append(states[i])
            batch_actions.append(actions[i])
            batch_old_log_probs.append(log_probs[i].detach())
            batch_returns.append(returns[i])

    states = torch.stack(batch_states)
    actions = torch.stack(batch_actions)
    old_log_probs = torch.stack(batch_old_log_probs)
    returns = torch.tensor(batch_returns)

    # normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # PPO update
    logits = policy(states)
    probs = torch.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)

    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * returns
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * returns

    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete.")

# -------------------------------
# 10. TEST AGENT
# -------------------------------
def run_agent(query):

    current_input = query

    print(f"\nQuery: {query}")

    for step in range(2):

        s = encode(current_input, step)
        logits = policy(s)
        action = torch.argmax(logits).item()

        tool_name = ["Calculator", "Search"][action]
        print(f"Step {step} → {tool_name}")

        if action == 0:
            result = calculator_tool(current_input)
        else:
            result = search_tool(current_input)

        print(f"Result: {result}")
        current_input = result

# -------------------------------
# RUN TEST
# -------------------------------
print("\n--- TESTING ---")

run_agent("5 * 6")
run_agent("Who invented electricity?")


```




## Dataset paper

```


# ============================================================
# AUTO-CYBER-DATASET.PY
#
# Proof-of-concept:
# Autonomous Cybersecurity Behavioral Dataset Generation
# + Simple Learning Demonstration
#
# PURPOSE
# -------
# 1. Generate controlled attacker/defender interactions
# 2. Automatically record behavioral trajectories
# 3. Save the resulting dataset
# 4. Train a simple model using the generated data
# 5. Demonstrate improvement on held-out data
#
# IMPORTANT
# ---------
# This is designed ONLY for an isolated VM cyber range that
# you own/control.
#
# Python packages:
#     pip install paramiko pandas numpy torch
#
# No CrewAI/Ollama is required for this first proof-of-concept.
# That is intentional: first prove that the automatically
# generated behavioral dataset contains learnable information.
#
# Later:
#     - replace rule attacker with LLM attacker
#     - replace simple defender with LLM defender
#     - replace classifier with PPO / imitation learning / LLM
#     - add additional VMs and tools
#
# ============================================================

import paramiko
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
import json
import uuid
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

# SSH connection to YOUR ISOLATED LAB VM.
#
# Example if VirtualBox forwards host port 2222 -> guest SSH 22:
#
#     SSH_HOST = "127.0.0.1"
#     SSH_PORT = 2222
#
# Enter your own lab username/password.

SSH_HOST = "127.0.0.1"
SSH_PORT = 2222

SSH_USERNAME = "YOUR_USERNAME"
SSH_PASSWORD = "YOUR_PASSWORD"


# Number of automatically generated cyber episodes.
#
# Start small.
# Once everything works, try:
#
#     100
#     500
#     1000
#     5000

NUM_EPISODES = 100


DATASET_CSV = "cyber_behavior_dataset.csv"
DATASET_JSONL = "cyber_behavior_dataset.jsonl"


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# BEHAVIORAL DATASET
# ============================================================

behavior_log = []


def log_event(
    episode_id,
    step,
    agent_role,
    scenario,
    observation,
    decision,
    tool_selected,
    tool_input,
    command,
    command_output,
    environment_state,
    other_agent_previous_action,
    success,
    reward,
    next_action
):

    event = {

        "episode_id": episode_id,

        "timestamp": datetime.now().isoformat(),

        "step": step,

        "agent_role": agent_role,

        "scenario": scenario,

        "observation": str(observation),

        "agent_decision": str(decision),

        "tool_selected": str(tool_selected),

        "tool_input": str(tool_input),

        "command": str(command),

        "command_output": str(command_output),

        "environment_state": str(environment_state),

        "other_agent_previous_action":
            str(other_agent_previous_action),

        "success": int(success),

        "reward": float(reward),

        "next_action": str(next_action)
    }

    behavior_log.append(event)


# ============================================================
# SSH
# ============================================================

def connect_ssh():

    client = paramiko.SSHClient()

    client.set_missing_host_key_policy(
        paramiko.AutoAddPolicy()
    )

    client.connect(
        SSH_HOST,
        port=SSH_PORT,
        username=SSH_USERNAME,
        password=SSH_PASSWORD,
        timeout=10
    )

    return client


def ssh_command(client, command):

    try:

        stdin, stdout, stderr = client.exec_command(
            command,
            timeout=10
        )

        output = stdout.read().decode(
            errors="ignore"
        )

        error = stderr.read().decode(
            errors="ignore"
        )

        return output + error

    except Exception as e:

        return "SSH_ERROR: " + str(e)


# ============================================================
# CYBER RANGE TOOLS
# ============================================================
#
# These are intentionally constrained.
#
# They create/observe simple security-relevant events without
# turning this proof-of-concept into an exploitation framework.
#
# ============================================================


def tool_system_status(client):

    command = "uptime"

    output = ssh_command(
        client,
        command
    )

    return command, output


def tool_recent_ssh_logs(client):

    command = (
        "journalctl -u ssh "
        "--no-pager -n 30"
    )

    output = ssh_command(
        client,
        command
    )

    return command, output


def tool_connections(client):

    command = "ss -tn"

    output = ssh_command(
        client,
        command
    )

    return command, output


def tool_processes(client):

    command = (
        "ps -eo pid,comm "
        "--sort=-pid | head -20"
    )

    output = ssh_command(
        client,
        command
    )

    return command, output


# ============================================================
# SYNTHETIC ATTACKER ACTIONS
# ============================================================
#
# For the first experiment we want KNOWN ground truth.
#
# The attacker therefore selects from controlled behavioral
# scenarios.
#
# This gives us automatically generated labels.
#
# Later these can be replaced with actual LLM decisions.
#
# ============================================================

ATTACK_ACTIONS = [

    "normal_activity",

    "recon_activity",

    "authentication_probe",

    "repeated_authentication_probe"
]


def attacker_choose_action():

    return random.choice(
        ATTACK_ACTIONS
    )


# ============================================================
# CONTROLLED ATTACKER BEHAVIOR
# ============================================================
#
# Instead of attacking an external machine, the experiment
# creates controlled behavioral events inside the lab.
#
# ============================================================


def attacker_execute(client, action):

    if action == "normal_activity":

        command = "echo NORMAL_ACTIVITY"

        output = ssh_command(
            client,
            command
        )

        suspicious = 0


    elif action == "recon_activity":

        command = "ss -tn"

        output = ssh_command(
            client,
            command
        )

        suspicious = 1


    elif action == "authentication_probe":

        command = (
            "journalctl -u ssh "
            "--no-pager -n 10"
        )

        output = ssh_command(
            client,
            command
        )

        suspicious = 1


    elif action == "repeated_authentication_probe":

        command = (
            "journalctl -u ssh "
            "--no-pager -n 30"
        )

        output = ssh_command(
            client,
            command
        )

        suspicious = 1


    else:

        command = "echo UNKNOWN"

        output = ssh_command(
            client,
            command
        )

        suspicious = 0


    return command, output, suspicious


# ============================================================
# DEFENDER OBSERVATION
# ============================================================

def defender_observe(client):

    ssh_cmd, ssh_logs = tool_recent_ssh_logs(
        client
    )

    conn_cmd, connections = tool_connections(
        client
    )

    proc_cmd, processes = tool_processes(
        client
    )

    observation = {

        "ssh_log_length":
            len(ssh_logs),

        "connection_length":
            len(connections),

        "process_length":
            len(processes),

        "failed_count":
            ssh_logs.lower().count("failed"),

        "invalid_count":
            ssh_logs.lower().count("invalid"),

        "accepted_count":
            ssh_logs.lower().count("accepted"),

        "ssh_count":
            ssh_logs.lower().count("ssh")
    }

    return observation


# ============================================================
# FEATURE EXTRACTION
# ============================================================
#
# IMPORTANT:
#
# This is intentionally simple.
#
# The purpose is NOT to claim this is the world's best
# intrusion detection representation.
#
# The purpose is to demonstrate:
#
#       GENERATED DATA
#              |
#              V
#       LEARNED POLICY
#              |
#              V
#        BETTER DECISIONS
#
# ============================================================


def make_features(
    observation,
    attacker_action
):

    features = [

        observation[
            "ssh_log_length"
        ] / 5000.0,

        observation[
            "connection_length"
        ] / 1000.0,

        observation[
            "process_length"
        ] / 1000.0,

        observation[
            "failed_count"
        ] / 20.0,

        observation[
            "invalid_count"
        ] / 20.0,

        observation[
            "accepted_count"
        ] / 20.0,

        observation[
            "ssh_count"
        ] / 50.0,

        int(
            attacker_action ==
            "normal_activity"
        ),

        int(
            attacker_action ==
            "recon_activity"
        ),

        int(
            attacker_action ==
            "authentication_probe"
        ),

        int(
            attacker_action ==
            "repeated_authentication_probe"
        )
    ]

    return features


# ============================================================
# DATA COLLECTION
# ============================================================


def run_episode(
    client,
    episode_number
):

    episode_id = str(
        uuid.uuid4()
    )

    attacker_action = (
        attacker_choose_action()
    )


    # --------------------------------------------------------
    # STEP 1
    # ATTACKER ACTS
    # --------------------------------------------------------

    command, output, suspicious = (
        attacker_execute(
            client,
            attacker_action
        )
    )


    log_event(

        episode_id=episode_id,

        step=1,

        agent_role="attacker",

        scenario=attacker_action,

        observation="lab environment",

        decision=attacker_action,

        tool_selected="ssh_command",

        tool_input=attacker_action,

        command=command,

        command_output=output,

        environment_state=
            "attacker action completed",

        other_agent_previous_action=
            "none",

        success=1,

        reward=1,

        next_action=
            "defender observation"
    )


    # --------------------------------------------------------
    # STEP 2
    # DEFENDER OBSERVES ENVIRONMENT
    # --------------------------------------------------------

    observation = defender_observe(
        client
    )


    # --------------------------------------------------------
    # Ground truth
    #
    # 0 = benign
    # 1 = suspicious
    #
    # This is known because the experiment controller knows
    # which behavior was generated.
    # --------------------------------------------------------

    label = suspicious


    features = make_features(
        observation,
        attacker_action
    )


    # --------------------------------------------------------
    # Dataset-specific training record.
    # --------------------------------------------------------

    log_event(

        episode_id=episode_id,

        step=2,

        agent_role="defender",

        scenario=attacker_action,

        observation=json.dumps(
            observation
        ),

        decision="observe",

        tool_selected=
            "security_telemetry",

        tool_input=json.dumps(
            features
        ),

        command=
            "journalctl + ss + ps",

        command_output=json.dumps(
            observation
        ),

        environment_state=
            "telemetry collected",

        other_agent_previous_action=
            attacker_action,

        success=1,

        reward=1,

        next_action=
            "classify_behavior"
    )


    return {

        "episode_id":
            episode_id,

        "scenario":
            attacker_action,

        "features":
            features,

        "label":
            label
    }


# ============================================================
# GENERATE DATASET
# ============================================================


def generate_dataset():

    print()
    print("=" * 60)
    print("AUTONOMOUS CYBER DATA GENERATION")
    print("=" * 60)

    client = connect_ssh()

    training_records = []

    try:

        for episode in range(
            NUM_EPISODES
        ):

            record = run_episode(
                client,
                episode
            )

            training_records.append(
                record
            )

            print(
                f"Episode "
                f"{episode + 1:4d}/"
                f"{NUM_EPISODES}"
                f"   "
                f"{record['scenario']}"
            )

            time.sleep(0.05)

    finally:

        client.close()


    return training_records


# ============================================================
# SAVE BEHAVIORAL DATASET
# ============================================================


def save_behavior_dataset():

    df = pd.DataFrame(
        behavior_log
    )

    df.to_csv(
        DATASET_CSV,
        index=False
    )


    with open(
        DATASET_JSONL,
        "w",
        encoding="utf-8"
    ) as f:

        for row in behavior_log:

            f.write(
                json.dumps(row)
                + "\n"
            )


    print()
    print(
        "Behavioral events:",
        len(df)
    )

    print(
        "Saved:",
        DATASET_CSV
    )

    print(
        "Saved:",
        DATASET_JSONL
    )


# ============================================================
# SIMPLE DEFENDER POLICY NETWORK
# ============================================================
#
# We deliberately keep this tiny.
#
# Dataset contribution remains the focus.
#
# ============================================================


class DefenderPolicy(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                8
            ),

            nn.ReLU(),

            nn.Linear(
                8,
                2
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(x)


# ============================================================
# PREPARE TRAIN / TEST DATA
# ============================================================


def prepare_data(records):

    X = np.array(
        [
            r["features"]
            for r in records
        ],
        dtype=np.float32
    )


    y = np.array(
        [
            r["label"]
            for r in records
        ],
        dtype=np.int64
    )


    indices = np.arange(
        len(X)
    )

    np.random.shuffle(
        indices
    )


    split = int(
        0.80 * len(X)
    )


    train_indices = (
        indices[:split]
    )

    test_indices = (
        indices[split:]
    )


    X_train = torch.tensor(
        X[train_indices]
    )

    y_train = torch.tensor(
        y[train_indices]
    )


    X_test = torch.tensor(
        X[test_indices]
    )

    y_test = torch.tensor(
        y[test_indices]
    )


    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# ACCURACY
# ============================================================


def accuracy(
    model,
    X,
    y
):

    model.eval()

    with torch.no_grad():

        logits = model(X)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct = (
            predictions == y
        ).float()

        return (
            correct.mean().item()
        )


# ============================================================
# RANDOM BASELINE
# ============================================================


def random_baseline(y):

    predictions = torch.randint(
        0,
        2,
        y.shape
    )

    return (
        (
            predictions == y
        )
        .float()
        .mean()
        .item()
    )


# ============================================================
# MAJORITY BASELINE
# ============================================================


def majority_baseline(
    y_train,
    y_test
):

    counts = torch.bincount(
        y_train
    )

    majority_class = torch.argmax(
        counts
    )

    predictions = torch.full_like(
        y_test,
        majority_class
    )

    return (
        (
            predictions == y_test
        )
        .float()
        .mean()
        .item()
    )


# ============================================================
# TRAIN MODEL
# ============================================================


def train_model(
    model,
    X_train,
    y_train,
    epochs=200
):

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )

    criterion = (
        nn.CrossEntropyLoss()
    )


    for epoch in range(
        epochs
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            X_train
        )

        loss = criterion(
            logits,
            y_train
        )

        loss.backward()

        optimizer.step()


        if (
            epoch + 1
        ) % 25 == 0:

            print(
                f"Epoch "
                f"{epoch + 1:3d}"
                f"   "
                f"Loss = "
                f"{loss.item():.4f}"
            )


# ============================================================
# CONFUSION MATRIX
# ============================================================


def confusion_matrix(
    model,
    X,
    y
):

    model.eval()

    with torch.no_grad():

        predictions = torch.argmax(
            model(X),
            dim=1
        )


    TP = int(
        (
            (predictions == 1)
            &
            (y == 1)
        ).sum()
    )

    TN = int(
        (
            (predictions == 0)
            &
            (y == 0)
        ).sum()
    )

    FP = int(
        (
            (predictions == 1)
            &
            (y == 0)
        ).sum()
    )

    FN = int(
        (
            (predictions == 0)
            &
            (y == 1)
        ).sum()
    )


    return TP, TN, FP, FN


# ============================================================
# PRECISION / RECALL / F1
# ============================================================


def classification_metrics(
    TP,
    TN,
    FP,
    FN
):

    precision = (
        TP /
        (TP + FP + 1e-8)
    )

    recall = (
        TP /
        (TP + FN + 1e-8)
    )

    f1 = (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
            + 1e-8
        )
    )

    detection_rate = recall

    false_positive_rate = (
        FP /
        (
            FP
            + TN
            + 1e-8
        )
    )


    return (
        precision,
        recall,
        f1,
        detection_rate,
        false_positive_rate
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================


def main():

    print()
    print("=" * 60)
    print("CYBER AGENT BEHAVIORAL DATASET EXPERIMENT")
    print("=" * 60)


    # --------------------------------------------------------
    # PART 1
    # Automatically generate behavioral data
    # --------------------------------------------------------

    records = generate_dataset()


    # --------------------------------------------------------
    # PART 2
    # Save complete trajectory dataset
    # --------------------------------------------------------

    save_behavior_dataset()


    # --------------------------------------------------------
    # PART 3
    # Prepare machine-learning dataset
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test

    ) = prepare_data(
        records
    )


    print()
    print("=" * 60)
    print("DATASET")
    print("=" * 60)

    print(
        "Training examples:",
        len(X_train)
    )

    print(
        "Held-out examples:",
        len(X_test)
    )


    # --------------------------------------------------------
    # PART 4
    # Baselines
    # --------------------------------------------------------

    random_acc = (
        random_baseline(
            y_test
        )
    )


    majority_acc = (
        majority_baseline(
            y_train,
            y_test
        )
    )


    print()
    print("=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)

    print(
        f"Random policy accuracy: "
        f"{random_acc * 100:.2f}%"
    )

    print(
        f"Majority baseline accuracy: "
        f"{majority_acc * 100:.2f}%"
    )


    # --------------------------------------------------------
    # PART 5
    # Create defender policy
    # --------------------------------------------------------

    input_size = (
        X_train.shape[1]
    )

    model = DefenderPolicy(
        input_size
    )


    # --------------------------------------------------------
    # PART 6
    # Train using AUTOMATICALLY GENERATED DATA
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "TRAINING DEFENDER USING "
        "GENERATED DATA"
    )
    print("=" * 60)


    train_model(
        model,
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PART 7
    # Evaluate on HELD-OUT generated episodes
    # --------------------------------------------------------

    trained_acc = accuracy(
        model,
        X_test,
        y_test
    )


    TP, TN, FP, FN = (
        confusion_matrix(
            model,
            X_test,
            y_test
        )
    )


    (
        precision,
        recall,
        f1,
        detection_rate,
        false_positive_rate

    ) = classification_metrics(
        TP,
        TN,
        FP,
        FN
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)


    print(
        f"Trained accuracy: "
        f"{trained_acc * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 score: "
        f"{f1:.4f}"
    )

    print(
        f"Detection rate: "
        f"{detection_rate * 100:.2f}%"
    )

    print(
        f"False positive rate: "
        f"{false_positive_rate * 100:.2f}%"
    )


    print()
    print("Confusion Matrix")
    print("----------------")

    print(
        "TP:",
        TP
    )

    print(
        "TN:",
        TN
    )

    print(
        "FP:",
        FP
    )

    print(
        "FN:",
        FN
    )


    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    improvement_random = (
        trained_acc
        - random_acc
    )


    improvement_majority = (
        trained_acc
        - majority_acc
    )


    print()
    print("=" * 60)
    print("PROOF-OF-CONCEPT RESULT")
    print("=" * 60)


    print(
        f"Random baseline: "
        f"{random_acc * 100:.2f}%"
    )

    print(
        f"Majority baseline: "
        f"{majority_acc * 100:.2f}%"
    )

    print(
        f"Trained policy: "
        f"{trained_acc * 100:.2f}%"
    )


    print()

    print(
        "Improvement over random: "
        f"{improvement_random * 100:+.2f} "
        "percentage points"
    )

    print(
        "Improvement over majority: "
        f"{improvement_majority * 100:+.2f} "
        "percentage points"
    )


    print()
    print("=" * 60)

    if trained_acc > max(
        random_acc,
        majority_acc
    ):

        print(
            "RESULT: GENERATED DATA "
            "PROVIDED A LEARNING BENEFIT."
        )

    else:

        print(
            "RESULT: NO CLEAR LEARNING "
            "BENEFIT YET."
        )

    print("=" * 60)


    # --------------------------------------------------------
    # Save trained policy
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        "defender_policy.pt"
    )

    print()
    print(
        "Saved trained policy: "
        "defender_policy.pt"
    )

    print()
    print("Experiment complete.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()


```







