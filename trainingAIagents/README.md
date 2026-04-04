## AI Agents and Training 


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











