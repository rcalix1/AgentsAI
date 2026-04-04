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
