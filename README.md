# Agents AI: Tools and media 

* ( Agents )
* AGS - The Future of Gen AI: Tools and media 
* AIML
* AI for Business Intelligence and Analytics - Integration of AI tools and practical application
* link
* https://agents4science.github.io/curriculum.html

## Karpathy - Autosearch

* https://github.com/karpathy/autoresearch

## CloudAI

* Replicate is here 
* https://github.com/rcalix1/CloudComputing

## Andrew Ng - AI Agents

* https://github.com/andrewyng/aisuite/tree/main
* https://github.com/andrewyng/aisuite/blob/main/examples/agents/snake_game_generator.ipynb
* https://github.com/andrewyng/aisuite/blob/main/examples/agents/world_weather_dashboard.ipynb


# AI Agents Masterclass — CrewAI and SmolAgents

Welcome to the **AI Agents Masterclass**, where we explore how to build and orchestrate intelligent agents using **CrewAI** and **SmolAgents**.

---

# AI Agents for Business Intelligence and Automation



---

## 🎯 Overview

This session explores how AI agents can automate business intelligence and analytics workflows using open-source frameworks. Through live Python demos and analogies, participants learn how to:

* Understand what AI agents are and how they differ from chatbots
* Set up **local LLMs** with **Ollama** (no cloud cost)
* Use **CrewAI** to structure AI agents, tasks, and tools
* Deploy multi-agent workflows (e.g., researcher + writer)
* Connect AI agents with tools like web scraping and file operations

---

## 🧩 Core Concepts

### 1. **AI Agents Defined**

AI agents are autonomous systems that use large language models (LLMs) to perform goal-oriented tasks through reasoning, collaboration, and tool use.

> **Analogy:** Like a marine biologist exploring the ocean — the LLM is the biologist, the code wrapper (CrewAI) is the wetsuit, and the tools (scrapers, search engines) are the instruments.

---

### 2. **Real-World Context**

Modern enterprise platforms like **Slack (Salesforce)** and **Microsoft Teams** are integrating AI agents into daily workflows — introducing the idea of **digital coworkers** that can assist, summarize, and automate communication tasks.

> Example: A fifth 'digital team member' in Slack who can research, summarize, and deliver insights on command.

---

## ⚙️ Tools and Frameworks

| Tool / Library                | Purpose                                       |
| ----------------------------- | --------------------------------------------- |
| **Ollama**                    | Runs open-source LLMs like Llama 3.2 locally  |
| **CrewAI**                    | Framework for defining and managing AI agents |
| **SmolAgents (HuggingFace)**  | Alternative agent orchestration library       |
| **Llama 3.2**                 | Open-source LLM by Meta, used via Ollama      |
| **BrowserTool / ScraperTool** | Allows agents to retrieve data from the web   |

---

## 🧠 Session Flow

### **1. Setting Up Ollama**

* Download Ollama for Windows/Mac/Linux from [ollama.com/download](https://ollama.com/download)
* Run: `ollama run llama3.2`
* Acts as a **local API** (localhost:11434)
* Use this to avoid cloud fees and latency

---

### **2. Installing CrewAI**

```bash
pip install crewai
pip install crewai-tools
```

Import in Python:

```python
from crewai import Agent, Task, Crew
from crewai.tools import ScrapeWebsiteTool
```

---

### **3. Defining an Agent**

```python
researcher_agent = Agent(
    role="Researcher",
    goal="Find and summarize the latest AI news",
    backstory="You are an AI analyst providing insights to the business.",
    llm=Llm(model="llama3.2", base_url="http://localhost:11434")
)
```

---

### **4. Creating a Task**

```python
research_task = Task(
    description="Conduct comprehensive research on the latest AI developments.",
    expected_output="A concise report highlighting key AI breakthroughs.",
    agent=researcher_agent
)
```

---

### **5. Launching a Crew**

```python
my_crew = Crew(agents=[researcher_agent], tasks=[research_task])
result = my_crew.kickoff()
print(result)
```

This creates and runs a single AI agent performing a defined task using Llama 3.2 locally.

---

## 🤖 Multi-Agent Example

```python
researcher = Agent(role="Researcher", goal="Gather information on black holes", ...)
writer = Agent(role="Writer", goal="Create a humorous blog post about black holes", ...)

research_task = Task(description="Find info on black holes", agent=researcher)
writing_task = Task(description="Write a funny blog post based on the research", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task], process="sequential")
result = crew.kickoff(inputs={"topic": "black holes"})
```

**Output:**

> *"The Mysterious World of Black Holes: A Humorous Guide"* — complete with playful analogies and witty commentary.

---

## 🔧 Tool Integration Example

Use the `ScrapeWebsiteTool` to give your agent access to live web data:

```python
from crewai.tools import ScrapeWebsiteTool
scraper = ScrapeWebsiteTool()
context = scraper.run("https://en.wikipedia.org/wiki/Artificial_intelligence")

agent = Agent(role="Educator", goal="Explain NLP using context from the web", llm=Llm(model="llama3.2"))
task = Task(description="Based on the context, explain what NLP is.", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
crew.kickoff(inputs={"context": context})
```

---

## 💬 Common Questions

| Question                              | Insight                                                                    |
| ------------------------------------- | -------------------------------------------------------------------------- |
| **Can we use ChatGPT instead?**       | Yes — replace Ollama with `OpenAI` API + token.                            |
| **Can agents read files or syslogs?** | Yes, using file reader tools. Provide context as input.                    |
| **Are tools secure?**                 | Some (like `eval`) pose security risks. Always sanitize inputs.            |
| **Can multiple LLMs run?**            | You can create multiple Ollama instances or agents using different models. |
| **Do prompts still matter?**          | Absolutely — role, goal, and backstory are crucial.                        |

---

## 🧩 Key Takeaways

* AI agents represent **structured automation**, not just chatbots.
* **Local-first AI** (Ollama + CrewAI) makes enterprise experimentation feasible.
* **Multi-agent setups** mimic team workflows — researcher, writer, analyst, etc.
* **Tools** (web, file, search) make agents capable of interacting with real-world data.
* **Prompt engineering** remains the heart of quality agent behavior.

---

## 💡 Final Thoughts

AI agents are the next layer of applied AI — bringing reasoning, delegation, and collaboration to automation. Frameworks like CrewAI make it accessible to small teams and educators alike.

> “The book on AI agents is still being written — but you can already start contributing chapters.”



---

## 📘 Table of Contents

1. [What Are AI Agents?](#what-are-ai-agents)
2. [Key Concepts](#key-concepts)
3. [Agent Frameworks Overview](#agent-frameworks-overview)
4. [CrewAI Basics](#crewai-basics)
5. [SmolAgents Basics](#smolagents-basics)
6. [Real-World Use Cases](#real-world-use-cases)
7. [Design Patterns for Agents](#design-patterns-for-agents)
8. [Advanced Topics](#advanced-topics)
9. [Resources](#resources)

---

## 🧬 What Are AI Agents?

**AI Agents** are autonomous programs that perceive their environment, reason about it, and take actions to achieve specific goals.

In our context, they:

* Use **LLMs (like GPT-4)** for reasoning.
* Interact via **tools, APIs, or prompts**.
* Operate **individually** or in **multi-agent systems (MAS)**.

---

## 🔑 Intuitions

* Marine biologist
* LLM and text based game

---

## 🔑 Key Concepts

| Concept          | Definition                                                         |
| ---------------- | ------------------------------------------------------------------ |
| **Agent**        | A program that can observe, decide, and act to fulfill a goal.     |
| **Tool**         | An external API, function, or service the agent can call.          |
| **LLM**          | The language model (e.g. GPT-4) that powers the agent's reasoning. |
| **Memory**       | Stores past interactions or decisions for context-aware reasoning. |
| **Environment**  | The space in which agents act (can be abstract or interactive).    |
| **Autonomy**     | Degree to which agents act without human intervention.             |
| **Coordination** | How multiple agents collaborate or negotiate tasks.                |

---

## 🛠 Agent Frameworks Overview

### ✅ CrewAI

* Designed for **multi-agent collaboration**.
* Inspired by a "movie crew" metaphor: Director, Researcher, Writer, etc.
* Focus on **structured agent roles** and **delegation**.

### ✅ SmolAgents

* Lightweight, **minimalist agent systems**.
* Emphasizes simplicity and chain-of-thought interactions.
* Good for **local-first**, single-agent logic.

---

## 👥 CrewAI Basics

```python
from crewai import Agent, Crew
```

* **Agents** are defined by roles, goals, and tools.
* **Crew** handles orchestration and task management.

**Example Roles:**

* `ResearchAgent`: Fetches data via search APIs.
* `WriterAgent`: Generates structured summaries.

**Key Concepts:**

* Task delegation
* Parallelism
* Agent communication

---

## 🧱 SmolAgents Basics

```python
from smolagents import SmolAgent
```

* Build **single-purpose agents** that use **functions as tools**.
* Focus on **reasoning chains** and **context prompts**.

**Example Workflow:**

* Give agent a goal → Agent reasons through steps → Calls functions.

---

## 🔍 Real-World Use Cases

* Research assistants
* Code generation and refactoring
* Data summarization and extraction
* Automated proposal drafting
* AI-driven agents for customer support or product development

---

## ↺ Design Patterns for Agents

| Pattern              | Description                               |
| -------------------- | ----------------------------------------- |
| **Reflex Agent**     | Reacts to current input (stateless)       |
| **Utility Agent**    | Chooses actions based on expected utility |
| **Planner Agent**    | Plans a sequence of steps to meet a goal  |
| **Team Agents**      | Coordinate tasks among multiple agents    |
| **Chain-of-Thought** | Use intermediate reasoning steps          |

---

## 🚀 Advanced Topics

* Tool execution tracing
* Memory-enabled agents (e.g., via vector DBs)
* Long-term planning
* Autonomy toggling & human-in-the-loop
* Fine-tuning tools and prompts
* Agent evaluation frameworks

---

## 📚 Resources

* [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
* [SmolAgents GitHub](https://github.com/smol-ai/smol-agents)
* [LangChain Agents](https://docs.langchain.com/docs/components/agents/)
* [AutoGPT / BabyAGI](https://github.com/Torantulino/Auto-GPT) for historical reference
* [ReAct Prompting](https://arxiv.org/abs/2210.03629) — foundation for reasoning+acting


## Agents AI - Definitions

* https://huggingface.co/blog/smolagents
* https://www.anthropic.com/research/building-effective-agents
* https://huggingface.co/docs/smolagents/en/conceptual_guides/intro_agents

## How to Build Effective AI Agents in Pure Python 

* https://www.youtube.com/watch?v=bZzyPscbtI8
* 

## Options

* [crewai.com](https://www.crewai.com)
* not diamond .com

## Repos of AI agent related subjects such as tools, etc

* https://github.com/ALucek?tab=repositories&q=&type=&language=&sort=stargazers

## Gemini AI

* [cook book](https://github.com/google-gemini/cookbook)


## 👋 About

Maintained by [Ricardo Calix](https://www.rcalix.com), author and AI consultant. This repository supports interactive workshops and masterclasses on **AI Agents**. Contact: rcalix@rcalix.com

## 📘 Featured Book 

<a href="https://amzn.to/3QmKKwC" target="_blank">
  <img src="https://m.media-amazon.com/images/I/71F2QLFMCFL._SL1233_.jpg" alt="Books" width="300" style="border-radius:10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);" />
</a>

➡️ **[Grab your copy on Amazon »](https://amzn.to/3QmKKwC)**

---

## ⚠️ Disclaimer

- 🤖 Portions of this content were generated or assisted by AI.
- 🔗 This post includes [Amazon affiliate links](https://amzn.to/3QmKKwC). Purchases made through them may earn a small commission at no extra cost to you.






