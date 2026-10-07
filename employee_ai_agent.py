"""
Small Employee Agent (LangGraph + Groq free key)

Install:  pip install -U langchain-groq langchain-core langgraph
Key:      set GROQ_API_KEY in your terminal (free key: console.groq.com/keys)
Run:      python employee_ai_agent.py
"""

import os
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

# 1) Our data: a simple list of employees (no database needed)
employees = [
    {"name": "Amit",   "department": "IT",    "salary": 60000},
    {"name": "Priya",  "department": "HR",    "salary": 45000},
    {"name": "Rahul",  "department": "IT",    "salary": 75000},
    {"name": "Sneha",  "department": "Sales", "salary": 50000},
]


# 2) Tools: small functions the AI can call. The text inside """ """ tells the AI what each one does.
@tool
def list_employees(department: str = "") -> str:
    """List all employees. Give a department (IT, HR, Sales) to filter, or leave empty for everyone."""
    result = [e for e in employees if not department or e["department"].lower() == department.lower()]
    return str(result) if result else "No employees found."


@tool
def get_employee(name: str) -> str:
    """Get details of one employee by name."""
    for e in employees:
        if e["name"].lower() == name.lower():
            return str(e)
    return "Employee not found."


@tool
def average_salary(department: str) -> str:
    """Find the average salary of a department."""
    salaries = [e["salary"] for e in employees if e["department"].lower() == department.lower()]
    if not salaries:
        return "No employees in that department."
    return f"Average salary in {department}: {sum(salaries) / len(salaries):.0f}"


@tool
def give_raise(name: str, percent: float) -> str:
    """Increase an employee's salary by a percentage."""
    for e in employees:
        if e["name"].lower() == name.lower():
            e["salary"] = round(e["salary"] * (1 + percent / 100))
            return f"{e['name']}'s new salary is {e['salary']}"
    return "Employee not found."


tools = [list_employees, get_employee, average_salary, give_raise]

# 3) The AI model (Groq). Model names change, so you can override with GROQ_MODEL.
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)
llm_with_tools = llm.bind_tools(tools)  # tell the AI which tools it may use


# 4) The "agent" step: the AI reads the chat and decides: answer, or call a tool?
def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 5) Build the graph (the flow):  START -> agent -> (tool -> agent)... -> answer
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))                 # this node runs the chosen tool
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)    # tool needed? go to tools, else finish
graph.add_edge("tools", "agent")                         # after a tool, go back to the AI
app = graph.compile()

# 6) Chat loop
if __name__ == "__main__":
    print("Employee agent ready. Type 'exit' to quit.")
    print("Try: 'List IT employees', 'Average salary in HR', 'Give Sneha a 10% raise'\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        try:
            result = app.invoke({"messages": [HumanMessage(content=question)]})
            print("Agent:", result["messages"][-1].content, "\n")
        except Exception as e:
            print("Error:", e, "\n")