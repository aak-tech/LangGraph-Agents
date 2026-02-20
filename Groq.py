from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities import SerpAPIWrapper
from typing import TypedDict, List, Optional
import os
from dotenv import load_dotenv

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

# --- Langfuse setup ---
try:
    Langfuse()
    langfuse = get_client()
    if langfuse.auth_check():
        print("Langfuse connected")
    else:
        print(" Langfuse auth failed — check your keys")
except Exception as e:
    print(f"Langfuse not connected: {e}")
    langfuse = None

# --- State ---
class AgentState(TypedDict):
    messages: List
    response: str
    search_results: Optional[str]   # stores web search output
    needs_search: bool              # flag to trigger search agent

# --- LLM ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# --- SerpAPI ---
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

# --- Node 1: Router — decides if web search is needed ---
def router_node(state: AgentState) -> AgentState:
    print("\n[Router] Deciding if search is needed...")

    last_message = state["messages"][-1].content

    handler = CallbackHandler()

    decision = llm.invoke(
        [
            SystemMessage(content="""You decide if a question needs a web search or not.
Reply with ONLY 'search' or 'chat'. Nothing else.
- Use 'search' for: current events, news, prices, weather, recent facts
- Use 'chat' for: casual talk, general knowledge, opinions, advice"""),
            HumanMessage(content=last_message)
        ],
        config={"callbacks": [handler]}
    )

    needs_search = "search" in decision.content.lower()
    state["needs_search"] = needs_search
    print(f"[Router] Decision: {'search' if needs_search else 'chat'}")
    return state

# --- Node 2: Web Search Agent ---
def search_node(state: AgentState) -> AgentState:
    print("\n[Search Agent] Fetching from web...")

    query = state["messages"][-1].content

    try:
        results = search.run(query)
        state["search_results"] = results
        print(f"[Search Agent] Got results: {results[:200]}...")
    except Exception as e:
        state["search_results"] = f"Search failed: {e}"
        print(f"[Search Agent] Error: {e}")

    return state

# --- Node 3: Chat Agent ---
def chat_node(state: AgentState) -> AgentState:
    print("\n[Chat Node] Thinking...")

    search_context = ""
    if state.get("search_results"):
        search_context = f"\n\nWeb search results:\n{state['search_results']}\n\nUse this to answer."

    messages = [
        SystemMessage(content=f"You are a chill, friendly assistant. Keep replies short and casual.{search_context}"),
        *state["messages"]
    ]

    handler = CallbackHandler()

    response = llm.invoke(
        messages,
        config={
            "callbacks": [handler],
            "metadata": {
                "langfuse_user_id": "aakash",
                "langfuse_session_id": "session_001",
                "langfuse_tags": ["groq", "casual-chat"]
            }
        }
    )

    state["messages"].append(AIMessage(content=response.content))
    state["response"] = response.content
    state["search_results"] = None  # reset for next call

    print(f"[Chat Node] Done: {response.content}")
    return state

# --- Routing Logic ---
def route_decision(state: AgentState) -> str:
    return "search" if state.get("needs_search") else "chat"

# --- Build Graph ---
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("search", search_node)
graph.add_node("chat", chat_node)

graph.set_entry_point("router")

graph.add_conditional_edges("router", route_decision, {
    "search": "search",
    "chat": "chat"
})

graph.add_edge("search", "chat")  # after search, always go to chat
graph.add_edge("chat", END)

app = graph.compile()

# --- Run ---
def chat(user_input: str):
    result = app.invoke({
        "messages": [HumanMessage(content=user_input)],
        "response": "",
        "search_results": None,
        "needs_search": False
    })
    print(f"\nAssistant: {result['response']}\n")
    print("-" * 50)

# --- Try it ---
chat("hey what's up?")                         
# chat("what's the latest news about AI today?")
# chat("what's the best way to learn python?")    
chat("what is the current price of bitcoin?")   
