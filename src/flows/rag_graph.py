import functools
from typing import Any, Dict
from langgraph.graph import StateGraph, START, END

from src.core.state import AgentState
from src.agents.nodes import retrieve_node, generate_node
from src.agents.routers import route_question
from src.utils.factory import ComponentFactory

def build_rag_graph(config: Dict[str, Any]):
    """
    The central orchestrator. Reads the config, instantiates components via the factory,
    injects them into the nodes, and compiles the LangGraph state machine.
    """
    # 1. Instantiate the components using your "boom, done" factory
    llm = ComponentFactory.create_instance(config["llm"])
    retriever = ComponentFactory.create_instance(config["vector_db"])

    # 2. Bind the concrete components to our pure node functions
    # This locks the 'retriever' and 'llm' arguments so LangGraph only has to pass 'state'
    bound_retrieve = functools.partial(retrieve_node, retriever=retriever)
    bound_generate = functools.partial(generate_node, llm=llm)

    # 3. Initialize the StateGraph with our strict schema
    workflow = StateGraph(AgentState)

    # 4. Add the nodes to the graph
    workflow.add_node("retrieve", bound_retrieve)
    workflow.add_node("generate", bound_generate)

    # 5. Define the edges and routing logic
    # The router decides whether to go to 'retrieve' or straight to 'generate'
    workflow.add_conditional_edges(
            START,
            route_question,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )
    
    # If we route to the database, we must generate an answer afterward
    workflow.add_edge("retrieve", "generate")
    
    # After generating the final response, the cycle ends
    workflow.add_edge("generate", END)

    # 6. Compile and return the executable engine
    return workflow.compile()