from typing import Dict, List, Tuple, Any
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from pydantic import BaseModel, Field
import json
from datetime import datetime

class Scheme(BaseModel):
    title: str = Field(description="Title of the scheme")
    description: str = Field(description="Description of the scheme")
    eligibility: str = Field(description="Eligibility criteria")
    benefits: str = Field(description="Benefits of the scheme")
    source: str = Field(description="Source of the scheme (Central/State)")
    state: str = Field(description="State name if applicable")
    last_updated: str = Field(description="Last updated date")

class SchemeAgent:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        self.schemes: List[Scheme] = []
        
    def search_schemes(self, query: str) -> List[Dict[str, Any]]:
        """Search for agriculture-related schemes using DuckDuckGo"""
        search_query = f"{query} agriculture scheme government"
        results = self.search.run(search_query)
        return self._parse_search_results(results)
    
    def _parse_search_results(self, results: str) -> List[Dict[str, Any]]:
        """Parse search results into structured scheme data"""
        schemes = []
        try:
            # Split results into individual scheme entries
            entries = results.split("\n\n")
            for entry in entries:
                if "scheme" in entry.lower():
                    # Determine if it's a central or state scheme
                    is_central = any(keyword in entry.lower() for keyword in ["central", "pm", "prime minister", "union"])
                    is_state = any(keyword in entry.lower() for keyword in ["state", "government of"])
                    
                    scheme = {
                        "title": entry.split("\n")[0] if entry.split("\n") else "Unknown",
                        "description": entry,
                        "eligibility": self._extract_eligibility(entry),
                        "benefits": self._extract_benefits(entry),
                        "source": "Central Government" if is_central else "State Government" if is_state else "Government",
                        "state": self._extract_state(entry) if is_state else "All India",
                        "last_updated": datetime.now().strftime("%Y-%m-%d")
                    }
                    schemes.append(scheme)
        except Exception as e:
            print(f"Error parsing results: {e}")
        return schemes
    
    def _extract_eligibility(self, text: str) -> str:
        """Extract eligibility criteria from text"""
        # Look for common eligibility indicators
        eligibility_indicators = ["eligible", "eligibility", "qualify", "qualification", "requirements"]
        sentences = text.split(".")
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in eligibility_indicators):
                return sentence.strip()
        return "To be determined"
    
    def _extract_benefits(self, text: str) -> str:
        """Extract benefits from text"""
        # Look for common benefit indicators
        benefit_indicators = ["benefit", "benefits", "provide", "offers", "gives", "assistance"]
        sentences = text.split(".")
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in benefit_indicators):
                return sentence.strip()
        return "To be determined"
    
    def _extract_state(self, text: str) -> str:
        """Extract state name from text"""
        # List of Indian states
        states = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
                 "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
                 "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Madhya Pradesh", "Maharashtra",
                 "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
                 "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
                 "Uttar Pradesh", "Uttarakhand", "West Bengal"]
        
        for state in states:
            if state.lower() in text.lower():
                return state
        return "All India"

def create_scheme_graph() -> Graph:
    """Create a LangGraph for processing agriculture schemes"""
    # Define the state
    class AgentState(BaseModel):
        query: str
        schemes: List[Dict[str, Any]]
        current_step: str
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Define nodes
    def search_node(state: AgentState) -> AgentState:
        agent = SchemeAgent()
        schemes = agent.search_schemes(state.query)
        state.schemes = schemes
        state.current_step = "search_complete"
        return state
    
    def process_node(state: AgentState) -> AgentState:
        # Process and enrich the schemes with additional information
        for scheme in state.schemes:
            # Add more processing logic here
            pass
        state.current_step = "processing_complete"
        return state
    
    # Add nodes to the graph
    workflow.add_node("search", search_node)
    workflow.add_node("process", process_node)
    
    # Define edges
    workflow.add_edge("search", "process")
    
    # Set entry point
    workflow.set_entry_point("search")
    
    return workflow.compile()

def get_agriculture_schemes(query: str = "agriculture schemes") -> List[Dict[str, Any]]:
    """Main function to get agriculture schemes"""
    # Create and run the graph
    graph = create_scheme_graph()
    
    # Initialize state
    initial_state = {
        "query": query,
        "schemes": [],
        "current_step": "start"
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    return final_state["schemes"]

def process_schemes_query() -> Dict[str, Any]:
    """Process schemes query and return formatted response for API"""
    schemes = get_agriculture_schemes()
    
    # Separate central and state schemes
    central_schemes = [scheme for scheme in schemes if scheme["source"] == "Central Government"]
    state_schemes = [scheme for scheme in schemes if scheme["source"] == "State Government"]
    
    # Generate analysis
    analysis = f"""
    Total Schemes Found: {len(schemes)}
    Central Government Schemes: {len(central_schemes)}
    State Government Schemes: {len(state_schemes)}
    
    Key Findings:
    - Most schemes focus on {schemes[0]["title"] if schemes else "No schemes found"}
    - Schemes are available across {len(set(scheme["state"] for scheme in schemes))} states
    - Latest update: {datetime.now().strftime("%Y-%m-%d")}
    """
    
    return {
        "central_schemes_count": len(central_schemes),
        "state_schemes_count": len(state_schemes),
        "central_schemes": central_schemes,
        "state_schemes": state_schemes,
        "analysis": analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    # Example usage
    schemes = get_agriculture_schemes()
    print(f"Found {len(schemes)} schemes:")
    for scheme in schemes:
        print(f"\nTitle: {scheme['title']}")
        print(f"Description: {scheme['description'][:200]}...")
        print(f"Source: {scheme['source']}")
        print(f"State: {scheme['state']}")
        print(f"Eligibility: {scheme['eligibility']}")
        print(f"Benefits: {scheme['benefits']}")
        print("-" * 80) 