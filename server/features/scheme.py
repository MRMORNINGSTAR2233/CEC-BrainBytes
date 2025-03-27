from typing import Dict, List, Tuple, Any, Optional
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.2-3b-preview",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

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
        self.llm = llm
        self.schemes: List[Scheme] = []
        
    def search_schemes(self, query: str) -> List[Dict[str, Any]]:
        """Search for agriculture-related schemes using DuckDuckGo"""
        search_query = f"{query} agriculture scheme government"
        results = self.search.run(search_query)
        return self._parse_search_results(results)
    
    def _parse_search_results(self, results: str) -> List[Dict[str, Any]]:
        """Parse search results into structured scheme data using LLM"""
        prompt = f"""
        Analyze these agriculture scheme search results and extract structured information:
        {results}
        
        For each scheme mentioned, extract:
        1. Title
        2. Description
        3. Eligibility criteria
        4. Benefits
        5. Source (Central/State)
        6. Applicable state
        
        Format the response as a JSON array of objects with these fields.
        """
        
        try:
            response = self.llm.invoke(prompt)
            schemes = json.loads(response.content)
            return schemes
        except Exception as e:
            print(f"Error parsing results with LLM: {e}")
            return self._fallback_parse(results)
    
    def _fallback_parse(self, results: str) -> List[Dict[str, Any]]:
        """Fallback parsing method if LLM parsing fails"""
        schemes = []
        try:
            entries = results.split("\n\n")
            for entry in entries:
                if "scheme" in entry.lower():
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
            print(f"Error in fallback parsing: {e}")
        return schemes

    def _extract_eligibility(self, text: str) -> str:
        """Extract eligibility criteria using LLM"""
        prompt = f"Extract eligibility criteria from this text: {text}"
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            return "To be determined"

    def _extract_benefits(self, text: str) -> str:
        """Extract benefits using LLM"""
        prompt = f"Extract benefits from this text: {text}"
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            return "To be determined"

    def _extract_state(self, text: str) -> str:
        """Extract state name using LLM"""
        prompt = f"Extract the Indian state name from this text: {text}"
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            return "All India"

def create_scheme_graph() -> Graph:
    """Create a LangGraph for processing agriculture schemes"""
    # Define the state
    class AgentState(BaseModel):
        query: str
        schemes: List[Dict[str, Any]]
        current_step: str
        error: Optional[str] = None
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Define nodes
    def search_node(state: AgentState) -> AgentState:
        try:
            agent = SchemeAgent()
            schemes = agent.search_schemes(state.query)
            state.schemes = schemes
            state.current_step = "search_complete"
            state.error = None
        except Exception as e:
            state.error = f"Error in search: {str(e)}"
            state.current_step = "error"
        return state
    
    def process_node(state: AgentState) -> AgentState:
        try:
            if state.error:
                return state
                
            agent = SchemeAgent()
            # Enrich schemes with additional information using LLM
            for scheme in state.schemes:
                # Add more detailed analysis using LLM
                analysis_prompt = f"""
                Analyze this agriculture scheme and provide additional insights:
                {json.dumps(scheme)}
                
                Focus on:
                1. Key benefits and impact
                2. Implementation challenges
                3. Success factors
                4. Recommendations for farmers
                """
                
                try:
                    response = agent.llm.invoke(analysis_prompt)
                    scheme["additional_analysis"] = response.content
                except:
                    scheme["additional_analysis"] = "Analysis not available"
            
            state.current_step = "processing_complete"
            state.error = None
        except Exception as e:
            state.error = f"Error in processing: {str(e)}"
            state.current_step = "error"
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