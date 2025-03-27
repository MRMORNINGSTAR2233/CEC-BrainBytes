from typing import Dict, List, Tuple, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
import requests
from datetime import datetime, timedelta
import json
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768",
    temperature=0.7
)

def fetch_news(query: str, days: int = 7) -> List[Dict]:
    """Fetch news articles using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            # Calculate the date range
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Search for news articles
            news_results = list(ddgs.news(
                query,
                region="wt-wt",
                time='d',  # Last day
                max_results=10
            ))
            
            # Format the results
            articles = []
            for result in news_results:
                article = {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "date": result.get("date", ""),
                    "source": result.get("source", ""),
                    "thumbnail": result.get("thumbnail", "")
                }
                articles.append(article)
            
            return articles
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def analyze_news(articles: List[Dict]) -> str:
    """Analyze news articles using Groq"""
    try:
        prompt = f"""Analyze these farm-related news articles and provide a comprehensive summary:
        {json.dumps(articles, indent=2)}
        
        Focus on:
        1. Key trends and patterns
        2. Impact on farmers and agriculture
        3. Market implications
        4. Policy changes
        5. Recommendations for stakeholders
        
        Format the response in a clear, structured manner with bullet points.
        """
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error analyzing news: {str(e)}")
        return "Error analyzing news articles."

# Define the agent nodes
def search_node(state: Dict) -> Dict:
    """Node for searching news"""
    query = state["query"]
    articles = fetch_news(query)
    state["articles"] = articles
    return state

def analyze_node(state: Dict) -> Dict:
    """Node for analyzing news"""
    analysis = analyze_news(state["articles"])
    state["analysis"] = analysis
    return state

def format_output_node(state: Dict) -> Dict:
    """Node for formatting final output"""
    output = {
        "query": state["query"],
        "articles_count": len(state["articles"]),
        "articles": state["articles"],
        "analysis": state["analysis"],
        "timestamp": datetime.now().isoformat()
    }
    state["output"] = output
    return state

# Create the graph
def create_farm_news_graph() -> Graph:
    workflow = StateGraph(StateType=Dict)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("format", format_output_node)
    
    # Add edges
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", "format")
    
    # Set entry point
    workflow.set_entry_point("search")
    
    return workflow.compile()

def process_news_query(query: str) -> Dict:
    """Process a news query and return results"""
    graph = create_farm_news_graph()
    
    initial_state = {
        "query": query,
        "articles": [],
        "analysis": "",
        "output": None
    }
    
    result = graph.invoke(initial_state)
    return result["output"]

def main():
    # Example query
    query = "agriculture farming crops market prices"
    result = process_news_query(query)
    
    # Print results
    print("\nFarm News Analysis:")
    print("=" * 50)
    print(f"Query: {result['query']}")
    print(f"Articles analyzed: {result['articles_count']}")
    print("\nAnalysis:")
    print(result['analysis'])

if __name__ == "__main__":
    main() 