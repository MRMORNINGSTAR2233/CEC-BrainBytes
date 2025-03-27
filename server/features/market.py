from typing import Dict, List, Tuple, Any
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM with Groq
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # Using Mixtral model for better performance
    temperature=0,
    max_tokens=32768,
    top_p=0.95,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

class MarketScraper:
    def __init__(self):
        self.market_data = {}
        self.categories = {
            "crops": ["wheat", "rice", "cotton", "sugarcane", "maize", "soybean"],
            "vegetables": ["tomato", "potato", "onion", "cauliflower", "cabbage", "brinjal"],
            "fruits": ["apple", "banana", "orange", "mango", "grapes", "watermelon"],
            "fertilizers": ["urea", "dap", "npk", "potash", "calcium", "zinc"],
            "seeds": ["wheat_seeds", "rice_seeds", "cotton_seeds", "vegetable_seeds"],
            "pesticides": ["insecticide", "herbicide", "fungicide", "weedicide"],
            "equipment": ["tractor", "harvester", "irrigation_pump", "sprayer"],
            "livestock": ["cattle", "poultry", "sheep", "goat"]
        }
        
    def scrape_market_prices(self, item: str) -> Dict[str, Any]:
        """
        Scrape market prices for a given item from various sources
        """
        # Determine the category of the item
        category = self._get_item_category(item)
        
        # Example implementation - you would need to replace with actual market data sources
        # This is a placeholder that simulates market data with more realistic structure
        return {
            "item": item,
            "category": category,
            "current_price": self._get_simulated_price(item, category),
            "unit": self._get_unit(item, category),
            "market": self._get_market_name(category),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price_trend": self._get_simulated_trend(),
            "availability": self._get_simulated_availability(),
            "quality_grade": self._get_simulated_quality(),
            "supplier_info": self._get_simulated_supplier()
        }
    
    def _get_item_category(self, item: str) -> str:
        """Determine the category of an item"""
        for category, items in self.categories.items():
            if item.lower() in [i.lower() for i in items]:
                return category
        return "other"
    
    def _get_simulated_price(self, item: str, category: str) -> float:
        """Generate simulated price based on category"""
        base_prices = {
            "crops": 100,
            "vegetables": 50,
            "fruits": 80,
            "fertilizers": 500,
            "seeds": 200,
            "pesticides": 300,
            "equipment": 5000,
            "livestock": 1000
        }
        return base_prices.get(category, 100) * (1 + hash(item) % 100 / 100)
    
    def _get_unit(self, item: str, category: str) -> str:
        """Get appropriate unit based on category"""
        units = {
            "crops": "kg",
            "vegetables": "kg",
            "fruits": "kg",
            "fertilizers": "kg",
            "seeds": "kg",
            "pesticides": "litre",
            "equipment": "piece",
            "livestock": "piece"
        }
        return units.get(category, "kg")
    
    def _get_market_name(self, category: str) -> str:
        """Get market name based on category"""
        markets = {
            "crops": "Agricultural Produce Market Committee (APMC)",
            "vegetables": "Local Vegetable Market",
            "fruits": "Fruit Wholesale Market",
            "fertilizers": "Agricultural Input Store",
            "seeds": "Seed Distribution Center",
            "pesticides": "Agro-Chemical Store",
            "equipment": "Farm Equipment Dealer",
            "livestock": "Livestock Market"
        }
        return markets.get(category, "Local Market")
    
    def _get_simulated_trend(self) -> str:
        """Generate simulated price trend"""
        trends = ["stable", "increasing", "decreasing"]
        return trends[hash(datetime.now().strftime("%Y-%m-%d")) % len(trends)]
    
    def _get_simulated_availability(self) -> str:
        """Generate simulated availability status"""
        statuses = ["high", "medium", "low"]
        return statuses[hash(datetime.now().strftime("%Y-%m-%d")) % len(statuses)]
    
    def _get_simulated_quality(self) -> str:
        """Generate simulated quality grade"""
        grades = ["A", "B", "C"]
        return grades[hash(datetime.now().strftime("%Y-%m-%d")) % len(grades)]
    
    def _get_simulated_supplier(self) -> Dict[str, str]:
        """Generate simulated supplier information"""
        return {
            "name": f"Supplier_{hash(datetime.now().strftime('%Y-%m-%d')) % 1000}",
            "location": "Local Market",
            "rating": f"{(hash(datetime.now().strftime('%Y-%m-%d')) % 5) + 1}/5"
        }

class MarketAnalysisAgent:
    def __init__(self):
        self.scraper = MarketScraper()
        
    def analyze_market(self, state: Dict) -> Dict:
        """
        Analyze market prices for the requested items
        """
        items = state["items"]
        results = {}
        
        for item in items:
            market_data = self.scraper.scrape_market_prices(item)
            results[item] = market_data
            
        return {"market_data": results}

class PriceRecommendationAgent:
    def __init__(self):
        self.llm = llm
        
    def generate_recommendations(self, state: Dict) -> Dict:
        """
        Generate price recommendations based on market data using Groq
        """
        market_data = state["market_data"]
        
        # Create a prompt for the LLM to analyze the market data
        prompt = f"""
        As an agricultural market expert, analyze the following market data and provide detailed recommendations for the farmer:
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Please provide a comprehensive analysis including:
        1. Current market trends and their implications for each category
        2. Recommended buying/selling prices with justification
        3. Best time to buy/sell based on market conditions
        4. Detailed market conditions and factors affecting prices
        5. Risk factors and mitigation strategies
        6. Historical price context if available
        7. Quality considerations for each item
        8. Supplier recommendations and reliability
        9. Cost optimization strategies
        10. Seasonal factors affecting prices
        
        Format your response in a clear, structured manner that a farmer can easily understand.
        Group the recommendations by category (crops, vegetables, fruits, fertilizers, etc.) for better organization.
        """
        
        response = self.llm.invoke(prompt)
        recommendations = response.content
        
        return {"recommendations": recommendations}

def create_market_analysis_graph() -> Graph:
    """
    Create a graph for market analysis workflow
    """
    # Create the graph
    workflow = StateGraph(StateType=Dict)
    
    # Add nodes
    workflow.add_node("market_analysis", MarketAnalysisAgent().analyze_market)
    workflow.add_node("price_recommendations", PriceRecommendationAgent().generate_recommendations)
    
    # Add edges
    workflow.add_edge("market_analysis", "price_recommendations")
    
    # Set entry point
    workflow.set_entry_point("market_analysis")
    
    # Compile the graph
    return workflow.compile()

def get_market_insights(items: List[str]) -> Dict[str, Any]:
    """
    Get market insights for a list of items
    """
    # Create the graph
    graph = create_market_analysis_graph()
    
    # Initialize the state
    initial_state = {
        "items": items,
        "market_data": {},
        "recommendations": ""
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    return final_state

def main():
    # Example usage with more comprehensive items
    items = [
        # Crops
        "wheat", "rice", "cotton",
        # Vegetables
        "tomato", "potato", "onion",
        # Fruits
        "apple", "banana", "orange",
        # Fertilizers
        "urea", "dap", "npk",
        # Seeds
        "wheat_seeds", "rice_seeds",
        # Pesticides
        "insecticide", "herbicide",
        # Equipment
        "tractor", "irrigation_pump"
    ]
    
    insights = get_market_insights(items)
    
    print("\nMarket Analysis Results:")
    print("=" * 50)
    print("\nMarket Data:")
    print(json.dumps(insights["market_data"], indent=2))
    print("\nRecommendations:")
    print(insights["recommendations"])

if __name__ == "__main__":
    main()
