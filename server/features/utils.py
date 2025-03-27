from typing import Any
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool

def setup_web_search() -> Tool:
    """Setup web search tool using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    return Tool(
        name="web_search",
        func=search.run,
        description="Search the web for agricultural information. Input should be a search query."
    )

def get_web_search_results(query: str, tool: Tool) -> str:
    """Get web search results for a query."""
    try:
        results = tool.run(query)
        return results
    except Exception as e:
        return f"Error performing web search: {str(e)}" 