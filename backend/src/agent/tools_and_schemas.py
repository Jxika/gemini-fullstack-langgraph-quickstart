import os
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import tool
from dotenv import load_dotenv
from google.genai import Client
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from agent.configuration import Configuration
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.logger import get_logger
logger=get_logger(__name__)


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


def escape_md(value):
    if not value:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


@tool("web_search",return_direct=False)
def web_search(query:str):
    """
    Performs web search using Tavily and returns sources and results."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    
    tavily_search = TavilySearch(api_key=tavily_api_key)
    search_results = tavily_search.invoke(query) 
    
    if isinstance(search_results, str):
       modified_text = f"### 网页搜索结果（来自工具 web_search）\n\n查询：{query}\n\n{search_results}"
       sources_gathered = []
    elif isinstance(search_results, list):
       sources_gathered = []
       logger.info(f"传入的query={query}的搜索结果数量：{len(search_results["results"])}" )
       modified_text = f"### 网页搜索结果（来自工具 web_search）\n\n查询：{query}\n\n" 
       for i, result in enumerate(search_results, 1):
            if isinstance(result, dict):
               title = result.get('title', f'结果 {i}')
               content = result.get('content', str(result))
               url = result.get('url', '')
               modified_text += f"**{i}. {title}**\n\n{content}\n\n来源：{url}\n\n"
            
               sources_gathered.append({
                  'label': title,
                  'short_url': url,
                  'value': url
                })
            else:
                modified_text += f"**{i}.** {result}\n\n"
    else:
       modified_text = f"### 网页搜索结果（来自工具 web_search）\n\n查询：{query}\n\n{str(search_results)}"
       sources_gathered = []   
    return {
        "query": query,
        "modified_text": modified_text,
        "sources_gathered": sources_gathered
    }
    
    
@tool("get_clinical_results",return_direct=False)
def get_clinical_results(keywords:str):
    """
    Query the global clinical trial results dataset.
    return markdown table so final report can render it directly.
    """
    clinical_results=[     ]
    
    headers = ["登记号", "试验药", "适应症", "试验状态", "试验分期"]
    table_header = "| " + " | ".join(headers) + " |\n"
    table_sep = "| " + " | ".join(["---"] * len(headers)) + " |\n"

    table_rows=""
    for row in clinical_results:
        table_rows += (
            "| "
            + " | ".join([
                escape_md(row.get("登记号","")),
                escape_md(row.get("试验药","")),
                escape_md(row.get("适应症","")),
                escape_md(row.get("试验状态","")),
                escape_md(row.get("试验分期","")),
            ])
            + " |\n"
        )
    markdown_table=table_header + table_sep + table_rows
   

    # html_table = "<table>\n<tr><th>登记号</th><th>试验药</th><th>适应症</th><th>试验状态</th><th>试验分期</th></tr>\n"

    # for row in clinical_results:
    #     html_table += "<tr>"
    #     for key in ["登记号","试验药","适应症","试验状态","试验分期"]:
    #        value = str(row.get(key,"")).replace("|","|")  # 可按需转义
    #        html_table += f"<td>{value}</td>"
    # html_table += "</tr>\n"

    # html_table += "</table>"

    
    modified_text = f"### 本地临床试验数据（来自工具 get_clinical_results）\n\n{markdown_table}\n\n来源：[PharmaOne](http://pharma1.pharmadl.test/)"
    sources_gathered = [{'label':'PharmaOne','short_url':'http://pharma1.pharmadl.test/','value':'http://pharma1.pharmadl.test/'}]
    return {
        "query":keywords,
        "modified_text":modified_text,
        "sources_gathered":sources_gathered
    }



