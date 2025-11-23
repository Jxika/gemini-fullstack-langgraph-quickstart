import os
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import tool
from dotenv import load_dotenv
from google.genai import Client
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from agent.configuration import Configuration
from agent.utils import resolve_urls,get_citations,insert_citation_markers
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
load_dotenv()
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

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


@tool("web_search_googlesearch",return_direct=False)
def web_search_googlesearch(query:str,config: RunnableConfig,state: WebSearchState):
    """Performs Google search and returns sources and results."""
    configurable = Configuration.from_runnable_config(config)
    response=genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=f"{query}",
        config={
            "tools":[{"google_search":{}}],
            "temperature":0,
        }
    )
    
    resolved_urls = resolve_urls(
             response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
         )

    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]


    return {"query": query, "modified_text": modified_text,"sources_gathered":sources_gathered}

@tool("web_search",return_direct=False)
def web_search(query:str,config:RunnableConfig,state:WebSearchState):
    """
    Performs web search using Tavily and returns sources and results."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    
    tavily_search = TavilySearch(api_key=tavily_api_key)
    
    # Perform search
    search_results = tavily_search.invoke(query)
    
    # Extract search results and sources
    # Debug: print the actual result type and content
    print(f"search_results type: {type(search_results)}")
    print(f"search_results content: {search_results}")

    # Handle different response formats
    if isinstance(search_results, str):
       # If it's a string, treat it as the content
       modified_text = f"### 网页搜索结果（来自工具 web_search）\n\n查询：{query}\n\n{search_results}"
       sources_gathered = []
    elif isinstance(search_results, list):
       # If it's a list, process each result
       sources_gathered = []
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
       # Fallback for other types
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
    clinical_results=[
                                            {"登记号":"NCT06667141","试验药":"PKMYT1 抑制剂(Acrivon Therapeutics)","适应症":"晚期恶性实体瘤","试验状态":"Recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT06612203","试验药":"Zedoresertib、戈沙妥组单抗","适应症":"三阴性乳腺癌 、HR阳性/HER2阴性乳腺癌 、晚期乳腺癌","试验状态":"Recruiting","试验分期":"临床1/2期"},
                                            {"登记号":"CTIS2024-516322-60-00","试验药":"Zedoresertib 、 戈沙妥组单抗","适应症":"肿瘤 、 HR阳性/HER2阴性乳腺癌","试验状态":"Recruiting","试验分期":"临床1/2期"},
                                            {"登记号":"NCT06476808","试验药":"BMS-986463","适应症":"子宫浆液性癌 、 非小细胞肺癌 、 晚期癌症 、 卵巢高级别浆液性腺癌","试验状态":"Recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT06463340","试验药":"SGR-3515","适应症":"晚期恶性实体瘤","试验状态":"Recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT06369155","试验药":"Azenosertib","适应症":"子宫浆液性癌","试验状态":"Recruiting","试验分期":"临床2期"},
                                            {"登记号":"NCT06364410","试验药":"德曲妥珠单抗 、 Azenosertib","适应症":"局部晚期恶性实体瘤 、 转移性实体瘤 、 HER2阳性胃食管交界处癌","试验状态":"Recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT06363552","试验药":"SC-0191 、 亚叶酸钙 、 贝伐珠","适应症":"转移性结直肠癌","试验状态":"Not yet recruiting","试验分期":"临床2期"},
                                            {"登记号":"NCT06351332","试验药":"卡铂 、 帕博利珠单抗 、 Azenosertib","适应症":"三阴性乳腺癌 、 转移性三阴性乳腺癌","试验状态":"Recruiting","试验分期":"临床1/2期"},
                                            {"登记号":"CTIS2024-511227-33-00","试验药":"LB-100 、 Azenosertib","适应症":"肿瘤 、 结直肠癌 、 转移性结直肠癌","试验状态":"Not yet recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT06260514","试验药":"APR-105","适应症":"晚期恶性实体瘤","试验状态":"Recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT06109883","试验药":"LB-100 、 Azenosertib","适应症":"转移性结直肠癌","试验状态":"Not yet recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT06055348","试验药":"SC-0191 、 紫杉醇 、 盐酸吉西他滨","适应症":"卵巢癌 、 复发性卵巢癌 、 铂耐药性卵巢癌","试验状态":"Not yet recruiting","试验分期":"临床1/2期"},
                                            {"登记号":"NCT06015659","试验药":"盐酸吉西他滨 、 Azenosertib","适应症":"胰腺导管腺癌 、 晚期胰腺腺癌","试验状态":"Recruiting","试验分期":"临床2期"},
                                            {"登记号":"NCT05815160","试验药":"Zedoresertib 、 依托泊苷 、 卡铂","适应症":"复发性小细胞肺癌","试验状态":"Recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT05765812","试验药":"Zedoresertib 、 替莫唑胺","适应症":"胶质母细胞瘤，IDH野生型 、 复发性胶质母细胞瘤","试验状态":"Recruiting","试验分期":"临床1/2期"},
                                            {"登记号":"NCT05743036","试验药":"西妥昔单抗 、 恩考芬尼 、 Azenosertib","适应症":"转移性结直肠癌","试验状态":"Terminated","试验分期":"临床1期"},
                                            {"登记号":"NCT05682170","试验药":"Asaretoclax 、 Azenosertib","适应症":"急性髓性白血病","试验状态":"Terminated","试验分期":"临床1/2期"},
                                            {"登记号":"NCT05431582","试验药":"贝伐珠单抗 、 帕博利珠单抗 、 Azenosertib","适应症":"肺癌 、 卵巢癌 、 转移性实体瘤 、 乳腺癌 、 胰腺癌","试验状态":"Withdrawn","试验分期":"临床1期"},
                                            {"登记号":"CTR20221135","试验药":"亚叶酸钙 、 贝伐珠单抗 、 Azenosertib 、 氟尿嘧啶","适应症":"结直肠癌","试验状态":"Terminated","试验分期":"临床1/2期"},
                                            {"登记号":"NCT05368506","试验药":"Azenosertib","适应症":"原发性腹膜癌 、 卵巢癌 、 转移性实体瘤 ","试验状态":"Withdrawn","试验分期":"早期临床1期"},
                                            {"登记号":"NCT05291182","试验药":"SY-4835","适应症":"晚期恶性实体瘤","试验状态":"Unknown status","试验分期":"临床1期"},
                                            {"登记号":"NCT05212025","试验药":"Adavosertib 、 盐酸吉西他滨","适应症":"晚期胰腺腺癌","试验状态":"Withdrawn","试验分期":"临床2期"},
                                            {"登记号":"NCT05198804","试验药":"甲苯磺酸尼拉帕利 、 Azenosertib","适应症":"原发性腹膜癌 、 复发性高级别输卵管浆液性腺癌 、 铂耐药性卵巢癌 、 输卵管癌","试验状态":"Active, not recruiting","试验分期":"临床1/2期"},
                                            {"登记号":"NCT05128825","试验药":"Azenosertib","适应症":"原发性腹膜癌 、 卵巢浆液性肿瘤 、 输卵管癌","试验状态":"Recruiting","试验分期":"临床2期"},
                                            {"登记号":"NCT05109975","试验药":"Zedoresertib","适应症":"局部晚期恶性实体瘤 、晚期恶性实体瘤","试验状态":"Recruiting","试验分期":"临床1期"},
                                            {"登记号":"NCT05008913","试验药":"Adavosertib","适应症":"局部晚期恶性实体瘤 、 晚期恶性实体瘤","试验状态":"Terminated","试验分期":"临床1期"},
                                            {"登记号":"CTR20211986","试验药":"Azenosertib","适应症":"卵巢癌 、 腹膜癌 、 输卵管癌","试验状态":"Terminated","试验分期":"临床1期"},
                                            {"登记号":"NCT04972422","试验药":"Azenosertib","适应症":"实体瘤","试验状态":"Unknown status","试验分期":"临床1期"},
                                            {"登记号":"NCT04959266","试验药":"伊曲康唑、奥美拉唑 、Adavosertib 、 利福平","适应症":"局部晚期恶性实体瘤 、 晚期恶性实体瘤","试验状态":"Terminated","试验分期":"临床1期"},
                                           ]
    
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

