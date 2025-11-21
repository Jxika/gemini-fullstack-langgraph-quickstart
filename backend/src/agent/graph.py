import os

from agent.tools_and_schemas import SearchQueryList, Reflection,get_clinical_results,web_search
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from google import genai
import re
import json
#原生的googlesdk，支持直接调用"tools"例如google搜索。
from google.genai import Client
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)

from agent.configuration import Configuration
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from agent.logger import get_logger
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    web_searcher_instructions_hybrid
)

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

logger=get_logger(__name__)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))
gemini_baseurl="https://generativelanguage.googleapis.com/v1beta/openai/"

deepseek_baseurl="https://api.deepseek.com/v1"


print(f"时间{get_current_date()}")

def extract_json(text):
    if '```json' not in text:
        return text
    text = text.split('```json')[1].split('```')[0].strip()
    return text

def extract_answer(text):
    if '</think>' in text:
        answer = text.split("</think>")[-1]
        return answer.strip()
    
    return text

def parse_tools(text, start_flag, end_flag):
  
    tools = text.split(start_flag)
    tools = [tool for tool in tools if end_flag in tool]
    if tools:
        tools = [tool.split(end_flag)[0].strip() for tool in tools]
    return tools

def get_tools(response):
    logger.info(f"get_tools|llm返回:{extract_answer(response['content'])}")
    if response['tool_calls']:
        tools = response['tool_calls']
    else:
        content = extract_answer(response['content'])
        if '<tool_call>' in content:
            print("----------<tool_call>------------")
            tools = parse_tools(content, '<tool_call>', '</tool_call>')
            
        elif '<function_call>' in content:
            print("----------<function_call>------------")
            tools = parse_tools(content, '<function_call>', '</function_call>')
            
        elif '```json\n[' in content:
            print("----------<```json>------------")
            tools = parse_tools(content, '```json\n[', ']\n```')
            
        elif '```json[' in content:
            print("----------<```json>------------")
            tools = parse_tools(content, '```json[', ']```')
            
        elif '```json' in content and ('name' in content and ("args" in content or "arguments" in content)):
            print("----------<```json>------------")
            tools = parse_tools(content, '```json', '```')      
        else:
            tools = [] 
    logger.info(f"get_tools|llm返回工具:{tools}")  
    logger.info(f"----------------------------")  
    return tools

def get_llm_client(configurable:Configuration,task_model_name:str,temperature:float=0.0,max_retries:int=2)->BaseChatModel:
    provider=configurable.llm_provider.lower()
    
    if provider=="gemini":
        gemini_api_key=os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set for Gemini provider, either via LLM_API_KEY or GEMINI_API_KEY environment variable.")
        return ChatOpenAI(
            model=task_model_name,
            temperature=temperature,
            max_retries=max_retries,
            api_key=gemini_api_key,
            base_url=gemini_baseurl
        )
    elif provider=="deepseek":
        deepseek_key=os.getenv("DEEP_SEEK_KEY")
        print(f"deepseek_key{deepseek_key}")
        if not deepseek_key:
            raise ValueError("DEEP_SEEK_KEY must be set for deepseek provider, either via LLM_API_KEY or DEEP_SEEK_KEY environment variable.")
        return ChatDeepSeek(
            model=task_model_name,
            temperature=temperature,
            max_retries=max_retries,
            api_key=deepseek_key,
            base_url="https://api.deepseek.com/v1"
        )
    else:
        raise ValueError(f"Unsupported LLM provider:{configurable.llm_provider}")


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """
       LangGraph node 基于用户问题生成查询需求。

       Args:
          state:当前的图状态了用户问题
          config:一些配置
       
       Returns:
          更新state字典，包括生成的问题。
    """

    configurable = Configuration.from_runnable_config(config)
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    formatted_prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )

    #llm=get_llm_client(configurable,configurable.query_generator_model,1.0,2)
    llm=ChatDeepSeek(model="deepseek-chat",
                   temperature=1,
                   max_retries=2,
                   api_key=os.getenv("DEEP_SEEK_KEY"),
                   base_url=deepseek_baseurl
                   )
    structured_llm = llm.with_structured_output(SearchQueryList)
    result=structured_llm.invoke(formatted_prompt)
    return {"search_query":result.query}

    # resp=genai_client.models.generate_content(
    #     model=configurable.query_generator_model,
    #     contents=formatted_prompt,
    #     config={"temperature": 1.0}
    # )
    # text_output=resp.candidates[0].content.parts[0].text
    # clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_output, flags=re.DOTALL)
    # query_json=json.loads(clean_text)

    # Generate the search queries
    #result = structured_llm.invoke(formatted_prompt)
    #logger.info(f"🧠generate_query|search_query={query_json['query']}")
    #return {"search_query": query_json["query"]}

def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node."""
    '''LangGraph特性：返回一个send()列表，意味这可以并行运行多个子节点。''' 
    for idx, query in enumerate(state["search_query"]):
        logger.info(f"🔧continue_to_web_research|📄任务 {idx}: search_query='{query}'")

    send_tasks=[
            Send("web_research", {"search_query": search_query, "id": int(idx)})
            for idx, search_query in enumerate(state["search_query"])
    ]
    return send_tasks

#调用Google GenAI 原生接口 进行真实网络搜索
def web_research_only(state: WebSearchState, config: RunnableConfig)->OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    response = genai_client.models.generate_content(
        model=configurable.reflection_model,
        contents=formatted_prompt,
        config={
            "tools": [
                {"google_search": {}},   
            ],  
            "temperature": 0,
        },
    ) 
    resolved_urls = resolve_urls(
            response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    logger.info(f"🧩sources_gathered={sources_gathered}")
    logger.info(f"🧩search_query={[state["search_query"]]}")
    logger.info(f"🧩web_research_result={[modified_text]}")
    return {
           "sources_gathered": sources_gathered,  
           "search_query": [state["search_query"]],
           "web_research_result": [modified_text],
        }


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:

    configurable = Configuration.from_runnable_config(config)

    formatted_prompt = web_searcher_instructions_hybrid.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    '''
       response.candidates[0].grounding_metadata.grounding_chunks 是 Google Generative AI（Gemini）
       返回结果中的 “grounding 数据” —— 也就是模型在回答时引用的外部来源（如网页搜索结果、文档、或其他上下文）
       的具体片段（chunks）。
    '''
    #llm=get_llm_client(configurable,configurable.query_generator_model,0,2)
    llm=ChatDeepSeek(model="deepseek-chat",
                   temperature=0,
                   max_retries=2,
                   api_key=os.getenv("DEEP_SEEK_KEY"),
                   base_url=deepseek_baseurl  
                   )
    messages=[HumanMessage(content=formatted_prompt)]
    web_research_result = []

    tools = {"web_search": web_search, "get_clinical_results": get_clinical_results}

    while True:
            response=llm.bind_tools([web_search, get_clinical_results]).invoke(messages)
            response = response.model_dump_json(indent=4, exclude_none=True)
            response = json.loads(response)
            extract_tools=get_tools(response)
            if extract_tools:
                for tool in extract_tools:
                    if isinstance(tool, str):
                        try:
                            tool = json.loads(tool)  
                        except Exception as e:
                            messages += [HumanMessage(content=f"{tool}json格式错误:{e}")]
                            break
                        
                        if isinstance(tool, list):
                            tool = tool[0]
                    try:
                        tool_name = tool['name']
                        keys = list(tool.keys())
                        tool_args = tool[keys[1]]
                    except Exception as e:
                        messages += [HumanMessage(content=f"{tool}工具调用格式错误:{e}")]
                        break
                    logger.info(f"web_research|调用工具{tool_name},参数{tool_args}")
                    tool_result = tools[tool_name].invoke(tool_args)
                
                    web_research_result.append(tool_result)
                    messages += [HumanMessage(content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")]
            else:
                break
    
    all_sources = []
    all_texts = []

    for r in web_research_result:
        if "sources_gathered" in r:
            all_sources.extend(r["sources_gathered"])
        if "modified_text" in r:
            all_texts.append(r["modified_text"])

    return {
         "sources_gathered": all_sources,  
         "search_query": [state["search_query"]],
         "web_research_result": all_texts,
    }


#反思当前研究的内容是否充分，并生成下一轮查询。
'''
   增加research_loop_count;
   将所有web_research_result 拼接成总结
   使用reflection_instructions prompt；
   使用输出结构化JSON；
   {
      "is_sufficient":False,
      "knowledge_gap":"No data on pediatric TB therapies",
      "follow_up_queries": ["pediatric TB treatment 2025", "TB vaccine trials"]
   }
'''
def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    #reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    logger.info(f"🤔summaries={"\n\n---\n\n".join(state["web_research_result"])}")
    
    llm=ChatDeepSeek(
            model="deepseek-chat",
            temperature=1,
            max_retries=2,
            api_key=os.getenv("DEEP_SEEK_KEY"),
            base_url=deepseek_baseurl
    )
    result=llm.with_structured_output(Reflection).invoke(formatted_prompt)
    #llm=get_llm_client(configurable,configurable.reflection_model,1)
    # response=genai_client.models.generate_content(
    #     model=reasoning_model,
    #     contents=formatted_prompt,
    #     config={"temperature": 1.0},
    # )
    # text_output=response.candidates[0].content.parts[0].text
    # clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_output, flags=re.DOTALL)
    # result = json.loads(clean_text)
    # logger.info(f"""🤔is_sufficient={result["is_sufficient"]},
    #             knowledge_gap={result["knowledge_gap"]},
    #             follow_up_queries={result["follow_up_queries"]},
    #             research_loop_count={state["research_loop_count"]},
    #             number_of_ran_queries={len(state["search_query"]),state["search_query"]}
    #             """)
    logger.info(f"""🤔is_sufficient={result.is_sufficient},
                 knowledge_gap={result.knowledge_gap},
                 follow_up_queries={result.follow_up_queries},
                 research_loop_count={state["research_loop_count"]},
                 number_of_ran_queries={len(state["search_query"]),state["search_query"]}
                """)
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }

#根据反思结果判断要不要继续研究：
#若 is_sufficient=True 或 已达到 max_research_loops → 进入 finalize_answer
#否则->根据 follow_up_queries 生成新的 send("web_research")
def evaluate_research(state: ReflectionState,config: RunnableConfig,) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    logger.info(f"🔁research_loop_count={state["research_loop_count"]}")
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:   
        return "finalize_answer"
    else:
        logger.info(f"🔁发现{len(state["follow_up_queries"])}个新的follow-up查询。")
        logger.info(f"🔁当前累计已运行查询数：{state["number_of_ran_queries"]}")
        for idx,q in enumerate(state["follow_up_queries"]):
            logger.info(f"🔁Follow-up #{idx}:'{q}'(id={state["number_of_ran_queries"]+idx})")

        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]

'''
合并所有研究结果，生成带引用的最终总结报告。
'''
def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    #reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    
    logger.info(f"⚡⚡⚡formatted_prompt={formatted_prompt}")
    
    #llm=get_llm_client(configurable,configurable.answer_model,0,2)
    llm=ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
        api_key=os.getenv("DEEP_SEEK_KEY"),
        base_url=deepseek_baseurl
    )
    result=llm.invoke(formatted_prompt)
    # response = genai_client.models.generate_content(
    #     model=reasoning_model,
    #     contents=formatted_prompt,
    #     config={"temperature": 0},
    # )
    # raw_text = response.candidates[0].content.parts[0].text
    # clean_text=re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL)


    unique_sources = []

    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content=result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    logger.info("🚀==============================END=================================🚀")
    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    #可供跳转的目标节点名称列表
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent") #给生成的图实例一个标识名称

#内部注册机制（用于Agent管理或多模态调度）
#manager.register(graph) 这样orchestrator 就能通过 "pro-search-agent"来调用它。