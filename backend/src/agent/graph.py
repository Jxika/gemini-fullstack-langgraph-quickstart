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
    web_searcher_instructions,
    query_writer_instructions_deepseek,
    web_searcher_instructions_hybrid_deepseek,
    reflection_instructions_deepseek,
    answer_instructions_deepseek
)

from langchain_openai import ChatOpenAI
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

def generate_query(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    formatted_prompt = query_writer_instructions_deepseek.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    llm=ChatDeepSeek(model="deepseek-chat",
                   temperature=1,
                   max_retries=2,
                   api_key=os.getenv("DEEP_SEEK_KEY"),
                   base_url=deepseek_baseurl
                   )
    structured_llm = llm.with_structured_output(SearchQueryList)
    result=structured_llm.invoke(formatted_prompt)
    return {"generated_query":result.query}

def continue_to_web_research(state: OverallState):
    for idx, query in enumerate(state["generated_query"]):
        logger.info(f"🔧continue_to_web_research|📄任务 {idx}: generated_query='{query}'")

    send_tasks=[
            Send("web_research", {"search_query": search_query, "id": int(idx)})
            for idx, search_query in enumerate(state["generated_query"])
    ]
    return send_tasks


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:

    configurable = Configuration.from_runnable_config(config)

    formatted_prompt = web_searcher_instructions_hybrid_deepseek.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
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
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    current_date = get_current_date()
    formatted_prompt = reflection_instructions_deepseek.format(
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


def evaluate_research(state: ReflectionState,config: RunnableConfig,) -> OverallState:
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
    configurable = Configuration.from_runnable_config(config)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions_deepseek.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    
    logger.info(f"⚡⚡⚡formatted_prompt={formatted_prompt}")
    
    llm=ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
        api_key=os.getenv("DEEP_SEEK_KEY"),
        base_url=deepseek_baseurl
    )
    result=llm.invoke(formatted_prompt)
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



builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent") 

#内部注册机制（用于Agent管理或多模态调度）
#manager.register(graph) 这样orchestrator 就能通过 "pro-search-agent"来调用它。