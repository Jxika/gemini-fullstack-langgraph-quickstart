import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

#原生的googlesdk，支持直接调用"tools"例如google搜索。
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration

from agent.logger import get_logger
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)


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



#根据用户的问题生成若干条优化后的搜索查询
#核心逻辑：
'''
   .初始化Gemini flash模型；
   ·使用query_writer_instructions 模板生成prompt；
   ·调用structured_llm.invoke(),产出结构化输出 SearchQueryList；
   ·返回：
      {"search_query":["tuberculosis treatment pipeline 2025","new TB drugs"]}
'''
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    logger.info(f"graph.py|generate_query1|{configurable.query_generator_model},{os.getenv("GEMINI_API_KEY")}" )

    #结构化输出：
    '''
    {"search_query":["tuberculosis treatment pipeline 2025","new TB drugs"]}
    '''
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
     
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    logger.info(f"graph.py|generate_query2|{formatted_prompt}" )

    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    logger.info(f"graph.py|generate_query3|{result.query}")
    return {"search_query": result.query}

#将上一步生成的多条搜索查询，分发成多个"web research"任务
def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    '''LangGraph特性：返回一个send()列表，意味这可以并行运行多个子节点。'''
     
    for idx, query in enumerate(state["search_query"]):
        logger.info(f"graph.py|continue_to_web_research|任务 {idx}: search_query='{query}'")

    send_tasks=[
            Send("web_research", {"search_query": search_query, "id": int(idx)})
            for idx, search_query in enumerate(state["search_query"])
    ]
    logger.info(f"graph.py|continue_to_web_research|[continue_to_web_research] 已构建 Send 任务列表，共 {len(send_tasks)} 个。")
    return send_tasks

#调用Google GenAI 原生接口 进行真实网络搜索
def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],   ##原生的googlesdk，支持直接调用"tools"例如google搜索。
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,  
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
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
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    logger.info(f"graph.py|reflection|formatted_prompt={formatted_prompt}")

    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    logger.info(f"""graph.py|reflection|is_sufficient={result.is_sufficient},
                knowledge_gap={result.knowledge_gap},
                follow_up_queries={result.follow_up_queries},
                research_loop_count={state["research_loop_count"]},
                number_of_ran_queries={len(state["search_query"])}
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
def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
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

    logger.info(f"graph.py|evaluate_research|is_sufficient={state["is_sufficient"]},research_loop_count={state["research_loop_count"]}")
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        logger.info(f"graph.py|evaluate_research|finalize_answer")      
        return "finalize_answer"
    else:
        logger.info(f"graph.py|evaluate_research|发现{len(state["follow_up_queries"])}个新的follow-up查询。")
        logger.info(f"graph.py|evaluate_research|当前累计已运行查询数：{state["number_of_ran_queries"]}")
        for idx,q in enumerate(state["follow_up_queries"]):
            logger.info(f"Follow-up #{idx}:'{q}'(id={state["number_of_ran_queries"]+idx})")

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
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    
    logger.info(f"graph.py|finalize_answer|formatted_prompt={formatted_prompt}")
    
    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    logger.info(f"graph.py|finalize_answer|messages={result.content},sources_gathered={unique_sources}")
    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
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