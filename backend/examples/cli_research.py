import argparse
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """Run the research agent from the command line."""
    #创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")

    parser.add_argument("question", help="Research question")

    #可选参数：初始搜索查询数量，默认值为3.
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    #可选参数：最大的循环次数，即研究-搜索-总结的最大迭代次数
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="Maximum number of research loops",
    )
    #可选参数：用于最终回答的推理模型
    parser.add_argument(
        "--reasoning-model",
        default="gemini-2.5-pro-preview-05-06",
        help="Model for the final answer",
    )

    #执行后args是一个命名空间对象，包含上面定义的参数值，比如：
    #
    args = parser.parse_args()

    #初始状态
    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "reasoning_model": args.reasoning_model,
    }

    result = graph.invoke(state)
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
