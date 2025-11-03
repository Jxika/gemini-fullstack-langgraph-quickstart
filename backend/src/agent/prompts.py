from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


'''
你的目标是生成复杂多样化的网页搜索查询。
这些查询将用于一个高级的自动化网页研究工具，该工具能够分析复杂的结果、跟踪链接并综合信息。

指令：
·优先只生成一个搜索查询；只有当原始问题包含多个方面或元素、一个查询不足以涵盖时，才生成额外的查询。
·每个查询应聚焦于原始问题的一个特定方面。
·不要生成超过{number_queries}个查询。
·查询应具有多样化；如果主题较广，可以生成多个查询。
·不要生成多个相似的查询，一个就够了。
·查询应确保获取最新的信息。当前日期为{current_date}。

格式：
你的回答应时一个JSON对象，包含以下两个完全相同名称的建；
·"rationale":简要说明这些查询为何与研究主题相关
·"query":一个包含搜索查询字符串的列表

'''

query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


'''
  执行有针对性的google搜索，以收集关于"{research_topic}"的最新且可信的信息，并将其综合为一个可验证的文本成果。
  指令：
  ·搜索查询应确保获取到最新的信息。当前日期为{current_date}.
  ·进行多次、多样化的搜索，以便获取全面的信息。
  ·整合关键信息，并仔细记录每一条具体信息对应的来源。
  ·输出内容应为一个基于搜索结果撰写的高质量摘要或报告。
  ·仅包含在搜索结果中找到信息，不得编造或虚构任何内容

  研究主题（research Topic）:
  {research_topic}
'''
web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""







'''
   你是一个专家研究助理，负责分析有关{research_topic}的摘要。
   指令：
   ·识别知识空白或需要更深入探讨的领域  ，并生成后续查询。（可为1条或多条）。
   ·如果提供的摘要已足以回答用户的问题，则不要生成后续查询。
   ·如果存在知识空白，生成一个能帮助扩展理解的后续查询。
   ·关注未充分涵盖的技术细节、实现细节或新兴趋势。
   要求：
   ·确保后续查询是自包含的，并包含进行网页搜索所需要的必要上下文。
   输出格式：
   ·将你的回答格式化为一个json对象，并包含以下完全相同的键：
      "is_sufficient"：true 或 false
      "knowledge_gap"：描述缺失或需要澄清的信息（如果 is_sufficient 为 true，则留空字符串 ""）
      "follow_up_queries"：写出用于解决该空白的具体问题（如果 is_sufficient 为 true，则为空数组 []）

    示例：
    {
    "is_sufficient": true, // 或 false
    "knowledge_gap": "摘要缺少有关性能指标和基准测试的信息", // 若 is_sufficient 为 true 则写 ""
    "follow_up_queries": ["用于评估[specific technology]的常见性能基准和指标是什么？"] // 若 is_sufficient 为 true 则写 []
    }

    请仔细反思所给的摘要以识别知识空白并生成后续查询。然后按照上述 JSON 格式输出：
    摘要（Summaries）：
    {summaries}
'''
reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""



'''
根据所提供的摘要，为用户的问题生成一个高质量的回答。
指令：
  当前日期为 {current_date}。
  你是一个多步骤研究流程中的最后一步，但不要在回答中提及这一点。
  你可以访问前面步骤中收集的所有信息。
  你可以访问用户提出的问题。
  基于所提供的摘要和用户的问题，生成一个高质量、准确且全面的回答。
  在回答中正确引用摘要中的信息来源，并使用 Markdown 格式（例如：[apnews](https://vertexaisearch.cloud.google.com/id/1-0)）。
  ⚠️ 这是必须做到的。

  用户上下文：
    {research_topic}
  摘要（Summaries）：
    {summaries}
'''
answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [apnews](https://vertexaisearch.cloud.google.com/id/1-0)). THIS IS A MUST.

User Context:
- {research_topic}

Summaries:
{summaries}"""
