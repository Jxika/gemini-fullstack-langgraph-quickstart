from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

query_writer_instructions_deepseek="""你的目标是生成复杂且多样化的网络搜索查询。这些查询将用于一个高级的自动化网页研究工具，该工具能够分析复杂的结果、跟踪链接并综合信息。
   
   指令：
   -优先只生成一个搜索查询；只有当原始问题包含多个方面或元素、一个查询不足以涵盖时，才生成额外的查询。
   -每个查询应聚焦于原始问题的一个特定方面。
   -不要产生超过{number_queries}个查询。
   -查询应具有多样化；如果主题较广，可以生成多个查询。
   -不要生成多个相似的查询，一个就够了。
   -查询应确保获取最新的信息。当前日期为{current_date}。
   
   格式：
   -将您的回复格式化为具有所有两个确切键的JSON对象：
      -"rationale":简要说明这些查询为何与研究主题相关
      -"query":搜索查询的列表
   
   示例：

   主题：去年苹果股票和iphone购买数量哪个增长更快
   ```json
  {{
    "rationale": "为准确回答此对比增长问题，需要苹果股票表现和iPhone销售数据的具体指标。这些查询精确指向所需财务信息：公司收入趋势、产品具体销量数据和同期股价变动，以便直接对比。",
    "query": ["苹果2024财年总收入增长", "iPhone 2024财年销量增长", "苹果股票2024财年涨幅"],
  }}
  ```

  上下文：{research_topic}"""


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


web_searcher_instructions_hybrid_deepseek="""
执行针对性的谷歌搜索，若主题涉及临床试验或患者/试验相关详细信息，也可调用本地工具获取临床试验数据。将{current_date}设为当前所用日期。

指令：
-查询必须确保获取到最新消息。当前日期是{current_date}.
-执行多次、不同方向的搜索以收集全面信息。
-整理关键发现时，必须严格记录每条信息对应的来源。
-输出内容应基于搜索结果撰写成结构良好的总结或报告。
-只包含搜索结果中发现的信息，不得杜撰任何内容。

研究主题：
{research_topic}
"""


web_searcher_instructions_hybrid="""
Conduct targeted Google searches and, when the topic concerns clinical trials or patient/trial details, call the local tool get_clinical_results. Use {current_date} as the current date.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""


reflection_instructions_deepseek="""
    你是一个专家研究助理，负责分析有关{research_topic}的摘要。

    指令:
      -识别知识空白或需要更深入探讨的领域  ，并生成后续查询。（可为1条或多条）。
      -如果提供的摘要已足以回答用户的问题，则不要生成后续查询。
      -如果存在知识空白，生成一个能帮助扩展理解的后续查询。
      -关注未充分涵盖的技术细节、实现细节或新兴趋势。
    
    要求：
      -确保后续查询是自洽的（self-contained），并包含进行网页搜索所需的必要上下文。

    输出格式:
      -将你的回答格式化一个JSON对象，且必须包含以下键：
         - "is_sufficient":true 或 false
         - "knowledge_gap":描述缺少哪些信息或哪些内容需要澄清
         - "follow_up_queries":写出一个用于解决上述
      
    Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

    请仔细反思所给的摘要以识别知识空白并生成后续查询。然后按照上述 JSON 格式输出：

    Summaries:
    {summaries}
"""

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


answer_instructions_deepseek="""
    基于提供的摘要内容，为用户的问题生成高质量回答。
    
    指令：
      -当前日期是{current_date}.
      -你是一个多步骤研究流程中的最后一步，但不要在回答中提及这一点。
      -你可以访问前面步骤中收集的所有信息。
      -你可以访问用户提出的问题。
      -基于提供的摘要内容以及用户的问题，生成高质量的回答。
      -在回答中正确引用摘要的来源，并使用 Markdown 格式。（例如：[apnews](https://tavily.search/id/1-0)）。这是必须做到的
      
    引用格式说明：
      -对于网页搜索结果：使用 [来源标题](链接) 格式引用
      -对于本地工具数据（如临床试验数据）：保持原始表格结构，并在表格下方注明数据来源
      
    重要要求：
      -摘要包含两种类型的数据：
        (1) 网页搜索结果
        (2) 本地结构化数据（如表格、管道或格式化的临床试验列表）
      -使用本地结构化数据时，必须尽可能完整地保留原始结构和内容：
        * 不要删除行、列、试验ID、药物名称、阶段或状态
        * 不要压缩或缩短结构化列表或表格
        * 可以重新格式化布局以提高可读性，但信息必须保持完整和忠实于来源
        * **在回答中包含本地结构化数据时，必须将其格式化为 Markdown 表格**，以便在前端正确渲染
      -网页搜索结果可以根据需要进行总结，但本地工具生成的数据必须保持详细和完整
      -不要编造引用

    用户上下文：
      {research_topic}
      
    摘要：
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

CRITICAL REQUIREMENTS:
- Summaries include two types of data:
  (1) Web search results
  (2) Local structured data (e.g., tables, pipelines, or formatted clinical trial lists produced by internal tools)
- When using LOCAL structured data, you MUST preserve the original structure and content as completely as possible.
  * Do NOT drop rows, columns, trial IDs, drug names, phases, or statuses.
  * Do NOT compress or shorten structured lists or tables.
  * You may reformat the layout for readability, but the information must remain complete and faithful to the source.
  * **When including local structured data in your answer, you MUST format it as a Markdown table**, so it can be properly rendered in the front-end.
- Web search results may be summarized if needed, but local tool-generated data must remain detailed and intact.
- Do NOT invent citations.


User Context:
- {research_topic}

Summaries:
{summaries}"""
