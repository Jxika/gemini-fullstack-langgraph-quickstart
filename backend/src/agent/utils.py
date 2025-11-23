from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from agent.logger import get_logger
logger=get_logger(__name__)

"""
   从对话消息中提取研究的问题
   从消息列表中提取研究主题。
   输入一般是聊天记录（LangChain格式：HumanMessage、AIMessage）
   如果是多轮对话，就是把每条消息前加上"User:"或"Assistant："，拼接成完整上下文。
"""
def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1: 
        research_topic = messages[-1].content
        logger.info(f"💬len(messages)=1,research_topic={research_topic}")
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
        logger.info(f"💬research_topic={research_topic}")
    
    return research_topic

'''
   为Tavily搜索结果生成短链接ID，用于可视化或Markdown引用
   将Tavily搜索结果中的URL转为短链接
'''
def resolve_urls(search_results: List[dict], id: int) -> Dict[str, str]:
    """
    Create a map of Tavily search result URLs to a short url with a unique id for each URL.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    
    Args:
        search_results: List of search result dictionaries from Tavily
        id: Unique identifier for this search batch
    
    Returns:
        Dictionary mapping original URLs to shortened URLs
    """
    prefix = f"https://tavily.search/id/"
    resolved_map = {}
    
    # Extract URLs from Tavily search results
    urls = []
    if isinstance(search_results, list):
        for result in search_results:
            if isinstance(result, dict) and 'url' in result:
                urls.append(result['url'])
    
    # Create a dictionary that maps each unique URL to its first occurrence index
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"
    return resolved_map

'''
   在文本中插入Markdown引用的链接
   在回答文本中插入Markdown引用标记（如[source](short_url)）。
   对于Tavily搜索结果，由于没有位置信息，将在文本末尾添加引用列表。
'''
def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.
    For Tavily results (which lack position info), appends citations at the end.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segments' with citation information.

    Returns:
        str: The text with citation markers inserted.
    """
    if not citations_list:
        return text
    
    # Check if we have position information (from Google Search) or not (from Tavily)
    has_position_info = any(
        citation.get("start_index", 0) != 0 or citation.get("end_index", 0) != 0 
        for citation in citations_list
    )
    
    if has_position_info:
        # Original logic for Google Search results with position info
        # Sort citations by end_index in descending order.
        # If end_index is the same, secondary sort by start_index descending.
        # This ensures that insertions at the end of the string don't affect
        # the indices of earlier parts of the string that still need to be processed.
        sorted_citations = sorted(
            citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
        )

        modified_text = text
        for citation_info in sorted_citations:
            # These indices refer to positions in the *original* text,
            # but since we iterate from the end, they remain valid for insertion
            # relative to the parts of the string already processed.
            end_idx = citation_info["end_index"]
            marker_to_insert = ""
            for segment in citation_info["segments"]:
                marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
            # Insert the citation marker at the original end_idx position
            modified_text = (
                modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
            )
        
        return modified_text
    else:
        # New logic for Tavily results without position info
        # Append citations at the end of the text
        modified_text = text
        
        # Add citation section if we have citations
        if citations_list:
            modified_text += "\n\n## 参考来源\n"
            
            for idx, citation in enumerate(citations_list, 1):
                for segment in citation["segments"]:
                    title = segment.get('label', f'来源 {idx}')
                    url = segment.get('short_url', segment.get('value', ''))
                    modified_text += f"\n{idx}. [{title}]({url})"
        
        return modified_text

'''
    从Tavily搜索结果中提取引用信息
    作用：从Tavily搜索结果中提取引用元数据，用于生成引用标记。
    👇主要逻辑：
      1.从Tavily搜索结果列表中提取每个结果的信息。
      2.提取：
          ·title：网页标题
          ·url：网页链接
          ·content：网页内容摘要
      3.使用 resolved_urls_map 将长URL映射为短链接。
      4.构建一个 citation 字典：
         {
            "start_index": 0,  # Tavily不提供位置信息，设为0
            "end_index": 0,    # Tavily不提供位置信息，设为0
            "segments": [
               {"label": "网页标题", "short_url": "https://tavily.search/id/1-0", "value": "原始URL"}
            ]
        }
      5.返回一个 citation 列表，用于传给 insert_citation_markers()。
'''
def get_citations(search_results, resolved_urls_map):
    """
    Extracts and formats citation information from Tavily search results.

    This function processes Tavily search results to construct a list of 
    citation objects. Each citation object includes the title, URL, and 
    formatted markdown links to the web sources.

    Args:
        search_results: List of search result dictionaries from Tavily
        resolved_urls_map: Dictionary mapping original URLs to shortened URLs

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): Set to 0 since Tavily doesn't provide position info
              - "end_index" (int): Set to 0 since Tavily doesn't provide position info
              - "segments" (list[dict]): List of citation segments with label, short_url, and value
              Returns an empty list if no valid search results are found.
    """
    citations = []

    if not search_results:
        return citations

    # Handle different response formats from Tavily
    if isinstance(search_results, str):
        # If it's a string, we can't extract structured citations
        logger.info("get_citations: search_results is a string, cannot extract structured citations")
        return citations
    
    if not isinstance(search_results, list):
        logger.info(f"get_citations: unexpected search_results type: {type(search_results)}")
        return citations

    for idx, result in enumerate(search_results):
        if not isinstance(result, dict):
            continue
            
        # Extract essential information
        title = result.get('title', f'来源 {idx + 1}')
        url = result.get('url', '')
        content = result.get('content', '')
        
        if not url:
            continue
            
        # Create citation entry
        citation = {
            "start_index": 0,  # Tavily doesn't provide position information
            "end_index": 0,    # Tavily doesn't provide position information
            "segments": []
        }
        
        # Get the shortened URL from the resolved map
        short_url = resolved_urls_map.get(url, url)
        
        # Add segment with title and URL
        citation["segments"].append({
            "label": title,
            "short_url": short_url,
            "value": url
        })
        
        citations.append(citation)
    
    return citations


##########################################以下是googlesearch用的方法#############################################

def resolve_urls_googlesearch(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers_googlesearch(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segment_string' (the marker to insert).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations_googlesearch(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.

    This function processes the grounding metadata provided in the response to
    construct a list of citation objects. Each citation object includes the
    start and end indices of the text segment it refers to, and a string
    containing formatted markdown links to the supporting web chunks.

    Args:
        response: The response object from the Gemini model, expected to have
                  a structure including `candidates[0].grounding_metadata`.
                  It also relies on a `resolved_map` being available in its
                  scope to map chunk URIs to resolved URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
    citations = []

    # Ensure response and necessary nested structures are present
    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}

        # Ensure segment information is present
        if not hasattr(support, "segment") or support.segment is None:
            continue  # Skip this support if segment info is missing

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # Ensure end_index is present to form a valid segment
        if support.segment.end_index is None:
            continue  # Skip if end_index is missing, as it's crucial

        # Add 1 to end_index to make it an exclusive end for slicing/range purposes
        # (assuming the API provides an inclusive end_index)
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # Handle cases where chunk, web, uri, or resolved_map might be problematic
                    # For simplicity, we'll just skip adding this particular segment link
                    # In a production system, you might want to log this.
                    pass
        citations.append(citation)
    return citations
