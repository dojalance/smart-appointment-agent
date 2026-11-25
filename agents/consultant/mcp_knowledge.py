import asyncio
import json
import logging
import os
import re
from pathlib import PurePosixPath

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


log = logging.getLogger(__name__)
_REFERENCES = re.compile(r"\*\*References \(JSON\):\*\*\s*```json\s*(.*?)\s*```", re.DOTALL)
_APPOINTMENT_SLOT = re.compile(r"(?:时间\s*)?\d{4}-\d{2}-\d{2}\s+\d{2}[:：]\d{2}(?:\s*[，,]\s*\d+\s*分钟)?")
_DEMO_SOURCES = {
    "r100-field-guide.pdf": ("R100", "光网络"),
    "p200-field-guide.pdf": ("P200", "打印设备"),
    "service-policy.pdf": (None, "售后政策"),
}


def retrieval_query(text):
    # The MCP query language reads `key:value` as a metadata filter.
    return " ".join(_APPOINTMENT_SLOT.sub(" ", text).replace(":", " ").replace("：", " ").split())


def documents_from_result(result):
    if getattr(result, "is_error", getattr(result, "isError", False)):
        return []
    structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
    if not isinstance(structured, dict) or "citations" not in structured:
        structured = {}
        for block in result.content:
            if block.type == "text" and (match := _REFERENCES.search(block.text)):
                try:
                    structured = json.loads(match[1])
                except ValueError:
                    return []
                break
    documents = []
    for item in structured.get("citations", []) if isinstance(structured, dict) else []:
        if not isinstance(item, dict) or not isinstance(item.get("source"), str):
            continue
        source = PurePosixPath(item["source"].replace("\\", "/")).name
        profile = _DEMO_SOURCES.get(source)
        snippet, chunk_id, index = item.get("text_snippet"), item.get("chunk_id"), item.get("index")
        if not profile or not isinstance(snippet, str) or not snippet or not chunk_id or type(index) is not int:
            continue
        product, skill = profile
        if product and (product not in snippet or skill not in snippet):
            continue
        citation = {"source": source, "chunk_id": chunk_id, "index": index}
        if type(item.get("page")) is int and item["page"] > 0:
            citation["page"] = item["page"]
        documents.append({"id": chunk_id, "category": skill, "content": snippet, "citation": citation})
    return documents


class McpAfterSalesKnowledge:
    def __init__(self, python, root, collection):
        self.collection = collection
        self.server = StdioServerParameters(command=python, args=["-m", "src.mcp_server.server"], cwd=root,
                                            env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})

    @classmethod
    def from_environment(cls):
        python = os.getenv("AFTER_SALES_RAG_PYTHON")
        root = os.getenv("AFTER_SALES_RAG_ROOT")
        if bool(python) != bool(root):
            raise ValueError("Set both AFTER_SALES_RAG_PYTHON and AFTER_SALES_RAG_ROOT")
        return cls(python, root, os.getenv("AFTER_SALES_RAG_COLLECTION", "after_sales_v1")) if python else None

    async def search(self, query, top_k=3):
        try:
            async with stdio_client(self.server) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    async with asyncio.timeout(30):
                        await session.initialize()
                        result = await session.call_tool("query_knowledge_hub", {
                            "query": retrieval_query(query), "top_k": top_k, "collection": self.collection,
                        })
            return documents_from_result(result)
        except Exception as error:
            log.warning("After-sales MCP lookup failed: %s", type(error).__name__)
            return []
