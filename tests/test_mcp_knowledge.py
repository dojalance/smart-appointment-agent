import json

from mcp.types import CallToolResult, TextContent

from agents.consultant.mcp_knowledge import documents_from_result, retrieval_query


def test_reads_upstream_reference_block_and_drops_uncited_or_unapproved_results():
    references = {"citations": [
        {"source": "C:/manuals/r100-field-guide.pdf", "index": 2, "chunk_id": "r100-c0", "page": 1,
         "text_snippet": "R100 红灯需光网络工程师"},
        {"source": "other.pdf", "index": 3, "chunk_id": "other-c0", "text_snippet": "R100 光网络"},
        {"source": "p200-field-guide.pdf", "index": 4, "chunk_id": "p200-c0", "text_snippet": "R100 光网络"},
    ]}
    response = CallToolResult(content=[TextContent(
        type="text", text=f"**References (JSON):**\n```json\n{json.dumps(references)}\n```"
    )])
    docs = documents_from_result(response)
    assert docs == [{"id": "r100-c0", "category": "光网络", "content": "R100 红灯需光网络工程师",
                     "citation": {"source": "r100-field-guide.pdf", "index": 2, "chunk_id": "r100-c0", "page": 1}}]
    assert not documents_from_result(CallToolResult(content=response.content, isError=True))
    assert not documents_from_result(CallToolResult(content=[TextContent(type="text", text="无引用")]))


def test_retrieval_query_omits_booking_slot_and_metadata_filter_delimiter():
    assert retrieval_query("R100 故障:PON 红灯。时间 2030-01-08 16:00，60分钟。") == "R100 故障 PON 红灯。 。"
