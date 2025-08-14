# Backend/api/app/pipeline/s04_customer_api/__init__.py
from __future__ import annotations
from typing import List, Dict, Any
from ...schema.process_information import ProcessInformation
from ...utils.pp_client import PPClient

class CustomerApiPipeline:
    def __init__(self, scheme_id: str | None = None, meeting_id: str | None = None, bearer_token: str | None = None):
        self.scheme_id = scheme_id
        self.meeting_id = meeting_id
        self.bearer = bearer_token

    @staticmethod
    def list_meetings(bearer_token: str) -> List[Dict[str, Any]]:
        """Return a flat list of meetings for all schemes the user has."""
        client = PPClient(bearer_token)
        me = client.current_user()
        schemes = me.get("schemes", [])
        items: List[Dict[str, Any]] = []
        for s in schemes:
            sid = str(s["id"])
            sname = s.get("name", "")
            for m in client.scheme_meetings(sid):
                items.append({
                    "scheme_id": sid,
                    "scheme_name": sname,
                    "meeting_id": str(m.get("id")),
                    "title": m.get("title") or m.get("name") or "",
                    "date": m.get("meetingDate") or m.get("date") or "",
                })
        return items

    def process(self, info: ProcessInformation) -> ProcessInformation:
        """Fetch and attach meeting detail for downstream steps."""
        if not (self.scheme_id and self.meeting_id and self.bearer):
            raise ValueError("scheme_id, meeting_id and bearer_token are required")

        client = PPClient(self.bearer)
        detail = client.meeting_detail(self.scheme_id, self.meeting_id)
        # 将会议详情挂到 info，后续分类/摘要可用
        info.customer_meeting_detail = detail
        return info
