import httpx
from ...config import app_config
from ...schema.process_information import ProcessInformation
from datetime import datetime


def parse_iso(dt_str):
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


class CustomerApiPipeline:
    def __init__(self, scheme_id: str = None, meeting_id: str = None):
        self.scheme_id = scheme_id
        self.meeting_id = meeting_id

    def process(self, info: ProcessInformation):
        # 登录获取token
        login_url = "https://pensionpal2test.azurewebsites.net/api/Logon"
        username = app_config.customer_api.username
        password = app_config.customer_api.password
        payload = {"username": username, "password": password}
        with httpx.Client() as client:
            resp = client.post(login_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("authentication_complete", False):
                raise RuntimeError("❌ 登录失败：未完成认证")
            token = data["bearer_token"]
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }

            if not self.scheme_id or not self.meeting_id:
                # 如果未指定 scheme_id 和 meeting_id，则获取第一个
                # scheme_id 和 meeting_id

                # 获取当前用户信息
                user_url = (
                    "https://pensionpal2test.azurewebsites.net/api/currentUser"
                )
                user_resp = client.get(user_url, headers=headers)
                user_resp.raise_for_status()
                user_data = user_resp.json()
                schemes = user_data.get("schemes", [])
                if not schemes:
                    raise RuntimeError("❌ 当前用户无关联Scheme")
                self.scheme_id = schemes[0]["id"]
                # 获取会议列表
                meetings_url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{self.scheme_id}/meetings"
                meetings_resp = client.get(meetings_url, headers=headers)
                meetings_resp.raise_for_status()
                meetings = meetings_resp.json()
                if not meetings:
                    raise RuntimeError("❌ Scheme下无会议")
                self.meeting_id = meetings[0]["id"]

            # 获取会议详情
            meeting_detail_url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{self.scheme_id}/meetings/{self.meeting_id}"
            detail_resp = client.get(meeting_detail_url, headers=headers)
            detail_resp.raise_for_status()
            meeting_detail = detail_resp.json()

        # === agenda分段合并 ===
        # info.transcription: 本地转录（含start/end/speaker等）
        # meeting_detail: API拉取的会议数据（含agenda）
        lines = info.transcription if hasattr(info, "transcription") else []
        meeting = meeting_detail
        meeting_start = parse_iso(meeting["startTime"])
        for item in meeting.get("agenda", []):
            start_str = item.get("calculatedStartTime") or item.get(
                "startTime"
            )
            if not start_str:
                continue
            item_start = parse_iso(start_str)
            delta_start = (item_start - meeting_start).total_seconds()
            delta_end = delta_start + item.get("lengthMinutes", 0) * 60
            seg_lines = [
                ln
                for ln in lines
                if ln.get("start", 0) < delta_end
                and ln.get("end", 0) > delta_start
            ]
            item["lines"] = seg_lines
        info.customer_meeting_detail = meeting
