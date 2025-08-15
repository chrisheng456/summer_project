# Backend/api/app/utils/pp_client.py
from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional


def _timeout() -> int:
    try:
        return int(os.getenv("CUSTOMER_API_TIMEOUT", "8"))
    except Exception:
        return 8


class PPClient:
    """
    轻量封装 PensionPal 客户 API。

    约定：
    - .env 里 `CUSTOMER_API_BASE_URL` 必须是“登录地址”，例如：
        https://pensionpal2test.azurewebsites.net/api/Logon
      我们会据此自动推导 API 根： https://.../api
    - login() 成功后返回远端的 bearer_token（一个 JWT 字符串）
    - 其余 GET 接口通过 Authorization: Bearer <token> 访问
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        base_login_url: Optional[str] = None,
    ) -> None:
        self.base_login_url = base_login_url or os.getenv("CUSTOMER_API_BASE_URL", "").strip()
        if not self.base_login_url:
            raise RuntimeError("CUSTOMER_API_BASE_URL 未配置；请在 .env 中设置为以 /Logon 结尾的登录地址。")

        # 统一出的 API 根（去掉最后一个路径段）
        if self.base_login_url.endswith("/"):
            self.base_login_url = self.base_login_url[:-1]
        self.api_root = self.base_login_url.rsplit("/", 1)[0]

        self.bearer = bearer_token
        self.session = requests.Session()

    # ---------------- 登录 ----------------

    def login(self, username: str, password: str) -> str:
        """
        调用 /Logon 登录，返回远端的 bearer_token（JWT）。
        """
        url = self.base_login_url
        resp = self.session.post(
            url,
            json={"username": username, "password": password},
            timeout=_timeout(),
        )
        resp.raise_for_status()
        data = resp.json()

        # 典型返回里会有这几个字段
        if not data.get("authentication_complete"):
            raise RuntimeError(f"login failed: {data}")

        token = data.get("bearer_token")
        if not token:
            # 兜底：有些实现把 token 放在其他键名
            for key in ("token", "access_token", "jwt", "BearerToken"):
                if data.get(key):
                    token = data[key]
                    break

        if not token:
            raise RuntimeError(f"cannot find token in login response: {data}")

        self.bearer = token
        return token

    # ---------------- 通用 ----------------

    def _headers(self) -> Dict[str, str]:
        if not self.bearer:
            raise RuntimeError("缺少 bearer_token，请先调用 login() 或在构造时传入。")
        return {"Authorization": f"Bearer {self.bearer}", "Accept": "application/json"}

    # ---------------- 业务 API ----------------

    def current_user(self) -> Dict[str, Any]:
        url = f"{self.api_root}/currentUser"
        r = self.session.get(url, headers=self._headers(), timeout=_timeout())
        r.raise_for_status()
        return r.json()

    def scheme_meetings(self, scheme_id: str) -> List[Dict[str, Any]]:
        url = f"{self.api_root}/scheme/{scheme_id}/meetings"
        r = self.session.get(url, headers=self._headers(), timeout=_timeout())
        r.raise_for_status()
        return r.json()

    def meeting_detail(self, scheme_id: str, meeting_id: str) -> Dict[str, Any]:
        url = f"{self.api_root}/scheme/{scheme_id}/meetings/{meeting_id}"
        r = self.session.get(url, headers=self._headers(), timeout=_timeout())
        r.raise_for_status()
        return r.json()

    # ---------------- 汇总封装：列会议 ----------------

    def list_meetings(self) -> List[Dict[str, Any]]:
        """
        读取“我的 schemes”，逐个拉取会议并做字段归一：
        返回元素结构统一为：
        {
          "scheme_id": "...",
          "scheme_name": "...",
          "meeting_id": "...",
          "title": "...",
          "date": "yyyy-mm-ddTHH:MM:SS"  # 原样透传/兜底
        }
        """
        user = self.current_user()
        # 兼容几种可能的字段命名
        schemes = (
            user.get("schemes")
            or user.get("Schemes")
            or user.get("data", {}).get("schemes")
            or []
        )

        out: List[Dict[str, Any]] = []
        for s in schemes:
            sid = str(s.get("id") or s.get("scheme_id") or s.get("SchemeId"))
            sname = s.get("name") or s.get("scheme_name") or s.get("SchemeName") or ""
            if not sid:
                continue
            try:
                ms = self.scheme_meetings(sid)
            except Exception:
                ms = []

            for m in ms or []:
                mid = str(
                    m.get("id")
                    or m.get("meeting_id")
                    or m.get("MeetingId")
                )
                title = (
                    m.get("title")
                    or m.get("name")
                    or m.get("meeting_title")
                    or ""
                )
                date = (
                    m.get("date")
                    or m.get("meeting_date")
                    or m.get("MeetingDate")
                    or m.get("Date")
                )
                out.append(
                    {
                        "scheme_id": sid,
                        "scheme_name": sname,
                        "meeting_id": mid,
                        "title": title,
                        "date": date,
                    }
                )
        return out
