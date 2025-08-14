# Backend/api/app/utils/pp_client.py
from __future__ import annotations
import requests
from typing import List, Dict, Any

BASE = "https://pensionpal2test.azurewebsites.net/api"

class PPClient:
    def __init__(self, bearer_token: str | None = None):
        self.bearer = bearer_token

    @staticmethod
    def login(username: str, password: str) -> str:
        url = f"{BASE}/Logon"
        resp = requests.post(url, json={"username": username, "password": password})
        resp.raise_for_status()
        data = resp.json()
        if not data.get("authentication_complete"):
            raise RuntimeError("login failed")
        return data["bearer_token"]

    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.bearer}",
        }

    # ----- user & schemes -----
    def current_user(self) -> Dict[str, Any]:
        url = f"{BASE}/currentUser"
        resp = requests.get(url, headers=self.headers())
        resp.raise_for_status()
        return resp.json()

    # ----- meetings -----
    def scheme_meetings(self, scheme_id: str) -> List[Dict[str, Any]]:
        url = f"{BASE}/scheme/{scheme_id}/meetings"
        r = requests.get(url, headers=self.headers())
        r.raise_for_status()
        return r.json()

    def meeting_detail(self, scheme_id: str, meeting_id: str) -> Dict[str, Any]:
        url = f"{BASE}/scheme/{scheme_id}/meetings/{meeting_id}"
        r = requests.get(url, headers=self.headers())
        r.raise_for_status()
        return r.json()
