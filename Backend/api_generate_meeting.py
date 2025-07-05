#!/usr/bin/env python3
# api_generate_all_meetings.py

import requests
import json
import os
from datetime import datetime

def get_bearer_token():
    """获取API认证令牌"""
    url = "https://pensionpal2test.azurewebsites.net/api/Logon"
    username = "ruixiong"
    password = "Ruixiong24937!"

    payload = {"username": username, "password": password}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if data.get("authentication_complete"):
        print("✅ 登录成功，已获取 Bearer Token")
        return data["bearer_token"]
    else:
        raise RuntimeError("❌ 登录失败：authentication_complete=False")

def get_current_user_info(headers):
    """获取当前用户信息及其关联的 Scheme 列表"""
    url = "https://pensionpal2test.azurewebsites.net/api/currentUser"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    schemes = data.get("schemes") or []
    print(f"→ 当前用户共关联 {len(schemes)} 个 Scheme")
    return schemes

def fetch_meetings_list(scheme_id, headers):
    """拉取某个 scheme 下的会议列表"""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def fetch_meeting_detail(scheme_id, meeting_id, headers):
    """拉取单个会议的完整详情"""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings/{meeting_id}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def main():
    # 1. 登录拿 Token
    token = get_bearer_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 2. 获取当前用户关联的 schemes
    schemes = get_current_user_info(headers)

    all_meetings = []
    # 3. 遍历每个 scheme，拉会议列表 & 详情
    for scheme in schemes:
        sid  = scheme["id"]
        name = scheme.get("name", "")
        print(f"\n=== Scheme {sid} «{name}» ===")
        meetings = fetch_meetings_list(sid, headers)
        print(f"  • 共 {len(meetings)} 个 meeting")

        for m in meetings:
            mid = m["id"]
            print(f"    - Meeting {mid}: ", end="", flush=True)
            try:
                detail = fetch_meeting_detail(sid, mid, headers)
            except Exception as e:
                print(f"❌ 拉取失败 ({e})")
                continue
            # 给 detail 里打标记，方便后续追溯
            detail["_scheme_id"]   = sid
            detail["_scheme_name"] = name
            all_meetings.append(detail)
            print("✔")

    # 4. 保存到文件
    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_schemes":  len(schemes),
        "total_meetings": len(all_meetings),
        "meetings":       all_meetings
    }
    here = os.path.dirname(__file__)
    out_path = os.path.join(here, "all_meetings_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已将 {len(all_meetings)} 条会议详情写入 `{out_path}`")

if __name__ == "__main__":
    main()
