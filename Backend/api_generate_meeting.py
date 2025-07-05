#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests
from datetime import datetime

# ——————————————————————————————————————————————————————————————
# 1. 获取 Token / 用户信息（无需修改）
# ——————————————————————————————————————————————————————————————

def get_bearer_token():
    """获取 API 认证令牌"""
    url = "https://pensionpal2test.azurewebsites.net/api/Logon"
    username = "ruixiong"
    password = "Ruixiong24937!"
    payload = {"username": username, "password": password}

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("authentication_complete", False):
        raise RuntimeError("❌ 登录失败：未完成认证")
    print("✅ 登录成功！Token 已获取")
    return data["bearer_token"]

def get_current_user_info(headers):
    """获取当前用户信息及其关联 Schemes"""
    url = "https://pensionpal2test.azurewebsites.net/api/currentUser"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    schemes = data.get("schemes", [])
    names = [s.get("name") for s in schemes]
    print(f"👤 用户 {data.get('username')} 关联 Schemes: {names}")
    return schemes

# ——————————————————————————————————————————————————————————————
# 2. 会议相关接口
# ——————————————————————————————————————————————————————————————

def get_scheme_meetings(scheme_id, headers):
    """拉取 Scheme 下的所有“会议”对象"""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"⚠️ 无法获取 Scheme={scheme_id} 的会议列表 (状态码 {resp.status_code})")
        return []
    return resp.json()

def get_meeting_details(scheme_id, meeting_id, headers):
    """拉取单个会议详情"""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings/{meeting_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"❌ 会议 {meeting_id} 详情获取失败 (状态码 {resp.status_code})")
        return None
    return resp.json()

# ——————————————————————————————————————————————————————————————
# 3. Document Vault（文档）相关接口
# ——————————————————————————————————————————————————————————————

def get_scheme_documents(scheme_id, headers):
    """拉取 Scheme 下的所有 Document Vault 文档元数据"""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/documents"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"⚠️ 无法获取 Scheme={scheme_id} 的文档列表 (状态码 {resp.status_code})")
        return []
    return resp.json()

def download_document_files(scheme_name, scheme_id, doc_meta, headers, base_dir="downloaded_json"):
    """
    根据文档元数据下载其所有附件：
      doc_meta 示例：
        {
          "id": 17812,
          "name": "Stewardship Report",
          "lengthInBytes": 11447,
          "attachment": [
            { "file": { "id": 234, "name": "report.pdf", ... } },
            ...
          ]
        }
    """
    folder = os.path.join(base_dir, scheme_name, "documents")
    os.makedirs(folder, exist_ok=True)

    for att in doc_meta.get("attachment", []):
        file_info = att.get("file", {})
        file_id   = file_info.get("id")
        file_name = file_info.get("name") or f"{file_id}"
        dl_url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/documents/{file_id}/file"
        r = requests.get(dl_url, headers=headers)
        if r.status_code == 200:
            path = os.path.join(folder, file_name)
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"📄 附件下载成功：{path}")
        else:
            print(f"❌ 附件 {file_id} 下载失败 (状态码 {r.status_code})")

# ——————————————————————————————————————————————————————————————
# 4. 主流程
# ——————————————————————————————————————————————————————————————

def main():
    # 1) 登录、构造 headers
    token = get_bearer_token()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # 2) 拿到所有 Scheme
    schemes = get_current_user_info(headers)

    # 3) 依次处理每个 Scheme
    for scheme in schemes:
        scheme_id   = scheme["id"]
        scheme_name = scheme["name"]
        print(f"\n=== 处理 Scheme: {scheme_name} (ID={scheme_id}) ===")

        # 3.1 会议列表 & 详情
        meetings = get_scheme_meetings(scheme_id, headers)
        for m in meetings:
            mid = m.get("id")
            detail = get_meeting_details(scheme_id, mid, headers)
            if not detail:
                continue

            out_dir = os.path.join("downloaded_json", scheme_name, "meetings")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{mid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(detail, f, ensure_ascii=False, indent=2)
            print(f"✔️ 会议详情已保存：{out_path}")

        # 3.2 Document Vault 文档列表 & 附件下载
        docs = get_scheme_documents(scheme_id, headers)
        for doc in docs:
            doc_id   = doc.get("id")
            # 保存元数据
            meta_dir  = os.path.join("downloaded_json", scheme_name, "documents")
            os.makedirs(meta_dir, exist_ok=True)
            meta_path = os.path.join(meta_dir, f"doc_{doc_id}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            print(f"✔️ 文档元数据已保存：{meta_path}")

            # 下载该文档的所有附件
            download_document_files(scheme_name, scheme_id, doc, headers)

    print("\n🎉 全部下载完成！")

if __name__ == "__main__":
    main()
