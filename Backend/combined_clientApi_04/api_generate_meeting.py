#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests
from datetime import datetime

# ============================================================
# 1. Authentication / User Info
# ============================================================

def get_bearer_token():
    """Authenticate and fetch an API bearer token."""
    url = "https://pensionpal2test.azurewebsites.net/api/Logon"
    username = "ruixiong"
    password = "Ruixiong24937!"
    payload = {"username": username, "password": password}

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("authentication_complete", False):
        raise RuntimeError("Login failed: authentication not complete")
    print("Login successful! Token retrieved.")
    return data["bearer_token"]


def get_current_user_info(headers):
    """Fetch the current user info along with associated Schemes."""
    url = "https://pensionpal2test.azurewebsites.net/api/currentUser"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    schemes = data.get("schemes", [])
    names = [s.get("name") for s in schemes]
    print(f" User {data.get('username')} is linked to Schemes: {names}")
    return schemes


# ============================================================
# 2. Meeting-related endpoints
# ============================================================

def get_scheme_meetings(scheme_id, headers):
    """Fetch all meetings under a given Scheme."""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f" Failed to fetch meetings for Scheme={scheme_id} (status {resp.status_code})")
        return []
    return resp.json()


def get_meeting_details(scheme_id, meeting_id, headers):
    """Fetch details for a specific meeting."""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings/{meeting_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f" Failed to fetch details for meeting {meeting_id} (status {resp.status_code})")
        return None
    return resp.json()


# ============================================================
# 3. Document Vault (documents) endpoints
# ============================================================

def get_scheme_documents(scheme_id, headers):
    """Fetch all Document Vault metadata under a given Scheme."""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/documents"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f" Failed to fetch documents for Scheme={scheme_id} (status {resp.status_code})")
        return []
    return resp.json()


def download_document_files(scheme_name, scheme_id, doc_meta, headers, base_dir="downloaded_json"):
    """
    Download all attachments for a given document metadata object.

    Example doc_meta:
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
            print(f" Attachment downloaded: {path}")
        else:
            print(f" Failed to download attachment {file_id} (status {r.status_code})")


# ============================================================
# 4. Main process
# ============================================================

def main():
    # 1) Authenticate and build request headers
    token = get_bearer_token()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    # 2) Fetch all Schemes linked to the current user
    schemes = get_current_user_info(headers)

    # 3) Iterate over each Scheme
    for scheme in schemes:
        scheme_id   = scheme["id"]
        scheme_name = scheme["name"]
        print(f"\n=== Processing Scheme: {scheme_name} (ID={scheme_id}) ===")

        # 3.1 Fetch meeting list and details
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
            print(f" Meeting detail saved: {out_path}")

        # 3.2 Fetch Document Vault list and download attachments
        docs = get_scheme_documents(scheme_id, headers)
        for doc in docs:
            doc_id   = doc.get("id")
            # Save metadata
            meta_dir  = os.path.join("downloaded_json", scheme_name, "documents")
            os.makedirs(meta_dir, exist_ok=True)
            meta_path = os.path.join(meta_dir, f"doc_{doc_id}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            print(f" Document metadata saved: {meta_path}")

            # Download attachments for this document
            download_document_files(scheme_name, scheme_id, doc, headers)

    print("\n All downloads completed!")


if __name__ == "__main__":
    main()
