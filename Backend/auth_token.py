import requests

# 👇 你的 token（可以也从上一步保存）
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJVc2VySWQiOiI1MTgiLCJBdXRoVHlwZSI6IkJlYXJlciIsImV4cCI6MTc1MTMwMjQ5MiwiaXNzIjoiUGVuc2lvblBhbCBMdGQiLCJhdWQiOiJQZW5zaW9uUGFsMiBUZXN0IFNpdGUifQ.4qfgbrK0Ne4Dv_Ch0HtkhX_HQ2kAoPEJ4QaRsoQt0Cs"

# 设置请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {bearer_token}"
}

# 请求 currentUser
url = "https://pensionpal2test.azurewebsites.net/api/currentUser"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("✅ 当前用户信息：")
    print("用户名：", data["userName"])
    print("用户ID：", data["id"])
    for scheme in data["schemes"]:
        print("📦 Scheme Name:", scheme["name"])
        print("🆔 Scheme ID:", scheme["id"])
else:
    print("❌ 获取失败，状态码：", response.status_code)

import requests

def get_bearer_token():
    url = "https://pensionpal2test.azurewebsites.net/api/Logon"
    username = "ruixiong"
    password = "Ruixiong24937!"

    payload = {
        "username": username,
        "password": password
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        if data.get("authentication_complete"):
            return data["bearer_token"]
        else:
            raise Exception("❌ 认证失败，未返回 token")
    else:
        raise Exception(f"❌ 获取 token 失败，状态码：{response.status_code}")
