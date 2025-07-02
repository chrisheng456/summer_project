#api_token.py

import requests
import json

# ✅ 封装获取 token 的函数
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
            token = data["bearer_token"]
            expiry = data.get("bearer_token_expiry")
            print("✅ 登录成功！")
            print("🔐 Token:", token)
            print("⏳ 过期时间:", expiry)
            return token
        else:
            raise Exception("❌ 登录失败，未完成认证")
    else:
        raise Exception(f"❌ 网络错误，状态码: {response.status_code}")


# ✅ 使用 token 获取当前用户信息
def get_current_user_info(token):
    url = "https://pensionpal2test.azurewebsites.net/api/currentUser"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print("\n✅ 当前用户信息：")
        print("👤 用户名：", data.get("userName"))
        print("🆔 用户ID：", data.get("id"))
        print("📦 所属 Schemes：")
        for scheme in data.get("schemes", []):
            print("  📦 Scheme Name:", scheme.get("name"))
            print("  🆔 Scheme ID:", scheme.get("id"))
    else:
        print("❌ 获取失败，状态码：", response.status_code)


# ✅ 主程序执行
if __name__ == "__main__":
    try:
        bearer_token = get_bearer_token()
        get_current_user_info(bearer_token)
    except Exception as e:
        print(e)