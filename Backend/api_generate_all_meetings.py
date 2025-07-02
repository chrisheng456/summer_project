import requests

# 1. 获取 Token（可以复用你的 auth_token.py）
from auth_token import get_bearer_token

token = get_bearer_token()
headers = {
     "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

# 2. 获取当前用户信息，拿到所有 scheme 和会议 ID
url_user = "https://pensionpal2test.azurewebsites.net/api/currentUser"
response = requests.get(url_user, headers=headers)
data = response.json()
schemes = data['schemes']

for scheme in schemes:
    scheme_name = scheme["name"]
    scheme_id = scheme["id"]
    # 3. 获取会议列表
    url_meetings = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings"
    res = requests.get(url_meetings, headers=headers)
    if res.status_code != 200:
        continue
    meetings = res.json()

    for meeting in meetings:
        meeting_id = meeting["id"]
        # 4. 调用你已有的 markdown 生成函数
        from markdown_generator import generate_markdown
        generate_markdown(scheme_id, scheme_name, meeting_id, headers)

