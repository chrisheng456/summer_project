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

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        if data.get("authentication_complete"):
            print("✅ 登录成功！Token已获取")
            return data["bearer_token"]
        else:
            raise Exception("❌ 登录失败：未完成认证")
    else:
        raise Exception(f"❌ 获取Token失败，状态码: {response.status_code}")


def get_current_user_info(headers):
    """获取当前用户信息及其关联的Scheme"""
    url = "https://pensionpal2test.azurewebsites.net/api/currentUser"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"❌ 无法获取用户信息 (状态码 {response.status_code})")

    return response.json()


def get_scheme_meetings(scheme_id, headers):
    """获取特定Scheme下的所有会议列表"""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"⚠️ 无法获取Scheme {scheme_id}的会议列表 (状态码 {response.status_code})")
        return []

    return response.json()


def get_meeting_details(scheme_id, scheme_name, meeting_id, headers):
    """获取单个会议的详细信息"""
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings/{meeting_id}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"❌ 无法获取会议 {meeting_id} 详情 (状态码 {response.status_code})")
        return None

    meeting_data = response.json()

    # 结构化会议数据
    structured_meeting = {
        "scheme_id": scheme_id,
        "scheme_name": scheme_name,
        "meeting_id": meeting_id,
        "title": meeting_data.get("name", "未命名会议"),
        "start_time": meeting_data.get("startTime", "N/A"),
        "location": meeting_data.get("location", "N/A"),
        "participants": [],
        "agenda_items": []
    }

    # 处理参与者
    for person in meeting_data.get("attendees", []):
        structured_meeting["participants"].append({
            "name": person.get("name", "未知"),
            "attending": person.get("attending", False)
        })

    # 处理议程项目
    for item in meeting_data.get("agenda", []):
        structured_meeting["agenda_items"].append({
            "title": item.get("title", "未命名议程"),
            "owner": item.get("owner", "N/A"),
            "duration_minutes": item.get("lengthMinutes", 0)
        })

    return structured_meeting


def main():
    try:
        print("🚀 开始获取会议数据...")

        # 获取Token并设置headers
        token = get_bearer_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # 获取当前用户信息
        user_data = get_current_user_info(headers)
        schemes = user_data.get('schemes', [])

        if not schemes:
            print("⚠️ 用户没有关联任何scheme")
            return

        print(f"🔍 找到 {len(schemes)} 个scheme")

        # 创建所有会议数据的集合
        all_meetings = []
        total_meeting_count = 0

        # 遍历所有scheme
        for scheme in schemes:
            scheme_name = scheme.get("name", "未命名Scheme")
            scheme_id = scheme.get("id")

            if not scheme_id:
                continue

            print(f"\n📂 处理scheme: {scheme_name} (ID: {scheme_id})")

            # 获取当前scheme的所有会议列表
            meetings = get_scheme_meetings(scheme_id, headers)
            print(f"  找到 {len(meetings)} 个会议")

            # 处理每个会议
            for meeting in meetings:
                meeting_id = meeting.get("id")

                if not meeting_id:
                    continue

                meeting_details = get_meeting_details(scheme_id, scheme_name, meeting_id, headers)

                if meeting_details:
                    all_meetings.append(meeting_details)
                    total_meeting_count += 1
                    print(f"  已获取会议: {meeting_details['title']} (ID: {meeting_id})")

        # 创建最终结果结构
        result = {
            "generated_at": datetime.now().isoformat(),
            "total_schemes": len(schemes),
            "total_meetings": total_meeting_count,
            "meetings": all_meetings
        }

        # 保存为单个JSON文件
        filename = "all_meetings_data.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 完成! 共处理 {len(schemes)} 个scheme, {total_meeting_count} 个会议")
        print(f"📁 所有数据已保存到: {filename}")

    except Exception as e:
        print(f"❌ 程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()