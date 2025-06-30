import requests

def generate_markdown(scheme_id, scheme_name, meeting_id, headers):
    # 请求会议详情
    url = f"https://pensionpal2test.azurewebsites.net/api/scheme/{scheme_id}/meetings/{meeting_id}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"❌ 无法获取会议详情：{meeting_id}")
        return

    data = response.json()
    print("🧪 原始数据结构：", data)

    # ✅ 正确提取字段（只保留一次）
    meeting_title = data["name"]  # 正确字段
    start_time = data["startTime"]
    location = data.get("location", "N/A")
    participants = data.get("attendees", [])
    agenda_items = data.get("agenda", [])  # ✅ 是 agenda，不是 agendaItems

    # ✅ 生成 Markdown 文本
    markdown_text = f"""# 📝 {meeting_title}
🏦 Scheme: {scheme_name}  
📅 Date: {start_time}  
📍 Location: {location}

## 👥 Participants
"""

    for person in participants:
        name = person["name"]
        is_present = person.get("attending", False)  # ✅ 修复出席字段
        status = "✅ 出席" if is_present else "❌ 缺席"
        markdown_text += f"- {name} ({status})\n"

    markdown_text += "\n## 🗂 Agenda Items\n"

    for i, item in enumerate(agenda_items, 1):
        title = item["title"]
        owner = item.get("owner", "N/A")  # ✅ 是字符串，不是 dict
        duration = item.get("lengthMinutes", 0)  # ✅ 修正字段名
        markdown_text += f"### {i}. {title}\n👤 Owner: {owner}  ⏱ Duration: {duration} mins\n\n"

    # ✅ 写入 Markdown 文件
    filename = f"meeting_minutes_{scheme_id}_{meeting_id}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"✅ Markdown 报告已保存: {filename}")
