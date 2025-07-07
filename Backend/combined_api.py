import json
from datetime import datetime

def parse_iso(dt_str):
    """
    将 ISO8601 字符串（支持带 Z 的 UTC）转成 datetime。
    """
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    return datetime.fromisoformat(dt_str)

def segment_transcript(all_meet_file, transcript_file, output_file):
    # 1. 读取 all_meetings_data.json
    with open(all_meet_file, 'r', encoding='utf-8') as f:
        all_meet = json.load(f)

    # 2. 读取转写结果 JSON
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    lines = transcript.get('lines', [])

    # 3. 按每个会议、每个议程项切分
    for meeting in all_meet.get('meetings', []):
        meeting_start = parse_iso(meeting['startTime'])

        for item in meeting.get('agenda', []):
            # —— 后备字段跳过逻辑 ——
            # 优先取 calculatedStartTime，否则 fallback 到 startTime
            start_str = item.get('calculatedStartTime') or item.get('startTime')
            if not start_str:
                # 两者都没有，打印警告并跳过
                print(f"⚠️ 跳过议程项（无开始时间字段）：{item.get('title', '<no-title>')}")
                continue

            item_start = parse_iso(start_str)
            delta_start = (item_start - meeting_start).total_seconds()
            delta_end   = delta_start + item.get('lengthMinutes', 0) * 60

            # 收集与该时间段有交集的所有转录行
            seg_lines = [
                ln for ln in lines
                if ln.get('start', 0) < delta_end and ln.get('end', 0) > delta_start
            ]
            item['lines'] = seg_lines

    # 4. 写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_meet, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成：{output_file}")

if __name__ == '__main__':
    segment_transcript(
        # 请根据实际路径调整：
        'all_meetings_data.json',                            # 你的 all_meetings_data.json（注意文件名要与磁盘一致）
        'Trustee Meeting Recording (30 June 2025) V1.json',      # 你的转写文件
        'segmented_meeting_data.json'                            # 输出文件
    )
