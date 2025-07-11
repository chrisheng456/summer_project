#!/usr/bin/env python3
# json_to_mysql.py

import json
import argparse
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(description="Ingest meeting JSON into MySQL")
    p.add_argument("json_file", help="Path to summarized_classified_segmented_meeting_data.json")
    p.add_argument("--db_host", default="localhost", help="MySQL host")
    p.add_argument("--db_port", default=3306, type=int, help="MySQL port")
    p.add_argument("--db_user", default="root", help="MySQL user")
    p.add_argument("--db_pass", default="", help="MySQL password")
    p.add_argument("--db_name", default="meetings_db", help="MySQL database name")
    return p.parse_args()

def main():
    args = parse_args()

    # 1. 读取 JSON
    with open(args.json_file, 'r', encoding='utf-8') as f:
        meeting = json.load(f)

    # 2. 连接数据库
    try:
        conn = mysql.connector.connect(
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_pass,
            database=args.db_name,
            charset='utf8mb4'
        )
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("❌ 用户名或密码错误")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("❌ 数据库不存在，请先创建")
        else:
            print(err)
        return

    cursor = conn.cursor()

    try:
        # ------------ 插入 event ------------
        evt_id = meeting['id']
        evt_name = meeting['name']
        evt_date = meeting.get('date')
        evt_start = meeting.get('startTime')
        evt_loc = meeting.get('location')

        cursor.execute("""
            INSERT INTO event (event_id, name, date, start_time, location)
            VALUES (%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              name=VALUES(name),
              date=VALUES(date),
              start_time=VALUES(start_time),
              location=VALUES(location)
        """, (
            evt_id,
            evt_name,
            evt_date.replace("T", " ") if evt_date else None,
            evt_start.replace("T", " ") if evt_start else None,
            evt_loc
        ))

        # ------------ 插入 attendee ------------
        for a in meeting.get('attendees', []):
            cursor.execute("""
                INSERT INTO attendee (attendee_id, event_id, name, attending, user_can_edit)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  name=VALUES(name),
                  attending=VALUES(attending),
                  user_can_edit=VALUES(user_can_edit)
            """, (
                a['id'],
                evt_id,
                a['name'],
                1 if a.get('attending') else 0,
                1 if a.get('userCanEdit') else 0
            ))

        # ------------ 插入 agenda_item ------------
        # 先清除本会议的旧议程项
        cursor.execute("DELETE FROM agenda_item WHERE event_id = %s", (evt_id,))

        for item in meeting.get('agenda', []):
            agenda_id = item['id']
            num        = item.get('number')
            title      = item.get('title')
            indent     = item.get('indent')
            cstart     = item.get('calculatedStartTime')
            length     = item.get('lengthMinutes')
            owner      = item.get('owner')
            label      = item.get('label')
            score      = item.get('label_score')
            summary    = item.get('summary')
            explain    = item.get('explanation')

            cursor.execute("""
                INSERT INTO agenda_item
                  (agenda_id, event_id, number, title, indent, calculated_start,
                   length_minutes, owner, label, label_score, summary, explanation)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                agenda_id, evt_id, num, title, indent,
                cstart.replace("T", " ") if cstart else None,
                length, owner, label, score, summary, explain
            ))

        # ------------ 插入 speech_segment ------------
        # 先清除本会议所有旧片段
        cursor.execute("""
            DELETE s FROM speech_segment s
            JOIN agenda_item a USING(agenda_id)
            WHERE a.event_id = %s
        """, (evt_id,))

        # JSON 最后一级结构：每条 “segments” 属于某个 agenda_item
        for item in meeting.get('agenda', []):
            aid = item['id']
            for seg in item.get('lines', []):  # 如果你的最终 JSON 里字段不是 lines，请替换
                speaker = seg.get('speaker')
                st = seg.get('start')
                ed = seg.get('end')
                text = seg.get('text')

                cursor.execute("""
                    INSERT INTO speech_segment
                      (agenda_id, speaker, start_time, end_time, text)
                    VALUES (%s,%s,%s,%s,%s)
                """, (
                    aid, speaker, st, ed, text
                ))

        conn.commit()
        print("✅ 数据已成功写入数据库！")

    except mysql.connector.Error as e:
        print("❌ 执行出错，回滚：", e)
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()