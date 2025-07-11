"""
Updated JSON to SQL import script for the revised schema including action, decision, and conflict per speech segment:
- Inserts into event, attendee
- Inserts into agenda_item (with label/summary fields)
- Inserts original lines into agenda_line
- Inserts speech segments (with summary, action, decision, conflict) into speech_segment
"""
import json
import mysql.connector
from mysql.connector import errorcode


def main():
    # 1. 读取 JSON 数据
    with open('summarized_classified_segmented_meeting_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 连接数据库
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='1234',
            database='meeting_details',
            charset='utf8mb4'
        )
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("用户名或密码错误")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("数据库不存在，请先创建 meeting_result 数据库")
        else:
            print(err)
        return

    cursor = conn.cursor()

    try:
        # 3. 插入 event
        evt = data
        cursor.execute(
            """
            INSERT IGNORE INTO event (id, name, date, start_time, location)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                evt['id'],
                evt['name'],
                evt['date'].replace('T', ' '),
                evt['startTime'].replace('T', ' '),
                evt['location']
            )
        )

        # 4. 插入 attendee
        for a in evt.get('attendees', []):
            cursor.execute(
                """
                INSERT IGNORE INTO attendee (id, event_id, name, attending, userCanEdit)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    a['id'],
                    evt['id'],
                    a['name'],
                    1 if a.get('attending') else 0,
                    1 if a.get('userCanEdit') else 0
                )
            )

        # 5. 插入 agenda_item，以及相关的 lines 与 segments
        for item in evt.get('agenda', []):
            cursor.execute(
                """
                INSERT IGNORE INTO agenda_item
                  (id, event_id, number, title, owner, label, label_score, summary, explanation)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    item['id'],
                    evt['id'],
                    item.get('number'),
                    item.get('title'),
                    item.get('owner'),
                    item.get('label'),
                    item.get('label_score'),
                    item.get('summary'),
                    item.get('explanation')
                )
            )
            # 原始逐行发言插入 agenda_line
            for line in item.get('lines', []):
                cursor.execute(
                    """
                    INSERT IGNORE INTO agenda_line
                      (agenda_item_id, start_time, end_time, speaker, `text`)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        item['id'],
                        line.get('start'),
                        line.get('end'),
                        line.get('speaker'),
                        line.get('text')
                    )
                )
                # 逐段发言插入 speech_segment，包含 action, decision, conflict
                cursor.execute(
                    """
                    INSERT IGNORE INTO speech_segment
                      (agenda_item_id, speaker, start_time, end_time, `text`, summary, action, decision, conflict)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        item['id'],
                        line.get('speaker'),
                        line.get('start'),
                        line.get('end'),
                        line.get('text'),
                        line.get('summary'),
                        line.get('action'),
                        line.get('decision'),
                        1 if line.get('conflict') else 0
                    )
                )

        # 6. 提交事务
        conn.commit()
        print("数据导入完成！")

    except mysql.connector.Error as err:
        print(f"导入过程中出现错误: {err}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    main()
