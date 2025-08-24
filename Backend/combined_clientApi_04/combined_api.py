import json
from datetime import datetime

def parse_iso(dt_str):
    """
    Parse ISO8601 string (with optional trailing Z) into a datetime.
    """
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    return datetime.fromisoformat(dt_str)

def segment_transcript(all_meet_file, transcript_file, output_file):
    # 1. load all_meet_file
    with open(all_meet_file, 'r', encoding='utf-8') as f:
        all_meet_raw = json.load(f)

    # 2. normalize to a list of meetings
    if isinstance(all_meet_raw, dict) and 'meetings' in all_meet_raw:
        meetings = all_meet_raw['meetings']
        wrap_in_list = True
    elif isinstance(all_meet_raw, dict):
        meetings = [all_meet_raw]
        wrap_in_list = False
    else:
        raise ValueError(f"Unsupported format in {all_meet_file}")

    # 3. load transcript (with speaker tags)
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    lines = transcript.get('lines', [])

    # 4. for each meeting / agenda item, slice out lines
    for meeting in meetings:
        meeting_start = parse_iso(meeting['startTime'])

        for item in meeting.get('agenda', []):
            # pick start time field
            start_str = item.get('calculatedStartTime') or item.get('startTime')
            if not start_str:
                print(f"⚠ skip agenda item without start time: {item.get('title')}")
                continue

            item_start = parse_iso(start_str)
            delta_start = (item_start - meeting_start).total_seconds()
            delta_end   = delta_start + item.get('lengthMinutes', 0) * 60

            # collect overlapping transcript lines
            seg_lines = [
                ln for ln in lines
                if ln.get('start', 0) < delta_end and ln.get('end', 0) > delta_start
            ]
            item['lines'] = seg_lines

    # 5. write back, preserving original top‐level shape
    if wrap_in_list:
        out_data = {'meetings': meetings}
    else:
        out_data = meetings[0]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f" generated {output_file}")

if __name__ == '__main__':
    # example invocation for single‐meeting 714.json:
    segment_transcript(
        '../714.json',
        'Trustee Meeting Recording (30 June 2025) V1_V2_with_speakers.json',
        'segmented_meeting_data.json'
    )
