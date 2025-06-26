# Backend/icsi_words2json.py
"""
ICSI  Words-XML  ➜  句子级 JSON
用法:
    python icsi_words2json.py                 # 批量转换全部会议
    python icsi_words2json.py --meeting Bdb001
"""

import argparse, glob, json, pathlib
from   lxml import etree          # pip install lxml

# ----------------------------------------------------------------------
# 把单个 *.words.xml 解析成若干句子
# ----------------------------------------------------------------------
def parse_words_xml(xf: pathlib.Path):
    tree  = etree.parse(str(xf))
    words = []
    for w in tree.findall(".//w"):
        tok = (w.attrib.get("orth") or w.attrib.get("punc") or
               w.attrib.get("pron") or (w.text or "").strip())
        if not tok:
            continue

        st   = float(w.attrib.get("starttime", 0) or 0)
        et_r = w.attrib.get("endtime")
        et   = float(et_r) if et_r and et_r.strip() else None

        spk  = (w.attrib.get("speaker") or w.attrib.get("who")
                or w.getparent().attrib.get("speaker", "")
                or w.getparent().attrib.get("who", "")
                or "UNK")
        words.append((st, et, spk, tok))

    # 用下一词起始时间填补缺失 endtime
    for i in range(len(words) - 1):
        if words[i][1] is None:
            words[i] = (*words[i][:1], words[i + 1][0], *words[i][2:])
    if words and words[-1][1] is None:           # 最后一词仍缺失
        st, _, spk, tok = words[-1]
        words[-1] = (st, st, spk, tok)

    # 简单按终止符分句
    sents, cur = [], []
    for st, et, spk, tok in words:
        cur.append((st, et, spk, tok))
        if tok.endswith((".", "!", "?", "--")):
            sents.append(_merge(cur)); cur = []
    if cur:
        sents.append(_merge(cur))
    return sents


def _merge(chunk):
    st, _, spk, _ = chunk[0]
    _,  et, _,  _ = chunk[-1]
    txt = " ".join(tok for *_, tok in chunk)
    return {"speaker": spk or "UNK", "start": st, "end": et, "text": txt}

# ----------------------------------------------------------------------
if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent      # Backend/
    default_ann = here / "dataset" / "ICSI" / "annotations" / "ICSI" / "Words"
    default_out = here / "dataset" / "ICSI" / "icsi_json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", help="会议 ID，如 Bdb001")
    ap.add_argument("--ann_dir", default=str(default_ann),
                    help="Words XML 根目录")
    ap.add_argument("--outdir",  default=str(default_out),
                    help="输出 JSON 目录")
    args = ap.parse_args()

    ann_dir = pathlib.Path(args.ann_dir)
    outdir  = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 要转换哪些会议？
    if args.meeting:
        meetings = [args.meeting]
    else:
        meetings = sorted({pathlib.Path(f).stem.split(".")[0]
                           for f in ann_dir.glob("*.words.xml")})
        print(f"🗂  发现 {len(meetings)} 个会议，将全部转换…")

    # 2) 逐会议处理
    for m in meetings:
        xml_files = sorted(ann_dir.glob(f"{m}.*.words.xml"))
        if not xml_files:
            print(f"⚠️  {m}: 未找到 *.words.xml，跳过")
            continue

        all_sents = []
        for xf in xml_files:
            all_sents.extend(parse_words_xml(xf))

        out_file = outdir / f"{m}.json"
        out_file.write_text(json.dumps(all_sents, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"✅ {m}: {len(all_sents)} sentences → {out_file}")
