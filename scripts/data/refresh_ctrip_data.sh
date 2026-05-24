#!/usr/bin/env bash
set -euo pipefail

out="${1:-data/raw/ctrip_jd_raw.jsonl}"
db="${2:-data/jobmatch_tune.sqlite3}"
page_size="${CTRIP_PAGE_SIZE:-100}"
max_pages="${CTRIP_MAX_PAGES:-}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

write_payload() {
  local page_index="$1"
  python - "$workdir/payload_${page_index}.json" "$page_index" "$page_size" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
page_index = int(sys.argv[2])
page_size = int(sys.argv[3])
payload = {
    "head": {"language": "zh-CN", "version": "1"},
    "condition": {"pageIndex": page_index, "pageSize": page_size},
}
path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
PY
}

fetch_page() {
  local page_index="$1"
  write_payload "$page_index"
  curl --compressed -L "https://careers.ctrip.com/api/hrrecruit/getJobAd" \
    -H "Content-Type: application/json" \
    --data-binary "@$workdir/payload_${page_index}.json" \
    -o "$workdir/page_${page_index}.json"
}

fetch_page 1

total_pages="$(python - "$workdir/page_1.json" "$page_size" "$max_pages" <<'PY'
from pathlib import Path
import json
import math
import sys

path = Path(sys.argv[1])
page_size = int(sys.argv[2])
max_pages = sys.argv[3]
data = json.loads(path.read_text(encoding="utf-8"))
total = int((data.get("retValue") or {}).get("total") or 0)
pages = math.ceil(total / page_size) if page_size > 0 else 0
if max_pages:
    pages = min(pages, int(max_pages))
print(pages)
PY
)"

if [[ "$total_pages" -gt 1 ]]; then
  for ((page=2; page<=total_pages; page++)); do
    fetch_page "$page"
  done
fi

PYTHONPATH=src python - "$workdir" "$out" "$db" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from jobmatch_tune.crawler.ctrip_careers import convert_ctrip_job, is_probably_tech_job
from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.utils.io import read_jsonl, write_jsonl

workdir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
db_path = sys.argv[3]

crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
rows: list[dict] = []
seen_ids: set[str] = set()

for page_file in sorted(workdir.glob("page_*.json")):
    data = json.loads(page_file.read_text(encoding="utf-8"))
    posts = (data.get("retValue") or {}).get("recruitJobAdList") or []
    for post in posts:
        if not is_probably_tech_job(post):
            continue
        post_id = str(post.get("id") or "").strip()
        if not post_id or post_id in seen_ids:
            continue
        row = convert_ctrip_job(post, crawl_time=crawl_time)
        if not row.get("raw_text"):
            continue
        rows.append(row)
        seen_ids.add(post_id)

jsonl_rows = [{key: value for key, value in row.items() if key != "html"} for row in rows]
merged_rows = jsonl_rows
by_id: dict[str, dict] = {}
try:
    for row in read_jsonl(str(out_path)):
        row_id = str(row.get("id") or "")
        if row_id:
            by_id[row_id] = row
except FileNotFoundError:
    pass
for row in jsonl_rows:
    by_id[str(row["id"])] = row
merged_rows = list(by_id.values())

write_jsonl(str(out_path), merged_rows)
init_db(db_path)
upsert_jd_raw(db_path, rows)
print(f"crawled {len(rows)} Ctrip tech-like posts")
print(f"wrote raw JSONL: {out_path} ({len(merged_rows)} rows)")
print(f"upserted SQLite: {db_path}")
PY
