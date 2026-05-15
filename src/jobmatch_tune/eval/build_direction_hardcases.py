from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import compose_jd_input_text
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


HARDCASE_DIRECTIONS = {
    "tencent_1987809143616069632": "前端开发",
    "tencent_1936827639495000064": "前端开发",
    "tencent_1988523500201205760": "前端开发",
    "tencent_1931180913459306496": "前端开发",
    "tencent_1983062025584140288": "后端开发",
    "tencent_2026648959719735296": "后端开发",
    "tencent_1906553057487982592": "后端开发",
    "tencent_2029541842982432768": "后端开发",
    "tencent_1946505217747058688": "后端开发",
    "tencent_2017120081338851328": "后端开发",
    "tencent_1959981063380029440": "后端开发",
    "tencent_2004170771991252992": "后端开发",
    "tencent_2006202335759585280": "后端开发",
    "tencent_1981562228847042560": "后端开发",
    "tencent_2020809468387946496": "测试开发",
    "tencent_2039181783093903360": "测试开发",
    "tencent_2027183384949846016": "测试开发",
    "tencent_1991322809237921792": "测试开发",
    "tencent_2015977043774300160": "测试开发",
    "tencent_2043623844203364352": "测试开发",
    "tencent_2034471887597371392": "测试开发",
    "tencent_2034277345287897088": "测试开发",
    "tencent_2052012609846292480": "测试开发",
    "tencent_2027612552653078528": "测试开发",
    "tencent_2027658886831570944": "算法工程",
    "tencent_2037101969994313728": "算法工程",
    "tencent_2042508679395311616": "算法工程",
    "tencent_2028325900583600128": "算法工程",
    "tencent_1976549654753599488": "算法工程",
    "tencent_2027548243965145088": "算法工程",
    "tencent_1934984934909386752": "算法工程",
    "tencent_2043890152685858816": "算法工程",
    "tencent_2042431102441910272": "算法工程",
    "tencent_2029126452007567360": "算法工程",
    "tencent_1985973381228552192": "AI应用开发",
    "tencent_2010621139876995072": "AI应用开发",
    "tencent_2020821173599895552": "AI应用开发",
    "tencent_2035224441180553216": "AI应用开发",
    "tencent_2034114862627581952": "AI应用开发",
    "tencent_2034117544100655104": "AI应用开发",
    "tencent_2046210778876510208": "AI应用开发",
    "tencent_1977555296700223488": "AI应用开发",
    "tencent_2052685072754196480": "AI应用开发",
    "tencent_2033551186275233792": "AI应用开发",
    "tencent_2044415716324700160": "AI应用开发",
    "tencent_2038877531406499840": "AI应用开发",
    "tencent_2028299866760966144": "AI应用开发",
    "tencent_2034114867123875840": "AI应用开发",
    "tencent_2039533180888973312": "AI应用开发",
    "tencent_2037177915745136640": "AI应用开发",
    "tencent_2043950303379877888": "前端开发",
    "tencent_1891443603608215552": "前端开发",
    "tencent_2013906220049653760": "前端开发",
    "tencent_2028682235867201536": "前端开发",
    "tencent_1988523500201205760": "前端开发",
    "tencent_1931180913459306496": "前端开发",
    "tencent_2029541840893669376": "后端开发",
    "tencent_1985649233327448064": "后端开发",
    "tencent_2039142871956877312": "后端开发",
    "tencent_1989232491273281536": "后端开发",
    "tencent_1966029487942553600": "后端开发",
    "tencent_1859161690281631744": "后端开发",
    "tencent_2030831082567528448": "后端开发",
    "tencent_1964869584070795264": "后端开发",
    "tencent_1998283271099797504": "后端开发",
    "tencent_1923261618968457216": "后端开发",
    "tencent_2017094358532247552": "测试开发",
    "tencent_2034471887597371392": "测试开发",
    "tencent_2043623844203364352": "测试开发",
    "tencent_2039181783093903360": "测试开发",
    "tencent_2027183384949846016": "测试开发",
    "tencent_1991322809237921792": "测试开发",
    "tencent_2015977043774300160": "测试开发",
    "tencent_2034277345287897088": "测试开发",
    "tencent_2052012609846292480": "测试开发",
    "tencent_2027612552653078528": "测试开发",
    "tencent_2027658886831570944": "算法工程",
    "tencent_2037101969994313728": "算法工程",
    "tencent_2042508679395311616": "算法工程",
    "tencent_2028325900583600128": "算法工程",
    "tencent_1976549654753599488": "算法工程",
    "tencent_2027548243965145088": "算法工程",
    "tencent_1934984934909386752": "算法工程",
    "tencent_2043890152685858816": "算法工程",
    "tencent_2042431102441910272": "算法工程",
    "tencent_2029126452007567360": "算法工程",
    "tencent_2010621139876995072": "AI应用开发",
    "tencent_2020821173599895552": "AI应用开发",
    "tencent_2035224441180553216": "AI应用开发",
    "tencent_2034114862627581952": "AI应用开发",
    "tencent_2034117544100655104": "AI应用开发",
    "tencent_2046210778876510208": "AI应用开发",
    "tencent_1977555296700223488": "AI应用开发",
    "tencent_2052685072754196480": "AI应用开发",
    "tencent_2033551186275233792": "AI应用开发",
    "tencent_2038877531406499840": "AI应用开发",
    "tencent_1968944641277579264": "AI应用开发",
    "tencent_1927639956445069312": "AI应用开发",
    "tencent_2021936804462231552": "AI应用开发",
    "tencent_2045025466083078144": "AI应用开发",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean.jsonl")
    parser.add_argument("--out", default="data/eval/jd_direction_hardcases_100.jsonl")
    args = parser.parse_args()

    rows = {row["id"]: row for row in read_jsonl(args.input)}
    output: list[dict[str, Any]] = []
    for row_id, direction in HARDCASE_DIRECTIONS.items():
        row = rows[row_id]
        output.append(
            {
                "id": f"{row_id}_jd_parse",
                "source_id": row_id,
                "job_title": row.get("job_title", ""),
                "task": "jd_parse",
                "text": compose_jd_input_text(row),
                "label": {"岗位方向": direction},
            }
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, output)
    print(f"wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
