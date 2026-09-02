from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobmatch_tune.resume.ingest import ingest_resume
from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d(?:[-\s]?\d){8}(?!\d)")
INTERNATIONAL_PHONE_RE = re.compile(
    r"(?<![\w.-])(?:\+\d{1,3}[\s.-]*)?(?:\(\d{2,4}\)|\d{2,4})"
    r"[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w.-])"
)
EMAIL_RE = re.compile(r"[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WECHAT_RE = re.compile(r"(?i)\b(?:微信|wechat|vx)[:：]?\s*([A-Za-z][A-Za-z0-9_-]{5,})")
QQ_RE = re.compile(r"(?:QQ|qq)[:：]?\s*([1-9][0-9]{4,11})")
ID_NUMBER_RE = re.compile(r"(?<!\d)(?:\d{17}[0-9Xx]|\d{15})(?!\d)")
AGE_RE = re.compile(r"(?<!\d)(?<!绝对差)(?<!误差)([1-5]?\d)\s*岁")
PERSONAL_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")
COPYRIGHT_LINE_RE = re.compile(r"(?i)(?:copyright\s*)?©")
NAME_LINE_RE = re.compile(r"^(?:姓名[:：]?\s*)?([\u4e00-\u9fff]{2,4})$")
_COMMON_CHINESE_SURNAMES = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平"
    "黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝"
    "董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊"
    "胡凌霍虞万支柯管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔"
    "吉龚程嵇邢滑裴陆荣翁荀羊甄麴家封芮羿储靳汲邴糜松井段富巫乌焦巴弓"
    "牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙"
    "叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻"
    "莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀浦尚农温别庄"
    "晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇"
    "广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养"
    "鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
)
_COMPOUND_CHINESE_SURNAMES = "欧阳|太史|端木|上官|司马|东方|独孤|南宫|万俟|闻人|夏侯|诸葛|尉迟|公羊|赫连|澹台|皇甫|宗政|濮阳|公冶|太叔|申屠|公孙|慕容|仲孙|钟离|长孙|宇文|司徒|鲜于|司空|闾丘|子车|亓官|司寇|巫马|公西|颛孙|壤驷|公良|漆雕|乐正|宰父|谷梁|拓跋|夹谷|轩辕|令狐|段干|百里|呼延|东郭|南门|羊舌|微生|梁丘|左丘|东门|西门"
_CHINESE_PERSON_NAME = (
    rf"(?:(?:{_COMPOUND_CHINESE_SURNAMES})[\u4e00-\u9fff]{{1,2}}|"
    rf"[{_COMMON_CHINESE_SURNAMES}][\u4e00-\u9fff]{{1,2}})"
)
SELF_INTRO_NAME_RE = re.compile(
    rf"(?:大家好[，,！!。]?\s*)?(?:我是|我叫|我的名字是|本人姓名是)"
    rf"(?:一名)?(?:程序员|软件工程师|工程师|开发者|博主)?[：:\s]*"
    rf"({_CHINESE_PERSON_NAME})(?=[，,。！!\s]|$)"
)
CONTEXTUAL_NAME_RE = re.compile(
    rf"({_CHINESE_PERSON_NAME})(?=(?:的)?(?:个人)?(?:简历|博客|主页|作品集|GitHub|Gitee|CSDN))",
    re.IGNORECASE,
)
COPYRIGHT_NAME_RE = re.compile(
    r"(?i)(?:copyright\s*)?©\s*(?:(?:19|20)\d{2}\s+)?"
    r"([\u4e00-\u9fff]{2,4}|[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3})"
    r"(?=\s*(?:[.·|©]|\r?\n|$))"
)
SOCIAL_HANDLE_RE = re.compile(
    r"(?i)\b(?:github|gitee|gitlab|linkedin)\s*(?:@|[:：])\s*([A-Za-z0-9][A-Za-z0-9_.-]{1,38})"
)
PRIVATE_CONTEXT_RE = re.compile(r"(电话|手机|邮箱|年龄|政治面貌|\[手机号\]|\[邮箱\]|\[年龄\])")
ENGLISH_NAME_FIELD_RE = re.compile(r"(?i)^\s*(?:full\s+name|candidate\s+name|name)\s*:\s*(.+)$")
ENGLISH_PRIVATE_FIELD_RE = re.compile(
    r"(?i)^\s*(?:phone|mobile|telephone|email|e-mail|address|home address|date of birth|dob|"
    r"marital status|gender|linkedin|github|portfolio|website)\s*:\s*"
)
RESUME_FILE_NAME_RE = re.compile(r"(?<=[-_])[\u4e00-\u9fff]{2,4}(?=\.)")
PRIVATE_PROFILE_FIELDS = {
    "姓名": "name",
    "年龄": "age",
    "性别": "gender",
    "婚姻状况": "marital_status",
    "户口地": "hukou",
    "籍贯": "hukou",
    "身体状况": "health_status",
    "残障情况": "health_status",
    "政治面貌": "political_status",
    "出生日期": "birth_date",
    "民族": "ethnicity",
    "地址": "address",
    "住址": "address",
    "居住地址": "address",
    "现居住地": "address",
    "通讯地址": "address",
    "身份证": "id_number",
    "身份证号": "id_number",
    "证件号码": "id_number",
    "个人账号": "personal_account",
    "社交账号": "personal_account",
    "个人主页": "personal_account",
    "GitHub": "personal_account",
    "Github": "personal_account",
    "Gitee": "personal_account",
    "LinkedIn": "personal_account",
    "博客": "personal_account",
}
TRAINING_PRIVATE_FIELDS = tuple(
    PRIVATE_PROFILE_FIELDS
    | {
        "电话": "phone",
        "手机": "phone",
        "联系电话": "phone",
        "邮箱": "email",
        "联系邮箱": "email",
        "微信": "wechat",
        "QQ": "qq",
    }
)


@dataclass(frozen=True)
class PiiFinding:
    kind: str
    value: str


def detect_resume_pii(text: str) -> list[PiiFinding]:
    findings: list[PiiFinding] = []
    patterns = [
        ("phone", PHONE_RE),
        ("international_phone", INTERNATIONAL_PHONE_RE),
        ("email", EMAIL_RE),
        ("wechat", WECHAT_RE),
        ("qq", QQ_RE),
        ("id_number", ID_NUMBER_RE),
        ("age", AGE_RE),
        ("personal_url", PERSONAL_URL_RE),
        ("copyright_name", COPYRIGHT_NAME_RE),
        ("social_handle", SOCIAL_HANDLE_RE),
        ("self_intro_name", SELF_INTRO_NAME_RE),
        ("contextual_name", CONTEXTUAL_NAME_RE),
    ]
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group(0)
            findings.append(PiiFinding(kind=kind, value=value))

    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or any(token in stripped for token in ["教育", "实习", "项目", "技能"]):
            continue
        match = NAME_LINE_RE.fullmatch(stripped)
        previous_line = lines[index - 1] if index > 0 else ""
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        near_private_context = bool(PRIVATE_CONTEXT_RE.search(previous_line + "\n" + next_line))
        explicit_name_field = stripped.startswith("姓名")
        if match and (explicit_name_field or near_private_context):
            findings.append(PiiFinding(kind="name", value=match.group(1)))
        normalized_line = re.sub(r"^[\s#>*+\-]+", "", stripped).replace("**", "")
        profile_match = re.match(r"([^：:]+)\s*[：:]\s*(.+)$", normalized_line)
        if profile_match:
            field = profile_match.group(1).strip()
            kind = PRIVATE_PROFILE_FIELDS.get(field)
            if kind == "age" and AGE_RE.search(stripped):
                continue
            finding = PiiFinding(kind=kind, value=profile_match.group(2).strip()) if kind else None
            if finding and finding not in findings:
                findings.append(finding)
        english_name = ENGLISH_NAME_FIELD_RE.match(normalized_line)
        if english_name:
            finding = PiiFinding(kind="name", value=english_name.group(1).strip())
            if finding not in findings:
                findings.append(finding)
        if ENGLISH_PRIVATE_FIELD_RE.match(normalized_line):
            kind = normalized_line.split(":", 1)[0].strip().lower().replace(" ", "_")
            finding = PiiFinding(kind=kind, value=normalized_line.split(":", 1)[-1].strip())
            if finding not in findings:
                findings.append(finding)
    return findings


def redact_resume_pii(text: str) -> str:
    redacted = PHONE_RE.sub("[手机号]", text)
    redacted = INTERNATIONAL_PHONE_RE.sub("[PHONE]", redacted)
    redacted = EMAIL_RE.sub("[邮箱]", redacted)
    redacted = WECHAT_RE.sub(lambda match: match.group(0).replace(match.group(1), "[微信]"), redacted)
    redacted = QQ_RE.sub(lambda match: match.group(0).replace(match.group(1), "[QQ]"), redacted)
    redacted = ID_NUMBER_RE.sub("[证件号码]", redacted)
    redacted = AGE_RE.sub("[年龄]", redacted)
    redacted = PERSONAL_URL_RE.sub("[个人链接]", redacted)
    redacted = COPYRIGHT_NAME_RE.sub(
        lambda match: match.group(0).replace(match.group(1), "[姓名]"), redacted
    )
    redacted = SOCIAL_HANDLE_RE.sub(
        lambda match: match.group(0).replace(match.group(1), "[个人账号]"), redacted
    )
    redacted = SELF_INTRO_NAME_RE.sub(
        lambda match: match.group(0).replace(match.group(1), "[姓名]"), redacted
    )
    redacted = CONTEXTUAL_NAME_RE.sub("[姓名]", redacted)

    source_lines = redacted.splitlines()
    lines: list[str] = []
    for index, line in enumerate(source_lines):
        stripped = line.strip()
        match = NAME_LINE_RE.fullmatch(stripped)
        previous_line = source_lines[index - 1] if index > 0 else ""
        next_line = source_lines[index + 1] if index + 1 < len(source_lines) else ""
        near_private_context = bool(PRIVATE_CONTEXT_RE.search(previous_line + "\n" + next_line))
        explicit_name_field = stripped.startswith("姓名")
        if match and (explicit_name_field or near_private_context) and not any(
            token in stripped for token in ["教育", "实习", "项目", "技能"]
        ):
            lines.append(line.replace(match.group(1), "[姓名]"))
        else:
            lines.append(line)
    return "\n".join(lines)


def sanitize_resume_text_for_training(text: str) -> str:
    redacted = redact_resume_pii(text)
    kept_lines = []
    for line in redacted.splitlines():
        normalized_line = re.sub(r"^[\s#>*+\-]+", "", line.strip()).replace("**", "")
        if COPYRIGHT_LINE_RE.search(normalized_line):
            continue
        if normalized_line in {"[姓名]", "[手机号]", "[邮箱]"} or any(
            marker in normalized_line for marker in ("[姓名]", "[个人账号]")
        ):
            continue
        if any(re.match(rf"{re.escape(field)}\s*[：:]", normalized_line) for field in TRAINING_PRIVATE_FIELDS):
            continue
        if ENGLISH_NAME_FIELD_RE.match(normalized_line) or ENGLISH_PRIVATE_FIELD_RE.match(
            normalized_line
        ):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def redact_resume_metadata(value: str) -> str:
    value = RESUME_FILE_NAME_RE.sub("[姓名]", value)
    value = re.sub(
        r"(^|/)(?!个人)([\u4e00-\u9fff]{2,4})(?=简历)",
        lambda match: f"{match.group(1)}[姓名]",
        value,
    )
    return value


def summarize_pii(findings: list[PiiFinding]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return {
        "has_pii": bool(findings),
        "total": len(findings),
        "counts": counts,
    }


def sanitize_resume_row(row: dict[str, Any]) -> dict[str, Any]:
    text_fields = ["raw_text", "clean_text", "normalized_text", "text"]
    combined_text = "\n".join(str(row.get(field) or "") for field in text_fields)
    findings = detect_resume_pii(combined_text)
    sanitized = dict(row)
    for field in text_fields:
        if field in sanitized and isinstance(sanitized[field], str):
            sanitized[field] = sanitize_resume_text_for_training(sanitized[field])
    for field in ["file_name", "file_path"]:
        if field in sanitized and isinstance(sanitized[field], str):
            sanitized[field] = redact_resume_metadata(sanitized[field])
    if isinstance(sanitized.get("sections"), dict):
        sanitized["sections"] = {
            key: sanitize_resume_text_for_training(str(value))
            for key, value in sanitized["sections"].items()
        }
    sanitized["privacy"] = summarize_pii(findings)
    sanitized["privacy"]["redacted"] = bool(findings)
    return sanitized


def build_resume_privacy_report(
    *,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row_reports = []
    aggregate_counts: dict[str, int] = {}
    for row in rows:
        text = "\n".join(
            str(row.get(field) or "") for field in ["raw_text", "clean_text", "normalized_text", "text"]
        )
        findings = detect_resume_pii(text)
        summary = summarize_pii(findings)
        for kind, count in summary["counts"].items():
            aggregate_counts[kind] = aggregate_counts.get(kind, 0) + int(count)
        row_reports.append(
            {
                "id": row.get("id", ""),
                "file_name": redact_resume_metadata(str(row.get("file_name", ""))),
                "has_pii": summary["has_pii"],
                "pii_total": summary["total"],
                "pii_counts": summary["counts"],
            }
        )
    rows_with_pii = sum(1 for item in row_reports if item["has_pii"])
    return {
        "row_count": len(rows),
        "rows_with_pii": rows_with_pii,
        "pii_row_rate": rows_with_pii / len(rows) if rows else 0.0,
        "pii_counts": aggregate_counts,
        "rows": row_reports,
    }


def _load_rows_from_file(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        return list(read_jsonl(file_path))
    row = ingest_resume(file_path)
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Resume file or JSONL rows")
    parser.add_argument("--out", default="")
    parser.add_argument("--report-out", default="outputs/eval_reports/resume_privacy_report.json")
    args = parser.parse_args()

    rows = _load_rows_from_file(args.input)
    report = build_resume_privacy_report(rows=rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.report_out, rendered + "\n")
    if args.out:
        write_jsonl(args.out, [sanitize_resume_row(row) for row in rows])


if __name__ == "__main__":
    main()
