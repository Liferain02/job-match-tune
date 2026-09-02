from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable
from urllib.request import Request, urlopen

import yaml

from jobmatch_tune.dataset.grouped_split import normalized_input_hash
from jobmatch_tune.preprocess.skill_canonicalization import extract_known_skills
from jobmatch_tune.resume.privacy import (
    detect_resume_pii,
    sanitize_resume_text_for_training,
)
from jobmatch_tune.resume.ocr import ocr_image_file, ocr_pdf_file
from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


DJINNI_SOURCE_URL = (
    "https://huggingface.co/datasets/lang-uk/"
    "recruitment-dataset-candidate-profiles-english"
)
DJINNI_REVISION = "86255174c6c378a2b52cd7e81d79002c64d899a6"
NETSOL_SOURCE_URL = "https://huggingface.co/datasets/netsol/resume-score-details"
NETSOL_REVISION = "f2b49384c133beaf0a927bad6f91acc633cd0326"
FAIRCV_SOURCE_URL = "https://huggingface.co/datasets/OhMyKing/FairCV"
FAIRCV_REVISION = "ece2a0e45439c0c1c5d70c92d30bf662756dd656"
PUBLIC_WEB_RESUME_MANIFEST = "configs/public_chinese_resume_sources.yaml"
PUBLIC_WEB_RESUME_CACHE = "data/external/public_chinese_resume_snapshots"
HUMAN_REVIEWED_PAIR_MANIFEST = "configs/human_reviewed_public_match_pairs.yaml"

_FAIRCV_DIRECTION_MAP = {
    "后端开发工程师": "后端开发",
    "前端开发工程师": "前端开发",
    "Android开发工程师": "客户端开发",
    "iOS开发工程师": "客户端开发",
    "全栈开发工程师": "后端开发",
    "机器学习工程师": "算法工程",
    "计算机视觉工程师": "算法工程",
    "自然语言处理工程师": "算法工程",
    "推荐算法工程师": "算法工程",
    "系统架构师": "后端开发",
    "安全架构师": "安全工程",
    "DevOps架构师": "运维开发",
}
_FAIRCV_SECTION_NAMES = {
    "个人信息",
    "教育背景",
    "专业技能",
    "工作经历",
    "项目经验",
    "其他亮点",
    "自我评价",
    "工作期望",
    "期望职位",
    "备注",
}

_EDUCATION_RE = re.compile(
    r"\b(?:education|university|college|bachelor(?:'s)?|master(?:'s)?|ph\.?d|degree)\b",
    re.I,
)
_PROJECT_RE = re.compile(r"\bprojects?\b", re.I)
_INTERNSHIP_RE = re.compile(r"\bintern(?:ship)?\b", re.I)
_CONDITION_WORDS_RE = re.compile(
    r"\b(?:year|experience|degree|bachelor|master|ph\.?d|education|location|salary)\b",
    re.I,
)

_WEB_SECTION_ALIASES = {
    "education": (
        "教育经历",
        "教育背景",
        "教育业号",
        "教育/培训履历",
        "Education",
        "教育",
        "学历背景",
    ),
    "skills": (
        "专业技能",
        "职业技能",
        "技能清单",
        "技术能力",
        "技术栈",
        "专业能力",
        "技术能力",
        "技术栈",
        "工作技能",
        "核心技能",
        "工具栈",
        "技能熟练度",
        "Skills",
        "技能",
    ),
    "work": (
        "工作经历",
        "工作经验",
        "实习经历",
        "职业经历",
        "工作履历",
        "Experience",
    ),
    "projects": (
        "项目经历",
        "项目经验",
        "个人项目",
        "开源项目",
        "科研经历",
        "科研项目",
        "科研与重大项目经历",
        "项目实践",
        "Projects",
        "作品集",
        "项目",
    ),
    "strengths": (
        "个人总结",
        "自我评价",
        "个人优势",
        "学术成果",
        "科研成果",
        "ABOUT ME",
        "竞赛经历",
        "获奖经历",
        "荣誉证书",
        "荣誉奖项",
        "其他亮点",
        "其他",
    ),
}
_WEB_PRIVATE_LINE_RE = re.compile(
    r"(?:姓名|性别|年龄|出生(?:日期|年月)?|民族|政治面貌|婚姻|籍贯|祖籍|户籍|"
    r"住址|居住地|现居住地|手机|电话|邮箱|邮件|E-?mail|微信|QQ|联系方式)\s*[:：]|"
    r"(?:个人主页|个人网站|GitHub|Gitee|LinkedIn|博客)(?:主页|地址|\s*[:：])|"
    r"^个人微信$",
    re.I,
)
_WEB_EDUCATION_EVIDENCE_RE = re.compile(
    r"大学|学院|本科|硕士|博士|学士|研究生|专科|专业|毕业"
)
_WEB_STRONG_EDUCATION_EVIDENCE_RE = re.compile(
    r"(?:大学|学院).{0,36}(?:本科|硕士|博士|学士|研究生|专科|专业|毕业)"
    r"|(?:本科|硕士|博士|学士|研究生|专科|专业|毕业).{0,36}(?:大学|学院)"
)
_WEB_PROJECT_EVIDENCE_RE = re.compile(
    r"项目|系统|平台|服务|应用|工具|插件|组件|网站|框架|"
    r"负责|实现|开发|设计|优化|搭建|重构|主导|参与|支持|制作|编写|维护"
)
_WEB_RESUME_SUMMARY_RE = re.compile(r"求职|专注于|擅长|正在寻找|活跃的?开源")
_WEB_PROJECT_ACTION_RE = re.compile(
    r"实现|设计|优化|搭建|重构|主导|参与|负责|解决|构建|采用|基于|"
    r"开发了|开发并|支持|制作|编写|维护|定制"
)
_WEB_WORK_EVIDENCE_RE = re.compile(
    r"负责|实现|设计|优化|搭建|重构|主导|参与|完成|开发|测试|定位|解决|构建"
)
_WEB_PUBLICATION_AUTHOR_RE = re.compile(
    r"\bet\s+al\.?\b|\b[A-Z][a-z]+,\s*[A-Z]\."
    r"|(?:\b[A-Z][A-Za-z'-]+\s+[A-Z][A-Za-z'-]+[*]?(?:,\s*|\s+and\s+)){2}"
    r"|共同一作|第一作者",
    re.I,
)
_WEB_IGNORED_SECTION_ALIASES = (
    "基本信息",
    "个人信息",
    "求职意向",
    "联系方式",
    "社会工作",
    "校园活动",
    "语言能力",
    "兴趣爱好",
    "关于我",
    "导航",
)


class _VisibleHTMLTextParser(HTMLParser):
    """Extract visible lines without bringing an HTML parsing dependency into data builds."""

    _BLOCK_TAGS = {
        "article",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "li",
        "p",
        "section",
        "td",
        "th",
        "tr",
    }
    _SKIP_TAGS = {"script", "style", "svg", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "\n".join(
            line for line in (_clean_text(line) for line in "".join(self._parts).splitlines()) if line
        )


def html_to_visible_text(content: bytes) -> str:
    decoded = ""
    for encoding in ("utf-8", "gb18030"):
        try:
            decoded = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not decoded:
        decoded = content.decode("utf-8", errors="ignore")
    parser = _VisibleHTMLTextParser()
    parser.feed(decoded)
    return parser.text()


def _public_section_name(line: str) -> str:
    normalized = re.sub(r"[\s/|·:：\-—_]+", "", line).casefold()
    if not normalized or len(normalized) > 24:
        return ""

    def is_heading_alias(alias: str) -> bool:
        candidate = alias.casefold()
        if normalized == candidate:
            return True
        # Long aliases may be followed by an English translation in bilingual
        # resumes. Short aliases such as “教育/项目” must match exactly so that
        # company names and sentences are not mistaken for section headings.
        return len(candidate) >= 4 and (
            normalized.startswith(candidate) or normalized.endswith(candidate)
        )

    for name, aliases in _WEB_SECTION_ALIASES.items():
        if any(is_heading_alias(alias) for alias in aliases):
            return name
    if any(is_heading_alias(alias) for alias in _WEB_IGNORED_SECTION_ALIASES):
        return "ignored"
    return ""


def extract_public_resume_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    current = ""
    for raw_line in text.splitlines():
        # Raw Markdown/Org resumes are useful public sources too.  Strip their
        # heading markers before applying the same section aliases as HTML/PDF.
        line = _clean_text(raw_line).strip("•●▪■◆◇★☆-—|#* ")
        if not line:
            continue
        section = _public_section_name(line)
        if section:
            current = "" if section == "ignored" else section
            continue
        if (
            not current
            or _WEB_PRIVATE_LINE_RE.search(line)
            or _WEB_PUBLICATION_AUTHOR_RE.search(line)
        ):
            continue
        sanitized = sanitize_resume_text_for_training(line).strip()
        if (
            not sanitized
            or any(marker in sanitized for marker in ("[手机号]", "[邮箱]", "[个人链接]", "[姓名]"))
            or detect_resume_pii(sanitized)
        ):
            continue
        if 4 <= len(sanitized) <= 360:
            sections[current].append(sanitized)
    return {name: _unique_texts(lines, limit=80) for name, lines in sections.items()}


def _is_public_project_evidence(line: str) -> bool:
    if len(line) < 18 or not _WEB_PROJECT_EVIDENCE_RE.search(line):
        return False
    return not _WEB_RESUME_SUMMARY_RE.search(line) or bool(_WEB_PROJECT_ACTION_RE.search(line))


def _pdf_to_text(content: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="jobmatch-public-resume-") as temp_dir:
        source = Path(temp_dir) / "source.pdf"
        output = Path(temp_dir) / "source.txt"
        source.write_bytes(content)
        subprocess.run(
            ["pdftotext", "-layout", str(source), str(output)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        text = output.read_text(encoding="utf-8", errors="ignore")
        # Some Chinese PDFs expose only digits and Latin tokens because their
        # embedded fonts have no usable Unicode map. OCR those pages instead of
        # silently producing a structurally empty resume.
        if len(re.findall(r"[\u4e00-\u9fff]", text)) < 20:
            return ocr_pdf_file(source)
        return text


def _image_to_text(content: bytes) -> str:
    """OCR an image without retaining the source asset on disk."""
    with tempfile.TemporaryDirectory(prefix="jobmatch-public-resume-") as temp_dir:
        source = Path(temp_dir) / "source.img"
        source.write_bytes(content)
        return ocr_image_file(source)


def _public_source_to_text(content: bytes, source_format: str) -> tuple[str, str]:
    normalized_format = source_format.strip().lower()
    if normalized_format == "pdf":
        return _pdf_to_text(content), "pdf_text"
    if normalized_format in {"image", "png", "jpg", "jpeg", "webp", "bmp"}:
        return _image_to_text(content), "image_ocr"
    return html_to_visible_text(content), "html_visible_text"


def _source_asset_urls(source: dict[str, Any]) -> list[str]:
    values = source.get("urls")
    if isinstance(values, list):
        return [str(value) for value in values if str(value).strip()]
    value = str(source.get("url") or "").strip()
    return [value] if value else []


def refresh_public_resume_snapshots(
    manifest_path: str | Path,
    cache_dir: str | Path,
    *,
    skip_existing: bool = False,
) -> dict[str, Any]:
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"requested": 0, "refreshed": 0, "failed": []}
    for source in manifest.get("sources") or []:
        if source.get("training_decision") != "allowed_after_deidentification":
            continue
        report["requested"] += 1
        source_id = str(source["id"])
        cache_path = cache_root / f"{source_id}.json"
        if skip_existing and cache_path.exists():
            report["already_cached"] = int(report.get("already_cached", 0)) + 1
            continue
        try:
            asset_urls = _source_asset_urls(source)
            if not asset_urls:
                raise ValueError("public resume source has no url or urls")
            visible_parts = []
            content_hash = hashlib.sha256()
            extraction_methods = []
            for asset_url in asset_urls:
                request = Request(
                    asset_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 JobMatchTune local-research-data-audit/1.0"
                    },
                )
                with urlopen(request, timeout=25) as response:
                    content = response.read()
                visible_text, extraction_method = _public_source_to_text(
                    content,
                    str(source.get("format") or "html"),
                )
                visible_parts.append(visible_text)
                extraction_methods.append(extraction_method)
                content_hash.update(content)
            visible_text = "\n".join(visible_parts)
            # The raw page/PDF and its full visible text contain contact data. Keep
            # only deidentified, task-relevant sections after extraction.
            payload = {
                "id": source_id,
                "url": asset_urls[0],
                "asset_urls": asset_urls,
                "source_page_url": source.get("source_page_url") or asset_urls[0],
                "fetched_at": date.today().isoformat(),
                "content_sha256": content_hash.hexdigest(),
                "extraction_method": "+".join(dict.fromkeys(extraction_methods)),
                "sections": extract_public_resume_sections(visible_text),
            }
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            report["refreshed"] += 1
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            report["failed"].append({"id": source_id, "error": str(error)[:240]})
    return report


def build_public_web_resume_rows(
    manifest_path: str | Path,
    cache_dir: str | Path,
    *,
    schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    cache_root = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    counters: dict[str, Any] = defaultdict(int)
    for source in manifest.get("sources") or []:
        counters["manifest_sources"] += 1
        if source.get("training_decision") != "allowed_after_deidentification":
            counters["excluded_by_review"] += 1
            continue
        cache_path = cache_root / f"{source['id']}.json"
        if not cache_path.exists():
            counters["missing_snapshot"] += 1
            continue
        snapshot = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_sections = snapshot.get("sections")
        sections = (
            {
                str(name): [
                    str(line)
                    for line in lines
                    if not _WEB_PUBLICATION_AUTHOR_RE.search(str(line))
                    and not _WEB_PRIVATE_LINE_RE.search(str(line))
                ]
                for name, lines in cached_sections.items()
                if isinstance(lines, list)
            }
            if isinstance(cached_sections, dict)
            else extract_public_resume_sections(str(snapshot.get("visible_text") or ""))
        )
        target = _clean_text(source.get("target_direction"))
        all_section_lines = [
            line for values in sections.values() for line in values
        ]
        education = _unique_texts(
            line
            for line in sections.get("education", [])
            if len(line) >= 6 and _WEB_EDUCATION_EVIDENCE_RE.search(line)
        )[:3]
        if not education:
            # Real resumes and PDF OCR often merge headings such as
            # "工作/教育经历" or lose a heading entirely. Recover only lines
            # that explicitly contain both an institution and degree/major
            # evidence; never infer education from a job title or date alone.
            education = _unique_texts(
                line
                for line in all_section_lines
                if len(line) >= 8 and _WEB_STRONG_EDUCATION_EVIDENCE_RE.search(line)
            )[:3]
        projects = _unique_texts(
            line
            for line in sections.get("projects", [])
            if _is_public_project_evidence(line)
        )[:5]
        work_evidence = _unique_texts(
            line
            for line in sections.get("work", [])
            if len(line) >= 18 and _WEB_WORK_EVIDENCE_RE.search(line)
        )[:5]
        internships = _unique_texts(
            line for line in sections.get("work", []) if "实习" in line
        )[:3]
        strengths = _unique_texts(
            line for line in sections.get("strengths", []) if len(line) >= 12
        )[:3]
        rendered_sections = []
        for name, title in (
            ("education", "教育背景"),
            ("skills", "专业技能"),
            ("work", "工作经历"),
            ("projects", "项目经历"),
            ("strengths", "其他亮点"),
        ):
            values = sections.get(name, [])
            if values:
                rendered_sections.append(f"{title}：\n" + "\n".join(values))
        text = f"目标岗位：{target}\n" + "\n".join(rendered_sections)
        text = sanitize_resume_text_for_training(text).strip()[:6500]
        skills = extract_known_skills(text, schema)
        failure_reasons = []
        if len(text) < 240:
            failure_reasons.append("short_text")
        if len(skills) < 2:
            failure_reasons.append("insufficient_skills")
        if not education:
            failure_reasons.append("missing_education")
        if not projects and len(work_evidence) < 2:
            failure_reasons.append("missing_project_or_work_evidence")
        if failure_reasons:
            counters["quality_filtered"] += 1
            for reason in failure_reasons:
                counters[f"quality_filtered_{reason}"] += 1
            continue
        if detect_resume_pii(text):
            counters["privacy_filtered"] += 1
            continue
        asset_urls = _source_asset_urls(source)
        source_page_url = str(source.get("source_page_url") or asset_urls[0])
        source_hash = hashlib.sha1(source_page_url.encode("utf-8")).hexdigest()
        rows.append(
            {
                "id": f"public_web_resume_{source_hash}",
                "task": "resume_parse",
                "source_type": "public_real_self_published_anonymized",
                "source_group": f"public_web_resume:{source_hash}",
                "text": text,
                "label": {
                    "目标岗位": target,
                    "教育背景": education,
                    "核心技能": skills,
                    "实习经历": internships,
                    "项目经历": projects,
                    "优势标签": strengths,
                },
                "meta": {
                    "language": "zh",
                    "domain": "technical_jobs",
                    "source_url": source_page_url,
                    "source_asset_url": asset_urls[0],
                    "source_asset_urls": asset_urls,
                    "source_reference_id": source["id"],
                    "source_revision": snapshot.get("content_sha256"),
                    "snapshot_date": snapshot.get("fetched_at"),
                    "extraction_method": snapshot.get("extraction_method"),
                    "license_status": source.get(
                        "license_status", "not_stated_no_explicit_prohibition"
                    ),
                    "intended_usage": "sft_training",
                    "provenance": "self_published_public_technical_resume",
                    "provenance_status": "public_page_manually_reviewed",
                    "annotation_status": "human_target_mapping_plus_extractive_labels",
                    "label_method": "reviewed_target_mapping_and_verbatim_section_extraction",
                    "training_eligible": True,
                    "contains_real_person_data": True,
                    "privacy_status": "contact_and_private_sections_removed_local_scan_passed",
                },
            }
        )
    counters["selected_rows"] = len(rows)
    direction_counts = Counter(row["label"]["目标岗位"] for row in rows)
    extraction_method_counts = Counter(
        str(row["meta"].get("extraction_method") or "unknown") for row in rows
    )
    counters["selected_directions"] = len(direction_counts)
    counters["selected_direction_counts"] = dict(direction_counts.most_common())
    counters["selected_extraction_method_counts"] = dict(
        extraction_method_counts.most_common()
    )
    return rows, dict(counters)


def _clean_public_jd_text(text: str) -> str:
    cleaned = unescape(text)
    cleaned = re.sub(r"(?:^|\n)任务类型：从岗位中提取学历\s*(?:\n|$)", "\n", cleaned)
    cleaned = re.sub(r"(?:\s*\|\s*|\n)学历提示：[^|\n]+$", "", cleaned)
    return re.sub(r"\s*\|\s*", "\n", cleaned).strip()


def _human_pair_analysis(
    label: dict[str, Any],
    *,
    rationale: str,
    jd_direction: str,
    resume_direction: str,
) -> dict[str, Any]:
    strengths = []
    if label["岗位方向匹配"]:
        strengths.append(f"简历方向与 {jd_direction} 岗位一致")
    if label["学历匹配"]:
        strengths.append("简历中的学历证据满足岗位要求")
    if label["经验匹配"]:
        strengths.append("已有经历能够覆盖岗位要求的经验类型或年限")
    if label["命中技能"]:
        strengths.append("已提供技能证据：" + "、".join(label["命中技能"][:6]))
    if label["命中项目"]:
        strengths.append("相关项目证据：" + "；".join(label["命中项目"][:2]))
    gaps = []
    if not label["岗位方向匹配"]:
        gaps.append(f"简历主方向是{resume_direction}，与 {jd_direction} 不一致")
    if not label["学历匹配"]:
        gaps.append("简历学历证据未满足岗位硬性要求")
    if not label["经验匹配"]:
        gaps.append("简历中的相关经验类型或年限不足")
    if label["缺失技能"]:
        gaps.append("缺少直接证据的技能：" + "、".join(label["缺失技能"][:6]))
    suggestions = []
    if label["缺失技能"]:
        suggestions.append("只有实际使用过时，才补充对应技能及项目证据")
    if not label["经验匹配"]:
        suggestions.append("优先补充与目标岗位直接相关的实习、工作或完整项目经历")
    if not label["岗位方向匹配"]:
        suggestions.append(f"更适合优先投递{resume_direction}方向岗位")
    if not suggestions:
        suggestions.append("保留当前核心证据，并在项目中补充个人职责和可验证结果")
    return {
        "匹配结论": rationale,
        "匹配优势": strengths or ["简历提供了可供核验的基础信息"],
        "主要短板": gaps or ["未发现明显硬性短板"],
        "简历优化建议": suggestions,
        "推荐投递岗位方向": [jd_direction if label["岗位方向匹配"] else resume_direction],
    }


def build_human_reviewed_public_match_rows(
    manifest_path: str | Path,
    resume_rows: list[dict[str, Any]],
    jd_pool_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not Path(manifest_path).exists() or not Path(jd_pool_path).exists():
        return [], {"selected_pairs": 0, "reason": "pair_manifest_or_jd_pool_missing"}
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    resume_by_id = {
        str((row.get("meta") or {}).get("source_reference_id") or ""): row
        for row in resume_rows
        if (row.get("meta") or {}).get("source_reference_id")
    }
    jd_by_id = {str(row.get("id") or ""): row for row in read_jsonl(jd_pool_path)}
    rows = []
    counters: dict[str, Any] = defaultdict(int)
    for annotation in manifest.get("pairs") or []:
        counters["manifest_pairs"] += 1
        resume = resume_by_id.get(str(annotation.get("resume_source_id") or ""))
        jd = jd_by_id.get(str(annotation.get("jd_id") or ""))
        if resume is None or jd is None:
            counters["missing_join_entity"] += 1
            continue
        jd_text = _clean_public_jd_text(str(jd.get("raw_text") or ""))
        if len(jd_text) < 120:
            counters["weak_jd"] += 1
            continue
        label = {
            "raw_label": "human_reviewed_fit",
            "raw_score": {"低匹配": 1, "基本匹配": 2, "较匹配": 3, "高匹配": 4}[
                annotation["level"]
            ],
            "匹配等级": annotation["level"],
            "岗位方向匹配": bool(annotation["direction_match"]),
            "学历匹配": bool(annotation["education_match"]),
            "经验匹配": bool(annotation["experience_match"]),
            "命中技能": list(annotation.get("matched_skills") or []),
            "缺失技能": list(annotation.get("missing_skills") or []),
            "命中项目": list(annotation.get("matched_projects") or []),
        }
        jd_direction = str(annotation["jd_direction"])
        resume_direction = str(resume["label"]["目标岗位"])
        pair_hash = normalized_input_hash(f"{jd_text}\n---\n{resume['text']}")
        rows.append(
            {
                "id": f"human_reviewed_public_match_{annotation['id']}",
                "task": "match",
                "source_type": "human_reviewed_public_real_pair",
                "source_group": f"human_reviewed_public_match:{pair_hash}",
                "jd_text": jd_text,
                "resume_text": resume["text"],
                "label": label,
                "analysis": _human_pair_analysis(
                    label,
                    rationale=str(annotation["rationale"]),
                    jd_direction=jd_direction,
                    resume_direction=resume_direction,
                ),
                "meta": {
                    "language": "zh",
                    "domain": "technical_jobs",
                    "source_url": jd.get("meta", {}).get("source_file"),
                    "resume_source_url": resume["meta"]["source_url"],
                    "license_status": "not_stated_no_explicit_prohibition",
                    "intended_usage": "sft_training",
                    "provenance": "public_jd_plus_deidentified_self_published_resume",
                    "provenance_status": "both_input_sources_documented",
                    "annotation_status": "human_reviewed_pair_v1",
                    "annotation_provenance": "individual_evidence_review_no_teacher_model",
                    "pair_type": "human_reviewed_public_real_pair",
                    "training_eligible": True,
                    "contains_real_person_data": True,
                    "privacy_status": "resume_private_sections_removed_local_scan_passed",
                    "entity_split": "train",
                    "jd_entity_hash": normalized_input_hash(jd_text),
                    "resume_entity_hash": normalized_input_hash(resume["text"]),
                    "jd_direction": jd_direction,
                    "resume_direction": resume_direction,
                    "annotation_rationale": annotation["rationale"],
                    "observed_outcome": False,
                },
            }
        )
    counters["selected_pairs"] = len(rows)
    counters["level_counts"] = dict(
        sorted(defaultdict(int, {level: sum(row["label"]["匹配等级"] == level for row in rows) for level in ("低匹配", "基本匹配", "较匹配", "高匹配")}).items())
    )
    return rows, dict(counters)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique_texts(values: Iterable[Any], limit: int = 8) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
            if len(output) >= limit:
                break
    return output


def _extract_evidence_sentences(text: str, pattern: re.Pattern[str], limit: int) -> list[str]:
    pieces = re.split(r"[\r\n]+|(?<=[.!?])\s+", text)
    return _unique_texts(
        sanitize_resume_text_for_training(piece)
        for piece in pieces
        if pattern.search(piece) and 12 <= len(_clean_text(piece)) <= 420
    )[:limit]


def _evidence_preserving_excerpt(
    text: str,
    *,
    evidence: Iterable[str] = (),
    prefix: str = "",
    head_chars: int,
    max_chars: int,
) -> str:
    """Keep inputs trainable without dropping the source evidence used by labels."""
    parts = [prefix.strip(), text[:head_chars].strip()]
    current = "\n".join(part for part in parts if part)
    normalized_current = _clean_text(current).casefold()
    for item in _unique_texts(evidence, 12):
        item = item[:280].strip()
        if not item or item.casefold() in normalized_current:
            continue
        candidate = f"{current}\n{item}" if current else item
        if len(candidate) > max_chars:
            continue
        current = candidate
        normalized_current = _clean_text(current).casefold()
    return current[:max_chars].strip()


def _plain_markdown_line(line: str) -> str:
    text = re.sub(r"^[\s#>*-]+", "", line)
    text = text.replace("**", "").replace("__", "").strip()
    return re.sub(r"\s+", " ", text).strip(" -|：:")


def _faircv_sections(content: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    current = ""
    for raw_line in content.splitlines():
        line = _plain_markdown_line(raw_line)
        if not line or set(line) <= {"-", "="}:
            continue
        heading = line.rstrip("：:")
        if heading in _FAIRCV_SECTION_NAMES:
            current = heading
            continue
        if current:
            sections[current].append(line)
    return dict(sections)


def _section_evidence(
    sections: dict[str, list[str]],
    section_names: tuple[str, ...],
    *,
    pattern: re.Pattern[str] | None = None,
    limit: int,
) -> list[str]:
    values = []
    for section in section_names:
        for line in sections.get(section, []):
            if "{" in line or "}" in line or not 6 <= len(line) <= 180:
                continue
            if pattern is None or pattern.search(line):
                values.append(line)
    return _unique_texts(values, limit)


def build_faircv_resume_rows(
    source_path: str | Path,
    *,
    schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    source_rows = payload.get("resumes") or []
    rows = []
    counters = defaultdict(int)
    for source_row in source_rows:
        counters["source_rows"] += 1
        meta = source_row.get("metadata") or {}
        position = _clean_text(meta.get("position"))
        target = _FAIRCV_DIRECTION_MAP.get(position)
        if not target:
            counters["non_technical_or_product_rows"] += 1
            continue
        sections = _faircv_sections(str(source_row.get("content") or ""))
        # Demographic fields were intentionally varied for a fairness study.
        # They are irrelevant to technical resume parsing, so remove the whole section.
        kept_sections = []
        for name in ("教育背景", "专业技能", "工作经历", "项目经验", "其他亮点", "自我评价"):
            lines = [line for line in sections.get(name, []) if "{" not in line and "}" not in line]
            if lines:
                kept_sections.append(f"{name}：\n" + "\n".join(lines))
        text = sanitize_resume_text_for_training(
            f"目标岗位：{target}\n原岗位名称：{position}\n" + "\n".join(kept_sections)
        ).strip()
        education = _section_evidence(
            sections,
            ("教育背景",),
            pattern=re.compile(r"大学|学院|本科|硕士|博士|学历|专业"),
            limit=2,
        )
        projects = _section_evidence(
            sections,
            ("项目经验",),
            pattern=re.compile(r"项目|负责|完成|实现|开发|设计|优化|提升|降低|搭建|主导"),
            limit=3,
        )
        internships = _section_evidence(
            sections,
            ("工作经历", "项目经验"),
            pattern=re.compile(r"实习"),
            limit=2,
        )
        strengths = _section_evidence(
            sections,
            ("其他亮点", "自我评价"),
            pattern=re.compile(r"开源|论文|专利|竞赛|协作|学习|架构|优化|经验|能力"),
            limit=2,
        )
        skills = extract_known_skills(text, schema)
        if len(text) < 240 or len(skills) < 2 or not education or not projects:
            counters["weak_label_rows"] += 1
            continue
        if detect_resume_pii(text):
            counters["privacy_filtered_rows"] += 1
            continue
        source_hash = hashlib.sha1(
            f"{position}\n{meta.get('skill_level')}\n{source_row.get('content')}".encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "id": f"faircv_zh_tech_{source_hash}",
                "task": "resume_parse",
                "source_type": "public_synthetic_unique_template",
                "source_group": f"faircv_template:{source_hash}",
                "text": text,
                "label": {
                    "目标岗位": target,
                    "教育背景": education,
                    "核心技能": skills,
                    "实习经历": internships,
                    "项目经历": projects,
                    "优势标签": strengths,
                },
                "meta": {
                    "language": "zh",
                    "domain": "technical_jobs",
                    "source_url": FAIRCV_SOURCE_URL,
                    "source_revision": FAIRCV_REVISION,
                    "license_status": "source_declared_research_noncommercial",
                    "intended_usage": "sft_training",
                    "provenance": "faircv_unique_resume_templates",
                    "provenance_status": "documented_synthetic_template",
                    "annotation_status": "source_metadata_plus_extractive_labels",
                    "label_method": "position_mapping_and_verbatim_evidence_extraction",
                    "generator": "upstream_unique_template_authoring",
                    "training_eligible": True,
                    "contains_real_person_data": False,
                    "privacy_status": "demographic_section_removed_and_local_redaction_passed",
                },
            }
        )
    counters["selected_rows"] = len(rows)
    counters["selected_positions"] = len({row["label"]["目标岗位"] for row in rows})
    return rows, dict(counters)


def _djinni_target(row: dict[str, Any]) -> str:
    keyword = _clean_text(row.get("Primary Keyword"))
    if keyword and keyword.casefold() != "other":
        return keyword
    parts = [_clean_text(part) for part in str(row.get("Position") or "").split("||")]
    return next((part for part in reversed(parts) if part), "")[:120]


def _reserved_djinni_candidate_ids(path: str | Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {str(row.get("source_id") or "") for row in read_jsonl(path)}


def build_djinni_resume_rows(
    source_path: str | Path,
    *,
    schema: dict[str, Any],
    reserved_source_ids: set[str] | None = None,
    max_rows: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as parquet

    reserved = reserved_source_ids or set()
    candidates_by_keyword: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    counters = defaultdict(int)
    parquet_file = parquet.ParquetFile(source_path)
    columns = [
        "Position",
        "Primary Keyword",
        "English Level",
        "Experience Years",
        "Highlights",
        "CV",
        "CV_lang",
        "id",
    ]
    for batch_index in range(parquet_file.num_row_groups):
        for source_row in parquet_file.read_row_group(batch_index, columns=columns).to_pylist():
            counters["source_rows"] += 1
            source_id = str(source_row.get("id") or "")
            if not source_id or source_id in reserved:
                counters["reserved_evaluation_rows"] += 1
                continue
            if str(source_row.get("CV_lang") or "").lower() != "en":
                counters["non_english_rows"] += 1
                continue
            raw_text = str(source_row.get("CV") or "").strip()
            if not 300 <= len(raw_text) <= 7000:
                counters["length_filtered_rows"] += 1
                continue
            # Require extractive evidence for the fields this task claims to parse.
            if not (_EDUCATION_RE.search(raw_text) and _PROJECT_RE.search(raw_text)):
                counters["missing_evidence_rows"] += 1
                continue
            text = sanitize_resume_text_for_training(raw_text).strip()
            if len(text) < 250 or detect_resume_pii(text):
                counters["privacy_filtered_rows"] += 1
                continue
            education = [item[:280] for item in _extract_evidence_sentences(text, _EDUCATION_RE, 2)]
            projects = [item[:280] for item in _extract_evidence_sentences(text, _PROJECT_RE, 2)]
            if not education or not projects:
                counters["weak_label_rows"] += 1
                continue
            internships = [item[:280] for item in _extract_evidence_sentences(text, _INTERNSHIP_RE, 1)]
            highlights = _unique_texts(
                sanitize_resume_text_for_training(part)
                for part in re.split(r"[\r\n]+|(?<=[.!?])\s+", str(source_row.get("Highlights") or ""))
            )[:3]
            highlights = [item[:180] for item in highlights]
            experience_years = source_row.get("Experience Years")
            if not highlights and isinstance(experience_years, (int, float)):
                highlights = [f"{experience_years:g} years of professional experience"]
            target = _djinni_target(source_row)
            if not target or not highlights:
                counters["weak_label_rows"] += 1
                continue
            profile_prefix = "\n".join(
                [f"Target role: {target}", *(f"Profile highlight: {item}" for item in highlights)]
            )
            training_text = _evidence_preserving_excerpt(
                text,
                evidence=[*education, *projects, *internships],
                prefix=profile_prefix,
                head_chars=1800,
                max_chars=3200,
            )
            # Only supervise skills visible in the compact training input.
            skills = extract_known_skills(training_text, schema)
            if len(skills) < 2:
                counters["weak_label_rows"] += 1
                continue
            row = {
                "id": f"djinni_resume_{source_id}",
                "task": "resume_parse",
                "source_type": "public_real_anonymized",
                "source_group": f"djinni_candidate:{source_id}",
                "text": training_text,
                "label": {
                    "目标岗位": target,
                    "教育背景": education,
                    "核心技能": skills,
                    "实习经历": internships,
                    "项目经历": projects,
                    "优势标签": highlights,
                },
                "meta": {
                    "language": "en",
                    "source_url": DJINNI_SOURCE_URL,
                    "source_revision": DJINNI_REVISION,
                    "license_status": "confirmed_mit",
                    "intended_usage": "sft_training",
                    "provenance": "djinni_platform_anonymized_candidate_profile",
                    "provenance_status": "paper_documented",
                    "annotation_status": "source_fields_plus_extractive_labels",
                    "label_method": "source_metadata_and_verbatim_evidence_extraction",
                    "training_eligible": True,
                    "contains_real_person_data": True,
                    "privacy_status": "source_anonymized_and_local_redaction_passed",
                },
            }
            keyword = _clean_text(source_row.get("Primary Keyword")) or "Other"
            stable_key = hashlib.sha1(source_id.encode("utf-8")).hexdigest()
            candidates_by_keyword[keyword].append((stable_key, row))

    for values in candidates_by_keyword.values():
        values.sort(key=lambda item: item[0])
    # Round-robin sampling prevents JavaScript/QA, the largest Djinni categories,
    # from crowding out smaller job families.
    selected: list[dict[str, Any]] = []
    keywords = sorted(candidates_by_keyword)
    depth = 0
    while len(selected) < max_rows:
        added = False
        for keyword in keywords:
            values = candidates_by_keyword[keyword]
            if depth < len(values):
                selected.append(values[depth][1])
                added = True
                if len(selected) >= max_rows:
                    break
        if not added:
            break
        depth += 1
    counters["eligible_rows"] = sum(len(rows) for rows in candidates_by_keyword.values())
    counters["selected_rows"] = len(selected)
    counters["selected_keywords"] = len({row["label"]["目标岗位"] for row in selected})
    return selected, dict(counters)


def _format_structured_item(item: Any, fields: tuple[str, ...]) -> str:
    if not isinstance(item, dict):
        return _clean_text(item)
    return " | ".join(_clean_text(item.get(field)) for field in fields if _clean_text(item.get(field)))


def _match_level(score: float) -> str:
    if score < 3.0:
        return "低匹配"
    if score < 5.5:
        return "基本匹配"
    if score < 7.5:
        return "较匹配"
    return "高匹配"


def _score_value(output: dict[str, Any]) -> float | None:
    aggregated = ((output.get("scores") or {}).get("aggregated_scores") or {})
    values = [float(value) for value in aggregated.values() if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def _criteria_by_score(output: dict[str, Any]) -> tuple[list[str], list[str]]:
    score_rows = [
        *((output.get("scores") or {}).get("macro_scores") or []),
        *((output.get("scores") or {}).get("micro_scores") or []),
    ]
    matched: list[str] = []
    missing: list[str] = []
    for item in score_rows:
        if not isinstance(item, dict):
            continue
        criterion = _clean_text(item.get("criteria"))
        score = item.get("score")
        if not criterion or not isinstance(score, (int, float)) or _CONDITION_WORDS_RE.search(criterion):
            continue
        if float(score) >= 6.0:
            matched.append(criterion)
        elif float(score) <= 4.0:
            missing.append(criterion)
    return _unique_texts(matched, 10), _unique_texts(missing, 10)


def _condition_match(output: dict[str, Any], words: re.Pattern[str]) -> bool:
    requirements = ((output.get("scores") or {}).get("requirements") or [])
    relevant = [
        bool(item.get("meets"))
        for item in requirements
        if isinstance(item, dict) and words.search(_clean_text(item.get("criteria")))
    ]
    return all(relevant) if relevant else True


def _netsol_analysis(
    data: dict[str, Any], matched: list[str], missing: list[str], target: str
) -> dict[str, Any]:
    output = data.get("output") or {}
    justification = _unique_texts(
        sanitize_resume_text_for_training(item) for item in output.get("justification") or []
    )
    strengths = [f"Evidence supports: {item}" for item in matched[:6]]
    gaps = [f"Evidence is weak or missing for: {item}" for item in missing[:6]]
    if not strengths:
        strengths = ["The source assessment contains usable evidence for an initial fit review."]
    if not gaps:
        gaps = ["No explicit high-priority gap was identified in the source scoring criteria."]
    return {
        "匹配结论": " ".join(justification[:2])[:1200]
        or "The candidate-job fit follows the source assessment and requirement evidence.",
        "匹配优势": strengths,
        "主要短板": gaps,
        "简历优化建议": [
            "Add concrete, verifiable resume evidence for: " + ", ".join(missing[:6])
        ]
        if missing
        else ["Keep the strongest role-relevant evidence explicit and measurable."],
        "推荐投递岗位方向": [target or "Role aligned with the candidate's current experience"],
    }


def build_netsol_match_rows(source_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, tuple[int, dict[str, Any]]] = {}
    counters = defaultdict(int)
    for path in sorted(Path(source_dir).glob("*.json")):
        counters["source_files"] += 1
        if not (path.name.startswith("match_") or path.name.startswith("mismatch_")):
            counters["non_training_files"] += 1
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        output = data.get("output") or {}
        inputs = data.get("input") or {}
        if not output.get("valid_resume_and_jd"):
            counters["invalid_pairs"] += 1
            continue
        score = _score_value(output)
        jd_text = sanitize_resume_text_for_training(str(inputs.get("job_description") or "")).strip()
        resume_text = sanitize_resume_text_for_training(str(inputs.get("resume") or "")).strip()
        if score is None or len(jd_text) < 120 or len(resume_text) < 120:
            counters["weak_pairs"] += 1
            continue
        if detect_resume_pii(jd_text) or detect_resume_pii(resume_text):
            counters["privacy_filtered_pairs"] += 1
            continue
        matched, missing = _criteria_by_score(output)
        matched = matched[:6]
        missing = missing[:6]
        if not (matched or missing) or not output.get("justification"):
            counters["weak_pairs"] += 1
            continue
        details = data.get("details") or {}
        target = _clean_text((output.get("personal_info") or {}).get("current_position"))
        education_match = _condition_match(
            output, re.compile(r"\b(?:degree|bachelor|master|ph\.?d|education)\b", re.I)
        )
        experience_match = _condition_match(output, re.compile(r"\b(?:year|experience)\b", re.I))
        pair_hash = normalized_input_hash(f"{jd_text}\n---\n{resume_text}")
        jd_hash = normalized_input_hash(jd_text)
        resume_hash = normalized_input_hash(resume_text)
        jd_training_text = _evidence_preserving_excerpt(
            jd_text,
            evidence=[*matched, *missing],
            head_chars=900,
            max_chars=1100,
        )
        resume_training_text = _evidence_preserving_excerpt(
            resume_text,
            evidence=[*matched, *missing],
            head_chars=1100,
            max_chars=1400,
        )
        projects = _unique_texts(
            _format_structured_item(item, ("title", "description"))
            for item in details.get("projects") or []
        )[:2]
        level = _match_level(score)
        row = {
            "id": f"netsol_match_{pair_hash}",
            "task": "match",
            "source_type": "synthetic_teacher_labeled_public_pair",
            "source_group": f"netsol_pair:{pair_hash}",
            "jd_text": jd_training_text,
            "resume_text": resume_training_text,
            "label": {
                "raw_label": "match" if path.name.startswith("match_") else "mismatch",
                "raw_score": round(score, 4),
                "匹配等级": level,
                "岗位方向匹配": path.name.startswith("match_"),
                "学历匹配": education_match,
                "经验匹配": experience_match,
                "命中技能": matched,
                "缺失技能": missing,
                "命中项目": projects if level in {"较匹配", "高匹配"} else [],
            },
            "analysis": _netsol_analysis(data, matched, missing, target),
            "meta": {
                "language": "en",
                "source_url": NETSOL_SOURCE_URL,
                "source_revision": NETSOL_REVISION,
                "license_status": "source_declared_cc",
                "intended_usage": "sft_training",
                "provenance": "netsol_resume_score_details_gpt4o",
                "provenance_status": "documented_machine_generated",
                "annotation_status": "teacher_labeled",
                "annotation_provenance": "gpt4o_scores_requirements_and_justification",
                "generator": "gpt4o_teacher",
                "pair_type": "synthetic_teacher_labeled_pair",
                "condition_label_source": "source_teacher_assessment",
                "training_eligible": True,
                "contains_real_person_data": True,
                "privacy_status": "local_redaction_passed",
                "input_compaction": "head_plus_scored_criteria_evidence",
                # The upstream bipartite graph has one 848-pair connected component.
                # Keep it train-only and use independent local/Djinni evaluation sets.
                "entity_split": "train",
                "jd_entity_hash": jd_hash,
                "resume_entity_hash": resume_hash,
                "jd_direction": target,
                "resume_direction": target,
            },
        }
        quality = len(output.get("justification") or []) + len(matched) + len(missing)
        previous = candidates.get(pair_hash)
        if previous is None or quality > previous[0]:
            candidates[pair_hash] = (quality, row)
        else:
            counters["duplicate_pairs"] += 1
    rows = [item[1] for item in sorted(candidates.values(), key=lambda item: item[1]["id"])]
    counters["selected_pairs"] = len(rows)
    counters["match_level_counts"] = dict(
        sorted(
            {
                level: sum(row["label"]["匹配等级"] == level for row in rows)
                for level in ("低匹配", "基本匹配", "较匹配", "高匹配")
            }.items()
        )
    )
    return rows, dict(counters)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建中文技术岗公开简历训练输入")
    parser.add_argument(
        "--faircv-templates",
        default="data/external/faircv/data/resumes_template.json",
    )
    parser.add_argument(
        "--public-web-resume-sources",
        default=PUBLIC_WEB_RESUME_MANIFEST,
    )
    parser.add_argument(
        "--public-web-resume-cache",
        default=PUBLIC_WEB_RESUME_CACHE,
    )
    parser.add_argument(
        "--refresh-public-web-resumes",
        action="store_true",
        help="下载清单中的公开简历，只缓存脱敏后的任务相关段落；原始文件不落盘。",
    )
    parser.add_argument(
        "--refresh-missing-public-web-resumes",
        action="store_true",
        help="只下载尚无脱敏快照的公开简历，不重复 OCR 已有来源。",
    )
    parser.add_argument(
        "--djinni-candidates",
        default="data/private/djinni_real_ranking_v1/source/candidates.parquet",
    )
    parser.add_argument(
        "--ranking-candidates",
        default="data/private/djinni_real_ranking_v1/candidates.jsonl",
    )
    parser.add_argument(
        "--human-reviewed-pairs",
        default=HUMAN_REVIEWED_PAIR_MANIFEST,
    )
    parser.add_argument(
        "--public-jd-pool",
        default="data/eval/public_jd_candidate_pool.jsonl",
    )
    parser.add_argument(
        "--netsol-dir", default="data/external/netsol_resume_score_details"
    )
    parser.add_argument("--label-schema", default="configs/label_schema.yaml")
    parser.add_argument("--max-resumes", type=int, default=5000)
    parser.add_argument(
        "--resume-out", default="data/external/public_resume_imports_zh_tech.jsonl"
    )
    parser.add_argument(
        "--match-out", default="data/external/public_match_imports_zh_tech.jsonl"
    )
    parser.add_argument(
        "--build-english-auxiliary",
        action="store_true",
        help="显式构建英文 Djinni 简历与 Netsol 教师匹配辅助集；默认训练不使用。",
    )
    parser.add_argument(
        "--english-resume-out",
        default="data/external/public_resume_imports_en_aux.jsonl",
    )
    parser.add_argument(
        "--english-match-out",
        default="data/external/public_match_imports_en_teacher_aux.jsonl",
    )
    parser.add_argument(
        "--report-out", default="outputs/eval_reports/public_training_import_report.json"
    )
    args = parser.parse_args()

    schema = yaml.safe_load(Path(args.label_schema).read_text(encoding="utf-8")) or {}
    refresh_report: dict[str, Any] = {"enabled": False}
    if args.refresh_public_web_resumes or args.refresh_missing_public_web_resumes:
        refresh_report = {
            "enabled": True,
            **refresh_public_resume_snapshots(
                args.public_web_resume_sources,
                args.public_web_resume_cache,
                skip_existing=args.refresh_missing_public_web_resumes,
            ),
        }

    faircv_rows, faircv_report = build_faircv_resume_rows(
        args.faircv_templates,
        schema=schema,
    )
    web_rows, web_report = build_public_web_resume_rows(
        args.public_web_resume_sources,
        args.public_web_resume_cache,
        schema=schema,
    )
    resume_rows = faircv_rows + web_rows
    resume_report = {
        "selected_rows": len(resume_rows),
        "faircv_templates": faircv_report,
        "self_published_public_resumes": web_report,
        "snapshot_refresh": refresh_report,
    }
    match_rows, match_report = build_human_reviewed_public_match_rows(
        args.human_reviewed_pairs,
        web_rows,
        args.public_jd_pool,
    )
    write_jsonl(args.resume_out, resume_rows)
    write_jsonl(args.match_out, match_rows)
    auxiliary_report: dict[str, Any] = {"enabled": False}
    if args.build_english_auxiliary:
        english_resume_rows, english_resume_report = build_djinni_resume_rows(
            args.djinni_candidates,
            schema=schema,
            reserved_source_ids=_reserved_djinni_candidate_ids(args.ranking_candidates),
            max_rows=args.max_resumes,
        )
        english_match_rows, english_match_report = build_netsol_match_rows(args.netsol_dir)
        write_jsonl(args.english_resume_out, english_resume_rows)
        write_jsonl(args.english_match_out, english_match_rows)
        auxiliary_report = {
            "enabled": True,
            "resume": english_resume_report,
            "match": english_match_report,
        }
    report = {
        "resume": resume_report,
        "match": match_report,
        "english_auxiliary": auxiliary_report,
        "sources": {
            "faircv": {"url": FAIRCV_SOURCE_URL, "revision": FAIRCV_REVISION},
            "self_published_public_resumes": {
                "manifest": args.public_web_resume_sources,
                "cache": args.public_web_resume_cache,
            },
            "human_reviewed_public_pairs": {
                "manifest": args.human_reviewed_pairs,
                "jd_pool": args.public_jd_pool,
            },
            "djinni_auxiliary": {"url": DJINNI_SOURCE_URL, "revision": DJINNI_REVISION},
            "netsol_auxiliary": {"url": NETSOL_SOURCE_URL, "revision": NETSOL_REVISION},
        },
    }
    write_text(args.report_out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
