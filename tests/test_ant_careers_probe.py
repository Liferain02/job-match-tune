from __future__ import annotations

from jobmatch_tune.crawler.ant_careers_probe import (
    _extract_search_items,
    build_social_search_variants,
)


def test_extract_search_items_handles_condition_payload() -> None:
    payload = {
        "content": {
            "searchItems": [
                {"type": "category", "items": [{"label": "技术类", "value": "11"}]},
                {"type": "workCity", "items": [{"label": "杭州", "value": "330100"}]},
            ]
        }
    }
    items = _extract_search_items(payload)
    assert len(items) == 2
    assert items[0]["type"] == "category"


def test_build_social_search_variants_uses_discovered_values() -> None:
    payload = {
        "content": {
            "searchItems": [
                {"type": "dept", "items": [{"label": "平台技术事业群", "value": "19612"}]},
                {"type": "category", "items": [{"label": "技术类", "value": "11"}]},
                {"type": "workCity", "items": [{"label": "杭州", "value": "330100"}]},
            ]
        }
    }
    variants = build_social_search_variants(payload)
    assert len(variants) == 4
    assert variants[2]["category"] == ["11"]
    assert variants[3]["workCity"] == "330100"
    assert variants[3]["dept"] == "19612"
