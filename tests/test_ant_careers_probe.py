from __future__ import annotations

from jobmatch_tune.crawler.ant_careers_probe import (
    _extract_search_items,
    build_social_search_variants,
    build_position_id_search_variants,
    _extract_id_list,
    extract_endpoint_snippets,
    extract_script_urls,
    extract_social_chunk_hints,
    extract_talent_route_hints,
    extract_tern_site_config,
    select_candidate_bundle_urls,
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
    group_payload = {"content": [{"id": 24001, "name": "社会招聘"}]}
    talent_plan_payload = {"content": [{"id": 105, "name": "研究型实习生"}]}
    variants = build_social_search_variants(
        payload,
        position_group_payload=group_payload,
        talent_plan_payload=talent_plan_payload,
    )
    assert len(variants) == 10
    assert variants[2]["payload"]["category"] == ["11"]
    assert variants[3]["payload"]["workCity"] == "330100"
    assert variants[3]["payload"]["dept"] == "19612"
    assert variants[4]["payload"]["recruitType"] == "social_recruit"
    assert variants[5]["payload"]["recruitType"] == ["social_recruit"]
    assert variants[6]["payload"]["categoryList"] == ["11"]
    assert variants[7]["payload"]["deptIds"] == ["19612"]
    assert variants[8]["payload"]["positionGroupId"] == "24001"
    assert variants[9]["payload"]["talentPlanId"] == "105"


def test_extract_id_list_handles_content_rows() -> None:
    payload = {"content": [{"id": 1}, {"id": "2"}, {"name": "x"}, {"id": 1}]}
    assert _extract_id_list(payload) == ["1", "2"]


def test_extract_script_urls_normalizes_relative_and_protocol_relative() -> None:
    html = """
    <html><head>
    <script src="/assets/index.js"></script>
    <script src="//cdn.example.com/app.js"></script>
    </head></html>
    """
    urls = extract_script_urls(html)
    assert urls == [
        "https://hrcareersweb.antgroup.com/assets/index.js",
        "https://cdn.example.com/app.js",
    ]


def test_select_candidate_bundle_urls_prefers_app_like_assets() -> None:
    urls = [
        "https://a.com/runtime.js",
        "https://a.com/main.js",
        "https://a.com/index.js",
        "https://a.com/umi.js",
    ]
    selected = select_candidate_bundle_urls(urls)
    assert selected[:3] == [
        "https://a.com/main.js",
        "https://a.com/index.js",
        "https://a.com/umi.js",
    ]


def test_extract_endpoint_snippets_finds_context() -> None:
    text = 'xxx "/api/social/position/search" yyy "/api/social/position/search" zzz'
    snippets = extract_endpoint_snippets(text, "/api/social/position/search", radius=5)
    assert snippets
    assert "/api/social/position/search" in snippets[0]


def test_build_position_id_search_variants_uses_discovered_values() -> None:
    payload = {
        "content": {
            "searchItems": [
                {"type": "dept", "items": [{"label": "平台技术事业群", "value": "19612"}]},
                {"type": "category", "items": [{"label": "技术类", "value": "11"}]},
                {"type": "workCity", "items": [{"label": "杭州", "value": "330100"}]},
            ]
        }
    }
    variants = build_position_id_search_variants(payload)
    assert len(variants) == 7
    assert variants[3]["payload"]["category"] == "11"
    assert variants[4]["payload"]["dept"] == "19612"
    assert variants[5]["payload"]["workCity"] == "330100"


def test_extract_tern_site_config_parses_json() -> None:
    html = """
    <html><head>
    <script type="tern-app-config">{"presets":[{"props":{"hooksJSUrl":"https://a.com/hook.js"}}]}</script>
    </head></html>
    """
    parsed = extract_tern_site_config(html)
    assert parsed is not None
    assert parsed["presets"][0]["props"]["hooksJSUrl"] == "https://a.com/hook.js"


def test_extract_talent_route_hints_returns_unique_routes() -> None:
    html = """
    <a href="https://talent.antgroup.com/off-campus"></a>
    <a href="https://talent.antgroup.com/off-campus-home"></a>
    <a href="https://talent.antgroup.com/off-campus"></a>
    """
    assert extract_talent_route_hints(html) == ["/off-campus", "/off-campus-home"]


def test_extract_social_chunk_hints_returns_unique_chunk_names() -> None:
    html = "p__SocialRecruitment__SRList__index xxx p__SocialRecruitment__Home__index xxx p__SocialRecruitment__SRList__index"
    assert extract_social_chunk_hints(html) == [
        "p__SocialRecruitment__SRList__index",
        "p__SocialRecruitment__Home__index",
    ]
