from jobmatch_tune.inference.structured_output import build_response_format


def test_build_response_format_for_jd_parse():
    response_format = build_response_format("jd_parse")
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "JDParseResult"
    properties = response_format["json_schema"]["schema"]["properties"]
    assert "岗位方向" in properties
    assert "核心职责" in properties


def test_build_response_format_for_resume_parse():
    response_format = build_response_format("resume_parse")
    assert response_format["json_schema"]["name"] == "ResumeParseResult"
    properties = response_format["json_schema"]["schema"]["properties"]
    assert "目标岗位" in properties
    assert "项目经历" in properties


def test_build_response_format_for_match():
    response_format = build_response_format("match")
    assert response_format["json_schema"]["name"] == "MatchAnalysisResult"
    properties = response_format["json_schema"]["schema"]["properties"]
    assert "匹配结论" in properties
    assert "简历优化建议" in properties
