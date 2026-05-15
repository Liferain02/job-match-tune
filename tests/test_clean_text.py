from jobmatch_tune.preprocess.clean_text import clean_text


def test_clean_text_removes_html_and_private_info():
    html = "<html><body><script>x</script><h1>岗位</h1><p>联系 13812345678 test@example.com</p></body></html>"
    text = clean_text(html, is_html=True)
    assert "script" not in text.lower()
    assert "13812345678" not in text
    assert "test@example.com" not in text
    assert "[手机号]" in text
    assert "[邮箱]" in text
