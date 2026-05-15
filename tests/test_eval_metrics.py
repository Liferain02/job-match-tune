from jobmatch_tune.eval.metrics import precision_recall_f1, text_exact_match


def test_precision_recall_f1():
    score = precision_recall_f1(["Python", "RAG"], ["python", "Java"])
    assert score["precision"] == 0.5
    assert score["recall"] == 0.5
    assert score["f1"] == 0.5


def test_text_exact_match_ignores_case_and_spaces():
    assert text_exact_match(" 本科及以上 ", "本科及以上") == 1.0
    assert text_exact_match("三年以上 工作经验", "三年以上工作经验") == 0.0
