from __future__ import annotations

from typing import Any

from jobmatch_tune.schemas import JDParseResult, ResumeParseResult


TASK_SCHEMA_MODELS = {
    "jd_parse": JDParseResult,
    "resume_parse": ResumeParseResult,
}


def get_task_schema_model(task: str):
    try:
        return TASK_SCHEMA_MODELS[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported task for structured output: {task}") from exc


def build_response_format(task: str) -> dict[str, Any]:
    schema_model = get_task_schema_model(task)
    schema = schema_model.model_json_schema(by_alias=True)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "schema": schema,
        },
    }
