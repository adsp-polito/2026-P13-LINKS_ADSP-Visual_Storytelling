import json

import pytest

from p13.common.jsonl import JSONLError, detect_model, read_jsonl, validate_file, write_jsonl
from p13.common.schema import GeneratedStory, StoryRecord


def test_write_read_round_trip(tmp_path, story_dict):
    recs = [StoryRecord.model_validate({**story_dict, "id": f"s{i}"}) for i in range(3)]
    path = tmp_path / "sub" / "vist_val.jsonl"
    assert write_jsonl(path, recs) == 3
    assert read_jsonl(path, StoryRecord) == recs
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_unicode_preserved(tmp_path, story_dict):
    story_dict["story"] = "Un caffè a Torino – très bien."
    path = tmp_path / "u.jsonl"
    write_jsonl(path, [StoryRecord.model_validate(story_dict)])
    assert "caffè" in path.read_text(encoding="utf-8")


def test_bad_json_reports_line(tmp_path, story_dict):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(story_dict) + "\n\n{not json}\n", encoding="utf-8")
    with pytest.raises(JSONLError) as e:
        read_jsonl(path, StoryRecord)
    assert e.value.line_no == 3


def test_schema_error_reports_field(tmp_path, story_dict):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({**story_dict, "dataset": "coco"}) + "\n", encoding="utf-8")
    with pytest.raises(JSONLError, match="dataset"):
        read_jsonl(path, StoryRecord)


def test_duplicate_ids_rejected(tmp_path, story_dict):
    path = tmp_path / "dup.jsonl"
    path.write_text((json.dumps(story_dict) + "\n") * 2, encoding="utf-8")
    with pytest.raises(JSONLError, match="duplicate"):
        read_jsonl(path, StoryRecord)


def test_same_id_different_prompts_is_fine(tmp_path, generated_dict):
    recs = [GeneratedStory.model_validate({**generated_dict, "prompt_id": p}) for p in ("P0", "P1")]
    path = tmp_path / "gen.jsonl"
    write_jsonl(path, recs)
    assert validate_file(path) == 2
    assert detect_model(path) is GeneratedStory
