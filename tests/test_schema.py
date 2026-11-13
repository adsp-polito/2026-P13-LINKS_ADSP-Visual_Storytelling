import pytest
from pydantic import ValidationError

from p13.common.schema import GeneratedStory, StoryRecord


def test_valid_story(story_dict):
    r = StoryRecord.model_validate(story_dict)
    assert r.dataset == "vist" and len(r.images) == 3


@pytest.mark.parametrize("n", [2, 11])
def test_image_count_bounds(story_dict, n):
    story_dict["images"] = [f"{i}.jpg" for i in range(n)]
    with pytest.raises(ValidationError):
        StoryRecord.model_validate(story_dict)


@pytest.mark.parametrize(
    "field,value",
    [("dataset", "coco"), ("split", "dev"), ("story", ""), ("id", ""), ("images", ["a", "", "c"])],
)
def test_invalid_fields(story_dict, field, value):
    story_dict[field] = value
    with pytest.raises(ValidationError):
        StoryRecord.model_validate(story_dict)


def test_unknown_field_rejected(story_dict):
    story_dict["extra"] = 1
    with pytest.raises(ValidationError):
        StoryRecord.model_validate(story_dict)


def test_descriptions_must_align_with_images(story_dict):
    story_dict["dii"] = ["one", "two"]
    with pytest.raises(ValidationError, match="dii"):
        StoryRecord.model_validate(story_dict)
    story_dict["dii"] = ["one", "two", "three"]
    assert StoryRecord.model_validate(story_dict).dii[2] == "three"


def test_generated_zero_shot_and_adapter(generated_dict):
    assert GeneratedStory.model_validate(generated_dict).adapter is None
    generated_dict["adapter"] = "outputs/E1_lora_2b_6gb/adapter"
    g = GeneratedStory.model_validate(generated_dict)
    assert g.key == ("vist", "val", "s1")


def test_generated_blank_adapter_rejected(generated_dict):
    generated_dict["adapter"] = "  "
    with pytest.raises(ValidationError):
        GeneratedStory.model_validate(generated_dict)


def test_round_trip(story_dict, generated_dict):
    for model, d in ((StoryRecord, story_dict), (GeneratedStory, generated_dict)):
        obj = model.model_validate(d)
        assert model.model_validate_json(obj.model_dump_json()) == obj
