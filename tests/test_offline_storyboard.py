import json
import unittest

from node import (
    RH_OfflineStoryboardParser_Node,
    RH_OfflineStoryboardRequest_Node,
    parse_offline_storyboard_text,
)


class OfflineStoryboardParserTests(unittest.TestCase):
    def test_parses_strict_json_scenes(self):
        raw = json.dumps(
            {
                "title": "雨夜来信",
                "scenes": [
                    {"scene_id": 1, "prompt": "雨夜中的宋代庭院", "negative_prompt": "模糊"},
                    {"scene_id": 2, "prompt": "女孩展开来自未来的信"},
                ],
            },
            ensure_ascii=False,
        )
        positive, negative, storyboard = parse_offline_storyboard_text(raw, 2, "低质量")
        self.assertEqual(positive, ["雨夜中的宋代庭院", "女孩展开来自未来的信"])
        self.assertEqual(negative, ["模糊", "低质量"])
        self.assertEqual(len(storyboard["shots"]), 2)

    def test_parses_markdown_fenced_json(self):
        raw = """```json
{"scenes":[{"prompt":"wide shot"},{"prompt":"close-up"}]}
```"""
        positive, _, _ = parse_offline_storyboard_text(raw, 2)
        self.assertEqual(positive, ["wide shot", "close-up"])

    def test_parses_numbered_lines(self):
        raw = "1. 雨夜远景，女孩走入庭院\n2、面部特写，女孩读取信件"
        positive, _, _ = parse_offline_storyboard_text(raw, 2)
        self.assertEqual(positive, ["雨夜远景，女孩走入庭院", "面部特写，女孩读取信件"])

    def test_builds_prompt_from_scene_fields_when_prompt_is_missing(self):
        raw = json.dumps(
            {
                "scenes": [
                    {
                        "shot_type": "medium shot",
                        "location": "courtyard",
                        "action": "opens a letter",
                        "lighting": "moonlight",
                    }
                ]
            }
        )
        positive, _, _ = parse_offline_storyboard_text(raw, 1)
        self.assertIn("medium shot", positive[0])
        self.assertIn("opens a letter", positive[0])

    def test_rejects_incomplete_generation(self):
        with self.assertRaisesRegex(ValueError, "returned 1 usable storyboard prompts; expected 2"):
            parse_offline_storyboard_text('{"scenes":[{"prompt":"only one"}]}', 2)

    def test_request_requires_exact_count_and_json(self):
        request, count, language, ratio, width, height = RH_OfflineStoryboardRequest_Node().build_request(
            "A girl discovers a letter.",
            4,
            "English",
            "16:9",
            "Create a cinematic storyboard. Return JSON only.",
        )
        self.assertIn("exactly 4", request)
        self.assertIn('"scenes"', request)
        self.assertEqual((count, language, ratio), (4, "English", "16:9"))
        self.assertEqual((width, height), (1024, 576))

    def test_parser_connection_types_match_request_outputs(self):
        request_outputs = RH_OfflineStoryboardRequest_Node.RETURN_TYPES
        parser_inputs = RH_OfflineStoryboardParser_Node.INPUT_TYPES()["required"]
        self.assertEqual(request_outputs[2], parser_inputs["prompt_language"][0])
        self.assertEqual(request_outputs[3], parser_inputs["aspect_ratio"][0])


if __name__ == "__main__":
    unittest.main()

