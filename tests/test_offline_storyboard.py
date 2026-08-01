import json
import unittest
from pathlib import Path

from node import (
    RH_OfflineStoryboardParser_Node,
    RH_OfflineStoryboardRequest_Node,
    RH_OfflineStoryboardSceneRequests_Node,
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
        self.assertIn("Required scene count: 4", request)
        self.assertIn('"scene_id": 1', request)
        self.assertIn('"scene_id": 4', request)
        self.assertIn('"character_bible"', request)
        self.assertIn("must never change between scenes", request)
        self.assertEqual((count, language, ratio), (4, "English", "16:9"))
        self.assertEqual((width, height), (1024, 576))

    def test_parser_connection_types_match_request_outputs(self):
        request_outputs = RH_OfflineStoryboardRequest_Node.RETURN_TYPES
        parser_inputs = RH_OfflineStoryboardParser_Node.INPUT_TYPES()["required"]
        self.assertEqual(request_outputs[2], parser_inputs["prompt_language"][0])
        self.assertEqual(request_outputs[3], parser_inputs["aspect_ratio"][0])

    def test_parser_aggregates_per_scene_qwen_responses(self):
        raw = [
            json.dumps({"scenes": [{"scene_id": index, "prompt": f"scene {index}"}]})
            for index in range(1, 4)
        ]
        positive, negative, storyboard_json, count = RH_OfflineStoryboardParser_Node().parse_storyboard(
            raw,
            [3],
            ["English"],
            ["16:9"],
            ["low quality"],
        )
        self.assertEqual(positive, ["scene 1", "scene 2", "scene 3"])
        self.assertEqual(negative, ["low quality"] * 3)
        self.assertEqual(count, 3)
        self.assertEqual(len(json.loads(storyboard_json)["shots"]), 3)

    def test_two_pass_scene_requests_share_one_character_lock(self):
        outline = json.dumps(
            {
                "title": "Locked character",
                "character_bible": {
                    "identity": "young East Asian woman",
                    "hairstyle": "black blunt-bang bob with a white flower",
                    "clothing": "light beige embroidered gauze jacket and pink trousers",
                },
                "style_bible": {"visual_style": "cinematic realism"},
                "scenes": [
                    {"scene_id": 1, "action": "stands in a courtyard"},
                    {"scene_id": 2, "action": "opens a letter in another room"},
                ],
            }
        )
        requests, normalized, count, language, ratio = (
            RH_OfflineStoryboardSceneRequests_Node().build_scene_requests(
                outline,
                2,
                "English",
                "16:9",
                "Generate one shot. Return JSON only.",
            )
        )
        self.assertEqual(len(requests), 2)
        self.assertEqual((count, language, ratio), (2, "English", "16:9"))
        self.assertIn("black blunt-bang bob with a white flower", requests[0])
        self.assertIn("black blunt-bang bob with a white flower", requests[1])
        self.assertEqual(json.loads(normalized)["generation_settings"]["mode"], "offline_qwen_two_pass")

    def test_parser_prepends_identical_character_anchor_to_every_scene(self):
        outline = json.dumps(
            {
                "character_bible": {
                    "identity": "young East Asian woman",
                    "hairstyle": "black blunt-bang bob with a white flower",
                    "clothing": "light beige embroidered gauze jacket and pink trousers",
                },
                "style_bible": {"visual_style": "cinematic realism"},
                "scenes": [
                    {"scene_id": 1, "action": "stands in a courtyard"},
                    {"scene_id": 2, "action": "reads a letter indoors"},
                ],
            }
        )
        raw = [
            json.dumps({"scene_id": 1, "prompt": "wide shot in a courtyard"}),
            json.dumps({"scene_id": 2, "prompt": "close-up while reading a letter"}),
        ]
        positive, _, storyboard_json, _ = RH_OfflineStoryboardParser_Node().parse_storyboard(
            raw,
            [2],
            ["English"],
            ["16:9"],
            ["low quality"],
            [outline],
        )
        prefix = "16:9, CHARACTER CONTINUITY LOCK"
        self.assertTrue(all(prompt.startswith(prefix) for prompt in positive))
        self.assertTrue(all("black blunt-bang bob with a white flower" in prompt for prompt in positive))
        storyboard = json.loads(storyboard_json)
        self.assertEqual(storyboard["generation_settings"]["mode"], "offline_qwen_two_pass")
        self.assertEqual(storyboard["shots"][0]["raw_prompt"], "wide shot in a courtyard")

    def test_bundled_qwen_workflow_enables_default_template(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "RH_Krea2_Offline_Qwen3VL_KleinSwap_batch_v7.0_native_parser.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        text_generates = [node for node in workflow["nodes"] if node["type"] == "TextGenerate"]
        self.assertEqual(len(text_generates), 2)
        self.assertTrue(all(node["widgets_values"][-1] is True for node in text_generates))
        self.assertTrue(
            any(node["type"] == "RH_OFFLINE_STORYBOARD_SCENE_REQUESTS" for node in workflow["nodes"])
        )

    def test_bundled_workflow_has_no_rgthree_image_comparer(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "RH_Krea2_Offline_Qwen3VL_KleinSwap_batch_v7.0_native_parser.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        comparer_nodes = [
            node
            for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
            for node in subgraph.get("nodes", [])
            if node.get("type") == "Image Comparer (rgthree)"
        ]
        self.assertEqual(comparer_nodes, [])

    def test_bundled_workflow_has_no_temp_image_preview_nodes(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "RH_Krea2_Offline_Qwen3VL_KleinSwap_batch_v7.0_native_parser.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        self.assertFalse(any(node.get("type") == "PreviewImage" for node in workflow["nodes"]))
        self.assertTrue(any(node.get("type") == "SaveImage" for node in workflow["nodes"]))


if __name__ == "__main__":
    unittest.main()

