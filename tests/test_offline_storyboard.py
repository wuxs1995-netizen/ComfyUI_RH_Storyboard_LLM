import base64
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from node import (
    RH_ConfigurableStoryboardContinuity_Node,
    RH_ConfigurableStoryboard_Node,
    RH_MultiSceneContinuityLLM_Node,
    RH_OfflineStoryboardParser_Node,
    RH_OfflineStoryboardRequest_Node,
    RH_OfflineStoryboardSceneRequests_Node,
    RH_StoryboardScenePrefixes_Node,
    _build_continuity_scene_request,
    _extract_api_image_bytes,
    _locked_continuity_prompt,
    _normalize_continuity_outline,
    _scene_save_prefix,
    parse_offline_storyboard_text,
)


class OfflineStoryboardParserTests(unittest.TestCase):
    def test_extracts_base64_image_api_response(self):
        expected = b"image-bytes"
        response = {"data": [{"b64_json": base64.b64encode(expected).decode("ascii")}]}
        self.assertEqual(_extract_api_image_bytes(response), expected)

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
        request, count, language, ratio, width, height, source_story = RH_OfflineStoryboardRequest_Node().build_request(
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
        self.assertIn('"supporting_characters"', request)
        self.assertIn('"ethnicity"', request)
        self.assertIn('"nationality"', request)
        self.assertIn('"skin_tone"', request)
        self.assertIn('"height"', request)
        self.assertIn('"body_build"', request)
        self.assertIn('"body_proportions"', request)
        self.assertIn('do not infer nationality from facial appearance alone', request)
        self.assertIn('"story_fact"', request)
        self.assertIn('"current_action"', request)
        self.assertIn('"must_not_show"', request)
        self.assertIn("THE ONLY AUTHORITATIVE SOURCE", request)
        self.assertIn("MUST NOT ADD OR REPLACE PLOT FACTS", request)
        self.assertIn("must never change between scenes", request)
        self.assertEqual((count, language, ratio), (4, "English", "16:9"))
        self.assertEqual((width, height), (1024, 576))
        self.assertEqual(source_story, "A girl discovers a letter.")

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
        self.assertEqual(
            positive,
            ["SCENE 01/03, scene 1", "SCENE 02/03, scene 2", "SCENE 03/03, scene 3"],
        )
        self.assertEqual(negative, ["low quality"] * 3)
        self.assertEqual(count, 3)
        self.assertEqual(len(json.loads(storyboard_json)["shots"]), 3)

    def test_two_pass_scene_requests_share_one_character_lock(self):
        outline = json.dumps(
            {
                "title": "Locked character",
                "character_bible": {
                    "identity": "young East Asian woman",
                    "ethnicity": "East Asian",
                    "nationality": "Chinese",
                    "skin_tone": "fair neutral skin tone",
                    "height": "165 cm",
                    "body_build": "slender athletic build",
                    "body_proportions": "narrow shoulders and long limbs",
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
                "A woman discovers a letter and reads it in another room.",
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
        self.assertIn("slender athletic build", requests[0])
        self.assertIn("slender athletic build", requests[1])
        self.assertIn("Chinese", requests[0])
        self.assertIn("Chinese", requests[1])
        self.assertIn("SOURCE STORY", requests[0])
        self.assertIn("CURRENT ACTION", requests[0])
        self.assertNotIn('"next_scene"', requests[0])
        self.assertEqual(json.loads(normalized)["generation_settings"]["mode"], "offline_qwen_two_pass")

    def test_parser_adds_supporting_character_only_when_present(self):
        outline = json.dumps(
            {
                "source_story": "A woman waits; a courier approaches and gives her a letter.",
                "character_bible": {
                    "character_id": "primary",
                    "identity": "young East Asian woman",
                    "hairstyle": "black bob with white flower",
                    "clothing": "beige coat",
                },
                "supporting_characters": [
                    {
                        "character_id": "courier",
                        "role": "courier",
                        "identity": "middle-aged man",
                        "clothing": "dark raincoat",
                    }
                ],
                "style_bible": {"visual_style": "cinematic realism"},
                "scenes": [
                    {
                        "scene_id": 1,
                        "story_fact": "The woman waits.",
                        "characters_present": ["primary"],
                    },
                    {
                        "scene_id": 2,
                        "story_fact": "The courier gives her a letter.",
                        "characters_present": ["primary", "courier"],
                    },
                ],
            }
        )
        raw = [
            json.dumps({"scene_id": 1, "prompt": "the woman waits"}),
            json.dumps({"scene_id": 2, "prompt": "the courier hands over the letter"}),
        ]
        positive, _, storyboard_json, _ = RH_OfflineStoryboardParser_Node().parse_storyboard(
            raw, [2], ["English"], ["16:9"], ["low quality"], [outline]
        )
        self.assertNotIn("middle-aged man", positive[0])
        self.assertIn("middle-aged man", positive[1])
        storyboard = json.loads(storyboard_json)
        self.assertEqual(storyboard["shots"][1]["story_fact"], "The courier gives her a letter.")

    def test_parser_prepends_identical_character_anchor_to_every_scene(self):
        outline = json.dumps(
            {
                "character_bible": {
                    "identity": "young East Asian woman",
                    "ethnicity": "East Asian",
                    "nationality": "Chinese",
                    "skin_tone": "fair neutral skin tone",
                    "height": "165 cm",
                    "body_build": "slender athletic build",
                    "body_proportions": "narrow shoulders and long limbs",
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
        self.assertTrue(all('"ethnicity":"East Asian"' in prompt for prompt in positive))
        self.assertTrue(all('"nationality":"Chinese"' in prompt for prompt in positive))
        self.assertTrue(all('"body_build":"slender athletic build"' in prompt for prompt in positive))
        self.assertTrue(
            all('"body_proportions":"narrow shoulders and long limbs"' in prompt for prompt in positive)
        )
        storyboard = json.loads(storyboard_json)
        self.assertEqual(storyboard["generation_settings"]["mode"], "offline_qwen_two_pass")
        self.assertEqual(storyboard["shots"][0]["raw_prompt"], "wide shot in a courtyard")
        self.assertEqual(storyboard["shots"][0]["scene_label"], "SCENE 01/02")

    def test_api_director_schema_locks_demographic_and_body_traits(self):
        request = RH_ConfigurableStoryboard_Node()._director_request(
            "A woman opens a letter.", 2, "English", "16:9"
        )
        for field in (
            '"ethnicity"',
            '"nationality"',
            '"skin_tone"',
            '"height"',
            '"body_build"',
            '"body_proportions"',
        ):
            self.assertIn(field, request)
        self.assertIn("国籍不得仅凭面部外观推断", request)

    def test_continuity_director_schema_requires_location_props_and_state_ledger(self):
        request = RH_ConfigurableStoryboardContinuity_Node()._director_request(
            "A girl breaks an old tree on a Song dynasty street.",
            2,
            "English",
            "16:9",
        )
        for field in (
            '"location_bible"',
            '"prop_bible"',
            '"location_id"',
            '"props_present"',
            '"state_before"',
            '"state_after"',
            '"must_not_show"',
        ):
            self.assertIn(field, request)
        self.assertIn("same location_id", request)
        self.assertIn("same prop_id", request)

    def test_continuity_prompt_programmatically_locks_shared_location_and_prop(self):
        outline = {
            "character_bible": {
                "character_id": "primary",
                "identity": "Chinese teenage girl",
                "hairstyle": "black twin braids",
                "clothing": "blue linen jacket",
            },
            "supporting_characters": [],
            "location_bible": {
                "song_street": {
                    "location_id": "song_street",
                    "architecture": "two-storey dark timber shops with grey tiled roofs",
                    "spatial_layout": "east-west stone road, tea stall on the south side",
                    "fixed_objects": "red wine banner and stone lion at the west entrance",
                }
            },
            "prop_bible": {
                "old_tree": {
                    "prop_id": "old_tree",
                    "name": "old locust tree",
                    "design": "thick forked trunk leaning left",
                    "color": "dark weathered brown",
                }
            },
            "style_bible": {"visual_style": "cinematic photorealism", "aspect_ratio": "16:9"},
        }
        scenes = [
            {
                "scene_id": 1,
                "characters_present": ["primary"],
                "location_id": "song_street",
                "props_present": ["old_tree"],
                "current_action": "the girl looks at the tree",
                "state_before": {"old_tree": "intact"},
                "state_after": {"old_tree": "intact"},
            },
            {
                "scene_id": 2,
                "characters_present": ["primary"],
                "location_id": "song_street",
                "props_present": ["old_tree"],
                "current_action": "lightning splits the tree",
                "state_after": {"old_tree": "trunk split and fallen toward the left side of the street"},
            },
        ]
        first, first_anchors = _locked_continuity_prompt(
            "wide establishing shot",
            outline,
            scenes,
            0,
            "16:9",
            "English",
        )
        second, second_anchors = _locked_continuity_prompt(
            "low-angle action shot",
            outline,
            scenes,
            1,
            "16:9",
            "English",
        )
        self.assertEqual(first_anchors["location_anchor"], second_anchors["location_anchor"])
        self.assertEqual(first_anchors["prop_anchor"], second_anchors["prop_anchor"])
        self.assertIn("LOCATION CONTINUITY LOCK", first)
        self.assertIn("PROP CONTINUITY LOCK", second)
        self.assertIn("old_tree: intact", second)
        self.assertIn("trunk split and fallen", second)

        request = _build_continuity_scene_request(outline, scenes, 1, "Generate one frame.")
        self.assertNotIn('"next_scene"', request)
        self.assertIn('"old_tree": "intact"', request)
        self.assertIn('"location_id": "song_street"', request)

    def test_continuity_outline_reuses_location_id_for_same_location_label(self):
        outline = {"character_bible": {"identity": "girl"}, "style_bible": {}}
        scenes = [
            {"location": "Song dynasty market street", "action": "walks into the street"},
            {"location": "Song dynasty market street", "action": "turns toward a tea stall"},
        ]
        normalized = _normalize_continuity_outline(
            outline,
            scenes,
            "A girl walks along a market street.",
            2,
            "English",
            "16:9",
        )
        self.assertEqual(normalized["scenes"][0]["location_id"], normalized["scenes"][1]["location_id"])
        self.assertEqual(normalized["generation_settings"]["mode"], "online_continuity_v84")
        self.assertEqual(len(normalized["location_bible"]), 1)

    def test_online_continuity_node_hard_prefixes_api_scene_responses(self):
        outline = {
            "title": "Locked street",
            "source_story": "A girl crosses one street and stops beside the same tree.",
            "character_bible": {"character_id": "primary", "identity": "girl", "clothing": "blue coat"},
            "supporting_characters": [],
            "location_bible": {
                "street": {
                    "location_id": "street",
                    "architecture": "dark timber shops",
                    "fixed_objects": "one red banner",
                }
            },
            "prop_bible": {"tree": {"prop_id": "tree", "design": "forked old tree"}},
            "style_bible": {"visual_style": "cinematic realism", "aspect_ratio": "16:9"},
            "generation_settings": {
                "mode": "online_continuity_v84",
                "scene_count": 2,
                "prompt_language": "English",
                "aspect_ratio": "16:9",
            },
            "scenes": [
                {
                    "scene_id": 1,
                    "characters_present": ["primary"],
                    "location_id": "street",
                    "props_present": ["tree"],
                    "current_action": "walks past the tree",
                    "state_before": {"tree": "intact"},
                    "state_after": {"tree": "intact"},
                },
                {
                    "scene_id": 2,
                    "characters_present": ["primary"],
                    "location_id": "street",
                    "props_present": ["tree"],
                    "current_action": "stops beside the tree",
                    "state_before": {"tree": "intact"},
                    "state_after": {"tree": "intact"},
                },
            ],
        }

        class FakeCompletions:
            def create(self, **kwargs):
                content = json.dumps(
                    {
                        "scene_id": 1,
                        "prompt": "a cinematic shot",
                        "negative_prompt": "low quality",
                    }
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch("node.OpenAI", FakeOpenAI):
            positive, negative, storyboard_json, count = (
                RH_MultiSceneContinuityLLM_Node().generate_scene_prompts(
                    json.dumps(outline),
                    "http://example.test/v1",
                    "test-key",
                    "test-model",
                    "",
                    "Generate one frame.",
                    0.1,
                    2,
                )
            )

        self.assertEqual(count, 2)
        self.assertTrue(all("LOCATION CONTINUITY LOCK" in prompt for prompt in positive))
        self.assertTrue(all("dark timber shops" in prompt for prompt in positive))
        self.assertTrue(all("forked old tree" in prompt for prompt in positive))
        self.assertTrue(all("changed room layout" in prompt for prompt in negative))
        storyboard = json.loads(storyboard_json)
        self.assertEqual(storyboard["shots"][0]["raw_prompt"], "a cinematic shot")
        self.assertEqual(
            storyboard["shots"][0]["location_anchor"],
            storyboard["shots"][1]["location_anchor"],
        )

    def test_bundled_v84_workflow_uses_continuity_director(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "RH_Krea2_OnlineAPI_Klein_10eros_i2v_v84_continuity_lock.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        director = next(node for node in workflow["nodes"] if node["id"] == 2)
        self.assertEqual(director["type"], "RH_CONFIGURABLE_STORYBOARD_CONTINUITY")
        self.assertEqual(director["properties"]["Node name for S&R"], director["type"])
        self.assertEqual(workflow["extra"]["rh_workflow_version"], "krea2-online-v8.4-continuity-lock-klein-10eros-i2v")
        continuity = workflow["extra"]["rh_character_continuity"]
        self.assertIn("location_id", continuity["location_lock"])
        self.assertIn("prop_id", continuity["prop_lock"])
        self.assertIn("state_before", continuity["state_ledger"])

    def test_numbered_scene_save_prefix(self):
        self.assertEqual(
            _scene_save_prefix("RH_Krea2_Offline_Storyboard", 3),
            "RH_Krea2_Offline_Storyboard_Scene_03",
        )

    def test_numbered_video_prefix_list(self):
        prefixes = RH_StoryboardScenePrefixes_Node().build_prefixes(
            3,
            "RH_Krea2_Offline_Storyboard_Video",
            2,
        )[0]
        self.assertEqual(
            prefixes,
            [
                "RH_Krea2_Offline_Storyboard_Video_Scene_02",
                "RH_Krea2_Offline_Storyboard_Video_Scene_03",
                "RH_Krea2_Offline_Storyboard_Video_Scene_04",
            ],
        )

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
        request_node = next(node for node in workflow["nodes"] if node["type"] == "RH_OFFLINE_STORYBOARD_REQUEST")
        scene_request_node = next(
            node for node in workflow["nodes"] if node["type"] == "RH_OFFLINE_STORYBOARD_SCENE_REQUESTS"
        )
        self.assertEqual(request_node["outputs"][6]["name"], "source_story")
        self.assertEqual(scene_request_node["inputs"][1]["name"], "source_story")
        self.assertIsNotNone(scene_request_node["inputs"][1]["link"])

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
        self.assertTrue(any(node.get("type") == "RH_STORYBOARD_SCENE_SAVE" for node in workflow["nodes"]))

    def test_optional_i2v_workflow_uses_resolution_master_and_has_no_temp_outputs(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "RH_Krea2_Offline_Qwen3VL_KleinSwap_10ErosI2V_batch_v8.1_resolution_master_video.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        i2v = workflow["extra"]["rh_i2v"]
        subgraph = next(
            item
            for item in workflow["definitions"]["subgraphs"]
            if item["id"] == i2v["subgraph_id"]
        )
        root_nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(root_nodes[107]["mode"], 4)
        self.assertEqual(root_nodes[108]["mode"], 4)
        self.assertEqual(root_nodes[109]["type"], "Fast Groups Bypasser (rgthree)")
        self.assertEqual(root_nodes[109]["properties"]["matchTitle"], i2v["group_title"])
        self.assertFalse(any(node.get("type") == "PreviewImage" for node in subgraph["nodes"]))
        self.assertFalse(any(node.get("type") in {"SetNode", "GetNode"} for node in subgraph["nodes"]))
        self.assertFalse(any(node.get("id") == 549 for node in subgraph["nodes"]))
        self.assertFalse(any(node.get("id") in {791, 792} for node in subgraph["nodes"]))
        self.assertEqual(
            [item["type"] for item in subgraph["inputs"]],
            ["IMAGE", "STRING", "INT", "INT", "STRING"],
        )
        root_links = {link[0]: link for link in workflow["links"]}
        self.assertEqual(root_links[30158][1:6], [38, 0, 107, 2, "INT"])
        self.assertEqual(root_links[30159][1:6], [38, 1, 107, 3, "INT"])
        final_video = next(node for node in subgraph["nodes"] if node.get("id") == 597)
        self.assertTrue(final_video["widgets_values"]["save_output"])
        final_upscale = next(node for node in subgraph["nodes"] if node.get("id") == 755)
        self.assertEqual(final_upscale["widgets_values"][1], 1.0)
        self.assertTrue(any(node.get("type") == "LTX2STGGuider" for node in subgraph["nodes"]))


if __name__ == "__main__":
    unittest.main()

