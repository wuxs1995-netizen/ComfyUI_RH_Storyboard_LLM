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
    RH_MiniMaxH3ModelSelector_Node,
    RH_MiniMaxH3StoryboardGuide_Node,
    RH_MiniMaxH3Settings_Node,
    RH_OfflineStoryboardContinuityParser_Node,
    RH_OfflineStoryboardContinuityRequest_Node,
    RH_OfflineStoryboardContinuitySceneRequests_Node,
    RH_OfflineStoryboardParser_Node,
    RH_OfflineStoryboardRequest_Node,
    RH_OfflineStoryboardSceneRequests_Node,
    RH_REF2VStoryboardPrompt_Node,
    RH_StoryboardScenePrefixes_Node,
    RH_StoryboardImageCollector_Node,
    RH_StoryboardPromptSource_Node,
    _build_continuity_scene_request,
    _extract_api_image_bytes,
    _locked_continuity_prompt,
    _normalize_continuity_outline,
    _scene_save_prefix,
    build_ref2v_prompt_fields,
    combine_storyboard_composition_prompt,
    parse_offline_storyboard_text,
    parse_manual_storyboard_prompts,
    replace_storyboard_character_global,
    separate_storyboard_global_prompt,
)


class FakeImageBatch:
    def __init__(self, indices, ndim=4):
        self.indices = list(indices)
        self.ndim = ndim
        self.shape = (
            (len(self.indices), 64, 64, 3)
            if ndim == 4
            else (64, 64, 3)
        )

    def unsqueeze(self, axis):
        if axis != 0 or self.ndim != 3:
            raise AssertionError("Unexpected fake tensor unsqueeze")
        return FakeImageBatch(self.indices, ndim=4)

    def __getitem__(self, item):
        return FakeImageBatch(self.indices[item], ndim=4)


class OfflineStoryboardParserTests(unittest.TestCase):
    def test_manual_prompt_parser_accepts_json_list(self):
        prompts = parse_manual_storyboard_prompts(
            '["SCENE 01/02, first shot", "SCENE 02/02, second shot"]'
        )
        self.assertEqual(
            prompts,
            ["SCENE 01/02, first shot", "SCENE 02/02, second shot"],
        )

    def test_manual_prompt_parser_accepts_lines_and_multiline_blocks(self):
        self.assertEqual(
            parse_manual_storyboard_prompts("1. first shot\n2、second shot"),
            ["first shot", "second shot"],
        )
        self.assertEqual(
            parse_manual_storyboard_prompts("first line\ncontinued detail\n---\nsecond shot"),
            ["first line\ncontinued detail", "second shot"],
        )

    def test_manual_prompt_source_skips_automatic_branch_in_manual_mode(self):
        source = RH_StoryboardPromptSource_Node()
        automatic_spec = source.INPUT_TYPES()["optional"]["automatic_prompts"]
        framing_spec = source.INPUT_TYPES()["required"]["framing_priority"]
        self.assertTrue(automatic_spec[1]["lazy"])
        self.assertTrue(automatic_spec[1]["forceInput"])
        self.assertEqual(framing_spec[1]["default"], "构图优先（推荐）")
        self.assertEqual(
            source.check_lazy_status(
                ["手动粘贴"], ["edited prompt"], [""], ["自动提取"], (None,)
            ),
            [],
        )
        self.assertEqual(
            source.check_lazy_status(
                ["自动 LLM"], [""], [""], ["自动提取"], (None,)
            ),
            ["automatic_prompts"],
        )
        self.assertEqual(
            source.check_lazy_status(
                ["自动 LLM"],
                [""],
                [""],
                ["自动提取"],
                ["generated prompt"],
            ),
            [],
        )
        generation, scenes, global_prompt, count, active = source.select_prompts(
            ["手动粘贴"],
            ["edited scene 1\nedited scene 2"],
            ["GLOBAL LOCK"],
            ["自动提取"],
        )
        self.assertEqual(scenes, ["edited scene 1", "edited scene 2"])
        self.assertTrue(generation[0].startswith("edited scene 1"))
        self.assertTrue(generation[1].startswith("edited scene 2"))
        self.assertTrue(all(prompt.endswith("GLOBAL LOCK") for prompt in generation))
        self.assertEqual(global_prompt, "GLOBAL LOCK")
        self.assertEqual((count, active), (2, "手动粘贴"))

    def test_manual_prompt_source_passes_through_automatic_list(self):
        generation, scenes, global_prompt, count, active = RH_StoryboardPromptSource_Node().select_prompts(
            ["自动 LLM"],
            [""],
            [""],
            ["自动提取"],
            ["auto scene 1", "auto scene 2"],
        )
        self.assertEqual(generation, ["auto scene 1", "auto scene 2"])
        self.assertEqual(scenes, generation)
        self.assertEqual(global_prompt, "")
        self.assertEqual((count, active), (2, "自动 LLM"))

    def test_prompt_source_separates_repeated_global_lock_from_scene_text(self):
        prefix = (
            '1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）：primary: '
            '{"character_id":"primary","identity":"18岁中国少女"}, '
            '视觉风格锁定：visual_style: 偷拍感; aspect_ratio: 1:1'
        )
        automatic = [
            f"{prefix}, 当前镜头：分镜 01/02，女孩站在窗边",
            f"{prefix}, 当前镜头：分镜 02/02，女孩回头看向镜头",
        ]
        global_prompt, scenes = separate_storyboard_global_prompt(automatic)
        self.assertEqual(global_prompt, prefix)
        self.assertEqual(
            scenes,
            ["分镜 01/02，女孩站在窗边", "分镜 02/02，女孩回头看向镜头"],
        )
        generation, selected, detected_global, count, _ = (
            RH_StoryboardPromptSource_Node().select_prompts(
                ["自动 LLM"], [""], [""], ["自动提取"], automatic
            )
        )
        self.assertTrue(generation[0].startswith(scenes[0]))
        self.assertTrue(generation[1].startswith(scenes[1]))
        self.assertIn("人物身份连续性锁定", generation[0])
        self.assertLess(generation[0].index(scenes[0]), generation[0].index("人物身份连续性锁定"))
        self.assertEqual(selected, scenes)
        self.assertEqual(detected_global, prefix)
        self.assertEqual(count, 2)

    def test_automatic_mode_can_override_character_but_keep_style_and_ratio(self):
        automatic_global = (
            '1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）：primary: '
            '{"identity":"旧人物"}, 视觉风格锁定：visual_style: 偷拍感; aspect_ratio: 1:1'
        )
        replaced = replace_storyboard_character_global(
            automatic_global,
            "18岁中国少女，齐肩短发，浅棕色皮肤，娇小丰满体型",
        )
        self.assertIn("18岁中国少女", replaced)
        self.assertNotIn("旧人物", replaced)
        self.assertIn("visual_style: 偷拍感", replaced)
        self.assertTrue(replaced.startswith("1:1,"))

        automatic = [f"{automatic_global}, 当前镜头：分镜 01/01，女孩看向窗外"]
        generation, scenes, global_prompt, count, _ = (
            RH_StoryboardPromptSource_Node().select_prompts(
                ["自动 LLM"],
                [""],
                ["18岁中国少女，齐肩短发"],
                ["手动人物覆盖（保留风格/画幅）"],
                automatic,
            )
        )
        self.assertIn("18岁中国少女", global_prompt)
        self.assertIn("视觉风格锁定", global_prompt)
        self.assertNotIn("旧人物", generation[0])
        self.assertEqual(scenes, ["分镜 01/01，女孩看向窗外"])
        self.assertEqual(count, 1)

    def test_automatic_mode_can_replace_or_append_complete_global_prompt(self):
        automatic = [
            "1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）：old, "
            "视觉风格锁定：realism, 当前镜头：分镜 01/01，女孩站立"
        ]
        generation, _, global_prompt, _, _ = RH_StoryboardPromptSource_Node().select_prompts(
            ["自动 LLM"],
            [""],
            ["CUSTOM GLOBAL"],
            ["手动覆盖全部 Global"],
            automatic,
        )
        self.assertEqual(global_prompt, "CUSTOM GLOBAL")
        self.assertTrue(generation[0].startswith("分镜 01/01，女孩站立"))
        self.assertTrue(generation[0].endswith("CUSTOM GLOBAL"))

        generation, _, global_prompt, _, _ = RH_StoryboardPromptSource_Node().select_prompts(
            ["自动 LLM"],
            [""],
            ["extra continuity rule"],
            ["追加到自动 Global"],
            automatic,
        )
        self.assertIn("人物连续性锁定", global_prompt)
        self.assertIn("extra continuity rule", global_prompt)
        self.assertIn("extra continuity rule", generation[0])

    def test_composition_priority_removes_face_details_from_wide_or_back_shots(self):
        global_prompt = (
            '1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）：primary: '
            '{"identity":"成年中国女性","age":"25","facial_features":"固定鹅蛋脸和杏眼",'
            '"hairstyle":"黑色齐肩短发","clothing":"蓝色夹克"}, '
            '视觉风格锁定：visual_style: 写实; aspect_ratio: 1:1'
        )
        scene = "SCENE 01/02，全身远景，人物背对镜头奔跑穿过车站"
        prompt = combine_storyboard_composition_prompt(
            global_prompt,
            scene,
            "构图优先（推荐）",
        )
        self.assertTrue(prompt.startswith(scene))
        self.assertNotIn("facial_features", prompt)
        self.assertIn("hairstyle", prompt)
        self.assertIn("不得为了展示人脸", prompt)
        self.assertGreater(prompt.index("人物身份连续性锁定"), prompt.index(scene))

    def test_composition_priority_keeps_face_details_for_explicit_face_closeup(self):
        global_prompt = (
            '1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）：primary: '
            '{"identity":"成年中国女性","age":"25","facial_features":"固定鹅蛋脸和杏眼",'
            '"hairstyle":"黑色齐肩短发"}, 视觉风格锁定：realism'
        )
        scene = "SCENE 02/02，脸部特写，正脸清晰可见，人物轻轻微笑"
        prompt = combine_storyboard_composition_prompt(
            global_prompt,
            scene,
            "构图优先（推荐）",
        )
        self.assertTrue(prompt.startswith(scene))
        self.assertIn('"facial_features":"固定鹅蛋脸和杏眼"', prompt)
        self.assertIn("不得改成居中证件照式构图", prompt)

    def test_balance_only_removes_face_details_when_scene_deemphasizes_face(self):
        global_prompt = (
            '1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）：primary: '
            '{"identity":"adult woman","facial_features":"fixed face","hairstyle":"bob"}, '
            '视觉风格锁定：realism'
        )
        medium_prompt = combine_storyboard_composition_prompt(
            global_prompt,
            "中景，人物坐在窗边阅读",
            "平衡",
        )
        wide_prompt = combine_storyboard_composition_prompt(
            global_prompt,
            "全身远景，人物背对镜头走入树林",
            "平衡",
        )
        self.assertIn("facial_features", medium_prompt)
        self.assertNotIn("facial_features", wide_prompt)

    def test_global_split_keeps_scene_specific_supporting_character_lock(self):
        primary = 'primary: {"character_id":"primary","identity":"girl"}'
        courier = 'courier: {"character_id":"courier","identity":"man"}'
        prompts = [
            (
                "1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）："
                f"{primary}, 视觉风格锁定：visual_style: realism, "
                "当前镜头：分镜 01/02，女孩独处"
            ),
            (
                "1:1, 人物连续性锁定（所有镜头必须完全一致，不得改动）："
                f"{primary}; {courier}, 视觉风格锁定：visual_style: realism, "
                "当前镜头：分镜 02/02，信使出现"
            ),
        ]
        global_prompt, scenes = separate_storyboard_global_prompt(prompts)
        self.assertIn(primary, global_prompt)
        self.assertNotIn(courier, global_prompt)
        self.assertNotIn("额外人物锁定", scenes[0])
        self.assertIn(f"本镜头额外人物锁定：{courier}", scenes[1])

    def test_minimax_model_selector_uses_guide_mode_and_lazy_model(self):
        selector = RH_MiniMaxH3ModelSelector_Node()
        self.assertEqual(
            selector.check_lazy_status({"mode": "REF2VA"}),
            ["ref2va_model"],
        )
        self.assertEqual(
            selector.check_lazy_status({"mode": "FL2VA"}),
            ["fl2va_model"],
        )
        self.assertEqual(
            selector.select_model(
                {"mode": "REF2VA"},
                fl2va_model="fl-model",
                ref2va_model="ref-model",
            ),
            ("ref-model", False, True),
        )
        self.assertEqual(
            selector.select_model(
                {"mode": "I2VA"},
                fl2va_model="fl-model",
                ref2va_model="ref-model",
            ),
            ("fl-model", True, False),
        )

    def test_minimax_settings_exposes_linkable_validated_values(self):
        result = RH_MiniMaxH3Settings_Node().values(
            "ref2va", 1024, 576, 18, "max", 99
        )
        self.assertEqual(result, ("REF2VA", 1024, 576, 18, "max", 9))
        self.assertEqual(
            RH_MiniMaxH3Settings_Node.RETURN_NAMES,
            (
                "mode",
                "width",
                "height",
                "duration",
                "ref_image_size",
                "max_reference_images",
            ),
        )
        self.assertEqual(
            RH_MiniMaxH3Settings_Node.RETURN_TYPES[0],
            ("T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"),
        )
        self.assertEqual(
            RH_MiniMaxH3Settings_Node.RETURN_TYPES[4],
            ("match", "max"),
        )

    def test_ref2v_storyboard_input_is_linkable_and_widget_serializable(self):
        storyboard_spec = RH_REF2VStoryboardPrompt_Node.INPUT_TYPES()["required"][
            "storyboard_texts"
        ]
        self.assertEqual(storyboard_spec[0], "STRING")
        self.assertTrue(storyboard_spec[1]["defaultInput"])
        self.assertEqual(storyboard_spec[1]["default"], "")
        self.assertTrue(storyboard_spec[1]["multiline"])

    def _build_minimax_guide(self, mode, image_count=4, max_refs=9):
        return RH_MiniMaxH3StoryboardGuide_Node().build_guide(
            [FakeImageBatch(range(image_count))],
            [mode],
            ["subject_definitions:\n\nsummary:\nStoryboard video"],
            ["[Shot 1] opening frame\n[Shot 4] ending frame"],
            ["footsteps and room tone"],
            ["N/A"],
            [1024],
            [576],
            [8],
            ["match"],
            [max_refs],
        )

    def test_minimax_storyboard_guide_maps_endpoint_modes(self):
        i2v, count, summary, *_ = self._build_minimax_guide("I2VA")
        self.assertEqual(count, 1)
        self.assertEqual(i2v["first_frame"].indices, [0])
        self.assertIsNone(i2v["last_frame"])
        self.assertIn("0.00 seconds", i2v["resolved_prompt"])
        self.assertIn("first_frame <- Scene 01", summary)

        l2v, count, summary, *_ = self._build_minimax_guide("L2VA")
        self.assertEqual(count, 1)
        self.assertIsNone(l2v["first_frame"])
        self.assertEqual(l2v["last_frame"].indices, [3])
        self.assertIn("Shot 4", l2v["resolved_prompt"])
        self.assertIn("last_frame <- Scene 04", summary)

        fl2v, count, summary, *_ = self._build_minimax_guide("FL2VA")
        self.assertEqual(count, 2)
        self.assertEqual(fl2v["first_frame"].indices, [0])
        self.assertEqual(fl2v["last_frame"].indices, [3])
        self.assertIn("Picture 2 (from Shot 4)", fl2v["resolved_prompt"])
        self.assertIn("last_frame <- Scene 04", summary)

    def test_minimax_storyboard_guide_t2va_ignores_images(self):
        guide, count, summary, *_ = self._build_minimax_guide("T2VA")
        self.assertEqual(count, 0)
        self.assertIsNone(guide["first_frame"])
        self.assertIsNone(guide["last_frame"])
        self.assertEqual(guide["ref_images"], {})
        self.assertIn("ignored 4 storyboard image", summary)
        self.assertIn("integrated_multimodal_description", guide["resolved_prompt"])

    def test_minimax_storyboard_guide_ref2va_evenly_samples_nine_images(self):
        guide, count, summary, *_ = self._build_minimax_guide("REF2VA", image_count=12)
        self.assertEqual(count, 9)
        self.assertEqual(len(guide["ref_images"]), 9)
        self.assertEqual(guide["ref_images"]["ref_image_1"].indices, [0])
        self.assertEqual(guide["ref_images"]["ref_image_9"].indices, [11])
        self.assertIn("<Picture 9>", guide["resolved_prompt"])
        self.assertIn("[Shot 12]", guide["resolved_prompt"])
        self.assertIn("Picture 9 <- Scene 12", summary)

    def test_minimax_storyboard_guide_requires_images_outside_t2va(self):
        with self.assertRaisesRegex(ValueError, "I2VA requires at least one storyboard image"):
            self._build_minimax_guide("I2VA", image_count=0)

    def test_storyboard_image_collector_flattens_list_mapped_batches(self):
        images, count = RH_StoryboardImageCollector_Node().collect(
            [FakeImageBatch([0, 1]), FakeImageBatch([2, 3])]
        )
        self.assertEqual(count, 4)
        self.assertEqual([image.indices for image in images], [[0], [1], [2], [3]])

    def test_ref2v_compiles_prompt_list_into_six_sections(self):
        result = build_ref2v_prompt_fields(
            ["wide shot of a woman entering the station", "close-up as she opens a letter"],
            seconds_per_shot=4.5,
        )
        full_prompt, subjects, summary, retention, details, soundscape, music, count = result
        self.assertEqual(count, 2)
        self.assertIn("<Picture 1>", subjects)
        self.assertIn("<Picture 2>", subjects)
        self.assertIn("[Shot 1] At 00:00.000", details)
        self.assertIn("[Shot 2] At 00:04.500", details)
        self.assertIn("subject_definitions:", full_prompt)
        self.assertIn("retention_analysis:", full_prompt)
        self.assertTrue(soundscape)
        self.assertEqual(music, "N/A")

    def test_ref2v_uses_raw_prompts_and_character_bible_from_storyboard_json(self):
        storyboard = json.dumps(
            {
                "character_bible": {
                    "character_id": "primary",
                    "identity": "Chinese woman",
                    "hairstyle": "short dark hair",
                    "clothing": "red coat",
                },
                "shots": [
                    {"raw_prompt": "walks through a rainy street", "prompt": "LOCKED PREFIX, rainy street"},
                    {"raw_prompt": "stops under a neon sign", "prompt": "LOCKED PREFIX, neon sign"},
                ],
            },
            ensure_ascii=False,
        )
        result = RH_REF2VStoryboardPrompt_Node().build_prompt(
            [storyboard],
            [5.0],
            [0.0],
            ["one_picture_per_shot"],
            ["city rain and footsteps"],
            ["N/A"],
            [""],
        )
        self.assertEqual(result[-1], 2)
        self.assertIn("short dark hair", result[1])
        self.assertIn("red coat", result[1])
        self.assertIn("walks through a rainy street", result[4])
        self.assertNotIn("LOCKED PREFIX", result[4])
        self.assertIn("[Shot 2] At 00:05.000", result[4])

    def test_ref2v_splits_bracketed_multishot_text(self):
        text = "[Shot 1] opening wide shot\n[Shot 2] medium shot\n[Shot 3] final close-up"
        result = build_ref2v_prompt_fields(text, reference_mode="text_only")
        self.assertEqual(result[-1], 3)
        self.assertNotIn("<Picture 1>", result[1])
        self.assertIn("[Shot 3] At 00:09.000", result[4])

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

    def test_offline_continuity_request_requires_shared_visual_asset_bibles(self):
        request, count, language, ratio, width, height, story = (
            RH_OfflineStoryboardContinuityRequest_Node().build_request(
                "A girl walks along one street and lightning splits the same old tree.",
                2,
                "English",
                "16:9",
                "Keep the story cinematic.",
            )
        )
        self.assertEqual((count, language, ratio, width, height), (2, "English", "16:9", 1024, 576))
        self.assertIn('"location_bible"', request)
        self.assertIn('"prop_bible"', request)
        self.assertIn('"state_before"', request)
        self.assertIn("planning pass", request)
        self.assertEqual(story, "A girl walks along one street and lightning splits the same old tree.")

    def test_offline_continuity_pipeline_hard_locks_location_prop_and_state(self):
        outline = json.dumps(
            {
                "title": "Old tree",
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
                        "architecture": "dark timber shops with grey tiled roofs",
                        "spatial_layout": "east-west stone street",
                        "fixed_objects": "one red wine banner at the west entrance",
                    }
                },
                "prop_bible": {
                    "old_tree": {
                        "prop_id": "old_tree",
                        "design": "thick forked trunk leaning left",
                        "material": "weathered bark",
                        "color": "dark brown",
                    }
                },
                "style_bible": {"visual_style": "cinematic realism"},
                "scenes": [
                    {
                        "scene_id": 1,
                        "story_fact": "The girl sees the old tree.",
                        "characters_present": ["primary"],
                        "location_id": "song_street",
                        "props_present": ["old_tree"],
                        "state_before": {"old_tree": "intact"},
                        "current_action": "the girl looks at the tree",
                        "state_after": {"old_tree": "intact"},
                        "must_not_show": "the tree breaking",
                    },
                    {
                        "scene_id": 2,
                        "story_fact": "Lightning splits the old tree.",
                        "characters_present": ["primary"],
                        "location_id": "song_street",
                        "props_present": ["old_tree"],
                        "current_action": "lightning splits the tree",
                        "state_after": {"old_tree": "split and fallen left"},
                        "must_not_show": "a restored intact tree after the strike",
                    },
                ],
            }
        )
        requests, normalized, count, language, ratio = (
            RH_OfflineStoryboardContinuitySceneRequests_Node().build_scene_requests(
                outline,
                "A girl sees an old tree and lightning splits it.",
                2,
                "English",
                "16:9",
                "Use concrete visual detail.",
            )
        )
        self.assertEqual((count, language, ratio), (2, "English", "16:9"))
        self.assertEqual(len(requests), 2)
        self.assertIn('"location_lock"', requests[0])
        self.assertIn('"prop_locks"', requests[1])
        self.assertIn('"old_tree": "intact"', requests[1])
        self.assertNotIn('"next_scene"', requests[1])
        self.assertEqual(json.loads(normalized)["generation_settings"]["mode"], "offline_qwen_continuity_v82")

        raw = [
            json.dumps({"scene_id": 1, "prompt": "wide shot of the girl beside a tree"}),
            json.dumps({"scene_id": 2, "prompt": "low-angle shot of lightning striking"}),
        ]
        positive, negative, storyboard_json, parsed_count = (
            RH_OfflineStoryboardContinuityParser_Node().parse_storyboard(
                raw,
                [2],
                ["English"],
                ["16:9"],
                ["low quality"],
                [normalized],
            )
        )
        self.assertEqual(parsed_count, 2)
        self.assertTrue(all("LOCATION CONTINUITY LOCK" in prompt for prompt in positive))
        self.assertTrue(all("dark timber shops" in prompt for prompt in positive))
        self.assertTrue(all("thick forked trunk leaning left" in prompt for prompt in positive))
        self.assertTrue(all("changed prop color" in item for item in negative))
        storyboard = json.loads(storyboard_json)
        self.assertEqual(storyboard["generation_settings"]["mode"], "offline_qwen_continuity_v82")
        self.assertEqual(
            storyboard["shots"][0]["location_anchor"],
            storyboard["shots"][1]["location_anchor"],
        )

    def test_bundled_offline_v82_replaces_only_continuity_planning_nodes(self):
        workflow_dir = Path(__file__).resolve().parents[1] / "workflows"
        base = json.loads(
            (workflow_dir / "RH_Krea2_Offline_Qwen3vl_Klein_10eros_i2v_v81.json").read_text(
                encoding="utf-8"
            )
        )
        workflow = json.loads(
            (workflow_dir / "RH_Krea2_Offline_Qwen3vl_Klein_10eros_i2v_v82_continuity_lock.json").read_text(
                encoding="utf-8"
            )
        )
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(nodes[99]["type"], "RH_OFFLINE_STORYBOARD_CONTINUITY_REQUEST")
        self.assertEqual(nodes[103]["type"], "RH_OFFLINE_STORYBOARD_CONTINUITY_SCENE_REQUESTS")
        self.assertEqual(nodes[100]["type"], "RH_OFFLINE_STORYBOARD_CONTINUITY_PARSER")
        self.assertEqual(base["links"], workflow["links"])
        self.assertEqual(base["definitions"], workflow["definitions"])
        self.assertEqual(
            workflow["extra"]["rh_workflow_version"],
            "krea2-offline-qwen3vl-v8.2-continuity-lock-klein-10eros-i2v",
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

    def test_manual_prompt_workflow_routes_parser_through_lazy_source(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "RH_Krea2_Manual_Prompt_Override.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        self.assertEqual(nodes[2704]["type"], "RH_STORYBOARD_PROMPT_SOURCE")
        self.assertEqual(nodes[2704]["widgets_values"][0], "自动 LLM")
        self.assertEqual(len(nodes[2704]["widgets_values"]), 5)
        self.assertEqual(nodes[2704]["widgets_values"][3], "自动提取")
        self.assertEqual(nodes[2704]["widgets_values"][4], "构图优先（推荐）")
        self.assertEqual(nodes[2705]["type"], "PreviewAny")
        self.assertEqual(links[30175][1:6], [100, 0, 2704, 0, "STRING"])
        self.assertEqual(links[30139][1:6], [2704, 0, 34, 1, "STRING"])
        self.assertEqual(links[30140][1:6], [2704, 1, 95, 0, "STRING"])
        self.assertEqual(links[30176][1:6], [2704, 2, 2705, 0, "STRING"])
        self.assertEqual(nodes[100]["outputs"][0]["links"], [30175])
        self.assertEqual(nodes[2704]["outputs"][0]["links"], [30139])
        self.assertEqual(nodes[2704]["outputs"][1]["links"], [30140])
        self.assertEqual(nodes[2704]["outputs"][2]["links"], [30176])
        self.assertNotIn(31, {group.get("id") for group in workflow.get("groups", [])})
        removed_klein_nodes = set(range(43, 79)) | {81, 96, 97, 98}
        self.assertTrue(removed_klein_nodes.isdisjoint(nodes))
        self.assertEqual(links[30129][1:6], [34, 0, 80, 0, "IMAGE"])
        self.assertEqual(nodes[80]["inputs"][0]["link"], 30129)
        self.assertIn(30129, nodes[34]["outputs"][0]["links"])

    def test_minimax_v91_workflow_links_storyboard_images_for_all_modes(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflows"
            / "RH_Krea2_Offline_Qwen3VL_Klein_MinimaxH3_StoryboardModes_v9.1.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(nodes[1392]["type"], "RH_STORYBOARD_IMAGE_COLLECTOR")
        self.assertEqual(nodes[1393]["type"], "RH_MINIMAX_H3_SETTINGS")
        self.assertEqual(nodes[1394]["type"], "RH_MINIMAX_H3_STORYBOARD_GUIDE")
        self.assertEqual(nodes[1393]["widgets_values"], ["FL2VA", 768, 768, 18, "match", 9])
        self.assertEqual(
            nodes[1393]["outputs"][0]["type"],
            ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"],
        )
        self.assertEqual(nodes[1393]["outputs"][4]["type"], ["match", "max"])
        self.assertEqual(len(nodes[1393]["outputs"]), 6)
        self.assertEqual(
            nodes[1394]["inputs"][1]["type"],
            ["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"],
        )
        self.assertEqual(nodes[1394]["inputs"][9]["type"], ["match", "max"])
        self.assertEqual(nodes[1394]["inputs"][0]["link"], 30175)
        self.assertEqual(
            [item["name"] for item in nodes[1389]["inputs"]],
            [
                "resolution_preset",
                "aspect_preset_when_not_image",
                "noise_seed",
                "filename_prefix",
                "guide",
            ],
        )
        self.assertEqual(nodes[1389]["inputs"][4]["link"], 30186)

        minimax = next(
            item
            for item in workflow["definitions"]["subgraphs"]
            if item["id"] == workflow["extra"]["rh_minimax_h3"]["subgraph_id"]
        )
        guide_links = [
            link
            for link in minimax["links"]
            if link["type"] == "MINIMAX_H3_DIRECTOR_GUIDE"
        ]
        self.assertEqual([link["id"] for link in guide_links], [6010, 6011])
        self.assertEqual({link["origin_id"] for link in guide_links}, {-10})
        self.assertEqual({link["target_id"] for link in guide_links}, {1512, 2704})
        selector = next(node for node in minimax["nodes"] if node["id"] == 2704)
        self.assertEqual(selector["type"], "RH_MINIMAX_H3_MODEL_SELECTOR")
        self.assertFalse(any(node.get("type") == "MiniMaxH3Director" for node in minimax["nodes"]))
        root_links = {link[0]: link for link in workflow["links"]}
        self.assertNotIn(30172, root_links)
        self.assertNotIn(30188, root_links)
        self.assertNotIn(30189, root_links)
        self.assertNotIn("10eros", workflow_path.read_text(encoding="utf-8").lower())

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

