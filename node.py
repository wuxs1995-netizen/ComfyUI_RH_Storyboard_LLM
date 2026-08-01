from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
import json
import os
import re
import time

import numpy as np
from openai import OpenAI
from PIL import Image


DEFAULT_SCENE_ROLE = """You are a professional cinematic storyboard prompt writer.
Turn exactly one scene into a production-ready image-generation prompt.
Preserve the shared character and visual style. Return JSON only."""

DEFAULT_DIRECTOR_ROLE = """你是一名专业影视分镜总导演。
根据用户故事和可选参考人物图，创建结构严格、镜头连续的短片分镜大纲。
你必须保持角色外貌、服装、身份、道具和环境状态的一致性，并且只返回合法 JSON。"""

DEFAULT_SCENE_INSTRUCTION = """Generate one storyboard frame for the current scene.

Requirements:
1. Preserve the character's face, age, hairstyle and clothing.
2. Describe only the current scene.
3. State shot size, composition, camera angle, lighting, environment and action.
4. Use concrete visual language rather than abstract literary language.
5. Write the positive and negative prompts in the requested prompt language.
6. Return valid JSON only in this shape:
{
  "scene_id": 1,
  "prompt": "...",
  "negative_prompt": "...",
  "camera": "...",
  "continuity_note": "..."
}"""

ASPECT_PRESETS = {
    "16:9": (1024, 576),
    "9:16": (576, 1024),
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (1152, 768),
    "2:3": (768, 1152),
    "21:9": (1344, 576),
}

DEFAULT_OFFLINE_DIRECTOR_INSTRUCTION = """You are a local cinematic storyboard director and Krea 2 prompt engineer.
Turn the supplied story into a chronological storyboard with exactly the requested number of scenes.
Keep character identity, face, age, hairstyle, clothing, props, environment, time, weather and visual style consistent.
Each scene must contain one clear visual action and a production-ready image prompt with shot size, camera angle, composition, lighting, environment, emotion and material detail.
Use the optional reference image only for character identity and appearance; never copy its background or pose.
Return valid JSON only. Do not use Markdown, explanations, comments or text outside the JSON object."""

OFFLINE_PROMPT_KEYS = ("prompt", "positive_prompt", "image_prompt", "visual_prompt")
OFFLINE_SCENE_DETAIL_KEYS = (
    "shot_type",
    "camera",
    "composition",
    "location",
    "environment",
    "time",
    "weather",
    "subject",
    "character",
    "action",
    "emotion",
    "lighting",
    "color_palette",
    "visual_style",
    "continuity",
    "continuity_note",
)


def encode_image_b64(ref_image):
    """Encode the first ComfyUI IMAGE tensor as an in-memory JPEG."""
    i = 255.0 * ref_image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_video_file_path(video):
    if hasattr(video, "_VideoFromFile__file"):
        path = getattr(video, "_VideoFromFile__file", None)
        if isinstance(path, str) and os.path.exists(path):
            return path

    if hasattr(video, "get_stream_source"):
        try:
            stream_source = video.get_stream_source()
            if isinstance(stream_source, str) and os.path.exists(stream_source):
                return stream_source
        except Exception:
            pass

    for attr in ("path", "file"):
        if hasattr(video, attr):
            path = getattr(video, attr, None)
            if isinstance(path, str) and os.path.exists(path):
                return path
    return None


def encode_video_b64(video):
    video_path = _get_video_file_path(video)
    if video_path:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if hasattr(video, "save_to"):
        temp_path = f"temp_video_{time.time()}.mp4"
        try:
            video.save_to(temp_path)
            with open(temp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    raise ValueError(f"Unable to read video data from object type: {type(video)}")


def _extract_json_text(value):
    """Accept raw JSON or a JSON object wrapped in a Markdown code fence."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("The outline is empty; expected a JSON object.")

    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object was found in the outline response.")
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"The director response is not valid JSON near line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("The outline JSON root must be an object.")
    return parsed


def _parse_outline(outline_json):
    outline = _extract_json_text(outline_json)
    scenes = outline.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("The outline JSON must contain a non-empty 'scenes' array.")

    normalized = []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"scenes[{index - 1}] must be a JSON object.")
        item = dict(scene)
        item.setdefault("scene_id", index)
        normalized.append(item)
    return outline, normalized


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_first_json_value(value):
    """Return the first JSON object/array from raw or Markdown-wrapped model text."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _offline_prompt_from_scene(scene):
    if isinstance(scene, str):
        return scene.strip(), "", "", ""
    if not isinstance(scene, dict):
        return "", "", "", ""

    prompt = ""
    for key in OFFLINE_PROMPT_KEYS:
        value = scene.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break

    if not prompt:
        details = []
        for key in OFFLINE_SCENE_DETAIL_KEYS:
            value = scene.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, str):
                rendered = value.strip()
            else:
                rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if rendered:
                details.append(rendered)
        prompt = ", ".join(details)

    negative = str(scene.get("negative_prompt", "") or "").strip()
    camera = str(scene.get("camera", "") or "").strip()
    continuity = str(scene.get("continuity_note", scene.get("continuity", "")) or "").strip()
    return prompt, negative, camera, continuity


def _offline_line_prompts(text):
    prompts = []
    for raw_line in str(text).replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("```") or line in {"{", "}", "[", "]"}:
            continue
        line = re.sub(
            r"^\s*(?:[-*•]\s*)?(?:(?:scene|shot|分镜|镜头)\s*)?\d+\s*[:：.)、-]\s*",
            "",
            line,
            flags=re.I,
        ).strip()
        line = re.sub(r'^\s*["\']?(?:prompt|positive_prompt)["\']?\s*[:：]\s*', "", line, flags=re.I)
        line = line.strip().rstrip(",").strip().strip('"').strip("'")
        if line:
            prompts.append(line)
    return prompts


def parse_offline_storyboard_text(generated_text, scene_count, default_negative_prompt=""):
    """Normalize Qwen JSON or numbered lines into ordered ComfyUI prompt lists."""
    parsed = _extract_first_json_value(generated_text)
    title = ""
    character_bible = {}
    style_bible = {}
    scene_values = None

    if isinstance(parsed, dict):
        title = str(parsed.get("title", "") or "")
        character_bible = parsed.get("character_bible", {})
        style_bible = parsed.get("style_bible", {})
        for key in ("scenes", "shots", "storyboard"):
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                scene_values = candidate
                break
    elif isinstance(parsed, list):
        scene_values = parsed

    shots = []
    if scene_values is not None:
        for index, scene in enumerate(scene_values, start=1):
            prompt, negative, camera, continuity = _offline_prompt_from_scene(scene)
            if not prompt:
                continue
            shots.append(
                {
                    "scene_id": index,
                    "prompt": prompt,
                    "negative_prompt": negative or default_negative_prompt,
                    "camera": camera,
                    "continuity_note": continuity,
                    "source_scene": scene,
                }
            )
    else:
        for index, prompt in enumerate(_offline_line_prompts(generated_text), start=1):
            shots.append(
                {
                    "scene_id": index,
                    "prompt": prompt,
                    "negative_prompt": default_negative_prompt,
                    "camera": "",
                    "continuity_note": "",
                }
            )

    if len(shots) < scene_count:
        raise ValueError(
            f"Local Qwen returned {len(shots)} usable storyboard prompts; expected {scene_count}. "
            "Increase TextGenerate max_length or lower the scene count, then run again."
        )

    shots = shots[:scene_count]
    for index, shot in enumerate(shots, start=1):
        shot["scene_id"] = index
    positive = [shot["prompt"] for shot in shots]
    negative = [shot["negative_prompt"] for shot in shots]
    storyboard = {
        "title": title,
        "character_bible": character_bible if isinstance(character_bible, dict) else {},
        "style_bible": style_bible if isinstance(style_bible, dict) else {},
        "shots": shots,
    }
    return positive, negative, storyboard


def _build_scene_request(outline, scenes, scene_index, instruction):
    scene = scenes[scene_index]
    previous_scene = scenes[scene_index - 1] if scene_index > 0 else None
    next_scene = scenes[scene_index + 1] if scene_index + 1 < len(scenes) else None
    payload = {
        "story_title": outline.get("title", ""),
        "character_bible": outline.get("character_bible", {}),
        "style_bible": outline.get("style_bible", {}),
        "generation_settings": outline.get("generation_settings", {}),
        "current_scene": scene,
        "previous_scene": previous_scene,
        "next_scene": next_scene,
    }
    return f"{instruction.strip()}\n\nSCENE PACKAGE:\n{_json_text(payload)}"


def _completion_text(completion):
    if completion is None or not getattr(completion, "choices", None):
        raise RuntimeError("The LLM API returned no choices.")
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("The LLM API returned an empty message.")
    return content


class RH_LLMAPI_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("describe",)
    FUNCTION = "rh_run_llmapi"
    CATEGORY = "Runninghub/LLM"

    def rh_run_llmapi(self, api_baseurl, api_key, model, role, prompt, temperature, seed, ref_image=None, video=None):
        client = OpenAI(api_key=api_key, base_url=api_baseurl)

        if video is not None:
            messages = [
                {"role": "system", "content": role},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{encode_video_b64(video)}"},
                        },
                    ],
                },
            ]
        elif ref_image is None:
            messages = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": role},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_b64(ref_image)}"},
                        },
                    ],
                },
            ]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
        )
        return (_completion_text(completion),)


class RH_SceneJSONSplitter_Node:
    """Extract one scene and its shared bibles for fixed-branch workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outline_json": ("STRING", {"forceInput": True}),
                "scene_number": ("INT", {"default": 1, "min": 1, "max": 999}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("scene_package", "scene_json", "character_bible", "style_bible", "scene_count")
    FUNCTION = "split_scene"
    CATEGORY = "Runninghub/Storyboard"

    def split_scene(self, outline_json, scene_number):
        outline, scenes = _parse_outline(outline_json)
        if scene_number > len(scenes):
            raise ValueError(f"Requested scene {scene_number}, but the outline contains {len(scenes)} scenes.")
        index = scene_number - 1
        package = _build_scene_request(outline, scenes, index, DEFAULT_SCENE_INSTRUCTION)
        return (
            package,
            _json_text(scenes[index]),
            _json_text(outline.get("character_bible", {})),
            _json_text(outline.get("style_bible", {})),
            len(scenes),
        )


class RH_StoryboardPromptSelector_Node:
    """Select one generated shot from RH Multi Scene LLM's storyboard JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True}),
                "scene_number": ("INT", {"default": 1, "min": 1, "max": 999}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "camera",
        "continuity_note",
        "shot_json",
        "scene_count",
        "active",
    )
    FUNCTION = "select_prompt"
    CATEGORY = "Runninghub/Storyboard"

    def select_prompt(self, storyboard_json, scene_number):
        storyboard = _extract_json_text(storyboard_json)
        shots = storyboard.get("shots")
        if not isinstance(shots, list) or not shots:
            raise ValueError("The storyboard JSON must contain a non-empty 'shots' array.")
        if scene_number > len(shots):
            return ("", "", "", "", "", len(shots), False)
        shot = shots[scene_number - 1]
        if not isinstance(shot, dict):
            raise ValueError(f"shots[{scene_number - 1}] must be a JSON object.")
        return (
            str(shot.get("prompt", "")),
            str(shot.get("negative_prompt", "")),
            str(shot.get("camera", "")),
            str(shot.get("continuity_note", "")),
            _json_text(shot),
            len(shots),
            True,
        )


class RH_OfflineStoryboardRequest_Node:
    """Build a strict local-Qwen request without contacting any API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "一个人物经历一次意外发现，并做出改变命运的选择。",
                    },
                ),
                "scene_count": ("INT", {"default": 8, "min": 1, "max": 12}),
                "prompt_language": (["中文", "English"], {"default": "中文"}),
                "aspect_ratio": (list(ASPECT_PRESETS.keys()), {"default": "16:9"}),
                "director_instruction": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_OFFLINE_DIRECTOR_INSTRUCTION},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "qwen_prompt",
        "scene_count",
        "prompt_language",
        "aspect_ratio",
        "width",
        "height",
    )
    FUNCTION = "build_request"
    CATEGORY = "Runninghub/Storyboard/Offline"

    def build_request(
        self,
        story,
        scene_count,
        prompt_language,
        aspect_ratio,
        director_instruction,
    ):
        language_rule = (
            "Every prompt value must be written entirely in Simplified Chinese."
            if prompt_language == "中文"
            else "Every prompt value must be written entirely in English."
        )
        request = f"""{director_instruction.strip()}

STORY:
{story.strip()}

SETTINGS:
- Exact scene count: {scene_count}
- Prompt language: {prompt_language}
- Aspect ratio: {aspect_ratio}

REQUIREMENTS:
1. The scenes array must contain exactly {scene_count} items, numbered from 1 to {scene_count}.
2. {language_rule}
3. Every scene.prompt must be a single line with no internal newline.
4. Every prompt must explicitly preserve the shared character identity and the aspect ratio {aspect_ratio}.
5. Return only this JSON shape:
{{
  "title": "",
  "character_bible": {{"appearance": "", "clothing": "", "identity": ""}},
  "style_bible": {{"visual_style": "", "color_palette": "", "aspect_ratio": "{aspect_ratio}"}},
  "scenes": [
    {{
      "scene_id": 1,
      "prompt": "",
      "negative_prompt": "blurry face, deformed anatomy, extra fingers, low quality",
      "camera": "",
      "continuity_note": ""
    }}
  ]
}}"""
        width, height = ASPECT_PRESETS[aspect_ratio]
        return request, scene_count, prompt_language, aspect_ratio, width, height


class RH_OfflineStoryboardParser_Node:
    """Convert local Qwen JSON/numbered text into ComfyUI prompt lists."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_text": ("STRING", {"forceInput": True}),
                "scene_count": ("INT", {"default": 8, "min": 1, "max": 12, "forceInput": True}),
                "prompt_language": (["中文", "English"], {"default": "中文", "forceInput": True}),
                "aspect_ratio": (list(ASPECT_PRESETS.keys()), {"default": "16:9", "forceInput": True}),
                "default_negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry face, deformed anatomy, extra fingers, low quality, watermark, text",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("positive_prompts", "negative_prompts", "storyboard_json", "scene_count")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "parse_storyboard"
    CATEGORY = "Runninghub/Storyboard/Offline"

    def parse_storyboard(
        self,
        generated_text,
        scene_count,
        prompt_language,
        aspect_ratio,
        default_negative_prompt,
    ):
        positive, negative, storyboard = parse_offline_storyboard_text(
            generated_text,
            scene_count,
            default_negative_prompt.strip(),
        )
        style_bible = storyboard.get("style_bible")
        if not isinstance(style_bible, dict):
            style_bible = {}
            storyboard["style_bible"] = style_bible
        style_bible["aspect_ratio"] = aspect_ratio
        storyboard["generation_settings"] = {
            "mode": "offline_qwen",
            "scene_count": scene_count,
            "prompt_language": prompt_language,
            "aspect_ratio": aspect_ratio,
        }
        return positive, negative, _json_text(storyboard), scene_count


class RH_MultiSceneLLM_Node:
    """Fan out one independent OpenAI-compatible request per scene."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outline_json": ("STRING", {"forceInput": True}),
                "api_baseurl": ("STRING", {"multiline": False, "default": "http://127.0.0.1:8000/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
                "role": ("STRING", {"multiline": True, "default": DEFAULT_SCENE_ROLE}),
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_SCENE_INSTRUCTION}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 16}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("positive_prompts", "negative_prompts", "storyboard_json", "scene_count")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "generate_scene_prompts"
    CATEGORY = "Runninghub/Storyboard"

    def generate_scene_prompts(
        self,
        outline_json,
        api_baseurl,
        api_key,
        model,
        role,
        instruction,
        temperature,
        max_workers,
        ref_image=None,
    ):
        outline, scenes = _parse_outline(outline_json)
        image_b64 = encode_image_b64(ref_image) if ref_image is not None else None

        def request_scene(index):
            scene = scenes[index]
            request_text = _build_scene_request(outline, scenes, index, instruction)
            if image_b64:
                user_content = [
                    {"type": "text", "text": request_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ]
            else:
                user_content = request_text

            client = OpenAI(api_key=api_key, base_url=api_baseurl)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
            )
            raw = _completion_text(completion)
            try:
                parsed = _extract_json_text(raw)
            except ValueError:
                parsed = {"scene_id": scene.get("scene_id", index + 1), "prompt": raw}

            parsed.setdefault("scene_id", scene.get("scene_id", index + 1))
            parsed.setdefault("prompt", raw)
            parsed.setdefault("negative_prompt", "")
            parsed["source_scene"] = scene
            return index, parsed

        ordered = [None] * len(scenes)
        worker_count = min(max_workers, len(scenes))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(request_scene, index) for index in range(len(scenes))]
            for future in as_completed(futures):
                index, result = future.result()
                ordered[index] = result

        positive = [str(item.get("prompt", "")) for item in ordered]
        negative = [str(item.get("negative_prompt", "")) for item in ordered]
        storyboard = {
            "title": outline.get("title", ""),
            "character_bible": outline.get("character_bible", {}),
            "style_bible": outline.get("style_bible", {}),
            "shots": ordered,
        }
        return positive, negative, _json_text(storyboard), len(scenes)


class RH_ConfigurableStoryboard_Node:
    """Generate a configurable outline and fan out one prompt request per scene."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": False, "default": "http://127.0.0.1:8000/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
                "story": ("STRING", {"multiline": True, "default": "一个人物经历一次意外发现，并做出改变命运的选择。"}),
                "scene_count": ("INT", {"default": 8, "min": 1, "max": 12}),
                "prompt_language": (["中文", "English"], {"default": "中文"}),
                "aspect_ratio": (list(ASPECT_PRESETS.keys()), {"default": "16:9"}),
                "director_role": ("STRING", {"multiline": True, "default": DEFAULT_DIRECTOR_ROLE}),
                "prompt_writer_role": ("STRING", {"multiline": True, "default": DEFAULT_SCENE_ROLE}),
                "outline_temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "prompt_temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 12}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "outline_json",
        "storyboard_json",
        "positive_prompts",
        "negative_prompts",
        "scene_count",
        "prompt_language",
        "aspect_ratio",
        "width",
        "height",
    )
    OUTPUT_IS_LIST = (False, False, True, True, False, False, False, False, False)
    FUNCTION = "generate_storyboard"
    CATEGORY = "Runninghub/Storyboard"

    def _director_request(self, story, scene_count, prompt_language, aspect_ratio):
        return f"""根据以下故事创建恰好 {scene_count} 个连续分镜。

故事：
{story}

最终图像提示词语言：{prompt_language}
所有镜头画面横宽比：{aspect_ratio}

要求：
1. scenes 数组必须恰好包含 {scene_count} 项，scene_id 从 1 连续编号。
2. 每个场景只包含一个明确动作，并记录地点、时间、景别、机位、动作、情绪和连续性。
3. character_bible 必须固定人物外貌、服装和身份。
4. style_bible 必须固定视觉风格、色彩和 aspect_ratio={aspect_ratio}。
5. generation_settings 必须原样记录 scene_count、prompt_language 和 aspect_ratio。
6. 只输出合法 JSON，不要 Markdown，不要解释。

输出结构：
{{
  "title": "故事标题",
  "character_bible": {{"appearance": "", "clothing": "", "identity": ""}},
  "style_bible": {{"visual_style": "", "color_palette": "", "aspect_ratio": "{aspect_ratio}"}},
  "generation_settings": {{"scene_count": {scene_count}, "prompt_language": "{prompt_language}", "aspect_ratio": "{aspect_ratio}"}},
  "scenes": [
    {{"scene_id": 1, "duration": 4, "location": "", "time": "", "shot_type": "", "camera": "", "action": "", "emotion": "", "dialogue": "", "continuity": ""}}
  ]
}}"""

    def _call_director(self, client, model, role, request_text, temperature, image_b64):
        if image_b64:
            user_content = [
                {"type": "text", "text": request_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        else:
            user_content = request_text
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        return _completion_text(completion)

    def generate_storyboard(
        self,
        api_baseurl,
        api_key,
        model,
        story,
        scene_count,
        prompt_language,
        aspect_ratio,
        director_role,
        prompt_writer_role,
        outline_temperature,
        prompt_temperature,
        max_workers,
        ref_image=None,
    ):
        client = OpenAI(api_key=api_key, base_url=api_baseurl)
        image_b64 = encode_image_b64(ref_image) if ref_image is not None else None
        request_text = self._director_request(story, scene_count, prompt_language, aspect_ratio)
        raw_outline = self._call_director(
            client, model, director_role, request_text, outline_temperature, image_b64
        )
        outline = _extract_json_text(raw_outline)
        scenes = outline.get("scenes")

        if not isinstance(scenes, list) or len(scenes) != scene_count:
            actual = len(scenes) if isinstance(scenes, list) else 0
            repair = f"""上一次 JSON 包含 {actual} 个场景，但必须是恰好 {scene_count} 个。
请保持故事、人物和风格不变，重新输出完整合法 JSON。scenes 必须恰好有 {scene_count} 项。
只输出 JSON。

上一次输出：
{raw_outline}"""
            raw_outline = self._call_director(
                client, model, director_role, repair, 0.1, image_b64
            )
            outline = _extract_json_text(raw_outline)
            scenes = outline.get("scenes")

        if not isinstance(scenes, list) or len(scenes) < scene_count:
            actual = len(scenes) if isinstance(scenes, list) else 0
            raise ValueError(f"Director returned {actual} scenes; expected exactly {scene_count}.")
        if len(scenes) > scene_count:
            scenes = scenes[:scene_count]
            outline["scenes"] = scenes

        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise ValueError(f"scenes[{index - 1}] must be a JSON object.")
            scene["scene_id"] = index

        style_bible = outline.get("style_bible")
        if not isinstance(style_bible, dict):
            style_bible = {}
            outline["style_bible"] = style_bible
        style_bible["aspect_ratio"] = aspect_ratio
        outline["generation_settings"] = {
            "scene_count": scene_count,
            "prompt_language": prompt_language,
            "aspect_ratio": aspect_ratio,
        }
        outline_json = _json_text(outline)

        language_instruction = (
            "Write every prompt entirely in Simplified Chinese."
            if prompt_language == "中文"
            else "Write every prompt entirely in English."
        )
        instruction = (
            DEFAULT_SCENE_INSTRUCTION
            + f"\n7. {language_instruction}"
            + f"\n8. Compose every frame for an aspect ratio of {aspect_ratio}."
        )
        positive, negative, storyboard_json, actual_count = RH_MultiSceneLLM_Node().generate_scene_prompts(
            outline_json,
            api_baseurl,
            api_key,
            model,
            prompt_writer_role,
            instruction,
            prompt_temperature,
            max_workers,
            ref_image,
        )
        width, height = ASPECT_PRESETS[aspect_ratio]
        return (
            outline_json,
            storyboard_json,
            positive,
            negative,
            actual_count,
            prompt_language,
            aspect_ratio,
            width,
            height,
        )
