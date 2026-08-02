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
Preserve the shared character identity, ethnicity, nationality, skin tone, height, body build, body proportions and visual style. Return JSON only."""

DEFAULT_DIRECTOR_ROLE = """你是一名专业影视分镜总导演。
根据用户故事和可选参考人物图，创建结构严格、镜头连续的短片分镜大纲。
你必须保持角色外貌、族裔、国籍、肤色、身高、身材体型、身体比例、服装、身份、道具和环境状态的一致性，并且只返回合法 JSON。"""

DEFAULT_SCENE_INSTRUCTION = """Generate one storyboard frame for the current scene.

Requirements:
1. Preserve the character's face, age, ethnicity, nationality, skin tone, height, body build, body proportions, hairstyle and clothing.
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

DEFAULT_OFFLINE_DIRECTOR_INSTRUCTION = """You are a local cinematic storyboard continuity director.
First create one canonical character bible from the story and optional reference image, then create a chronological scene outline.
The character's identity, face, age, ethnicity, nationality, skin tone, height, body build, body proportions, hairstyle, hair accessories, clothing and signature props are IMMUTABLE across every scene unless the story explicitly requires a change.
Nationality is a cultural/legal identity, not a facial feature: preserve it when stated by the story, and never infer or change it solely from appearance.
Location, action, emotion, shot size, camera angle, composition, lighting, time and weather are MUTABLE scene variables.
The source story is authoritative. Preserve every named or implied character, their relationships, the central action and the outcome. Do not replace, remove or invent major story events.
When a short story is expanded into many shots, subdivide the existing action into visual beats instead of adding unrelated actions or a new plot.
Use the optional reference image only for character identity and appearance; never copy its background or pose.
Return valid JSON only. Do not use Markdown, explanations, comments or text outside the JSON object."""

DEFAULT_OFFLINE_SCENE_INSTRUCTION = """You are a cinematic Krea 2 prompt engineer.
Generate exactly one production-ready image prompt for the supplied scene package.
Treat CHARACTER LOCK and STYLE LOCK as immutable source-of-truth data. Never shorten, reinterpret, replace or contradict them, including ethnicity, nationality, skin tone, height, body build and body proportions.
Treat SOURCE STORY and STORY FACT as authoritative. The shot must visualize that fact without changing the people, action, relationship or outcome.
Generate one still frame, not a montage or an action sequence. Depict exactly CURRENT ACTION and its resulting visible state.
Do not combine previous actions, future actions, transitions or the entire story into this prompt.
Describe only the current scene's mutable action, emotion, environment, camera, composition and lighting.
Return valid JSON only. Do not use Markdown or commentary."""

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


def _bible_anchor(bible):
    """Render a stable, reusable text anchor from a character/style bible."""
    if not isinstance(bible, dict):
        return ""
    values = []
    for key, value in bible.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            rendered = value.strip()
        else:
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if rendered:
            values.append(f"{key}: {rendered}")
    return "; ".join(values)


def _supporting_character_map(value):
    characters = value if isinstance(value, list) else []
    result = {}
    for index, character in enumerate(characters, start=1):
        if not isinstance(character, dict):
            continue
        character_id = str(
            character.get("character_id")
            or character.get("id")
            or character.get("name")
            or f"supporting_{index}"
        ).strip()
        if character_id:
            result[character_id] = character
    return result


def _scene_character_lock(primary_bible, supporting_characters, scene_outline):
    """Select only characters present in this shot while keeping descriptions canonical."""
    primary = primary_bible if isinstance(primary_bible, dict) else {}
    supporting = _supporting_character_map(supporting_characters)
    present = []
    if isinstance(scene_outline, dict):
        value = scene_outline.get("characters_present", [])
        if isinstance(value, str):
            present = [item.strip() for item in re.split(r"[,，;；]", value) if item.strip()]
        elif isinstance(value, list):
            present = [str(item).strip() for item in value if str(item).strip()]
    present_keys = {item.casefold() for item in present}

    selected = {}
    primary_id = str(primary.get("character_id") or "primary").strip()
    if primary and (not present_keys or primary_id.casefold() in present_keys or "primary" in present_keys):
        selected[primary_id] = primary
    for character_id, character in supporting.items():
        aliases = {
            character_id.casefold(),
            str(character.get("name", "")).strip().casefold(),
            str(character.get("role", "")).strip().casefold(),
        }
        aliases.discard("")
        if present_keys & aliases:
            selected[character_id] = character
    if not selected and primary:
        selected[primary_id] = primary
    return selected


def _locked_prompt(prompt, character_bible, style_bible, aspect_ratio, prompt_language):
    """Prefix every scene with the exact same immutable continuity anchors."""
    raw_prompt = str(prompt or "").strip()
    character_anchor = _bible_anchor(character_bible)
    style_anchor = _bible_anchor(style_bible)
    if not character_anchor:
        return raw_prompt, character_anchor, style_anchor

    if str(prompt_language).strip().lower() == "english":
        parts = [
            str(aspect_ratio).strip(),
            f"CHARACTER CONTINUITY LOCK (identical in every shot; do not alter): {character_anchor}",
        ]
        if style_anchor:
            parts.append(f"STYLE LOCK: {style_anchor}")
        parts.append(f"CURRENT SHOT: {raw_prompt}")
    else:
        parts = [
            str(aspect_ratio).strip(),
            f"人物连续性锁定（所有镜头必须完全一致，不得改动）：{character_anchor}",
        ]
        if style_anchor:
            parts.append(f"视觉风格锁定：{style_anchor}")
        parts.append(f"当前镜头：{raw_prompt}")
    return ", ".join(part for part in parts if part), character_anchor, style_anchor


def _scene_save_prefix(base_prefix, scene_number):
    base = str(base_prefix or "RH_Storyboard").strip().strip("/\\") or "RH_Storyboard"
    return f"{base}_Scene_{int(scene_number):02d}"


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


def _offline_scene_beat(scene_number, scene_count):
    if scene_count <= 1:
        return "Create the story's clearest defining visual moment with a complete beginning-to-end implication."
    progress = (scene_number - 1) / (scene_count - 1)
    if scene_number == 1:
        return "Establish the protagonist, location, time, weather and initial situation."
    if scene_number == scene_count:
        return "Show the final decision, consequence or resolution; make it visually distinct from the opening."
    if progress <= 0.25:
        return "Show the inciting discovery or first meaningful change; advance beyond the establishing shot."
    if progress <= 0.5:
        return "Develop the investigation or journey with a new action, location detail or piece of evidence."
    if progress <= 0.75:
        return "Escalate conflict, risk or emotion with a clearly different composition and action."
    return "Build toward the climax and force the protagonist toward the final choice."


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
        if scene_values is None and any(parsed.get(key) for key in OFFLINE_PROMPT_KEYS):
            scene_values = [parsed]
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
    """Build the first-pass character bible and storyboard-outline request."""

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

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "qwen_prompt",
        "scene_count",
        "prompt_language",
        "aspect_ratio",
        "width",
        "height",
        "source_story",
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
        scene_shape = ",\n    ".join(
            f'{{"scene_id": {scene_number}, "beat_role": "{_offline_scene_beat(scene_number, scene_count)}", '
            '"story_fact": "", "characters_present": ["primary"], "state_before": "", '
            '"current_action": "", "state_after": "", "must_not_show": "", '
            '"location": "", "time": "", "shot_type": "", "camera": "", '
            '"emotion": "", "lighting": "", "continuity": ""}'
            for scene_number in range(1, scene_count + 1)
        )
        request = f"""SOURCE STORY — THE ONLY AUTHORITATIVE SOURCE OF CHARACTERS, ACTIONS, RELATIONSHIPS AND OUTCOME:
{story.strip()}

DIRECTOR RULES — THESE MAY CONTROL STYLE, DETAIL AND FORMAT, BUT MUST NOT ADD OR REPLACE PLOT FACTS:
{director_instruction.strip()}

SETTINGS:
- Required scene count: {scene_count}
- Prompt language: {prompt_language}
- Aspect ratio: {aspect_ratio}

REQUIREMENTS:
1. Analyze the protagonist once and create exactly one canonical character_bible. Assign it character_id="primary".
2. {language_rule}
3. character_bible must explicitly lock identity, age, ethnicity, nationality, skin tone, height, body build, body proportions, facial features, hairstyle, hair accessories, clothing and signature props. These values must never change between scenes.
   Nationality must come from SOURCE STORY when stated; do not infer nationality from facial appearance alone. If it is not specified, use "unspecified".
4. Create one supporting_characters entry for every other recurring or action-relevant person in the source story. Never omit a person who performs or receives an action.
5. Each scene.story_fact must state which exact source-story fact that shot visualizes. characters_present must list the canonical character_id values visible in that shot.
6. Each scene.current_action must contain exactly one visible action or one held reaction. Never combine multiple temporal steps into one scene.
7. state_before and state_after define continuity; must_not_show must list actions reserved for other scenes. Later events must not appear early, and completed events must not be replayed unless the source story explicitly repeats them.
8. Scene entries may subdivide the source event, but must not add, remove, replace or reverse any person, action, relationship or outcome.
   Treat any plot-like content inside DIRECTOR RULES as non-authoritative when it is absent from SOURCE STORY.
9. scenes must form a progressive sequence: establishing state → approach/change → interaction/escalation → reaction/consequence. Do not place the climax or final state in every scene.
10. scenes must contain exactly {scene_count} items with scene_id values 1 through {scene_count}.
11. Do not write final image-generation prompts in this planning pass.
12. Return only this JSON shape:
{{
  "title": "",
  "character_bible": {{"character_id": "primary", "identity": "", "age": "", "ethnicity": "", "nationality": "", "skin_tone": "", "height": "", "body_build": "", "body_proportions": "", "facial_features": "", "hairstyle": "", "hair_accessories": "", "clothing": "", "signature_props": ""}},
  "supporting_characters": [{{"character_id": "supporting_1", "role": "", "identity": "", "age": "", "ethnicity": "", "nationality": "", "skin_tone": "", "height": "", "body_build": "", "body_proportions": "", "appearance": "", "hairstyle": "", "clothing": ""}}],
  "style_bible": {{"visual_style": "", "color_palette": "", "aspect_ratio": "{aspect_ratio}"}},
  "scenes": [
    {scene_shape}
  ]
}}"""
        width, height = ASPECT_PRESETS[aspect_ratio]
        return request, scene_count, prompt_language, aspect_ratio, width, height, story.strip()


class RH_OfflineStoryboardSceneRequests_Node:
    """Turn one locked outline into per-scene requests sharing the same bibles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outline_text": ("STRING", {"forceInput": True}),
                "source_story": ("STRING", {"forceInput": True}),
                "scene_count": ("INT", {"default": 8, "min": 1, "max": 12, "forceInput": True}),
                "prompt_language": ("STRING", {"forceInput": True}),
                "aspect_ratio": ("STRING", {"forceInput": True}),
                "scene_instruction": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_OFFLINE_SCENE_INSTRUCTION},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("qwen_prompts", "outline_json", "scene_count", "prompt_language", "aspect_ratio")
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "build_scene_requests"
    CATEGORY = "Runninghub/Storyboard/Offline"

    def build_scene_requests(
        self,
        outline_text,
        source_story,
        scene_count,
        prompt_language,
        aspect_ratio,
        scene_instruction,
    ):
        requested_count = int(scene_count)
        outline, scenes = _parse_outline(outline_text)
        if len(scenes) != requested_count:
            raise ValueError(
                f"Local Qwen outline returned {len(scenes)} scenes; expected exactly {requested_count}. "
                "Increase the planning TextGenerate max_length and run again."
            )
        character_bible = outline.get("character_bible")
        if not _bible_anchor(character_bible):
            raise ValueError("The local Qwen outline did not return a usable character_bible.")
        style_bible = outline.get("style_bible")
        if not isinstance(style_bible, dict):
            style_bible = {}
            outline["style_bible"] = style_bible
        style_bible["aspect_ratio"] = str(aspect_ratio)
        supporting_characters = outline.get("supporting_characters")
        if not isinstance(supporting_characters, list):
            supporting_characters = []
            outline["supporting_characters"] = supporting_characters
        outline["scenes"] = scenes
        outline["source_story"] = str(source_story).strip()
        outline["generation_settings"] = {
            "mode": "offline_qwen_two_pass",
            "scene_count": requested_count,
            "prompt_language": str(prompt_language),
            "aspect_ratio": str(aspect_ratio),
        }

        language_rule = (
            "Write prompt values entirely in English."
            if str(prompt_language).strip().lower() == "english"
            else "所有 prompt 字段必须完全使用简体中文。"
        )
        requests = []
        for index, scene in enumerate(scenes):
            previous_scene = scenes[index - 1] if index > 0 else None
            package = {
                "source_story": str(source_story).strip(),
                "story_fact": scene.get("story_fact", ""),
                "shot_number": index + 1,
                "total_shots": requested_count,
                "beat_role": scene.get("beat_role", _offline_scene_beat(index + 1, requested_count)),
                "character_lock": character_bible,
                "supporting_character_locks": supporting_characters,
                "characters_present": scene.get("characters_present", ["primary"]),
                "style_lock": style_bible,
                "continuity_before": (
                    previous_scene.get("state_after", previous_scene.get("continuity", ""))
                    if previous_scene
                    else "story opening state"
                ),
                "current_action": scene.get("current_action", scene.get("action", "")),
                "visible_state_after": scene.get("state_after", ""),
                "must_not_show": scene.get("must_not_show", ""),
                "current_scene": scene,
            }
            requests.append(
                f"""{scene_instruction.strip()}

{language_rule}
The final prompt must use aspect ratio {aspect_ratio}.
Do not invent or restate alternative character traits, including ethnicity, nationality, skin tone, height, body build or body proportions. The parser will prepend CHARACTER LOCK verbatim.
Visualize only STORY FACT from SOURCE STORY. Do not add a new action, remove an actor, change who acts on whom, or invent a different outcome.
This is shot {index + 1}/{requested_count}. Show exactly CURRENT ACTION as one frozen visual moment.
Do not depict anything in MUST NOT SHOW. Do not summarize previous or future shots. Do not write a sequence using "then", "after that", "gradually", or multiple consecutive actions.

SCENE PACKAGE:
{_json_text(package)}

Return only:
{{
  "scene_id": {index + 1},
  "prompt": "",
  "negative_prompt": "blurry face, deformed anatomy, extra fingers, low quality, inconsistent character, changed ethnicity, changed skin tone, changed body build, inconsistent body proportions, changed hairstyle, changed clothing",
  "camera": "",
  "continuity_note": ""
}}"""
            )
        return requests, _json_text(outline), requested_count, str(prompt_language), str(aspect_ratio)


class RH_OfflineStoryboardParser_Node:
    """Convert local Qwen JSON/numbered text into ComfyUI prompt lists."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_text": ("STRING", {"forceInput": True}),
                "scene_count": ("INT", {"default": 8, "min": 1, "max": 12, "forceInput": True}),
                "prompt_language": ("STRING", {"forceInput": True}),
                "aspect_ratio": ("STRING", {"forceInput": True}),
                "default_negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry face, deformed anatomy, extra fingers, low quality, watermark, text",
                    },
                ),
            },
            "optional": {
                "outline_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("positive_prompts", "negative_prompts", "storyboard_json", "scene_count")
    OUTPUT_IS_LIST = (True, True, False, False)
    INPUT_IS_LIST = True
    FUNCTION = "parse_storyboard"
    CATEGORY = "Runninghub/Storyboard/Offline"

    def parse_storyboard(
        self,
        generated_text,
        scene_count,
        prompt_language,
        aspect_ratio,
        default_negative_prompt,
        outline_json=None,
    ):
        def first(value):
            if isinstance(value, (list, tuple)):
                return value[0] if value else None
            return value

        requested_count = int(first(scene_count))
        language = str(first(prompt_language) or "中文")
        ratio = str(first(aspect_ratio) or "16:9")
        default_negative = str(first(default_negative_prompt) or "").strip()
        texts = list(generated_text) if isinstance(generated_text, (list, tuple)) else [generated_text]
        canonical_outline = None
        supplied_outline = first(outline_json)
        if isinstance(supplied_outline, str) and supplied_outline.strip():
            canonical_outline, outline_scenes = _parse_outline(supplied_outline)
            if len(outline_scenes) != requested_count:
                raise ValueError(
                    f"The locked outline contains {len(outline_scenes)} scenes; expected {requested_count}."
                )

        if len(texts) > 1:
            shots = []
            fallback_character_bible = {}
            fallback_style_bible = {}
            for scene_number, text in enumerate(texts, start=1):
                _, _, partial = parse_offline_storyboard_text(text, 1, default_negative)
                if not fallback_character_bible and partial.get("character_bible"):
                    fallback_character_bible = partial["character_bible"]
                if not fallback_style_bible and partial.get("style_bible"):
                    fallback_style_bible = partial["style_bible"]
                shot = dict(partial["shots"][0])
                shot["scene_id"] = scene_number
                shots.append(shot)
            if len(shots) < requested_count:
                raise ValueError(
                    f"Local Qwen returned {len(shots)} scene responses; expected {requested_count}. "
                    "Check the TextGenerate error log and run again."
                )
            shots = shots[:requested_count]
            positive = [shot["prompt"] for shot in shots]
            negative = [shot["negative_prompt"] for shot in shots]
            storyboard = {
                "title": canonical_outline.get("title", "") if canonical_outline else "",
                "character_bible": (
                    canonical_outline.get("character_bible", {})
                    if canonical_outline
                    else fallback_character_bible
                ),
                "style_bible": (
                    canonical_outline.get("style_bible", {})
                    if canonical_outline
                    else fallback_style_bible
                ),
                "supporting_characters": (
                    canonical_outline.get("supporting_characters", []) if canonical_outline else []
                ),
                "source_story": canonical_outline.get("source_story", "") if canonical_outline else "",
                "shots": shots,
            }
            if canonical_outline:
                for index, shot in enumerate(shots):
                    shot["scene_outline"] = canonical_outline["scenes"][index]
        else:
            positive, negative, storyboard = parse_offline_storyboard_text(
                first(texts),
                requested_count,
                default_negative,
            )
        style_bible = storyboard.get("style_bible")
        if not isinstance(style_bible, dict):
            style_bible = {}
            storyboard["style_bible"] = style_bible
        style_bible["aspect_ratio"] = ratio
        character_bible = storyboard.get("character_bible")
        if not isinstance(character_bible, dict):
            character_bible = {}
            storyboard["character_bible"] = character_bible
        if canonical_outline and not _bible_anchor(character_bible):
            raise ValueError("The locked outline does not contain a usable character_bible.")
        supporting_characters = storyboard.get("supporting_characters")
        if not isinstance(supporting_characters, list):
            supporting_characters = []
            storyboard["supporting_characters"] = supporting_characters

        locked_positive = []
        character_anchor = _bible_anchor(character_bible)
        style_anchor = ""
        for index, shot in enumerate(storyboard["shots"], start=1):
            raw_prompt = shot["prompt"]
            scene_outline = shot.get("scene_outline", {})
            current_action = ""
            if isinstance(scene_outline, dict):
                current_action = str(
                    scene_outline.get("current_action") or scene_outline.get("action") or ""
                ).strip()
            if language.strip().lower() == "english":
                scene_label = f"SCENE {index:02d}/{requested_count:02d}"
                isolated_prompt = (
                    f"{scene_label}, CURRENT ACTION ONLY: {current_action}, {raw_prompt}"
                    if current_action
                    else f"{scene_label}, {raw_prompt}"
                )
            else:
                scene_label = f"分镜 {index:02d}/{requested_count:02d}"
                isolated_prompt = (
                    f"{scene_label}，本镜头只表现：{current_action}，{raw_prompt}"
                    if current_action
                    else f"{scene_label}，{raw_prompt}"
                )
            selected_characters = _scene_character_lock(
                character_bible,
                supporting_characters,
                scene_outline,
            )
            locked, shot_character_anchor, style_anchor = _locked_prompt(
                isolated_prompt,
                selected_characters,
                style_bible,
                ratio,
                language,
            )
            shot["raw_prompt"] = raw_prompt
            shot["scene_label"] = scene_label
            shot["prompt"] = locked
            shot["character_anchor"] = shot_character_anchor
            if isinstance(scene_outline, dict):
                shot["story_fact"] = scene_outline.get("story_fact", "")
                shot["characters_present"] = scene_outline.get("characters_present", [])
            locked_positive.append(locked)
        positive = locked_positive
        storyboard["character_anchor"] = character_anchor
        storyboard["style_anchor"] = style_anchor
        storyboard["generation_settings"] = {
            "mode": "offline_qwen_two_pass" if canonical_outline else "offline_qwen",
            "scene_count": requested_count,
            "prompt_language": language,
            "aspect_ratio": ratio,
        }
        return positive, negative, _json_text(storyboard), requested_count


class RH_StoryboardSceneSave_Node:
    """Save list/batch images with explicit Scene_01, Scene_02... filenames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "RH_Krea2_Offline_Storyboard"}),
                "start_scene": ("INT", {"default": 1, "min": 1, "max": 999}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    CATEGORY = "Runninghub/Storyboard"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save_images(
        self,
        images,
        filename_prefix="RH_Krea2_Offline_Storyboard",
        start_scene=1,
        prompt=None,
        extra_pnginfo=None,
    ):
        from nodes import SaveImage

        def first(value, default=None):
            if isinstance(value, (list, tuple)):
                return value[0] if value else default
            return value if value is not None else default

        image_items = list(images) if isinstance(images, (list, tuple)) else [images]
        base_prefix = str(first(filename_prefix, "RH_Krea2_Offline_Storyboard"))
        scene_number = int(first(start_scene, 1))
        prompt_value = first(prompt)
        pnginfo_value = first(extra_pnginfo)
        saver = SaveImage()
        saved = []

        for image_item in image_items:
            if image_item is None:
                continue
            if getattr(image_item, "ndim", None) == 3:
                image_item = image_item.unsqueeze(0)
            batch_size = int(image_item.shape[0])
            for batch_index in range(batch_size):
                result = saver.save_images(
                    image_item[batch_index : batch_index + 1],
                    filename_prefix=_scene_save_prefix(base_prefix, scene_number),
                    prompt=prompt_value,
                    extra_pnginfo=pnginfo_value,
                )
                saved.extend(result.get("ui", {}).get("images", []))
                scene_number += 1
        if not saved:
            raise ValueError(
                "Numbered scene saver received no final images. Check the final Klein/Krea route switch."
            )
        return {"ui": {"images": saved}}


class RH_StoryboardScenePrefixes_Node:
    """Build a mapped filename-prefix list matching Scene_01, Scene_02... outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_count": ("INT", {"default": 8, "min": 1, "max": 999}),
                "base_prefix": (
                    "STRING",
                    {"default": "RH_Krea2_Offline_Storyboard_Video"},
                ),
                "start_scene": ("INT", {"default": 1, "min": 1, "max": 999}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefixes",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "build_prefixes"
    CATEGORY = "Runninghub/Storyboard"

    def build_prefixes(
        self,
        scene_count,
        base_prefix="RH_Krea2_Offline_Storyboard_Video",
        start_scene=1,
    ):
        count = max(1, int(scene_count))
        first_scene = max(1, int(start_scene))
        prefix = str(base_prefix).strip().strip("/\\") or "RH_Krea2_Offline_Storyboard_Video"
        return ([
            _scene_save_prefix(prefix, first_scene + offset)
            for offset in range(count)
        ],)


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
3. character_bible 必须固定人物身份、年龄、族裔、人种外观、国籍、肤色、身高、身材体型、身体比例、面部特征、发型、服装和标志性道具。国籍不得仅凭面部外观推断；故事未指定时填写“未指定”。
4. style_bible 必须固定视觉风格、色彩和 aspect_ratio={aspect_ratio}。
5. generation_settings 必须原样记录 scene_count、prompt_language 和 aspect_ratio。
6. 只输出合法 JSON，不要 Markdown，不要解释。

输出结构：
{{
  "title": "故事标题",
  "character_bible": {{"character_id": "primary", "identity": "", "age": "", "ethnicity": "", "nationality": "", "skin_tone": "", "height": "", "body_build": "", "body_proportions": "", "facial_features": "", "hairstyle": "", "clothing": "", "signature_props": ""}},
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
