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

DEFAULT_SCENE_INSTRUCTION = """Generate one storyboard frame for the current scene.

Requirements:
1. Preserve the character's face, age, hairstyle and clothing.
2. Describe only the current scene.
3. State shot size, composition, camera angle, lighting, environment and action.
4. Use concrete visual language rather than abstract literary language.
5. Write the positive prompt in English.
6. Return valid JSON only in this shape:
{
  "scene_id": 1,
  "prompt": "...",
  "negative_prompt": "...",
  "camera": "...",
  "continuity_note": "..."
}"""


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


def _build_scene_request(outline, scenes, scene_index, instruction):
    scene = scenes[scene_index]
    previous_scene = scenes[scene_index - 1] if scene_index > 0 else None
    next_scene = scenes[scene_index + 1] if scene_index + 1 < len(scenes) else None
    payload = {
        "story_title": outline.get("title", ""),
        "character_bible": outline.get("character_bible", {}),
        "style_bible": outline.get("style_bible", {}),
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
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xFFFFFFFF}),
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
        seed,
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
                seed=(seed + index) & 0xFFFFFFFF,
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
