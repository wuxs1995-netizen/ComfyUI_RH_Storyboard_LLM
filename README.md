# ComfyUI RH Storyboard LLM

An OpenAI-compatible ComfyUI custom node package for turning one story idea into a structured outline and then generating independent prompts for every storyboard scene in parallel.

This repository is based on [`HM-RunningHub/ComfyUI_RH_LLM_API`](https://github.com/HM-RunningHub/ComfyUI_RH_LLM_API) and adds storyboard-oriented nodes while retaining the original image/video-aware LLM node.

## Nodes

### RH Storyboard - Configurable Director

Runs the complete director pipeline from one node. Choose 1–12 scenes, Chinese or English final prompts, and a common aspect ratio. The node generates the outline, fans out the per-scene LLM calls, and returns positive/negative prompt lists plus width and height outputs that can connect to a latent-image node.

### Runninghub LLM API Node

Makes one OpenAI-compatible chat-completions request. It supports text, an optional reference image, or an optional video.

### RH Storyboard - Split One Scene

Parses a director response containing strict JSON and extracts:

- a ready-to-send scene package;
- the selected scene JSON;
- the shared character bible;
- the shared style bible;
- the total scene count.

Use this node when you want a fixed workflow with one visible branch per scene.

### RH Storyboard - Parallel Scene Prompts

Parses the outline's `scenes` array and sends one independent LLM request per scene. Requests run concurrently and are restored to scene order after completion.

Outputs:

- `positive_prompts`: a ComfyUI string list;
- `negative_prompts`: a matching string list;
- `storyboard_json`: all generated shots and continuity notes;
- `scene_count`: the number of generated scenes.

The prompt lists can be connected to text-encoding and image-generation nodes. ComfyUI maps downstream execution across list elements.

### RH Storyboard - Offline Qwen Request

Builds the first-pass planning request for ComfyUI's native `TextGenerate` node. The model analyzes the reference image once, creates one canonical primary-character bible plus supporting-character records, and outlines the requested scenes while separating immutable character traits from mutable scene details. The exact source story is also emitted for the second pass. This node performs no network request and has no API inputs.

Connect its `qwen_prompt` output to `TextGenerate.prompt`, and connect the same reference image directly to `TextGenerate.image` when using a vision-language model such as Qwen3VL.

### RH Storyboard - Offline Locked Scene Requests

Parses the first-pass outline and creates one second-pass request per scene. Every request receives the exact source story, the scene's required `story_fact`, the complete character records, a compact prior-state continuity value and only the current scene outline. Short source events may be divided into more camera beats, but the model is explicitly prohibited from replacing actors or inventing a different plot.

The v7.5 request format also separates `state_before`, one `current_action`, `state_after` and `must_not_show`. The second pass no longer receives complete previous/next scene payloads, preventing a small local model from combining the whole sequence into every prompt.

### RH Storyboard - Offline Qwen Parser

Collects the second-pass local model responses into ordered `positive_prompts` and `negative_prompts` lists for batch image generation. When connected to the locked outline, it programmatically prepends the identical canonical character and style anchors to every final prompt, so a small local model cannot silently omit or rewrite hairstyle, clothing, age or identity between shots. It also accepts the older single-pass format for compatibility.

The parser raises a clear error when the local model returns fewer usable prompts than requested instead of silently producing an empty storyboard. Increase `TextGenerate.max_length` or reduce the scene count if that happens.

### RH Storyboard - Save Numbered Scenes

Saves mapped image lists directly in the output root with explicit names such as `RH_Krea2_Offline_Storyboard_Scene_01_00001_.png`, making the corresponding storyboard shot visible in ComfyUI's asset list and on disk. The saver is forced to run on every queue; an empty final image route now raises an error instead of silently returning no assets.

### RH Storyboard - Select Scene Prompt

Takes the combined `storyboard_json` output and selects one scene by number. It returns separate positive prompt, negative prompt, camera, continuity note and shot JSON outputs. Duplicate this node for each visible storyboard branch when you want every scene connected to its own preview or image-generation workflow.

## Expected director JSON

```json
{
  "title": "Story title",
  "character_bible": {
    "appearance": "Fixed appearance",
    "clothing": "Fixed clothing",
    "identity": "Character identity"
  },
  "style_bible": {
    "visual_style": "Cinematic realism",
    "color_palette": "Muted warm colors",
    "aspect_ratio": "16:9"
  },
  "scenes": [
    {
      "scene_id": 1,
      "location": "Courtyard",
      "shot_type": "Medium shot",
      "camera": "Slow push-in",
      "action": "The character opens a wooden door",
      "emotion": "Curious",
      "continuity": "Rain continues"
    }
  ]
}
```

Markdown-fenced JSON is accepted, although prompting the director to return raw JSON is more reliable.

## Installation

Do not keep this repository and the original `ComfyUI_RH_LLM_API` enabled at the same time because both register `RH_LLMAPI_NODE`.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/wuxs1995-netizen/ComfyUI_RH_Storyboard_LLM.git
cd ComfyUI_RH_Storyboard_LLM
/path/to/ComfyUI/python -m pip install -r requirements.txt
```

For the Vast.ai environment used during development:

```bash
/venv/main/bin/python -m pip install -r requirements.txt
supervisorctl restart comfyui
```

After restarting ComfyUI, hard-refresh the browser and search for `RH Storyboard`.

## Bundled workflows

- [`workflows/RH_parallel_storyboard_demo.json`](workflows/RH_parallel_storyboard_demo.json) demonstrates the parallel prompt pipeline with eight separate scene previews.
- [`workflows/RH_configurable_director_1_to_12_scenes.json`](workflows/RH_configurable_director_1_to_12_scenes.json) adds selectable scene count, prompt language and aspect ratio, with up to twelve independently connectable scene branches.
- [`workflows/RH_configurable_director_Krea2Image_batch.json`](workflows/RH_configurable_director_Krea2Image_batch.json) maps the generated prompt list through the bundled Krea2Image subgraph definition, exposes `📐 Resolution Master` in the main configuration area for the actual generation width and height, removes the original list-incompatible metadata saver, and saves the resulting storyboard images through an external list-safe `SaveImage` node.
- [`workflows/RH_configurable_director_Krea2Image_ReActor_batch.json`](workflows/RH_configurable_director_Krea2Image_ReActor_batch.json) keeps the Krea2Image batch workflow and adds an enabled-by-default `ReActorFaceSwap` identity pass. The same uploaded character reference is sent to the director and broadcast as the ReActor source image for every generated storyboard frame. Separate previews show the swapped result, the original Krea image, and the Krea base image. Version 5.3 exports the workflow with ComfyUI schema version `0.4` for frontend link-rendering compatibility and uses the dedicated `30001+` namespace for root links so they cannot collide with links inside the bundled Krea subgraphs.
- [`workflows/RH_Krea2_API_Director_KleinSwap_batch_v6.4_no_temp_previews.json`](workflows/RH_Krea2_API_Director_KleinSwap_batch_v6.4_no_temp_previews.json) is the online API director workflow with KleinSwap and asset-library-safe final previews. Its API-key field is intentionally empty in the repository.
- [`workflows/RH_Krea2_Offline_Qwen3VL_KleinSwap_batch_v7.5_progressive_numbered.json`](workflows/RH_Krea2_Offline_Qwen3VL_KleinSwap_batch_v7.5_progressive_numbered.json) is the current fully offline Qwen3VL workflow. It adds progressive one-action-per-shot planning, prevents previous/future actions from being merged into the current prompt, prefixes prompts with `SCENE 01/xx`, and saves final images with matching `Scene_01`, `Scene_02` filenames. Character locks and source-story fidelity remain enabled.
- [`workflows/RH_Krea2_Offline_Qwen3VL_KleinSwap_batch_v7.0_native_parser.json`](workflows/RH_Krea2_Offline_Qwen3VL_KleinSwap_batch_v7.0_native_parser.json) is retained as the earlier native-parser offline workflow for compatibility.

For a fully offline workflow, use `Offline Qwen Request → TextGenerate (plan) → Offline Locked Scene Requests → TextGenerate (shots) → Offline Qwen Parser`.

Add a new API key locally after importing any workflow; exported workflow files intentionally contain no API key. The Krea2Image workflow also requires the models and custom nodes used by the original Krea2Image blueprint to be installed on the ComfyUI instance.

The ReActor workflow additionally requires [`ComfyUI-ReActor`](https://github.com/Gourieff/ComfyUI-ReActor) and `inswapper_128.onnx`. The default InsightFace/inswapper weights commonly carry non-commercial research restrictions; verify the model license before commercial use.

## Troubleshooting hidden links

If every node socket is connected but no wires are visible, the workflow JSON may still be valid. ComfyUI has a global `Comfy.LinkRenderMode` setting that can hide links across every workflow.

- Click the link-visibility button in the canvas toolbar; when links are hidden, its action is **Show Links**.
- Or open Settings and change **Link Render Mode** from **Hidden** to **Spline**, **Linear**, or **Straight**.
- Hard-refresh the browser after changing the setting.

This is a browser/user display setting, not a per-workflow connection setting.

## Security

Never commit API keys inside ComfyUI workflow JSON files. Rotate any key that has appeared in a screenshot, exported workflow, terminal log, or chat message.

## Validation

The implementation was validated with:

- Python syntax compilation;
- JSON parsing and scene extraction tests;
- mocked concurrent LLM calls with order restoration;
- ComfyUI `/object_info` registration checks after restart.

## Upstream attribution

The original single-request OpenAI-compatible node was created by [HM-RunningHub](https://github.com/HM-RunningHub). The upstream repository does not currently declare a license; this repository therefore does not add a license declaration of its own.
