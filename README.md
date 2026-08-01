# ComfyUI RH Storyboard LLM

An OpenAI-compatible ComfyUI custom node package for turning one story idea into a structured outline and then generating independent prompts for every storyboard scene in parallel.

This repository is based on [`HM-RunningHub/ComfyUI_RH_LLM_API`](https://github.com/HM-RunningHub/ComfyUI_RH_LLM_API) and adds storyboard-oriented nodes while retaining the original image/video-aware LLM node.

## Nodes

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

## Demo workflow

Import [`workflows/RH_parallel_storyboard_demo.json`](workflows/RH_parallel_storyboard_demo.json). Add a new API key locally after importing; the workflow intentionally contains no API key.

The demo stops at the ordered positive/negative prompt lists because checkpoint, conditioning, and sampler choices differ between installations. Connect those list outputs to your preferred image-generation pipeline to render one frame per scene.

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
