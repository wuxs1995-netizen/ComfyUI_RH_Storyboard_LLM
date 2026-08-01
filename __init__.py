from .node import (
    RH_LLMAPI_Node,
    RH_ConfigurableStoryboard_Node,
    RH_MultiSceneLLM_Node,
    RH_OfflineStoryboardParser_Node,
    RH_OfflineStoryboardRequest_Node,
    RH_OfflineStoryboardSceneRequests_Node,
    RH_SceneJSONSplitter_Node,
    RH_StoryboardPromptSelector_Node,
)


NODE_CLASS_MAPPINGS = {
    "RH_LLMAPI_NODE": RH_LLMAPI_Node,
    "RH_CONFIGURABLE_STORYBOARD": RH_ConfigurableStoryboard_Node,
    "RH_SCENE_JSON_SPLITTER": RH_SceneJSONSplitter_Node,
    "RH_MULTI_SCENE_LLM": RH_MultiSceneLLM_Node,
    "RH_OFFLINE_STORYBOARD_REQUEST": RH_OfflineStoryboardRequest_Node,
    "RH_OFFLINE_STORYBOARD_SCENE_REQUESTS": RH_OfflineStoryboardSceneRequests_Node,
    "RH_OFFLINE_STORYBOARD_PARSER": RH_OfflineStoryboardParser_Node,
    "RH_STORYBOARD_PROMPT_SELECTOR": RH_StoryboardPromptSelector_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_LLMAPI_NODE": "Runninghub LLM API Node",
    "RH_CONFIGURABLE_STORYBOARD": "RH Storyboard - Configurable Director",
    "RH_SCENE_JSON_SPLITTER": "RH Storyboard - Split One Scene",
    "RH_MULTI_SCENE_LLM": "RH Storyboard - Parallel Scene Prompts",
    "RH_OFFLINE_STORYBOARD_REQUEST": "RH Storyboard - Offline Qwen Request",
    "RH_OFFLINE_STORYBOARD_SCENE_REQUESTS": "RH Storyboard - Offline Locked Scene Requests",
    "RH_OFFLINE_STORYBOARD_PARSER": "RH Storyboard - Offline Qwen Parser",
    "RH_STORYBOARD_PROMPT_SELECTOR": "RH Storyboard - Select Scene Prompt",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
