from .node import (
    RH_GPTImageAPI_Node,
    RH_LLMAPI_Node,
    RH_ConfigurableStoryboard_Node,
    RH_ConfigurableStoryboardContinuity_Node,
    RH_MultiSceneLLM_Node,
    RH_MultiSceneContinuityLLM_Node,
    RH_OfflineStoryboardContinuityParser_Node,
    RH_OfflineStoryboardContinuityRequest_Node,
    RH_OfflineStoryboardContinuitySceneRequests_Node,
    RH_OfflineStoryboardParser_Node,
    RH_OfflineStoryboardRequest_Node,
    RH_OfflineStoryboardSceneRequests_Node,
    RH_SceneJSONSplitter_Node,
    RH_StoryboardPromptSelector_Node,
    RH_StoryboardScenePrefixes_Node,
    RH_StoryboardSceneSave_Node,
)


NODE_CLASS_MAPPINGS = {
    "RH_GPT_IMAGE_API": RH_GPTImageAPI_Node,
    "RH_LLMAPI_NODE": RH_LLMAPI_Node,
    "RH_CONFIGURABLE_STORYBOARD": RH_ConfigurableStoryboard_Node,
    "RH_CONFIGURABLE_STORYBOARD_CONTINUITY": RH_ConfigurableStoryboardContinuity_Node,
    "RH_SCENE_JSON_SPLITTER": RH_SceneJSONSplitter_Node,
    "RH_MULTI_SCENE_LLM": RH_MultiSceneLLM_Node,
    "RH_MULTI_SCENE_CONTINUITY_LLM": RH_MultiSceneContinuityLLM_Node,
    "RH_OFFLINE_STORYBOARD_CONTINUITY_REQUEST": RH_OfflineStoryboardContinuityRequest_Node,
    "RH_OFFLINE_STORYBOARD_CONTINUITY_SCENE_REQUESTS": RH_OfflineStoryboardContinuitySceneRequests_Node,
    "RH_OFFLINE_STORYBOARD_CONTINUITY_PARSER": RH_OfflineStoryboardContinuityParser_Node,
    "RH_OFFLINE_STORYBOARD_REQUEST": RH_OfflineStoryboardRequest_Node,
    "RH_OFFLINE_STORYBOARD_SCENE_REQUESTS": RH_OfflineStoryboardSceneRequests_Node,
    "RH_OFFLINE_STORYBOARD_PARSER": RH_OfflineStoryboardParser_Node,
    "RH_STORYBOARD_PROMPT_SELECTOR": RH_StoryboardPromptSelector_Node,
    "RH_STORYBOARD_SCENE_PREFIXES": RH_StoryboardScenePrefixes_Node,
    "RH_STORYBOARD_SCENE_SAVE": RH_StoryboardSceneSave_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_GPT_IMAGE_API": "RH Image API - OpenAI Compatible",
    "RH_LLMAPI_NODE": "Runninghub LLM API Node",
    "RH_CONFIGURABLE_STORYBOARD": "RH Storyboard - Configurable Director",
    "RH_CONFIGURABLE_STORYBOARD_CONTINUITY": "RH Storyboard - Continuity Director v84",
    "RH_SCENE_JSON_SPLITTER": "RH Storyboard - Split One Scene",
    "RH_MULTI_SCENE_LLM": "RH Storyboard - Parallel Scene Prompts",
    "RH_MULTI_SCENE_CONTINUITY_LLM": "RH Storyboard - Continuity-Locked Scene Prompts",
    "RH_OFFLINE_STORYBOARD_CONTINUITY_REQUEST": "RH Storyboard - Offline Continuity Request",
    "RH_OFFLINE_STORYBOARD_CONTINUITY_SCENE_REQUESTS": "RH Storyboard - Offline Continuity Scene Requests",
    "RH_OFFLINE_STORYBOARD_CONTINUITY_PARSER": "RH Storyboard - Offline Continuity Parser",
    "RH_OFFLINE_STORYBOARD_REQUEST": "RH Storyboard - Offline Qwen Request",
    "RH_OFFLINE_STORYBOARD_SCENE_REQUESTS": "RH Storyboard - Offline Locked Scene Requests",
    "RH_OFFLINE_STORYBOARD_PARSER": "RH Storyboard - Offline Qwen Parser",
    "RH_STORYBOARD_PROMPT_SELECTOR": "RH Storyboard - Select Scene Prompt",
    "RH_STORYBOARD_SCENE_PREFIXES": "RH Storyboard - Scene Filename Prefixes",
    "RH_STORYBOARD_SCENE_SAVE": "RH Storyboard - Save Numbered Scenes",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
