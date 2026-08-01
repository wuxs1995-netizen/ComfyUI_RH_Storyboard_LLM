from .node import RH_LLMAPI_Node, RH_MultiSceneLLM_Node, RH_SceneJSONSplitter_Node


NODE_CLASS_MAPPINGS = {
    "RH_LLMAPI_NODE": RH_LLMAPI_Node,
    "RH_SCENE_JSON_SPLITTER": RH_SceneJSONSplitter_Node,
    "RH_MULTI_SCENE_LLM": RH_MultiSceneLLM_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_LLMAPI_NODE": "Runninghub LLM API Node",
    "RH_SCENE_JSON_SPLITTER": "RH Storyboard - Split One Scene",
    "RH_MULTI_SCENE_LLM": "RH Storyboard - Parallel Scene Prompts",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
