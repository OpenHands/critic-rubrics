import copy
import json
import logging
from typing import Any, Dict, List

from litellm import (
    AllMessageValues as LiteLLMMessageType,
    ChatCompletionAssistantMessage,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionUserMessage,
    OpenAIMessageContent,
)


logger = logging.getLogger(__name__)


class FunctionCallConversionError(Exception):
    pass


def reformat_tools(original_tools):
    """
    Convert tools from LangFuse format to expected function schema.
    """

    def is_expected(tools):
        for t in tools:
            if not isinstance(t, dict) or t.get("type") != "function":
                return False
            f = t.get("function", {})
            if "name" not in f or "parameters" not in f:
                return False
            if not isinstance(f["parameters"], dict):
                return False
        return True

    if is_expected(original_tools):
        return original_tools
    reformatted = []
    for t in original_tools:
        assert "name" in t, "Tool must have a 'name'"
        assert "input_schema" in t, "Tool must have an 'input_schema'"
        assert isinstance(t["input_schema"], dict), "Tool 'input_schema' must be a dict"
        assert "description" in t, "Tool must have a 'description'"
        reformatted.append({"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""), "parameters": t.get("input_schema", {})}})
    assert is_expected(reformatted), "Reformatted tools do not match expected schema"
    return reformatted


def convert_tool_call_to_string(tool_call: dict) -> str:
    """Convert tool call to content in string format."""
    if "function" not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'function' key.")
    if "id" not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'id' key.")
    if "type" not in tool_call:
        raise FunctionCallConversionError("Tool call must contain 'type' key.")
    if tool_call["type"] != "function":
        raise FunctionCallConversionError("Tool call type must be 'function'.")

    ret = f"<function={tool_call['function']['name']}>\n"
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError as e:
        raise FunctionCallConversionError(f"Failed to parse arguments as JSON. Arguments: {tool_call['function']['arguments']}") from e
    for param_name, param_value in args.items():
        is_multiline = isinstance(param_value, str) and "\n" in param_value
        ret += f"<parameter={param_name}>"
        if is_multiline:
            ret += "\n"
        if isinstance(param_value, list) or isinstance(param_value, dict):
            ret += json.dumps(param_value)
        else:
            ret += f"{param_value}"
        if is_multiline:
            ret += "\n"
        ret += "</parameter>\n"
    ret += "</function>"
    return ret


def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ""
    for i, tool in enumerate(tools):
        assert tool["type"] == "function"
        fn = tool["function"]
        if i > 0:
            ret += "\n"
        ret += f"---- BEGIN FUNCTION #{i + 1}: {fn['name']} ----\n"
        ret += f"Description: {fn['description']}\n"

        if "parameters" in fn:
            ret += "Parameters:\n"
            properties = fn["parameters"].get("properties", {})
            required_params = set(fn["parameters"].get("required", []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                # Indicate required/optional in parentheses with type
                is_required = param_name in required_params
                param_status = "required" if is_required else "optional"
                param_type = param_info.get("type", "string")

                # Get parameter description
                desc = param_info.get("description", "No description provided")

                # Handle enum values if present
                if "enum" in param_info:
                    enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                    desc += f"\nAllowed values: [{enum_values}]"

                ret += f"  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n"
        else:
            ret += "No parameters are required for this function.\n"

        ret += f"---- END FUNCTION #{i + 1} ----\n"
    return ret


def transform_for_annotator(
    payload: Dict[str, Any],
    system_message: str,
    annotation_instruction_message: str,
) -> list[LiteLLMMessageType] | None:
    """
    - Prepend a synthetic system message (new_system_block + original system + tools description).
    - Keep every message as-is, EXCEPT:
        * assistant with tool_calls: append convert_tool_call_to_string(...) AFTER original content (prefix retained)
        * tool role messages: convert to user, prefix with "EXECUTION RESULT of [name]:\n", keep original content
        * tag last user with tag_last_user + annotation_instruction_message (append as extra text block)
        * tag last assistant with tag_last_assistant (append as extra text block)
    - DO NOT redact images or modify other blocks.
    """

    messages: List[Dict[str, Any]] = copy.deepcopy(payload.get("messages", []))
    raw_tools: List[Dict[str, Any]] = payload.get("tools", [])
    formatted_tools = reformat_tools(raw_tools)
    tools_desc = convert_tools_to_description(formatted_tools)

    # Get the original system message
    system_messages = [m for m in messages if m.get("role") == "system"]
    if len(system_messages) < 1:
        logger.info("did not find exactly one system message, found:", len(system_messages))
        return None

    original_system = system_messages[0]
    original_system = messages[0]
    assert original_system is not None, "No system message found in the payload."
    if isinstance(original_system.get("content"), str):
        original_system_text = original_system["content"]
    else:
        # content is list of blocks
        original_system_text = "\n".join(block["text"] for block in original_system.get("content", []) if block.get("type") == "text")
    if not original_system_text:
        logger.info("System message content is empty.")
        return None

    # remove system messages from the main list
    messages = [m for m in messages if m.get("role") != "system"]

    # build the synthetic system meta message
    transformed: list[LiteLLMMessageType] = [ChatCompletionSystemMessage(role="system", content=[{"type": "text", "text": system_message}])]

    # Filter initial empty assistant messages
    # {'content': [{'type': 'text', 'text': ''}], 'role': 'assistant'}
    if (
        messages
        and messages[0].get("role") == "assistant"
        and ("".join(block.get("text", "") for block in messages[0].get("content", [])) == "" if isinstance(messages[0].get("content"), list) else messages[0].get("content") == "")
    ):
        logger.info("Removing initial empty assistant message.")
        messages = messages[1:]  # remove the first empty assistant message

    # find last user & assistant index on the *original* messages
    last_user_idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    if last_user_idx is None:
        logger.info("No user messages found in the payload. This has to exists otherwise the conversation won't even start.")
        return None

    last_asst_idx = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=None)

    # If no assistant messages, we can't tag the last user message
    # This happens when the condeser JUST kicked up and removed all the previous agent history
    # And then the user sends a follow-up message
    if last_asst_idx is None:
        logger.info("No assistant messages found in the payload. Skipping this one.")
        return None
    assert last_asst_idx is not None, "No assistant messages found in the payload. This has to exists otherwise the conversation won't even start."

    # Modify each message as required
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content")
        if content is None:
            content = ""

        content_blocks: OpenAIMessageContent
        # unify content into list-of-blocks so we can append
        if isinstance(content, str):
            content_blocks = [
                ChatCompletionTextObject(
                    type="text", text=content
                )
            ]
        elif isinstance(content, list):
            content_blocks = content
        else:
            raise FunctionCallConversionError(f"Unexpected content type {type(content)}. Expected str or list. Message idx={i}, role={role}")

        # Remove all "cache_control"
        for block in content_blocks:
            block.pop("cache_control", None)

        if i == 0:
            content_blocks = (
                [
                    ChatCompletionTextObject(
                        type="text",
                        text=(
                            "<< BEGIN ORIGINAL SYSTEM MESSAGE>>\n"
                            f"{original_system_text}\n"
                            "<< END ORIGINAL SYSTEM MESSAGE >>\n"
                            "\n<< BEGIN TOOLS DESCRIPTION >>\n"
                            f"{tools_desc}\n"
                            "<< END TOOLS DESCRIPTION >>\n\n"
                        ),
                    ),
                    ChatCompletionTextObject(type="text", text="<< BEGIN FIRST USER MESSAGE >>\n"),
                ]
                + content_blocks
                + [
                    ChatCompletionTextObject(type="text", text="\n<< END FIRST USER MESSAGE >>"),
                ]
            )

        # 1) assistant with tool_calls -> append convert_tool_call_to_string after original content
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    tool_text = convert_tool_call_to_string(tc)
                except FunctionCallConversionError as e:
                    raise FunctionCallConversionError(f"Failed to convert tool call to string.\nCurrent tool call: {tc}\nRaw messages: {json.dumps(messages, indent=2, ensure_ascii=False)}") from e

                if content_blocks and content_blocks[-1].get("type") == "text":
                    content_blocks[-1]["text"] += "\n\n" + tool_text
                    content_blocks[-1]["text"] = content_blocks[-1]["text"].lstrip()
                else:
                    content_blocks.append({"type": "text", "text": tool_text})

        # 2) tool role -> follow your reference code: convert to user, prefix "EXECUTION RESULT of ..."
        elif role == "tool":
            tool_name = m.get("name", "function")
            prefix = f"EXECUTION RESULT of [{tool_name}]:\n"

            assert isinstance(content_blocks, list), "Tool content must be a list of blocks."
            assert len(content_blocks) > 0, "Tool content cannot be empty."
            content_blocks = [
                ChatCompletionTextObject(type="text", text=prefix)
            ] + content_blocks
            transformed.append(ChatCompletionUserMessage(role="user", content=content_blocks))
            continue  # we've already appended, move to next message

        # 3) Tag last user with instruction, last assistant with finish tag
        # (apply on the *original* roles/indices)
        if i == last_asst_idx and role == "assistant":
            # this typically happens when the agent didn't finish yet the user sent a follow-up
            content_blocks = (
                [
                    ChatCompletionTextObject(type="text", text="<< BEGIN LAST AGENT MESSAGE >>\n"),
                ]
                + content_blocks
                + [
                    ChatCompletionTextObject(type="text", text="\n<< END LAST AGENT MESSAGE >>"),
                ]
            )

        # Tag last user ONLY if it comes after the last assistant
        if i == last_user_idx and role == "user" and last_user_idx > last_asst_idx:
            content_blocks = (
                [
                    ChatCompletionTextObject(type="text", text="<< BEGIN LAST USER MESSAGE >>\n"),
                ]
                + content_blocks
                + [
                    ChatCompletionTextObject(type="text", text="<< END LAST USER MESSAGE >>\n"),
                    ChatCompletionTextObject(type="text", text=annotation_instruction_message.strip()),
                ]
            )

        if role == "user":
            transformed.append(ChatCompletionUserMessage(role="user", content=content_blocks))
        elif role == "assistant":
            transformed.append(ChatCompletionAssistantMessage(role="assistant", content=content_blocks))
        else:
            raise ValueError(f"Unexpected role {role}. Expected 'user', 'assistant', or 'tool'. Message idx={i}, content={content}")

    return transformed
