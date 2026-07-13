"""Small helpers shared by the DeepSeek and OpenAI compatibility adapters."""

import io
import json
import random


JSON_PROMPT_MARKERS = (
    '"current_answer"', '"current_problem"', '"calibration"', '"explicit_needs"'
)


def parse_json_object(text):
    last_error = ValueError("DeepSeek response contains no valid JSON object")
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as error:
            last_error = error
            continue
        if isinstance(value, dict):
            return value
    raise last_error


def prompt_requests_json(messages):
    prompt = "\n".join(str(message.get("content", "")) for message in messages)
    return any(marker in prompt for marker in JSON_PROMPT_MARKERS)


def bedrock_response(is_anthropic, text):
    if is_anthropic:
        envelope = {"content": [{"text": text}]}
    else:
        envelope = {"choices": [{"message": {"content": text}}]}
    return {"body": io.BytesIO(json.dumps(envelope, ensure_ascii=False).encode())}


def retry_delay(retry_after, attempt):
    try:
        delay = float(retry_after) if retry_after is not None else 2 * (attempt + 1)
    except ValueError:
        delay = 2 * (attempt + 1)
    return min(max(delay, 0), 60) + random.uniform(0, 1)
