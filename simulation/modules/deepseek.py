"""DeepSeek HTTP client plus the boto3-compatible simulation adapter."""

import json
import os
import time
import urllib.error
import urllib.request

from simulation.modules.provider_common import bedrock_response, parse_json_object
from simulation.modules.provider_common import prompt_requests_json, retry_delay

URL = "https://api.deepseek.com/chat/completions"


def chat_completion(
    messages,
    *,
    model=None,
    temperature=0.0,
    max_tokens=None,
    json_output=False,
    thinking=None,
    reasoning_effort=None,
    return_metadata=False,
):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    payload = {
        "model": model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    thinking = thinking or os.environ.get("DEEPSEEK_THINKING", "disabled")
    if thinking:
        payload["thinking"] = {"type": thinking}
    if thinking == "enabled":
        reasoning_effort = reasoning_effort or os.environ.get("DEEPSEEK_REASONING_EFFORT")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

    request = urllib.request.Request(
        URL,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(4):
        retry_after = None
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read())
                content = body["choices"][0]["message"]["content"] or ""
                if return_metadata:
                    return {
                        "content": content,
                        "usage": body.get("usage", {}),
                        "model": body.get("model"),
                        "response_id": body.get("id"),
                    }
                return content
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            if error.code not in (408, 409, 429) and error.code < 500:
                raise RuntimeError(f"DeepSeek HTTP {error.code}: {body[:800]}") from error
            last_error = f"DeepSeek HTTP {error.code}: {body[:800]}"
            retry_after = error.headers.get("Retry-After")
        except Exception as error:
            last_error = repr(error)

        if attempt == 3:
            raise RuntimeError(last_error)
        time.sleep(retry_delay(retry_after, attempt))


class _DeepSeekBedrockClient:
    def invoke_model(self, *, modelId, body):
        request_body = json.loads(body)
        is_anthropic = "anthropic_version" in request_body
        messages = request_body["messages"]
        json_output = prompt_requests_json(messages)
        retries = 1 if is_anthropic else int(os.environ.get("DEEPSEEK_JSON_RETRIES", "8"))

        for _ in range(retries):
            try:
                model = modelId if str(modelId).startswith("deepseek-") else os.environ.get(
                    "DEEPSEEK_MODEL", "deepseek-v4-pro"
                )
                raw = chat_completion(
                    messages,
                    model=model,
                    temperature=request_body.get("temperature", 0.0),
                    max_tokens=request_body.get("max_tokens")
                    or request_body.get("max_completion_tokens"),
                    json_output=json_output,
                    # Tracker uses the OpenAI-style request body; keep thinking
                    # enabled for the original user/assistant calls only.
                    thinking=None if is_anthropic else "disabled",
                    reasoning_effort=request_body.get("reasoning_effort"),
                )
                if json_output:
                    value = parse_json_object(raw)
                    response = value.get("response")
                    if response is not None:
                        if not isinstance(response, str) or response.strip() in {"", "...", "…"}:
                            raise ValueError("DeepSeek returned an empty response")
                    normalized = json.dumps(value, ensure_ascii=False)
                else:
                    if raw.strip() in {"", "...", "…"}:
                        raise ValueError("DeepSeek returned an empty response")
                    normalized = raw
                break
            except ValueError as error:
                last_error = error
        else:
            raise RuntimeError("DeepSeek returned invalid JSON") from last_error

        return bedrock_response(is_anthropic, normalized)


class DeepSeekBoto3:
    @staticmethod
    def client(service_name, region_name=None):
        if service_name != "bedrock-runtime":
            raise ValueError(f"Unsupported service: {service_name}")
        return _DeepSeekBedrockClient()
