"""OpenAI Responses client plus the boto3-compatible simulation adapter."""

import json
import os
import time
import urllib.error
import urllib.request

from simulation.modules.provider_common import bedrock_response, parse_json_object
from simulation.modules.provider_common import prompt_requests_json, retry_delay


URL = "https://api.openai.com/v1/responses"


def response_text(data):
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    parts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def responses_completion(
    messages,
    *,
    model=None,
    temperature=0.0,
    max_output_tokens=None,
    json_output=False,
    reasoning_effort=None,
    return_metadata=False,
    force_temperature=False,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    reasoning_effort = reasoning_effort or os.environ.get(
        "OPENAI_REASONING_EFFORT", "low"
    )
    payload = {
        "model": model or os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
        "input": messages,
        "reasoning": {"effort": reasoning_effort},
        "store": False,
    }
    if temperature is not None and (
        reasoning_effort == "none" or force_temperature
    ):
        payload["temperature"] = temperature
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if json_output:
        payload["text"] = {"format": {"type": "json_object"}}

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
                content = response_text(body)
                if not content.strip():
                    raise ValueError("OpenAI returned an empty response")
                if return_metadata:
                    usage = dict(body.get("usage") or {})
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    if isinstance(input_tokens, int):
                        usage.setdefault("prompt_tokens", input_tokens)
                    if isinstance(output_tokens, int):
                        usage.setdefault("completion_tokens", output_tokens)
                    if (
                        isinstance(input_tokens, int)
                        and isinstance(output_tokens, int)
                    ):
                        usage.setdefault("total_tokens", input_tokens + output_tokens)
                    return {
                        "content": content,
                        "usage": usage,
                        "model": body.get("model"),
                        "response_id": body.get("id"),
                    }
                return content
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            if error.code not in (408, 409, 429) and error.code < 500:
                raise RuntimeError(f"OpenAI HTTP {error.code}: {body[:800]}") from error
            last_error = f"OpenAI HTTP {error.code}: {body[:800]}"
            retry_after = error.headers.get("Retry-After")
        except Exception as error:
            last_error = repr(error)

        if attempt == 3:
            raise RuntimeError(last_error)
        time.sleep(retry_delay(retry_after, attempt))


class _OpenAIResponsesBedrockClient:
    def invoke_model(self, *, modelId, body):
        request_body = json.loads(body)
        is_anthropic = "anthropic_version" in request_body
        messages = request_body["messages"]
        json_output = prompt_requests_json(messages)
        retries = int(os.environ.get("OPENAI_JSON_RETRIES", "8"))

        for _ in range(retries):
            try:
                model = (
                    modelId
                    if str(modelId).startswith(("gpt-", "o1", "o3", "o4"))
                    else os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
                )
                raw = responses_completion(
                    messages,
                    model=model,
                    temperature=request_body.get("temperature", 0.0),
                    max_output_tokens=request_body.get("max_tokens")
                    or request_body.get("max_completion_tokens"),
                    json_output=json_output,
                    reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "low"),
                )
                if json_output:
                    value = parse_json_object(raw)
                    response = value.get("response")
                    if response is not None:
                        if not isinstance(response, str) or response.strip() in {"", "...", "…"}:
                            raise ValueError("OpenAI returned an empty response")
                    normalized = json.dumps(value, ensure_ascii=False)
                else:
                    normalized = raw
                break
            except ValueError as error:
                last_error = error
        else:
            raise RuntimeError("OpenAI returned invalid JSON") from last_error

        return bedrock_response(is_anthropic, normalized)


class OpenAIResponsesBoto3:
    @staticmethod
    def client(service_name, region_name=None):
        if service_name != "bedrock-runtime":
            raise ValueError(f"Unsupported service: {service_name}")
        return _OpenAIResponsesBedrockClient()
