from __future__ import annotations

import json
import logging
import boto3
from typing import List

from utils.template import parse_messages
from utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)


class LLMCollaborator(object):
    """
    Mirrors CollabLLM's LLMCollaborator but uses AWS Bedrock instead of litellm.

    method='none'   → baseline: just answers directly (no special prompt)
    method='proact' → collaborative: uses PROACT_MODEL_PROMPT (for later)
    """

    registered_prompts = {
        'none': None,
        'proact': None,  # will be filled in when we add proact support
    }

    def __init__(self, method='none', num_retries=10, region='us-east-1', **llm_kwargs):
        """
        Initialize the LLMCollaborator.
        """
        super().__init__()
        self.method = method
        assert method in self.registered_prompts, \
            f"Prompting method {method} not registered. Available methods: {list(self.registered_prompts.keys())}"
        self.num_retries = num_retries
        self.llm_kwargs = {"temperature": 0.8, "max_tokens": 2048, **llm_kwargs}
        self.client = boto3.client('bedrock-runtime', region_name=region)

    def __call__(self, messages: List[dict], **kwargs) -> str:
        """
        Forward pass of the LLMCollaborator.
        """
        assert messages[-1]['role'] == 'user'

        if self.method == 'none':
            if len(messages) and messages[0]['role'] == 'system':
                logger.info('System message detected.')
            input_messages = messages
        else:
            raise NotImplementedError("proact method not implemented yet.")

        for _ in range(self.num_retries):
            try:
                response_raw = self.client.invoke_model(
                    modelId=self.llm_kwargs.get('model'),
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.llm_kwargs.get('max_tokens', 2048),
                        "temperature": self.llm_kwargs.get('temperature', 0.8),
                        "messages": input_messages
                    })
                )

                full_response = json.loads(response_raw['body'].read())['content'][0]['text']

                try:
                    if isinstance(full_response, str) and not (self.method == 'none'):
                        full_response = extract_json(full_response)
                except Exception as e:
                    logger.error(f"[LLMCollaborator] Error extracting JSON: {e}")
                    continue

                if isinstance(full_response, dict):
                    keys = full_response.keys()
                    if {'current_problem', 'thought', 'response'}.issubset(keys):
                        response = full_response.pop('response')
                        break
                    else:
                        logger.error(f"[LLMCollaborator] Keys {keys} do not match expected keys. Retrying...")
                        continue
                else:
                    response = full_response
                    break

            except Exception as e:
                logger.error(f"[LLMCollaborator] Error on attempt: {e}")
                continue

        else:
            logger.error("[LLMCollaborator] All retries failed.")
            return ""

        return response.strip()