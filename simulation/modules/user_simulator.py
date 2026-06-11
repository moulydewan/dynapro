#simulation/modules/user_simulator.py
from __future__ import annotations

import json
import logging
import boto3
from typing import List

from utils.template import parse_messages
from utils.extract_json_reliable import extract_json
from simulation.prompts import USER_SIMULATOR_PROMPT, TERMINATION_SIGNAL

logger = logging.getLogger(__name__)


class UserSimulator(object):
    def __init__(self, task_desc='', single_turn_prompt='', num_retries=10, region='us-east-1', **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.task_desc = task_desc
        self.single_turn_prompt = single_turn_prompt
        self.num_retries = num_retries
        self.llm_kwargs = {"temperature": 1.0, "max_tokens": 1024, **llm_kwargs}
        self.client = boto3.client('bedrock-runtime', region_name=region)

        assert 'model' in self.llm_kwargs, "Model name must be provided in llm_kwargs"

    def __call__(self, messages: List[dict]):

        prompt = USER_SIMULATOR_PROMPT.format(
            task_desc=self.task_desc,
            single_turn_prompt=self.single_turn_prompt,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            terminal_signal=TERMINATION_SIGNAL,
        )
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.num_retries):
            try:
                response_raw = self.client.invoke_model(
                    modelId=self.llm_kwargs.get('model'),
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.llm_kwargs.get('max_tokens', 1024),
                        "temperature": self.llm_kwargs.get('temperature', 1.0),
                        "messages": messages
                    })
                )
                full_response = json.loads(response_raw['body'].read())['content'][0]['text']

                try:
                    if isinstance(full_response, str):
                        full_response = extract_json(full_response)
                except Exception as e:
                    logger.error(f"[UserSimulator] Error extracting JSON: {e}")
                    continue

                if isinstance(full_response, dict):
                    keys = full_response.keys()
                    if {'current_answer', 'thought', 'response'}.issubset(keys):
                        response = full_response.pop('response')
                        break
                    else:
                        logger.error(f"[UserSimulator] Keys {keys} do not match expected keys. Retrying...")
                        continue

            except Exception as e:
                logger.error(f"[UserSimulator] Error on attempt: {e}")
                continue

        else:
            logger.error("[UserSimulator] All retries failed. Returning termination signal.")
            return TERMINATION_SIGNAL

        return response.strip()