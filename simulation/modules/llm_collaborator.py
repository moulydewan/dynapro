# simulation/modules/llm_collaborator.py
from __future__ import annotations

import json
import logging
import boto3
from typing import List, Optional

from utils.template import parse_messages
from utils.extract_json_reliable import extract_json
from simulation.prompts import PROACT_MODEL_PROMPT
from simulation.prompts import DYNAPRO_ASSISTANT_PROMPT
from simulation.prompts import DYNAPRO_MEDICAL_ASSISTANT_PROMPT
from simulation.prompts import GENERIC_PROACT_PROMPT

logger = logging.getLogger(__name__)


class LLMCollaborator(object):
    """
    method='none'           → baseline: answers directly
    method='proact'         → CollabLLM proact prompt
    method='dynapro'        → DYNAPRO prompt with tracker state injected
    method='dynapro_medical'→ medical DYNAPRO prompt with the same I_t injection
    method='generic_proact' → generic proactive prompt
    """

    registered_prompts = {
        'none': None,
        'proact': PROACT_MODEL_PROMPT,
        'dynapro': DYNAPRO_ASSISTANT_PROMPT,
        'dynapro_medical': DYNAPRO_MEDICAL_ASSISTANT_PROMPT,
        'generic_proact': GENERIC_PROACT_PROMPT,
    }

    def __init__(
        self,
        method: str = 'none',
        num_retries: int = 10,
        region: str = 'us-east-1',
        intent_state: Optional[dict] = None,
        **llm_kwargs
    ):
        super().__init__()
        self.method = method
        assert method in self.registered_prompts, \
            f"Method {method} not registered. Available: {list(self.registered_prompts.keys())}"
        self.num_retries = num_retries
        self.llm_kwargs = {"temperature": 0.8, "max_tokens": 2048, **llm_kwargs}
        self.client = boto3.client('bedrock-runtime', region_name=region)
        self.intent_state = intent_state

    def __call__(self, messages: List[dict], **kwargs) -> str:
        assert messages[-1]['role'] == 'user'

        if self.method == 'none':
            input_messages = messages

        else:
            prompt_template = self.registered_prompts[self.method]

            if self.method in {'dynapro', 'dynapro_medical'} and self.intent_state is not None:
                additional_info = self._format_intent_state(self.intent_state)
            else:
                additional_info = ''

            prompt = prompt_template.format(
                chat_history=parse_messages(messages, strip_sys_prompt=True),
                max_new_tokens=self.llm_kwargs.get('max_tokens', 2048),
                additional_info=additional_info,
            )
            input_messages = [{"role": "user", "content": prompt}]

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
                    current_schema = {'current_problem', 'thought', 'response'}
                    legacy_schema = {'calibration', 'response'}
                    if current_schema.issubset(keys) or legacy_schema.issubset(keys):
                        response = full_response['response']
                        break
                    logger.error(f"[LLMCollaborator] Keys {keys} don't match. Retrying...")
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

    def _format_intent_state(self, state: dict) -> str:
        """
        Format the tracker fields used by the assistant prompt.

        Context and invalidated items stay in the tracker history but are not
        injected; the assistant receives goal, open needs, and resolved needs.
        """
        lines = ["[Current Intent State — use this to respond proactively]"]

        if state.get("goal"):
            lines.append(f"\nGoal: {state['goal']}")

        if state.get("explicit_needs"):
            lines.append("\nExplicit Needs (user directly asked for these — fulfill first):")
            for item in state["explicit_needs"]:
                lines.append(f"  - {item}")

        if state.get("latent_needs"):
            lines.append("\nLatent Needs (user hasn't asked for these — surface ONE if well-timed and valuable):")
            for item in state["latent_needs"]:
                lines.append(f"  - {item}")

        if state.get("resolved"):
            lines.append("\nResolved (already done — do NOT repeat these):")
            for item in state["resolved"]:
                lines.append(f"  - {item}")

        return "\n".join(lines)
