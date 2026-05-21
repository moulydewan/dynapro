from __future__ import annotations

import os
import copy
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from simulation.modules.user_simulator import UserSimulator
from simulation.modules.llm_collaborator import LLMCollaborator
from simulation.prompts import TERMINATION_SIGNAL

logger = logging.getLogger(__name__)


class ChatSessionSimulator:
    """
    Manages multiple simultaneous chat sessions.
    Mirrors CollabLLM's ChatSessionSimulator but uses AWS Bedrock
    instead of vllm/HuggingFace/litellm.
    """

    def run_chat_simulation(
        self,
        *,
        task_desc: str,
        single_turn_prompt: str,
        chat_history: List[Dict[str, str]],
        assistant_generation_kwargs: Dict[str, Any],
        user_generation_kwargs: Dict[str, Any],
        num_samples: int = 1,
        max_new_turns: int = 0,
        proact_prompt_ratio: float = 0.0,
        add_system_prompt_ratio: float = 0.0,
        region: str = 'us-east-1',
        max_workers: int = 8,
        verbose: bool = True,
    ) -> List[List[Dict[str, str]]]:
        """
        Simulate num_samples conversations in parallel.

        Returns
        -------
        List[List[Dict[str, str]]]
            A list of num_samples full chat transcripts.
        """
        # ------------------------------------------------------------------ #
        # 0. Validation                                                       #
        # ------------------------------------------------------------------ #
        self._validate_session_inputs(
            task_desc,
            single_turn_prompt,
            max_new_turns,
            assistant_generation_kwargs,
            user_generation_kwargs,
        )

        # ------------------------------------------------------------------ #
        # 1. Per-conversation state                                           #
        # ------------------------------------------------------------------ #
        sessions: List[List[Dict[str, str]]] = [
            copy.deepcopy(chat_history or []) for _ in range(num_samples)
        ]

        current_roles = [
            self._determine_starting_role(hist) for hist in sessions
        ]

        msg_budget = [max_new_turns for _ in range(num_samples)]
        active: set[int] = {i for i, b in enumerate(msg_budget) if b > 0}

        user_sims = [
            UserSimulator(
                task_desc=task_desc,
                single_turn_prompt=single_turn_prompt,
                region=region,
                **user_generation_kwargs,
            )
            for _ in range(num_samples)
        ]

        pbar = tqdm(total=max_new_turns, desc="Simulating chat", disable=not verbose)

        # ------------------------------------------------------------------ #
        # 2. Conversation loop                                                #
        # ------------------------------------------------------------------ #
        while active:

            # ---------- USER TURNS ---------------------------------------- #
            user_idx = [i for i in active if current_roles[i] == "user"]
            if user_idx:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    fut_to_i = {
                        pool.submit(user_sims[i], sessions[i]): i
                        for i in user_idx
                    }
                    for fut in as_completed(fut_to_i):
                        i = fut_to_i[fut]
                        resp = fut.result()
                        self._log_response(f"user (Turn {len(sessions[i])})", resp)
                        sessions[i].append({"role": "user", "content": resp})

                        msg_budget[i] -= 1

                        if msg_budget[i] == 0 or self._should_terminate_conversation(resp):
                            current_roles[i] = "terminated"
                            active.discard(i)
                        else:
                            current_roles[i] = "assistant"
                pbar.update(1)

            if not active:
                break

            # ---------- ASSISTANT TURNS ----------------------------------- #
            asst_idx = [i for i in active if current_roles[i] == "assistant"]
            if not asst_idx:
                continue

            num_asst = len(asst_idx)
            cutoff = int(num_asst * proact_prompt_ratio)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_to_i = {}
                for rank, i in enumerate(asst_idx):
                    method_i = "proact" if rank < cutoff else "none"
                    collab_i = LLMCollaborator(
                        method=method_i,
                        region=region,
                        **assistant_generation_kwargs,
                    )
                    fut = pool.submit(collab_i, sessions[i])
                    fut_to_i[fut] = i

                responses = {fut_to_i[f]: f.result() for f in fut_to_i}

            for i, resp in responses.items():
                self._log_response(f"assistant (Turn {len(sessions[i])})", resp)
                sessions[i].append({"role": "assistant", "content": resp})

                msg_budget[i] -= 1

                if msg_budget[i] == 0:
                    current_roles[i] = "terminated"
                    active.discard(i)
                else:
                    current_roles[i] = "user"
            pbar.update(1)

        pbar.close()
        return sessions

    # ------------------------------------------------------------------ #
    # Helper methods                                                      #
    # ------------------------------------------------------------------ #
    def _validate_session_inputs(
        self,
        task_desc: str,
        single_turn_prompt: str,
        max_new_turns: int,
        assistant_generation_kwargs: Optional[Dict[str, Any]] = None,
        user_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(task_desc, str) or not task_desc.strip():
            raise ValueError("`task_desc` must be a non-empty string.")
        if not isinstance(single_turn_prompt, str) or not single_turn_prompt.strip():
            raise ValueError("`single_turn_prompt` must be a non-empty string.")
        if not isinstance(max_new_turns, int) or max_new_turns < 0:
            raise ValueError("`max_new_turns` must be an integer >= 0.")
        if assistant_generation_kwargs.get("model") is None:
            raise ValueError("`assistant_generation_kwargs` must include a 'model' key.")
        if user_generation_kwargs.get("model") is None:
            raise ValueError("`user_generation_kwargs` must include a 'model' key.")

    def _determine_starting_role(self, chat_history: List[Dict[str, str]]) -> str:
        """Determine which role should start the conversation."""
        if chat_history and chat_history[-1]['role'] == 'user':
            return 'assistant'
        return 'user'

    def _should_terminate_conversation(self, response: str) -> bool:
        """Check if the response contains the termination signal."""
        try:
            return TERMINATION_SIGNAL in response
        except Exception as e:
            logger.error(f"Error checking for chat termination: {e}")
            return False

    def _log_response(self, role: str, response: str) -> None:
        """Log the response."""
        logger.info(f"[rank {os.environ.get('RANK', 0)}]{role.capitalize()}: {response}")

    # ------------------------------------------------------------------ #
    # Future methods (not implemented yet)                               #
    # ------------------------------------------------------------------ #
    def _batch_generate_with_vllm(self, *args, **kwargs):
        raise NotImplementedError("vllm support not implemented yet.")

    def _batch_generate_with_huggingface(self, *args, **kwargs):
        raise NotImplementedError("HuggingFace support not implemented yet.")