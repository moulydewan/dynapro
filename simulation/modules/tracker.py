# simulation/modules/tracker.py
"""
DYNAPRO Intent Tracker
- Default model: GPT-OSS 120B via AWS Bedrock; experiment runners may inject
  a DeepSeek or OpenAI-compatible client instead.
- Runs after every USER turn (before assistant responds)
- Output: {context, goal, explicit_needs, latent_needs, resolved, invalidated}
- Domain-aware: selects prompt based on task_desc
- Stateful: accepts previous_state to carry resolved/latent across turns
"""

import json
import re
import boto3
import textwrap

# ── Config ─────────────────────────────────────────────────────────────────────
TRACKER_MODEL_ID = "openai.gpt-oss-120b-1:0"
AWS_REGION       = "us-east-1"

# ── Shared prompt base ─────────────────────────────────────────────────────────
TRACKER_PROMPT_BASE = """You are an expert intent annotator for {domain_desc} in a multi-turn conversation.
Your job is to read the conversation history, analyze the conversation, understand the user's underlying goal, and identify both what the user has explicitly requested and the latent needs they have not yet considered.

You must output ONLY a valid JSON object with exactly these six fields:

{{
  "context": "Background facts about the user's situation, constraints, and preferences established so far",
  "goal": "The user's primary objective for this conversation",
  "explicit_needs": ["Explicit requests the user made that are NOT yet fulfilled."],
  "latent_needs": ["Latent needs the user has NOT asked for and may not be aware of that will help completing the goal."],
  "resolved": ["Needs fully addressed — both explicit requests and proactively surfaced needs."],
  "invalidated": ["Items that became irrelevant — things that were once explicit/latent but are no longer relevant"]
}}

The key distinction:
- "explicit_needs" = Track only what the user has directly and explicitly requested. Never infer here.
- "latent_needs" = Important considerations the user has not mentioned but that typically matter when completing similar {domain_short} tasks.

Each latent need item must:
  - not already appear in the conversation
  - not be safely inferable from what was said
  - not be a generic personal detail
  - represent a factor that genuinely affects the quality or success of the {domain_short} task

Latent need dimensions to consider for {domain_short}: {dimensions}

You are encouraged to surface out-of-the-box dimensions that a typical user may never think to articulate.
Extract genuine latent needs and present the 3 most important ones.

Rules:
- Be specific and concrete. No vague entries like "user may need help".
- "explicit_needs" and "latent_needs" should each have at max 3 items. Be selective.
- Once something moves to "resolved", it stays there unless explicitly reopened.
- "invalidated" is for items made irrelevant by new information.
- If a field has no entries, use an empty list [] or empty string "".
- Output ONLY the JSON object. No preamble, no explanation, no markdown fences."""

# ── Domain-specific dimensions ─────────────────────────────────────────────────
DIMENSIONS = {
    "document editing": {
    "domain_desc":  "document writing and editing",
    "domain_short": "document editing",
    "dimensions": textwrap.dedent("""\
        - Missing information: facts, details, or content the draft needs but doesn't have
        - Constraint awareness: limits like word count, format, or other rules the piece must follow including checking if a draft is far under a stated limit
        - Audience fit: whether tone and detail level match who is actually going to read it
        - Structure and flow: headings, section breaks, paragraph organization, whether it reads well
        - Discoverability and reach: whether the piece is set up to actually reach its intended readers (SEO, subject lines, titles, tags — depending on where it's going)
        - Sourcing and attribution: needed when quoting or borrowing someone else's work, tied to where it will be published"""),
    },
    "math tutoring": {
        "domain_desc":  "math tutoring",
        "domain_short": "math tutoring",
        "dimensions": textwrap.dedent("""\
            - Prerequisite knowledge gaps: concepts or techniques the problem depends on that the student may not have mastered
            - Conceptual vs procedural confusion: whether the student needs to understand WHY a method works, not just HOW to apply it
            - Solution approach selection: whether there is a more efficient method the student hasn't considered
            - Hint vs full solution: whether the student would benefit more from a nudge toward the next step rather than a complete walkthrough
            - Common error patterns: mistakes students typically make on this problem type that are worth flagging preemptively
            - Domain boundary awareness: unstated conditions the solution relies on, like assuming variables are positive, or that a function is continuous, or that an answer must be an integer.
            - Verification strategy: whether the student knows how to check their answer or detect errors in their own work
            - Generalization: whether understanding this problem unlocks a broader class of similar problems worth pointing out"""),
    },
    "code generation": {
    "domain_desc":  "code generation and programming assistance",
    "domain_short": "code generation",
    "dimensions": textwrap.dedent("""\
        - Missing information: unstated specifics the code needs but wasn't given — value ranges, output format, exact edge-case behavior, or which language/library to use
        - Constraint handling: exact function signatures, parameter names, default values, required imports, or naming conventions that haven't been given yet but are likely to be specified later — don't wait for the user to correct this after the fact
        - Robustness: how the code should behave on invalid, empty, negative, or unexpected input, and whether that's been defined
        - Testability: whether the code's behavior is precise enough to pass exact-match tests — specific return types, exact exception types, deterministic output
        - Algorithm and data structure choice: whether a more suitable approach exists for the problem, when the task is complex enough for this to matter
        - Performance: time or space complexity concerns, only when the input scale or task makes this relevant"""),
    },
    "medical diagnosis": {
        "domain_desc":  "medical consultation and diagnosis",
        "domain_short": "medical diagnosis",
        "dimensions": "goal intent, missing information, constraint awareness, uncertainty indicators, expertise level, urgency assessment, task structure",
    },
}

# ── Build prompts from base + dimensions ──────────────────────────────────────
def _build_prompt(domain_key: str) -> str:
    d = DIMENSIONS[domain_key]
    return TRACKER_PROMPT_BASE.format(
        domain_desc=d["domain_desc"],
        domain_short=d["domain_short"],
        dimensions=d["dimensions"],
    )

TRACKER_PROMPTS = {key: _build_prompt(key) for key in DIMENSIONS}


# ── Domain routing ─────────────────────────────────────────────────────────────
def get_tracker_prompt(task_desc: str) -> str:
    """Return the correct system prompt for the given task_desc."""
    prompt = TRACKER_PROMPTS.get(task_desc)
    if prompt is None:
        raise ValueError(
            f"No tracker prompt for task_desc='{task_desc}'. "
            f"Available: {list(TRACKER_PROMPTS.keys())}"
        )
    return prompt


# ── User templates ─────────────────────────────────────────────────────────────
TRACKER_USER_TEMPLATE_NO_PREV = """Here is the conversation so far:

{conversation}

Based on this conversation, output the current intent state JSON."""

TRACKER_USER_TEMPLATE_WITH_PREV = """Here is the conversation so far:

{conversation}

Previous intent state:
{previous_state}

Instructions:
1. Look at the last ASSISTANT message and move each item from previous explicit_needs and latent_needs to resolved if the assistant covered it partially or fully.
2. Update explicit_needs — remove fulfilled items, add any new explicit requests from the last USER message.
3. Update latent_needs — keep unaddressed ones from previous state, only replace addressed ones with new genuinely relevant latent needs.
4. resolved is cumulative — never remove items from it, only add to it.
5. invalidated is for items made irrelevant by new information.

Output the updated intent state JSON."""


# ── Bedrock Client ─────────────────────────────────────────────────────────────
def get_bedrock_client(region: str = AWS_REGION):
    return boto3.client("bedrock-runtime", region_name=region)


# ── Format conversation for tracker ───────────────────────────────────────────
def format_conversation(turns: list[dict], max_content_len: int = 2048) -> str:
    """
    Truncates long messages to prevent tracker from copying content
    verbatim into JSON, causing unterminated string errors.
    """
    lines = []
    for turn in turns:
        role = turn["role"].upper()
        content = turn["content"].strip()
        if len(content) > max_content_len:
            content = content[:max_content_len] + "... [truncated]"
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


# ── Parse JSON robustly ────────────────────────────────────────────────────────
def parse_json_robust(raw_text: str) -> dict:
    """
    Robustly parse JSON from tracker output.
    Handles: surrounding text, markdown fences, reasoning tags,
             trailing commas, unterminated strings from long content.
    """
    if "<reasoning>" in raw_text:
        raw_text = raw_text.split("</reasoning>")[-1].strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    if not raw_text.startswith("{"):
        start = raw_text.find("{")
        end   = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            raw_text = raw_text[start:end]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', raw_text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        last_brace = raw_text.rfind("}")
        if last_brace != -1:
            truncated = raw_text[:last_brace + 1]
            cleaned = re.sub(r',\s*([}\]])', r'\1', truncated)
            return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse tracker output as JSON: {raw_text[:200]}...")


# ── Call tracker ───────────────────────────────────────────────────────────────
def track_intent(
    turns: list[dict],
    client=None,
    task_desc: str = "document editing",
    previous_state: dict = None,
) -> dict:
    """
    Given conversation turns so far, return the intent state I_t.

    Args:
        turns:          list of {"role": "user"|"assistant", "content": "..."}
        client:         boto3 bedrock-runtime client
        task_desc:      domain string — must match a key in TRACKER_PROMPTS
        previous_state: intent state from the previous turn (None for turn 1)

    Returns:
        dict with keys: context, goal, explicit_needs, latent_needs,
                        resolved, invalidated
    """
    if client is None:
        client = get_bedrock_client()

    system_prompt     = get_tracker_prompt(task_desc)
    conversation_text = format_conversation(turns)

    if previous_state is not None:
        user_message = TRACKER_USER_TEMPLATE_WITH_PREV.format(
            conversation=conversation_text,
            previous_state=json.dumps(previous_state, indent=2),
        )
    else:
        user_message = TRACKER_USER_TEMPLATE_NO_PREV.format(
            conversation=conversation_text,
        )

    request_body = {
        "model": TRACKER_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        "max_completion_tokens": 8192,
        "temperature": 0.0,
        "reasoning_effort": "medium"
    }

    response = client.invoke_model(
        modelId=TRACKER_MODEL_ID,
        body=json.dumps(request_body)
    )

    response_body = json.loads(response["body"].read().decode("utf-8"))
    raw_text = response_body["choices"][0]["message"]["content"].strip()

    intent_state = parse_json_robust(raw_text)

    # Validate correct 6 keys
    required_keys = {"context", "goal", "explicit_needs", "latent_needs", "resolved", "invalidated"}
    missing = required_keys - set(intent_state.keys())
    if missing:
        raise ValueError(f"Tracker output missing keys: {missing}")

    # Safety: carry over resolved from previous state in case tracker drops items
    if previous_state and previous_state.get("resolved"):
        existing = set(previous_state["resolved"])
        new_resolved = set(intent_state.get("resolved", []))
        intent_state["resolved"] = list(existing | new_resolved)

    return intent_state


# ── Track full conversation turn by turn ──────────────────────────────────────
def track_conversation(
    conversation: list[dict],
    client=None,
    task_desc: str = "document editing",
) -> list[dict]:
    """
    Run tracker after every USER turn, returning list of intent states.
    Threads previous_state across turns for statefulness.

    Returns:
        list of dicts, one per user turn:
        [{"turn": 1, "after_user_message": "...", "intent_state": {...}}, ...]
    """
    if client is None:
        client = get_bedrock_client()

    intent_states  = []
    turns_so_far   = []
    previous_state = None

    for turn in conversation:
        turns_so_far.append(turn)
        if turn["role"] == "user":
            state = track_intent(
                turns_so_far,
                client=client,
                task_desc=task_desc,
                previous_state=previous_state,
            )
            intent_states.append({
                "turn": len(intent_states) + 1,
                "after_user_message": turn["content"][:100] + "...",
                "intent_state": state,
            })
            previous_state = state

    return intent_states


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_conversation = [
        {
            "role": "user",
            "content": "Hey, I need help writing an article about Nissan GTR wallpapers. It should be under 500 words and sound good for a general audience. I have some reference material but it's kind of messy. Can you help me clean it up and make it flow better?"
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help! Could you paste the reference material?"
        },
        {
            "role": "user",
            "content": "Yeah here's the stuff: basically it's about downloading GTR wallpapers in high quality. There's info about Nissan being founded in 1933 and six generations of GTR. Focus more on wallpapers than car maintenance. Make it professional but engaging, under 500 words."
        }
    ]

    print("Running DYNAPRO Intent Tracker (document editing)...\n")
    client = get_bedrock_client()
    states = track_conversation(sample_conversation, client=client, task_desc="document editing")

    for s in states:
        print(f"=== Turn {s['turn']} ===")
        print(f"After: {s['after_user_message']}")
        print(json.dumps(s['intent_state'], indent=2))
        print()
