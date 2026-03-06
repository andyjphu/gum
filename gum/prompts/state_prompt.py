# state_prompt.py
# Prompts for egocentric footage state extraction and multi-pass refinement
#
# Pass structure (alternating primitive/intent):
#   Pass 1: Primitive states (observable actions)
#   Pass 2: Hidden intents (informed by Pass 1)
#   Pass 3: Refined primitives (informed by Pass 1, 2)
#   Pass 4: Refined intents (informed by Pass 1, 2, 3)
#   Pass 5: Refined primitives (informed by Pass 2, 3, 4)  <- sliding window starts
#   Pass 6: Refined intents (informed by Pass 3, 4, 5)

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


# =============================================================================
# PASS 1: PRIMITIVE STATE EXTRACTION (Odd passes start here)
# =============================================================================

PRIMITIVE_STATE_PROMPT = """You are analyzing egocentric (first-person) footage to extract the user's OBSERVABLE behavioral state.

Focus ONLY on what is directly visible/observable - physical actions, body position, and object interactions.
Do NOT infer intent or purpose at this stage.

Given the image(s), identify:
1. The PRIMARY physical action (e.g., walking, typing, reaching, pointing)
2. Any CONCURRENT actions happening simultaneously (e.g., listening to music while typing)
3. Body position if visible (standing, sitting, leaning, etc.)
4. Objects being interacted with (phone, keyboard, door, etc.)

Examples of primitive states:
- "walking" (NOT "walking to the store")
- "typing_on_keyboard" (NOT "writing an email")
- "holding_phone_to_ear" (NOT "making a phone call")
- "reaching_for_shelf" (NOT "getting groceries")
- "looking_at_screen" (NOT "reading news")

Examples of concurrent states (multiple simultaneous activities):
- Primary: "typing_on_keyboard", Concurrent: ["listening_to_music", "drinking_coffee"]
- Primary: "walking", Concurrent: ["talking_on_phone"]
- Primary: "cooking_on_stove", Concurrent: ["watching_tv"]

Guidelines:
1. Use lowercase with underscores
2. Describe WHAT is happening, not WHY
3. Be specific about the action (e.g., "typing" vs "tapping" vs "swiping")
4. The primary state should be the DOMINANT activity receiving the most attention
5. List concurrent states ONLY for genuinely simultaneous activities visible in the frame — not sequential or speculative ones
6. If only one activity is visible, leave concurrent_states as an empty list
7. If unclear, use "unclear" or "transitioning"

Output JSON with:
- primitive_state: The dominant observable action string
- concurrent_states: List of additional simultaneous actions (empty list if only one activity)
- body_position: Body posture if visible (null if not visible)
- objects_interacted: List of objects being interacted with (empty list if none)
- confidence: 1-10 score (10 = clearly visible, 1 = obscured/uncertain)

{additional_fields}
"""

# =============================================================================
# PASS 2: HIDDEN INTENT INFERENCE (Even passes start here)
# =============================================================================

HIDDEN_INTENT_PROMPT = """You are inferring the user's HIDDEN INTENT from observed primitive states in egocentric footage.

## Context from Prior Pass (Primitive States):
{prior_summary}

## Temporally Nearest Frames' Primitive States:
{temporal_context}

## Current Frame Analysis:
Given the primitive states observed in this sequence, infer what the user is trying to accomplish.

Intent inference guidelines:
1. Look for PATTERNS across multiple primitive states
2. Consider the SEQUENCE of actions (what comes before/after)
3. Think about common GOALS that would explain the observed actions
4. If the frame has concurrent primitive states, consider whether they serve SEPARATE intents simultaneously
5. Assign lower confidence to speculative intents

Examples of intent inference:
- Primitives: [walking, holding_bag, looking_at_list] → Intent: "grocery_shopping"
- Primitives: [typing, looking_at_screen, drinking_coffee] → Intent: "working_on_computer_task"
- Primitives: [walking, holding_phone, looking_around] → Intent: "navigating_to_destination"

Examples of concurrent intents (from concurrent primitives):
- Primitives: [typing_on_keyboard + listening_to_music] → Intent: "coding", Concurrent: ["relaxing_with_background_music"]
- Primitives: [cooking_on_stove + watching_tv] → Intent: "preparing_meal", Concurrent: ["staying_entertained"]

Output JSON with:
- hidden_intent: The primary inferred goal/purpose (e.g., "commuting_to_work", "preparing_meal")
- concurrent_intents: Additional simultaneous goals if the user has multiple purposes at once (empty list if only one intent)
- supporting_primitives: List of primitive states that support this inference
- intent_confidence: 1-10 (1=highly speculative, 10=near certain)
- alternative_intents: Other possible intents if ambiguous (can be empty)

{additional_fields}
"""

# =============================================================================
# PASS 3+: REFINED PRIMITIVE STATE (Odd passes after Pass 1)
# =============================================================================

REFINED_PRIMITIVE_PROMPT = """You are RE-ANALYZING egocentric footage to extract REFINED primitive states.

You now have context from prior passes including both primitive states AND hidden intents.
Use this context to:
1. Be more precise in labeling the primitive state
2. Collapse semantically equivalent primitives (e.g., "walking_slowly" + "strolling" → "walking")
3. Distinguish primitives that LOOKED similar but serve different intents

## Prior Context:
{prior_context}

## Current Image(s):
Re-analyze with the enriched context above.

Refinement guidelines:
1. If the hidden intent suggests a more specific primitive, use it
   - Generic "typing" + intent "coding" → refined to "typing_code"
2. Collapse synonymous primitives to canonical forms
   - "strolling", "walking_slowly", "ambling" → "walking"
3. Split primitives that serve different intents
   - "looking_at_phone" for navigation vs entertainment should remain distinct if intent differs
4. Refine concurrent states: add, remove, or relabel concurrent activities based on enriched context
   - If prior passes missed a concurrent action now supported by intent, add it
   - If a concurrent state was speculative and context disproves it, remove it

Output JSON with:
- primitive_state: Refined observable action
- concurrent_states: Refined list of additional simultaneous actions (empty list if only one activity)
- body_position: Body posture if visible
- objects_interacted: List of objects
- confidence: 1-10 score
- refined_from: Original primitive if changed (null if unchanged)
- refinement_reason: Why refined (null if unchanged)
- informed_by_intent: Which intent informed this refinement (null if N/A)

{additional_fields}
"""

# =============================================================================
# PASS 4+: REFINED HIDDEN INTENT (Even passes after Pass 2)
# =============================================================================

REFINED_INTENT_PROMPT = """You are RE-ANALYZING to infer REFINED hidden intents with broader context.

You now have multiple passes of both primitives AND intents.
Use this to:
1. Validate or invalidate prior intent inferences
2. Discover higher-level intents (intent hierarchies)
3. Increase or decrease confidence based on subsequent observations

## Prior Context (Passes {context_passes}):
{prior_context}

## Temporal Window (50 nearest frames):
{temporal_context}

## Current Frame:
Re-analyze the intent with enriched context.

Refinement guidelines:
1. VALIDATE: If subsequent primitives confirm the intent, increase confidence
   - Prior intent "going_to_grocery_store" + later "picking_items_from_shelf" → validated
2. INVALIDATE: If subsequent primitives contradict, revise the intent
   - Prior intent "going_to_work" + later "entering_gym" → revise to "going_to_gym"
3. ELEVATE: Discover higher-level patterns
   - Sequence of [grocery_shopping, cooking, eating] → "preparing_and_having_meal"
4. Refine concurrent intents: if concurrent primitives reveal multiple simultaneous goals, include them
   - If prior passes missed a concurrent intent now supported by evidence, add it
   - If a concurrent intent was speculative and later context disproves it, remove it

Output JSON with:
- hidden_intent: Refined inferred intent
- concurrent_intents: Refined list of additional simultaneous goals (empty list if only one intent)
- supporting_primitives: Primitives supporting this inference
- intent_confidence: 1-10 score
- alternative_intents: Other possible intents
- refined_from: Original intent if changed (null if unchanged)
- refinement_reason: Why refined (null if unchanged)
- validated_by_outcome: true/false/null - did subsequent frames validate this?

{additional_fields}
"""

# =============================================================================
# SUMMARY PROMPTS
# =============================================================================

PRIMITIVE_SUMMARY_PROMPT = """Summarize the primitive states observed in this footage sequence.

## Primitive States (chronological):
{states}

Create a summary that:
1. Lists the distinct observable actions in order
2. Notes action frequencies (which actions repeated most)
3. Identifies action transitions (what tends to follow what)
4. Highlights any unclear or ambiguous observations

Output a concise summary (1-2 paragraphs) focusing on OBSERVABLE behaviors only.
"""

INTENT_SUMMARY_PROMPT = """Summarize the hidden intents inferred from this footage sequence.

## Primitive States Context:
{primitive_summary}

## Hidden Intents Inferred:
{intents}

Create a summary that:
1. Lists the inferred goals/purposes
2. Notes confidence levels and supporting evidence
3. Identifies intent sequences (goal → subgoal patterns)
4. Highlights ambiguous or conflicting intent inferences

Output a concise summary (1-2 paragraphs) about what the user was trying to accomplish.
"""

COMBINED_SUMMARY_PROMPT = """Create a comprehensive summary combining primitive states and hidden intents.

## Pass History:
{pass_summaries}

## Final Primitive States:
{final_primitives}

## Final Hidden Intents:
{final_intents}

Create a summary that:
1. Describes what the user DID (primitives) and WHY (intents)
2. Shows how the understanding evolved across passes
3. Highlights the most confident insights
4. Notes remaining uncertainties

This summary will inform subsequent analysis passes.
"""

# Legacy prompt for backward compatibility
RETRO_SUMMARY_PROMPT = """You are summarizing a sequence of user states and behavioral propositions from egocentric footage.

## States Observed (chronological):
{states}

## Propositions (behavioral insights):
{propositions}

Create a summary that:
1. Describes the overall activity flow (what did the user do in this sequence?)
2. Notes any patterns or repeated behaviors
3. Highlights key transitions between states
4. Connects states to the behavioral propositions

Output a concise summary (2-4 paragraphs) that captures:
- Main activities performed
- Time spent on different activities (if inferable from state counts)
- Any notable behavioral patterns

This summary will be used as context for re-analyzing the footage in subsequent passes.
"""

# Legacy prompts
STATE_EXTRACTION_PROMPT = PRIMITIVE_STATE_PROMPT  # Alias for backward compatibility

REANALYZE_PROMPT = """You are re-analyzing egocentric footage with additional context from a prior analysis pass.

## Prior Pass Summary:
{prior_summary}

## Prior States (deduplicated):
{prior_states_context}

## Current Image(s):
Analyze the provided image(s) considering the context above.

Your task:
1. Extract the user's current state (same format as before)
2. Consider whether this state should be COLLAPSED with a prior state
   - Collapse if semantically equivalent (e.g., "walking_the_dog" and "dog_walking" should become one)
   - Preserve distinctions that matter (e.g., "walking_dog" vs "walking_cat" stay separate)

Output JSON with:
- state: The state string (may match a prior state if collapsing)
- confidence: 1-10 score
- collapsed_from: null or the original state if this was collapsed
- collapse_reason: null or brief explanation if collapsed

{additional_fields}
"""

FINAL_SUMMARY_PROMPT = """You are creating the final refined summary after {num_passes} analysis passes.

## Pass History:
{pass_summaries}

## Final Collapsed States:
{final_states}

## Final Propositions:
{final_propositions}

Create a comprehensive summary that:
1. Describes the user's activity during this footage
2. Lists the distinct behavioral states identified (after all collapsing)
3. Notes the key behavioral insights (propositions)
4. Highlights any patterns across the session

This is the final output - make it informative and well-structured.
"""


# =============================================================================
# CONTEXT WINDOW HELPERS
# =============================================================================

def get_pass_type(pass_num: int) -> str:
    """Return 'primitive' for odd passes, 'intent' for even passes."""
    return "primitive" if pass_num % 2 == 1 else "intent"


def get_context_passes(current_pass: int) -> List[int]:
    """
    Get the pass numbers to use as context for the current pass.

    Logic:
    - Pass 1: No context (raw extraction)
    - Pass 2: [1]
    - Pass 3: [1, 2]
    - Pass 4: [1, 2, 3]
    - Pass 5+: Sliding window of previous 3 passes [N-3, N-2, N-1]

    Args:
        current_pass: The current pass number (1-indexed)

    Returns:
        List of pass numbers to use as context
    """
    if current_pass <= 1:
        return []
    elif current_pass <= 4:
        return list(range(1, current_pass))
    else:
        # Sliding window of 3
        return [current_pass - 3, current_pass - 2, current_pass - 1]


@dataclass
class _TemporalRun:
    """A run of consecutive frames with the same state-set."""
    start_idx: int
    end_idx: int
    state_str: str
    concurrent: List[str]
    conf_sum: float
    count: int
    pass_type: str

    @property
    def center_idx(self) -> int:
        return (self.start_idx + self.end_idx) // 2

    def format_line(self) -> str:
        avg_conf = round(self.conf_sum / self.count, 1)
        if self.count == 1:
            prefix = f"  [{self.start_idx}]"
        else:
            prefix = f"  [{self.start_idx}-{self.end_idx}] ({self.count} frames)"

        if self.pass_type == "primitive":
            line = f"{prefix} {self.state_str} (conf: {avg_conf})"
            if self.concurrent:
                line += f" + concurrent: [{', '.join(self.concurrent)}]"
        else:
            line = f"{prefix} Intent: {self.state_str} (conf: {avg_conf})"
            if self.concurrent:
                line += f" + concurrent: [{', '.join(self.concurrent)}]"
        return line


def _collapse_to_runs(
    pass_states: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    pass_type: str,
) -> List[_TemporalRun]:
    """Tier 1: Collapse consecutive frames with identical state-sets into runs."""
    window = pass_states[start_idx:end_idx]
    if not window:
        return []

    runs: List[_TemporalRun] = []
    for i, state in enumerate(window):
        frame_idx = start_idx + i
        if pass_type == "primitive":
            state_str = state.get("primitive_state", state.get("state", "unknown"))
            conf = float(state.get("confidence", 5))
            concurrent = sorted(state.get("concurrent_states", []))
        else:
            state_str = state.get("hidden_intent", "unknown")
            conf = float(state.get("intent_confidence", state.get("confidence", 5)))
            concurrent = sorted(state.get("concurrent_intents", []))

        # Extend current run if state-set matches
        if runs and runs[-1].state_str == state_str and runs[-1].concurrent == concurrent:
            runs[-1].end_idx = frame_idx
            runs[-1].conf_sum += conf
            runs[-1].count += 1
        else:
            runs.append(_TemporalRun(
                start_idx=frame_idx,
                end_idx=frame_idx,
                state_str=state_str,
                concurrent=concurrent,
                conf_sum=conf,
                count=1,
                pass_type=pass_type,
            ))
    return runs


def build_temporal_context(
    all_pass_states: Dict[int, List[Dict[str, Any]]],
    current_frame_idx: int,
    total_frames: int,
    context_passes: List[int],
    max_frames: int = 50,
) -> str:
    """
    Build temporal context from nearest frames with change-point compaction.

    Consecutive frames with identical state-sets are collapsed into run-length
    entries (lossless). No information is discarded.

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states
        current_frame_idx: Index of the current frame being analyzed
        total_frames: Total number of frames
        context_passes: Which passes to pull context from
        max_frames: Raw frame window size (default 50)

    Returns:
        Formatted context string
    """
    if not context_passes or not all_pass_states:
        return "No temporal context available."

    half_window = max_frames // 2
    start_idx = max(0, current_frame_idx - half_window)
    end_idx = min(total_frames, current_frame_idx + half_window)

    lines = []
    for pass_num in context_passes:
        if pass_num not in all_pass_states:
            continue

        pass_type = get_pass_type(pass_num)
        lines.append(f"\n--- Pass {pass_num} ({pass_type}) ---")

        runs = _collapse_to_runs(all_pass_states[pass_num], start_idx, end_idx, pass_type)

        for run in runs:
            lines.append(run.format_line())

    return "\n".join(lines) if lines else "No temporal context available."


def build_context_window(
    prior_states: List[Dict[str, Any]],
    prior_summaries: List[str],
    max_unique_states: int = 20
) -> str:
    """
    Build a deduplicated context window for re-analysis passes.

    Args:
        prior_states: List of state dicts from prior pass
        prior_summaries: List of summary strings from prior passes
        max_unique_states: Maximum unique states to include in context

    Returns:
        Formatted context string for the prompt
    """
    if not prior_states and not prior_summaries:
        return "No prior context available."

    # Count state occurrences (handle both old and new schema)
    state_counts: Dict[str, Dict] = {}
    for s in prior_states:
        # Support both old "state" field and new "primitive_state"/"hidden_intent" fields
        state_name = (
            s.get("primitive_state") or
            s.get("hidden_intent") or
            s.get("state", "unknown")
        )
        if state_name not in state_counts:
            state_counts[state_name] = {
                "count": 0,
                "first_confidence": s.get("confidence", s.get("intent_confidence", 5)),
                "example": s
            }
        state_counts[state_name]["count"] += 1

    # Sort by count (most frequent first) and limit
    sorted_states = sorted(state_counts.items(), key=lambda x: -x[1]["count"])
    limited_states = sorted_states[:max_unique_states]

    # Format context
    lines = []
    for state_name, info in limited_states:
        count = info["count"]
        conf = info["first_confidence"]
        lines.append(f"- {state_name} (occurred {count}x, confidence: {conf})")

    if lines:
        context = "Unique states from prior pass:\n" + "\n".join(lines)
    else:
        context = "No states extracted from prior pass."

    if prior_summaries:
        context += "\n\nPrior summaries:\n"
        for i, summary in enumerate(prior_summaries, 1):
            if summary:
                context += f"\n--- Pass {i} Summary ---\n{summary}\n"

    return context


def build_multi_pass_context(
    all_pass_states: Dict[int, List[Dict[str, Any]]],
    all_pass_summaries: Dict[int, str],
    context_passes: List[int],
    max_unique_per_pass: int = 15,
    meta_summary: str = "",
) -> str:
    """
    Build context from multiple prior passes for refined analysis.

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states
        all_pass_summaries: Dict mapping pass_num -> summary string
        context_passes: Which passes to include [e.g., [2, 3, 4] for pass 5]
        max_unique_per_pass: Max unique states to show per pass
        meta_summary: Compacted history from passes that fell out of the
            sliding window. Prepended when non-empty.

    Returns:
        Formatted multi-pass context string
    """
    if not context_passes and not meta_summary:
        return "No prior context available."

    sections = []

    if meta_summary:
        sections.append(
            f"=== Compacted history (earlier passes) ===\n{meta_summary}"
        )

    if not context_passes:
        return "\n".join(sections)

    for pass_num in context_passes:
        pass_type = get_pass_type(pass_num)

        section_lines = [f"\n=== Pass {pass_num} ({pass_type.upper()}) ==="]

        # Add summary if available
        if pass_num in all_pass_summaries and all_pass_summaries[pass_num]:
            section_lines.append(f"Summary: {all_pass_summaries[pass_num]}")

        # Add deduplicated states
        if pass_num in all_pass_states:
            states = all_pass_states[pass_num]
            state_counts: Dict[str, int] = {}

            for s in states:
                if pass_type == "primitive":
                    name = s.get("primitive_state", s.get("state", "unknown"))
                else:
                    name = s.get("hidden_intent", s.get("state", "unknown"))
                state_counts[name] = state_counts.get(name, 0) + 1

            sorted_states = sorted(state_counts.items(), key=lambda x: -x[1])[:max_unique_per_pass]

            if pass_type == "primitive":
                section_lines.append("Primitives:")
            else:
                section_lines.append("Intents:")

            for name, count in sorted_states:
                section_lines.append(f"  - {name} ({count}x)")

        sections.append("\n".join(section_lines))

    return "\n".join(sections)
