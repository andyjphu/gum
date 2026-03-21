# state_prompt.py
# Prompts for egocentric footage state extraction and multi-pass refinement.
# Odd passes extract primitives, even passes infer intents.
# Each pass sees the previous pass_window_size passes as context.

from dataclasses import dataclass
from typing import List, Dict, Any


# Pass 1: primitive state extraction (odd passes start here)
PRIMITIVE_STATE_PROMPT = """You are analyzing egocentric (first-person) footage to extract the user's OBSERVABLE behavioral state.

Focus ONLY on what is directly visible/observable - physical actions, body position, and object interactions.
Do NOT infer intent or purpose at this stage.

Given the image(s), identify:
1. The PRIMARY physical action (e.g., walking, typing, reaching, pointing)
2. Body position if visible (standing, sitting, leaning, etc.)
3. Objects being interacted with (phone, keyboard, door, etc.)

Examples of primitive states:
- "walking" (NOT "walking to the store")
- "typing_on_keyboard" (NOT "writing an email")
- "holding_phone_to_ear" (NOT "making a phone call")
- "reaching_for_shelf" (NOT "getting groceries")
- "looking_at_screen" (NOT "reading news")

Guidelines:
1. Use lowercase with underscores
2. Describe WHAT is happening, not WHY
3. Be specific about the action (e.g., "typing" vs "tapping" vs "swiping")
4. The primary state should be the DOMINANT activity receiving the most attention
5. If unclear, use "unclear" or "transitioning"

Output JSON with:
- primitive_state: The dominant observable action string
- body_position: Body posture if visible (null if not visible)
- objects_interacted: List of objects being interacted with (empty list if none)
- confidence: 1-10 score (10 = clearly visible, 1 = obscured/uncertain)
"""

# Pass 2: hidden intent inference (even passes start here)
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
4. Assign lower confidence to speculative intents

Examples of intent inference:
- Primitives: [walking, holding_bag, looking_at_list] → Intent: "grocery_shopping"
- Primitives: [typing, looking_at_screen, drinking_coffee] → Intent: "working_on_computer_task"
- Primitives: [walking, holding_phone, looking_around] → Intent: "navigating_to_destination"

Output JSON with:
- hidden_intent: The primary inferred goal/purpose (e.g., "commuting_to_work", "preparing_meal")
- supporting_primitives: List of primitive states that support this inference
- intent_confidence: 1-10 (1=highly speculative, 10=near certain)
- alternative_intents: Other possible intents if ambiguous (can be empty)
"""

# Pass 3+: refined primitive state (odd passes after pass 1)
REFINED_PRIMITIVE_PROMPT = """You are RE-ANALYZING egocentric footage to extract REFINED primitive states.

You now have context from prior passes including both primitive states AND hidden intents.
Use this context to:
1. Be more precise in labeling the primitive state
2. Collapse semantically equivalent primitives (e.g., "walking_slowly" + "strolling" → "walking")
3. Distinguish primitives that LOOKED similar but serve different intents

## Prior Context:
{prior_context}

## Temporal Context:
{temporal_context}

## Current Image(s):
Re-analyze with the enriched context above.

Refinement guidelines:
1. If the hidden intent suggests a more specific primitive, use it
   - Generic "typing" + intent "coding" → refined to "typing_code"
2. Collapse synonymous primitives to canonical forms
   - "strolling", "walking_slowly", "ambling" → "walking"
3. Split primitives that serve different intents
   - "looking_at_phone" for navigation vs entertainment should remain distinct if intent differs

Output JSON with:
- primitive_state: Refined observable action
- body_position: Body posture if visible
- objects_interacted: List of objects
- confidence: 1-10 score
- refined_from: Original primitive if changed (null if unchanged)
- refinement_reason: Why refined (null if unchanged)
- informed_by_intent: Which intent informed this refinement (null if N/A)
"""

# Pass 4+: refined hidden intent (even passes after pass 2)
REFINED_INTENT_PROMPT = """You are RE-ANALYZING to infer REFINED hidden intents with broader context.

You now have multiple passes of both primitives AND intents.
Use this to:
1. Validate or invalidate prior intent inferences
2. Discover higher-level intents (intent hierarchies)
3. Increase or decrease confidence based on subsequent observations

## Prior Context (Passes {context_passes}):
{prior_context}

## Temporal Context:
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

Output JSON with:
- hidden_intent: Refined inferred intent
- supporting_primitives: Primitives supporting this inference
- intent_confidence: 1-10 score
- alternative_intents: Other possible intents
- refined_from: Original intent if changed (null if unchanged)
- refinement_reason: Why refined (null if unchanged)
- validated_by_outcome: true/false/null - did subsequent frames validate this?
"""

# Summary prompts
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

CHUNK_PRIMITIVE_SUMMARY_PROMPT = """Summarize this CHUNK of primitive states from a longer footage sequence.

## Chunk {chunk_num} of {total_chunks} (frames {start_frame}-{end_frame}):
{states}

Create a concise summary (1-2 paragraphs) of this chunk:
1. List the distinct observable actions in this segment
2. Note action frequencies within this chunk
3. Note the dominant transitions

This is a partial summary that will be merged with other chunks.
"""

CHUNK_INTENT_SUMMARY_PROMPT = """Summarize this CHUNK of hidden intents from a longer footage sequence.

## Chunk {chunk_num} of {total_chunks} (frames {start_frame}-{end_frame}):
{intents}

Create a concise summary (1-2 paragraphs) of this chunk:
1. List the inferred goals/purposes in this segment
2. Note confidence levels
3. Note dominant intent transitions

This is a partial summary that will be merged with other chunks.
"""

MERGE_SUMMARIES_PROMPT = """Merge these partial summaries into a single coherent summary.

## Prior Pass Summary:
{prior_summary}

## Partial Summaries (chronological):
{chunk_summaries}

Create a unified summary (1-2 paragraphs) that:
1. Combines insights from all chunks into a coherent narrative
2. Notes overall action/intent frequencies and dominant patterns
3. Identifies transitions and flow across the full sequence
4. Preserves the most important details from each chunk

Output a concise summary covering the ENTIRE sequence.
"""


# Context window helpers

def get_pass_type(pass_num: int) -> str:
    """Return 'primitive' for odd passes, 'intent' for even passes."""
    return "primitive" if pass_num % 2 == 1 else "intent"


def get_context_passes(current_pass: int, pass_window_size: int = 2) -> List[int]:
    """
    Get the pass numbers to use as context for the current pass.

    Returns the last pass_window_size passes before current_pass.
    Pass 1 has no context. Early passes return fewer than pass_window_size
    if not enough prior passes exist.

    Args:
        current_pass: The current pass number (1-indexed)
        pass_window_size: Number of prior passes to include (default 2)

    Returns:
        List of pass numbers to use as context
    """
    if current_pass <= 1:
        return []
    start = max(1, current_pass - pass_window_size)
    return list(range(start, current_pass))


@dataclass
class _TemporalRun:
    """A run of consecutive frames with the same state."""
    start_idx: int
    end_idx: int
    state_str: str
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
        else:
            line = f"{prefix} Intent: {self.state_str} (conf: {avg_conf})"
        return line


def _collapse_to_runs(
    pass_states: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    pass_type: str,
) -> List[_TemporalRun]:
    """Collapse consecutive frames with identical primary states into runs."""
    window = pass_states[start_idx:end_idx]
    if not window:
        return []

    runs: List[_TemporalRun] = []
    for i, state in enumerate(window):
        frame_idx = start_idx + i
        if pass_type == "primitive":
            state_str = state.get("primitive_state", state.get("state", "unknown"))
            conf = float(state.get("confidence", 5))
        else:
            state_str = state.get("hidden_intent", "unknown")
            conf = float(state.get("intent_confidence", state.get("confidence", 5)))

        # Extend current run if primary state matches
        if runs and runs[-1].state_str == state_str:
            runs[-1].end_idx = frame_idx
            runs[-1].conf_sum += conf
            runs[-1].count += 1
        else:
            runs.append(_TemporalRun(
                start_idx=frame_idx,
                end_idx=frame_idx,
                state_str=state_str,
                conf_sum=conf,
                count=1,
                pass_type=pass_type,
            ))
    return runs


def build_temporal_context(
    all_pass_states: Dict[int, List[Dict[str, Any]]],
    current_frame_idx: int,
    total_frames: int,
    current_pass: int,
    context_window_size: int = 20,
) -> str:
    """
    Build temporal context around the current frame using an asymmetric window.

    Backward half: states already extracted by the current pass (frames before
    current_frame_idx). These are the freshest states.

    Forward half: states from the same-type previous pass (P-2), providing
    lookahead without leaking current-pass future information.

    Both halves are RLE-compressed (lossless) via _collapse_to_runs().

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states
        current_frame_idx: Index of the current frame being analyzed
        total_frames: Total number of frames
        current_pass: The current pass number
        context_window_size: Total window size (split equally backward/forward)

    Returns:
        Formatted context string
    """
    half_window = context_window_size // 2
    current_pass_type = get_pass_type(current_pass)
    lines = []

    # Backward: current pass's already-extracted states
    if current_pass in all_pass_states and current_frame_idx > 0:
        back_start = max(0, current_frame_idx - half_window)
        back_end = current_frame_idx
        runs = _collapse_to_runs(
            all_pass_states[current_pass], back_start, back_end, current_pass_type
        )
        if runs:
            lines.append(f"\n--- Current pass {current_pass} (backward, {current_pass_type}) ---")
            for run in runs:
                lines.append(run.format_line())

    # Forward: same-type previous pass (P-2)
    same_type_prior = current_pass - 2
    if same_type_prior >= 1 and same_type_prior in all_pass_states:
        fwd_start = current_frame_idx + 1
        fwd_end = min(total_frames, current_frame_idx + 1 + half_window)
        prior_type = get_pass_type(same_type_prior)
        runs = _collapse_to_runs(
            all_pass_states[same_type_prior], fwd_start, fwd_end, prior_type
        )
        if runs:
            lines.append(f"\n--- Pass {same_type_prior} (forward lookahead, {prior_type}) ---")
            for run in runs:
                lines.append(run.format_line())

    return "\n".join(lines) if lines else "No temporal context available."


def build_multi_pass_context(
    all_pass_states: Dict[int, List[Dict[str, Any]]],
    all_pass_summaries: Dict[int, str],
    context_passes: List[int],
    max_unique_per_pass: int = 15,
) -> str:
    """
    Build context from prior passes in the pass window.

    Includes summaries and top unique states from each pass in the window.

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states
        all_pass_summaries: Dict mapping pass_num -> summary string
        context_passes: Which passes to include (from get_context_passes)
        max_unique_per_pass: Max unique states to show per pass

    Returns:
        Formatted multi-pass context string
    """
    if not context_passes:
        return "No prior context available."

    sections = []

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
