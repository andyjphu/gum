# state_prompt.py
# Prompts for egocentric footage state extraction and multi-pass refinement.
# Odd passes extract activities (observable actions), even passes infer intents.
# Each pass sees the previous pass_window_size passes as context.

from dataclasses import dataclass
from typing import List, Dict, Any


# Pass 1: activity state extraction (odd passes start here)
PRIMITIVE_STATE_SYSTEM = """You extract observable activities from egocentric footage. Output a lowercase_snake_case label describing what the user is physically doing.

Focus ONLY on what is directly visible. Do NOT infer intent or purpose.

Identify the user's observable activity from:
1. Physical actions and body position (e.g., walking, typing, sitting)
2. Object interactions (e.g., phone, keyboard, tools)
3. Screen content — if the frame shows a screen without visible hands/body, describe what is on screen (e.g., "viewing_code", "browsing_web")

Examples:
- "typing_on_keyboard" (NOT "writing an email")
- "scrolling_through_code" (NOT "looking for a bug")
- "positioning_component_on_workbench" (NOT "repairing the device")
- "unclear" ONLY when the frame is a title card, logo, or truly uninterpretable

Guidelines:
1. Use lowercase with underscores
2. Describe WHAT is happening, not WHY
3. The label should be the DOMINANT activity receiving the most attention
4. Use "unclear" only as a last resort

Confidence scale: 1-3 = occluded/ambiguous. 4-6 = visible but multiple descriptions possible. 7-8 = clear with minor ambiguity. 9-10 = only one reasonable description.

Respond with the requested JSON fields."""

PRIMITIVE_STATE_PROMPT = """Analyze this frame and extract the observable activity."""

# Pass 2: hidden intent inference (even passes start here)
HIDDEN_INTENT_SYSTEM = """You infer the user's goal from observed activities in egocentric footage. Output a lowercase_snake_case label describing the user's purpose.

Guidelines:
1. Look for PATTERNS across multiple observed activities
2. Consider the SEQUENCE of actions (what comes before/after)
3. Think about common GOALS that would explain the observed actions
4. Assign lower confidence to speculative inferences

Examples:
- Activities: [typing, looking_at_screen, drinking_coffee] → "working_on_computer_task"
- Activities: [walking, holding_bag, looking_at_list] → "grocery_shopping"
- Activities: [typing, pausing, typing, deleting] → "editing_text" (could also be "debugging_code" — note alternatives)

The label names the USER'S GOAL, not a narration.
GOOD: "configuring_docker_environment"
BORDERLINE BAD: "debugging_tic_tac_toe_win_condition_logic" — too specific, use "debugging_game_logic"
BAD: "personalizing_nighttime_driving_environment_through_interior_lighting" — caption, not intent. Use "adjusting_car_settings"

If uncertain, choose the most likely interpretation with low confidence and list alternatives.

Confidence scale: 1-3 = highly speculative. 4-6 = reasonable but other interpretations possible. 7-8 = well-supported. 9-10 = only one plausible intent.

Respond with the requested JSON fields."""

HIDDEN_INTENT_PROMPT = """Context from prior observations:
{prior_summary}

Nearby frames:
{temporal_context}

Analyze this frame and infer the user's goal."""

# Pass 3+: refined activity state (odd passes after pass 1)
REFINED_PRIMITIVE_SYSTEM = """You verify or correct activity labels on egocentric footage. Output a lowercase_snake_case label describing an observable activity.

Set "changed" to true ONLY if:
1. WRONG: The prior label is factually incorrect for what is visible.
2. SYNONYM: The prior label duplicates another label in this video — use the canonical form.
3. UNDERSPECIFIED: The prior label is "unclear" but the frame has enough information for a real label.

If the prior label is already specific and correct, return it unchanged with changed=false.

The label must describe an observable activity. Never output metadata, section headers, or pass numbers as labels.

Example — keeping a good label:
  {{"changed": false, "primitive_state": "sitting", "confidence": 8}}
Example — fixing a wrong label:
  {{"changed": true, "primitive_state": "viewing_code", "confidence": 7, "refined_from": "unclear", "refinement_reason": "screen shows code editor"}}

Confidence scale: 1-3 = occluded/ambiguous. 4-6 = visible but multiple descriptions possible. 7-8 = clear with minor ambiguity. 9-10 = only one reasonable description.

Respond with the requested JSON fields."""

REFINED_PRIMITIVE_PROMPT = """Prior label: {prior_label}

Prior context:
{prior_context}

Nearby frames:
{temporal_context}

Verify or correct the prior label for this frame."""

# Pass 4+: refined hidden intent (even passes after pass 2)
REFINED_INTENT_SYSTEM = """You verify or correct intent labels on egocentric footage. Output a lowercase_snake_case label describing the user's goal.

Set "changed" to true ONLY if:
1. WRONG: Subsequent frames reveal the prior intent was incorrect. Name the general goal, not implementation details.
2. SYNONYM: The prior intent duplicates another label in this video — use the canonical form.
3. UNDERSPECIFIED: The prior intent is "unclear" but context provides enough information for a real label.

If the prior intent is already specific and correct, return it unchanged with changed=false.

The label must describe a user goal. Never output metadata, section headers, or pass numbers as labels.

Example — keeping a good label:
  {{"changed": false, "hidden_intent": "preparing_cocktail", "intent_confidence": 8}}
Example — fixing a wrong label:
  {{"changed": true, "hidden_intent": "writing_code", "intent_confidence": 7, "refined_from": "unclear", "refinement_reason": "context indicates coding activity"}}

Confidence scale: 1-3 = highly speculative. 4-6 = reasonable but other interpretations possible. 7-8 = well-supported. 9-10 = only one plausible intent.

Respond with the requested JSON fields."""

REFINED_INTENT_PROMPT = """Prior label: {prior_label}

Prior context (passes {context_passes}):
{prior_context}

Nearby frames:
{temporal_context}

Verify or correct the prior label for this frame."""

# Summary prompts
PRIMITIVE_SUMMARY_PROMPT = """Summarize the activities observed in this footage sequence.

## Activities (chronological):
{states}

In 2-3 sentences, describe the overall activity sequence at a high level. Note significant transitions between different activities. Do not list individual states or frequencies — those are captured in the structured data.
"""

INTENT_SUMMARY_PROMPT = """Summarize the hidden intents inferred from this footage sequence.

## Activities Context:
{primitive_summary}

## Hidden Intents Inferred:
{intents}

In 2-3 sentences, describe what the user was trying to accomplish at a high level. Note significant transitions between different goals. Do not list individual intents or frequencies — those are captured in the structured data.
"""

CHUNK_PRIMITIVE_SUMMARY_PROMPT = """Summarize this CHUNK of activities from a longer footage sequence.

## Chunk {chunk_num} of {total_chunks} (frames {start_frame}-{end_frame}):
{states}

In 2-3 sentences, describe the observable activity in this segment at a high level. Note significant transitions. This is a partial summary that will be merged with other chunks.
"""

CHUNK_INTENT_SUMMARY_PROMPT = """Summarize this CHUNK of hidden intents from a longer footage sequence.

## Chunk {chunk_num} of {total_chunks} (frames {start_frame}-{end_frame}):
{intents}

In 2-3 sentences, describe the user's goals in this segment at a high level. Note significant transitions. This is a partial summary that will be merged with other chunks.
"""

MERGE_SUMMARIES_PROMPT = """Merge these partial summaries into a single coherent summary.

## Prior Pass Summary:
{prior_summary}

## Partial Summaries (chronological):
{chunk_summaries}

In 2-3 sentences, combine these chunks into a unified narrative covering the entire sequence. Note the major activities and significant transitions.
"""


# Context window helpers

def get_pass_type(pass_num: int) -> str:
    """Return 'activity' for odd passes, 'intent' for even passes."""
    return "activity" if pass_num % 2 == 1 else "intent"


def _pass_type_to_file_key(pass_type: str) -> str:
    """Map a pass type to its on-disk file/field key.

    On-disk files use 'primitive' (e.g. states_primitive_P1.json) and JSON
    fields use 'primitive_state'. This helper translates the logical pass type
    back to the storage key so callers can build paths and read fields correctly.
    """
    return "primitive" if pass_type == "activity" else pass_type


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
    """A run of consecutive frames with the same state label."""
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
        if self.count == 1:
            return f"  [{self.start_idx}] {self.state_str}"
        return f"  [{self.start_idx}-{self.end_idx}] {self.state_str}"


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
        if pass_type == "activity":
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
            lines.append(f"\n<preceding frames=\"{current_pass}\">")
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
            lines.append(f"\n<surrounding frames=\"{same_type_prior}\">")
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

        section_lines = [f"\n<pass num=\"{pass_num}\">"]

        # Add summary if available
        if pass_num in all_pass_summaries and all_pass_summaries[pass_num]:
            section_lines.append(f"Summary: {all_pass_summaries[pass_num]}")

        # Add deduplicated states
        if pass_num in all_pass_states:
            states = all_pass_states[pass_num]
            state_counts: Dict[str, int] = {}

            for s in states:
                if pass_type == "activity":
                    name = s.get("primitive_state", s.get("state", "unknown"))
                else:
                    name = s.get("hidden_intent", s.get("state", "unknown"))
                state_counts[name] = state_counts.get(name, 0) + 1

            sorted_states = sorted(state_counts.items(), key=lambda x: -x[1])[:max_unique_per_pass]

            section_lines.append("Labels:")

            for name, count in sorted_states:
                section_lines.append(f"  - {name} ({count}x)")

        section_lines.append("</pass>")

        sections.append("\n".join(section_lines))

    return "\n".join(sections)
