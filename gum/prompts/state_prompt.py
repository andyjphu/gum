# state_prompt.py
# Prompts for egocentric footage state extraction and multi-pass refinement

# =============================================================================
# PASS 1: STATE EXTRACTION
# =============================================================================

STATE_EXTRACTION_PROMPT = """You are analyzing egocentric (first-person) footage to extract the user's current behavioral state.

Given the image(s), identify what the user is currently doing. Output a single compound state that captures all concurrent activities.

Examples of compound states:
- "walking"
- "cooking_dinner"
- "walking_while_on_phone"
- "sitting_at_desk_typing"
- "driving_car"
- "playing_video_game"
- "having_conversation_at_restaurant"

Guidelines:
1. Use lowercase with underscores for compound states
2. Be specific but concise (e.g., "cooking_pasta" not just "cooking" if identifiable)
3. Combine concurrent activities with "_while_" or "_and_" (e.g., "eating_while_watching_tv")
4. Focus on observable actions, not inferred intent
5. If the scene is unclear, use "unclear" or "transitioning"

Output JSON with:
- state: The compound state string
- confidence: 1-10 score (10 = very certain, 1 = uncertain)

{additional_fields}
"""

# =============================================================================
# PASS 1: RETRO SUMMARY (after state extraction)
# =============================================================================

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

# =============================================================================
# PASS 2+: RE-ANALYZE WITH CONTEXT
# =============================================================================

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

## TODO: Semantic Clustering Logic
# -------------------------------------------------------------------------
# MODIFY THIS SECTION to adjust how states are collapsed/merged:
#
# Current approach: Collapse states that are semantically equivalent
# - "walking_the_dog" + "dog_walking" → "walking_dog"
# - "cooking_dinner" + "making_dinner" → "cooking_dinner"
#
# Preserve when:
# - Different objects/targets: "walking_dog" ≠ "walking_cat"
# - Different locations matter: "working_at_office" ≠ "working_at_home"
# - Different intensity/mode: "jogging" ≠ "walking"
#
# Future improvements:
# - Add embedding-based similarity threshold
# - Use hierarchical state taxonomy
# - Allow user-defined collapse rules
# -------------------------------------------------------------------------

Output JSON with:
- state: The state string (may match a prior state if collapsing)
- confidence: 1-10 score
- collapsed_from: null or the original state if this was collapsed
- collapse_reason: null or brief explanation if collapsed

{additional_fields}
"""

# =============================================================================
# PASS K: FINAL REFINEMENT SUMMARY
# =============================================================================

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
# CONTEXT WINDOW HELPER
# =============================================================================

def build_context_window(prior_states: list, prior_summaries: list, max_unique_states: int = 20) -> str:
    """
    Build a deduplicated context window for re-analysis passes.

    Implements farsight-style logic:
    - If there were 100 frames of "cooking" and 1 frame of "coding",
      we want "cooking" and "coding" each represented once in context,
      not 100x cooking entries.

    Args:
        prior_states: List of state dicts from prior pass
        prior_summaries: List of summary strings from prior passes
        max_unique_states: Maximum unique states to include in context

    Returns:
        Formatted context string for the prompt
    """
    if not prior_states and not prior_summaries:
        return "No prior context available."

    # Count state occurrences
    state_counts: dict[str, dict] = {}
    for s in prior_states:
        state_name = s.get('state', 'unknown')
        if state_name not in state_counts:
            state_counts[state_name] = {
                'count': 0,
                'first_confidence': s.get('confidence', 5),
                'example': s
            }
        state_counts[state_name]['count'] += 1

    # Sort by count (most frequent first) and limit
    sorted_states = sorted(state_counts.items(), key=lambda x: -x[1]['count'])
    limited_states = sorted_states[:max_unique_states]

    # Format context
    lines = []
    for state_name, info in limited_states:
        count = info['count']
        conf = info['first_confidence']
        lines.append(f"- {state_name} (occurred {count}x, confidence: {conf})")

    if lines:
        context = "Unique states from prior pass:\n" + "\n".join(lines)
    else:
        context = "No states extracted from prior pass."

    if prior_summaries:
        context += "\n\nPrior summaries:\n"
        for i, summary in enumerate(prior_summaries, 1):
            if summary:  # Skip empty summaries
                context += f"\n--- Pass {i} Summary ---\n{summary}\n"

    return context
