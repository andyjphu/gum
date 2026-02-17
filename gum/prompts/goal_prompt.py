"""
Prompt and context builder for post-pass goal hypothesis generation.

After all multi-pass analysis is complete, this module aggregates
final-pass data into a structured context and prompts the LLM to
hypothesize 5 high-level emergent goals for the user.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Any, Tuple


GOAL_HYPOTHESIS_PROMPT = """You are a behavioral analyst specializing in emergent pattern recognition from egocentric (first-person) user activity data.

You have been given a comprehensive statistical summary of a user's observed behaviors (primitive states) and inferred intentions (hidden intents) from multi-pass video analysis. Your task is to hypothesize 5 HIGH-LEVEL GOALS that explain the user's overall behavior session.

## Important Guidelines

1. Goals should be NON-OBVIOUS and EMERGENT — not simply restating the most common state
   - BAD: "User is typing on keyboard" (just restating a primitive)
   - BAD: "User is working on computer" (too generic)
   - GOOD: "User is systematically testing a new software setup, evidenced by rapid alternation between configuration screens and testing actions"
   - GOOD: "User is context-switching between a primary creative task and reference-gathering, suggesting they are in an early exploratory phase of a project"

2. Use the CONTRAST between common and rare states to find insights
   - If typing is dominant but rare states include "looking_at_phone" — maybe taking breaks suggests sustained focus periods
   - If common intents are "coding" but rare intents include "debugging" — maybe the code is mostly working (not debugging-heavy)

3. Use TRANSITION PATTERNS to infer workflow structure
   - Frequent A→B→A cycles suggest iteration/refinement
   - Linear A→B→C→D suggests a procedural workflow
   - High self-transition rate suggests deep focus; low rate suggests exploration

4. Use CONCURRENT PATTERNS for multitasking insights
   - Regular concurrent states reveal habitual background activities
   - Changing concurrent patterns may reveal context shifts

5. Use ENTROPY and STABILITY metrics to characterize the session
   - High entropy + low stability = exploratory session
   - Low entropy + high stability = focused, routine session
   - High entropy + high stability = complex but well-understood behavior

6. Each goal should be DISTINCT — cover different aspects of the user's behavior

## Behavioral Data Summary

### Most Common Primitives (Top 5):
{top_primitives}

### Least Common Primitives (Bottom 5, random tie-breaker):
{bottom_primitives}

### Most Common Intents (Top 5):
{top_intents}

### Least Common Intents (Bottom 5, random tie-breaker):
{bottom_intents}

### Top Transitions (State Flow Patterns):
{top_transitions}

### Session Metrics:
{session_metrics}

### Concurrent Activity Patterns:
{concurrent_patterns}

### Primitive-Intent Alignment:
{alignment_summary}

### Convergence & Stability:
{convergence_summary}

## Task

Generate exactly 5 high-level goal hypotheses. For each goal:
1. State the goal clearly and specifically
2. Explain your reasoning with references to the data above
3. List specific supporting evidence (data points, frequencies, transition patterns)
4. Rate your confidence (1-10)
5. Rate the novelty/non-obviousness of the insight (1-10)

Also provide a brief meta-observation about the overall behavioral pattern.

Output JSON with:
- goals: List of exactly 5 goal hypothesis objects
- meta_observation: A 1-2 sentence overall behavioral characterization
"""


def _select_bottom_n(
    freq: Counter,
    n: int = 5,
    seed: str = "",
) -> List[Tuple[str, int]]:
    """Select N least common items with random tie-breaking.

    Groups items by count, takes from the lowest-count groups first.
    When a group has more items than remaining slots, randomly samples
    using a deterministic seed for reproducibility.

    Args:
        freq: Counter of item frequencies
        n: Number of items to select
        seed: Deterministic seed string (e.g., video name)

    Returns:
        List of (item, count) tuples, sorted by count ascending
    """
    if not freq:
        return []

    rng = random.Random(seed)

    # Group by count
    by_count: Dict[int, List[str]] = {}
    for item, count in freq.items():
        by_count.setdefault(count, []).append(item)

    result: List[Tuple[str, int]] = []
    for count in sorted(by_count.keys()):
        items = by_count[count]
        remaining = n - len(result)
        if remaining <= 0:
            break
        if len(items) <= remaining:
            result.extend((item, count) for item in sorted(items))
        else:
            sampled = rng.sample(items, remaining)
            result.extend((item, count) for item in sorted(sampled))

    return result


def build_goal_context(
    all_pass_states: Dict[int, List[Dict[str, Any]]],
    all_pass_summaries: Dict[int, str],
    metrics: Dict[str, Any],
    video_name: str = "",
) -> Dict[str, str]:
    """Aggregate final-pass data into prompt context sections.

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states
        all_pass_summaries: Dict mapping pass_num -> summary string
        metrics: Output of compute_all_metrics()
        video_name: Used as seed for random tie-breaking

    Returns:
        Dict of placeholder_name -> formatted string for GOAL_HYPOTHESIS_PROMPT
    """
    # Identify final primitive and intent passes
    primitive_passes = sorted(p for p in all_pass_states if p % 2 == 1)
    intent_passes = sorted(p for p in all_pass_states if p % 2 == 0)
    final_prim_num = primitive_passes[-1] if primitive_passes else None
    final_intent_num = intent_passes[-1] if intent_passes else None

    # --- Top/bottom primitives ---
    prim_freq: Counter = Counter()
    if final_prim_num and final_prim_num in all_pass_states:
        for s in all_pass_states[final_prim_num]:
            prim_freq[s.get("primitive_state", "unknown")] += 1

    top_prims = prim_freq.most_common(5)
    bottom_prims = _select_bottom_n(prim_freq, 5, seed=video_name)
    prim_total = sum(prim_freq.values()) or 1

    top_primitives = "\n".join(
        f"  {i+1}. {name} ({count} frames, {count/prim_total*100:.1f}%)"
        for i, (name, count) in enumerate(top_prims)
    ) if top_prims else "  No primitive data available"

    bottom_primitives = "\n".join(
        f"  {i+1}. {name} ({count} frames)"
        for i, (name, count) in enumerate(bottom_prims)
    ) if bottom_prims else "  No primitive data available"

    # --- Top/bottom intents ---
    intent_freq: Counter = Counter()
    if final_intent_num and final_intent_num in all_pass_states:
        for s in all_pass_states[final_intent_num]:
            intent_freq[s.get("hidden_intent", "unknown")] += 1

    top_ints = intent_freq.most_common(5)
    bottom_ints = _select_bottom_n(intent_freq, 5, seed=video_name)
    intent_total = sum(intent_freq.values()) or 1

    top_intents = "\n".join(
        f"  {i+1}. {name} ({count} frames, {count/intent_total*100:.1f}%)"
        for i, (name, count) in enumerate(top_ints)
    ) if top_ints else "  No intent data available"

    bottom_intents = "\n".join(
        f"  {i+1}. {name} ({count} frames)"
        for i, (name, count) in enumerate(bottom_ints)
    ) if bottom_ints else "  No intent data available"

    # --- Top transitions ---
    trans_lines: List[str] = []
    final_prim_key = f"pass_{final_prim_num}" if final_prim_num else None
    if final_prim_key and final_prim_key in metrics.get("passes", {}):
        pass_data = metrics["passes"][final_prim_key]
        transitions = pass_data.get("transitions", {})
        top_trans = transitions.get("top_transitions", [])[:10]
        self_rate = transitions.get("self_transition_rate", 0)
        for i, t in enumerate(top_trans):
            # top_transitions can be [str, int] or (str, int)
            if isinstance(t, (list, tuple)) and len(t) == 2:
                trans_lines.append(f"  {i+1}. {t[0]} ({t[1]} times)")
            else:
                trans_lines.append(f"  {i+1}. {t}")
        trans_lines.append(f"  Self-transition rate: {self_rate:.1%}")
    top_transitions = "\n".join(trans_lines) if trans_lines else "  No transition data available"

    # --- Session metrics ---
    session_lines: List[str] = []
    for pass_key in sorted(metrics.get("passes", {}).keys()):
        pass_info = metrics["passes"][pass_key]
        m = pass_info.get("metrics", {})
        ptype = pass_info.get("type", "?")
        session_lines.append(f"  {pass_key} ({ptype}):")
        session_lines.append(f"    Total frames: {m.get('total_frames', '?')}")
        entropy = m.get("entropy")
        norm_entropy = m.get("normalized_entropy")
        if entropy is not None:
            session_lines.append(
                f"    Entropy: {entropy:.3f}"
                + (f" (normalized: {norm_entropy:.3f})" if norm_entropy is not None else "")
            )
        avg_conf = m.get("avg_confidence")
        if avg_conf is not None:
            session_lines.append(f"    Avg confidence: {avg_conf:.2f}")
        high_conf = m.get("high_confidence_ratio")
        if high_conf is not None:
            session_lines.append(f"    High confidence ratio: {high_conf:.1%}")
        unique_p = m.get("unique_primitives")
        unique_i = m.get("unique_intents")
        if unique_p:
            session_lines.append(f"    Unique primitives: {unique_p}")
        if unique_i:
            session_lines.append(f"    Unique intents: {unique_i}")
    session_metrics = "\n".join(session_lines) if session_lines else "  No session metrics available"

    # --- Concurrent patterns ---
    conc_lines: List[str] = []
    for pass_key in sorted(metrics.get("passes", {}).keys()):
        pass_info = metrics["passes"][pass_key]
        m = pass_info.get("metrics", {})
        ratio = m.get("concurrent_ratio", 0)
        if ratio and ratio > 0:
            conc_lines.append(f"  {pass_key}: {ratio:.1%} of frames have concurrent states")
            top_conc = m.get("top_concurrent", [])
            for item in top_conc[:5]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    conc_lines.append(f"    - {item[0]} ({item[1]} frames)")
                else:
                    conc_lines.append(f"    - {item}")
    concurrent_patterns = "\n".join(conc_lines) if conc_lines else "  No concurrent activity detected"

    # --- Alignment ---
    alignment = metrics.get("alignment", {})
    align_lines: List[str] = []
    avg_prim_cons = alignment.get("avg_primitive_consistency")
    avg_int_cons = alignment.get("avg_intent_consistency")
    if avg_prim_cons is not None:
        align_lines.append(f"  Primitive-to-intent consistency: {avg_prim_cons:.3f}")
    if avg_int_cons is not None:
        align_lines.append(f"  Intent-to-primitive consistency: {avg_int_cons:.3f}")

    prim_to_intent = alignment.get("primitive_to_intent_mapping", {})
    if prim_to_intent:
        align_lines.append("  Top primitive-intent mappings:")
        for prim, info in list(prim_to_intent.items())[:5]:
            if isinstance(info, dict):
                top_intent = info.get("top_intent", "?")
                cons = info.get("consistency", 0)
                align_lines.append(f"    {prim} -> {top_intent} (consistency: {cons:.2f})")
    alignment_summary = "\n".join(align_lines) if align_lines else "  No alignment data available"

    # --- Convergence ---
    convergence = metrics.get("convergence", {})
    conv_lines: List[str] = []
    avg_ps = convergence.get("avg_primitive_stability")
    avg_is = convergence.get("avg_intent_stability")
    if avg_ps is not None:
        conv_lines.append(f"  Average primitive stability: {avg_ps:.3f}")
    if avg_is is not None:
        conv_lines.append(f"  Average intent stability: {avg_is:.3f}")
    prim_stab = convergence.get("primitive_stability", [])
    for entry in prim_stab:
        if isinstance(entry, dict):
            conv_lines.append(
                f"    P{entry['from_pass']}->P{entry['to_pass']}: {entry['stability']:.3f}"
            )
    convergence_summary = "\n".join(conv_lines) if conv_lines else "  No convergence data available"

    return {
        "top_primitives": top_primitives,
        "bottom_primitives": bottom_primitives,
        "top_intents": top_intents,
        "bottom_intents": bottom_intents,
        "top_transitions": top_transitions,
        "session_metrics": session_metrics,
        "concurrent_patterns": concurrent_patterns,
        "alignment_summary": alignment_summary,
        "convergence_summary": convergence_summary,
    }
