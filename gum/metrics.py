# metrics.py
# Clustering and analysis metrics for multi-pass primitive/intent extraction
#
# Computes metrics to understand:
# - State/intent distribution and diversity
# - Refinement patterns across passes
# - Convergence behavior
# - Intent confidence and validation

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import json


# =============================================================================
# PRIMITIVE STATE METRICS
# =============================================================================

def compute_primitive_metrics(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metrics for primitive state extraction.

    Args:
        states: List of primitive state dicts from a single pass

    Returns:
        Dict containing various primitive metrics
    """
    if not states:
        return {"error": "No states provided"}

    # Extract primitive states
    primitives = [s.get("primitive_state", "unknown") for s in states]
    confidences = [s.get("confidence", 5) for s in states]

    # Frequency distribution
    freq = Counter(primitives)
    total = len(primitives)

    # Entropy (diversity measure)
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalized entropy (0 = all same, 1 = uniform distribution)
    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Confidence statistics
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    high_conf_ratio = sum(1 for c in confidences if c >= 7) / total if total > 0 else 0
    low_conf_ratio = sum(1 for c in confidences if c <= 3) / total if total > 0 else 0

    # Refinement metrics (for passes > 1)
    refined_count = sum(1 for s in states if s.get("refined_from"))
    refinement_rate = refined_count / total if total > 0 else 0

    # Body position diversity
    body_positions = [s.get("body_position") for s in states if s.get("body_position")]
    body_freq = Counter(body_positions)

    # Object interaction diversity
    all_objects = []
    for s in states:
        objs = s.get("objects_interacted", [])
        if objs:
            all_objects.extend(objs)
    object_freq = Counter(all_objects)

    return {
        "total_frames": total,
        "unique_primitives": len(freq),
        "primitive_frequency": dict(freq.most_common()),
        "entropy": round(entropy, 3),
        "normalized_entropy": round(normalized_entropy, 3),
        "avg_confidence": round(avg_confidence, 2),
        "high_confidence_ratio": round(high_conf_ratio, 3),
        "low_confidence_ratio": round(low_conf_ratio, 3),
        "refinement_rate": round(refinement_rate, 3),
        "body_position_frequency": dict(body_freq.most_common()),
        "object_frequency": dict(object_freq.most_common(20)),
        "top_primitives": freq.most_common(10),
    }


def compute_primitive_transitions(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute state transition matrix for primitives.

    Args:
        states: List of primitive state dicts (chronological order)

    Returns:
        Transition matrix and related metrics
    """
    if len(states) < 2:
        return {"error": "Need at least 2 states for transitions"}

    primitives = [s.get("primitive_state", "unknown") for s in states]

    # Build transition counts
    transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(primitives) - 1):
        from_state = primitives[i]
        to_state = primitives[i + 1]
        transitions[from_state][to_state] += 1

    # Convert to regular dict for JSON serialization
    transition_matrix = {k: dict(v) for k, v in transitions.items()}

    # Compute self-transition rate (staying in same state)
    self_transitions = sum(
        transitions[s][s] for s in transitions if s in transitions[s]
    )
    total_transitions = len(primitives) - 1
    self_transition_rate = self_transitions / total_transitions if total_transitions > 0 else 0

    # Most common transitions
    all_transitions = []
    for from_s, to_dict in transitions.items():
        for to_s, count in to_dict.items():
            all_transitions.append((f"{from_s} → {to_s}", count))
    top_transitions = sorted(all_transitions, key=lambda x: -x[1])[:15]

    return {
        "transition_matrix": transition_matrix,
        "total_transitions": total_transitions,
        "self_transition_rate": round(self_transition_rate, 3),
        "top_transitions": top_transitions,
    }


# =============================================================================
# HIDDEN INTENT METRICS
# =============================================================================

def compute_intent_metrics(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metrics for hidden intent inference.

    Args:
        states: List of intent state dicts from a single pass

    Returns:
        Dict containing various intent metrics
    """
    if not states:
        return {"error": "No states provided"}

    # Extract intents
    intents = [s.get("hidden_intent", "unknown") for s in states]
    confidences = [s.get("intent_confidence", 5) for s in states]

    # Frequency distribution
    freq = Counter(intents)
    total = len(intents)

    # Entropy
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Confidence statistics
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    high_conf_ratio = sum(1 for c in confidences if c >= 7) / total if total > 0 else 0
    speculative_ratio = sum(1 for c in confidences if c <= 4) / total if total > 0 else 0

    # Validation metrics (for refined intents)
    validated = [s for s in states if s.get("validated_by_outcome") is True]
    invalidated = [s for s in states if s.get("validated_by_outcome") is False]
    validation_rate = len(validated) / total if total > 0 else 0
    invalidation_rate = len(invalidated) / total if total > 0 else 0

    # Alternative intent analysis (ambiguity measure)
    states_with_alternatives = sum(
        1 for s in states if s.get("alternative_intents") and len(s.get("alternative_intents", [])) > 0
    )
    ambiguity_rate = states_with_alternatives / total if total > 0 else 0

    # Supporting primitives analysis
    supporting_counts = []
    for s in states:
        supporting = s.get("supporting_primitives", [])
        if supporting:
            supporting_counts.append(len(supporting))
    avg_supporting = sum(supporting_counts) / len(supporting_counts) if supporting_counts else 0

    return {
        "total_frames": total,
        "unique_intents": len(freq),
        "intent_frequency": dict(freq.most_common()),
        "entropy": round(entropy, 3),
        "normalized_entropy": round(normalized_entropy, 3),
        "avg_confidence": round(avg_confidence, 2),
        "high_confidence_ratio": round(high_conf_ratio, 3),
        "speculative_ratio": round(speculative_ratio, 3),
        "validation_rate": round(validation_rate, 3),
        "invalidation_rate": round(invalidation_rate, 3),
        "ambiguity_rate": round(ambiguity_rate, 3),
        "avg_supporting_primitives": round(avg_supporting, 2),
        "top_intents": freq.most_common(10),
    }


# =============================================================================
# CROSS-PASS METRICS
# =============================================================================

def compute_pass_convergence(
    all_pass_states: Dict[int, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Compute convergence metrics across passes.

    Measures how states/intents stabilize across iterations.

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states

    Returns:
        Convergence metrics
    """
    if len(all_pass_states) < 2:
        return {"error": "Need at least 2 passes for convergence"}

    pass_nums = sorted(all_pass_states.keys())
    metrics = {
        "primitive_stability": [],
        "intent_stability": [],
        "unique_count_trend": {"primitives": [], "intents": []},
    }

    for i in range(1, len(pass_nums)):
        prev_pass = pass_nums[i - 1]
        curr_pass = pass_nums[i]

        prev_states = all_pass_states[prev_pass]
        curr_states = all_pass_states[curr_pass]

        if not prev_states or not curr_states:
            continue

        # Determine pass type
        is_primitive = curr_pass % 2 == 1

        if is_primitive:
            # Compare primitives
            prev_prims = [s.get("primitive_state", "unknown") for s in prev_states if "primitive_state" in s]
            curr_prims = [s.get("primitive_state", "unknown") for s in curr_states]

            if prev_prims and curr_prims and len(prev_prims) == len(curr_prims):
                matches = sum(1 for p, c in zip(prev_prims, curr_prims) if p == c)
                stability = matches / len(curr_prims)
                metrics["primitive_stability"].append({
                    "from_pass": prev_pass,
                    "to_pass": curr_pass,
                    "stability": round(stability, 3)
                })

            metrics["unique_count_trend"]["primitives"].append({
                "pass": curr_pass,
                "unique_count": len(set(curr_prims))
            })
        else:
            # Compare intents
            prev_intents = [s.get("hidden_intent", "unknown") for s in prev_states if "hidden_intent" in s]
            curr_intents = [s.get("hidden_intent", "unknown") for s in curr_states]

            if prev_intents and curr_intents and len(prev_intents) == len(curr_intents):
                matches = sum(1 for p, c in zip(prev_intents, curr_intents) if p == c)
                stability = matches / len(curr_intents)
                metrics["intent_stability"].append({
                    "from_pass": prev_pass,
                    "to_pass": curr_pass,
                    "stability": round(stability, 3)
                })

            metrics["unique_count_trend"]["intents"].append({
                "pass": curr_pass,
                "unique_count": len(set(curr_intents))
            })

    # Compute overall convergence score
    if metrics["primitive_stability"]:
        prim_stabilities = [m["stability"] for m in metrics["primitive_stability"]]
        metrics["avg_primitive_stability"] = round(sum(prim_stabilities) / len(prim_stabilities), 3)

    if metrics["intent_stability"]:
        intent_stabilities = [m["stability"] for m in metrics["intent_stability"]]
        metrics["avg_intent_stability"] = round(sum(intent_stabilities) / len(intent_stabilities), 3)

    return metrics


def compute_primitive_intent_alignment(
    primitive_states: List[Dict[str, Any]],
    intent_states: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute alignment between primitives and inferred intents.

    Args:
        primitive_states: List of primitive states
        intent_states: List of intent states (same frames)

    Returns:
        Alignment metrics
    """
    if len(primitive_states) != len(intent_states):
        return {"error": "Mismatched state counts"}

    total = len(primitive_states)
    if total == 0:
        return {"error": "No states provided"}

    # Build primitive-to-intent mapping
    prim_to_intents: Dict[str, List[str]] = defaultdict(list)
    intent_to_prims: Dict[str, List[str]] = defaultdict(list)

    for prim_s, intent_s in zip(primitive_states, intent_states):
        prim = prim_s.get("primitive_state", "unknown")
        intent = intent_s.get("hidden_intent", "unknown")
        prim_to_intents[prim].append(intent)
        intent_to_prims[intent].append(prim)

    # Compute primitive-intent consistency
    # (Do similar primitives map to similar intents?)
    prim_consistency = {}
    for prim, intents in prim_to_intents.items():
        intent_freq = Counter(intents)
        if len(intent_freq) > 0:
            top_intent, top_count = intent_freq.most_common(1)[0]
            consistency = top_count / len(intents)
            prim_consistency[prim] = {
                "top_intent": top_intent,
                "consistency": round(consistency, 3),
                "intent_distribution": dict(intent_freq)
            }

    # Compute intent-primitive consistency
    intent_consistency = {}
    for intent, prims in intent_to_prims.items():
        prim_freq = Counter(prims)
        if len(prim_freq) > 0:
            top_prim, top_count = prim_freq.most_common(1)[0]
            consistency = top_count / len(prims)
            intent_consistency[intent] = {
                "top_primitive": top_prim,
                "consistency": round(consistency, 3),
                "primitive_distribution": dict(prim_freq)
            }

    # Average consistency scores
    avg_prim_consistency = sum(
        v["consistency"] for v in prim_consistency.values()
    ) / len(prim_consistency) if prim_consistency else 0

    avg_intent_consistency = sum(
        v["consistency"] for v in intent_consistency.values()
    ) / len(intent_consistency) if intent_consistency else 0

    return {
        "primitive_to_intent_mapping": prim_consistency,
        "intent_to_primitive_mapping": intent_consistency,
        "avg_primitive_consistency": round(avg_prim_consistency, 3),
        "avg_intent_consistency": round(avg_intent_consistency, 3),
    }


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

def compute_all_metrics(
    all_pass_states: Dict[int, List[Dict[str, Any]]],
    all_pass_summaries: Dict[int, str]
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for all passes.

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states
        all_pass_summaries: Dict mapping pass_num -> summary string

    Returns:
        Complete metrics dictionary
    """
    metrics = {
        "pass_count": len(all_pass_states),
        "passes": {},
        "convergence": {},
        "alignment": {},
    }

    # Per-pass metrics
    for pass_num, states in sorted(all_pass_states.items()):
        pass_type = "primitive" if pass_num % 2 == 1 else "intent"

        if pass_type == "primitive":
            metrics["passes"][f"pass_{pass_num}"] = {
                "type": "primitive",
                "metrics": compute_primitive_metrics(states),
                "transitions": compute_primitive_transitions(states),
            }
        else:
            metrics["passes"][f"pass_{pass_num}"] = {
                "type": "intent",
                "metrics": compute_intent_metrics(states),
            }

    # Convergence metrics
    metrics["convergence"] = compute_pass_convergence(all_pass_states)

    # Alignment metrics (compare latest primitive and intent passes)
    primitive_passes = [p for p in all_pass_states.keys() if p % 2 == 1]
    intent_passes = [p for p in all_pass_states.keys() if p % 2 == 0]

    if primitive_passes and intent_passes:
        latest_prim = max(primitive_passes)
        latest_intent = max(intent_passes)
        metrics["alignment"] = compute_primitive_intent_alignment(
            all_pass_states[latest_prim],
            all_pass_states[latest_intent]
        )

    return metrics


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics to JSON file."""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


# =============================================================================
# VISUALIZATION DATA EXPORT
# =============================================================================

def export_for_visualization(
    all_pass_states: Dict[int, List[Dict[str, Any]]],
    all_pass_summaries: Dict[int, str]
) -> Dict[str, Any]:
    """
    Export data in a format suitable for visualization.

    Returns data structured for:
    - Timeline visualization (state over time)
    - Sankey diagram (state transitions)
    - Confidence heatmap
    - Pass comparison charts

    Args:
        all_pass_states: Dict mapping pass_num -> list of frame states
        all_pass_summaries: Dict mapping pass_num -> summary string

    Returns:
        Visualization-ready data structure
    """
    viz_data = {
        "timeline": [],
        "sankey": {"nodes": [], "links": []},
        "confidence_matrix": [],
        "pass_comparison": [],
    }

    # Build timeline data
    for pass_num, states in sorted(all_pass_states.items()):
        pass_type = "primitive" if pass_num % 2 == 1 else "intent"

        for idx, state in enumerate(states):
            if pass_type == "primitive":
                viz_data["timeline"].append({
                    "frame": idx,
                    "pass": pass_num,
                    "type": "primitive",
                    "value": state.get("primitive_state", "unknown"),
                    "confidence": state.get("confidence", 5),
                    "body_position": state.get("body_position"),
                    "objects": state.get("objects_interacted", []),
                })
            else:
                viz_data["timeline"].append({
                    "frame": idx,
                    "pass": pass_num,
                    "type": "intent",
                    "value": state.get("hidden_intent", "unknown"),
                    "confidence": state.get("intent_confidence", 5),
                    "supporting": state.get("supporting_primitives", []),
                    "alternatives": state.get("alternative_intents", []),
                })

    # Build Sankey diagram data for transitions
    # Use the latest primitive pass
    primitive_passes = [p for p in all_pass_states.keys() if p % 2 == 1]
    if primitive_passes:
        latest_prim = max(primitive_passes)
        states = all_pass_states[latest_prim]

        # Build nodes (unique states)
        unique_states = list(set(s.get("primitive_state", "unknown") for s in states))
        viz_data["sankey"]["nodes"] = [{"name": s} for s in unique_states]

        # Build links (transitions)
        state_to_idx = {s: i for i, s in enumerate(unique_states)}
        transitions: Dict[Tuple[int, int], int] = defaultdict(int)

        for i in range(len(states) - 1):
            from_s = states[i].get("primitive_state", "unknown")
            to_s = states[i + 1].get("primitive_state", "unknown")
            transitions[(state_to_idx[from_s], state_to_idx[to_s])] += 1

        viz_data["sankey"]["links"] = [
            {"source": src, "target": tgt, "value": count}
            for (src, tgt), count in transitions.items()
        ]

    # Build confidence matrix (pass x frame)
    for pass_num, states in sorted(all_pass_states.items()):
        pass_type = "primitive" if pass_num % 2 == 1 else "intent"
        conf_key = "confidence" if pass_type == "primitive" else "intent_confidence"

        confidences = [s.get(conf_key, 5) for s in states]
        viz_data["confidence_matrix"].append({
            "pass": pass_num,
            "type": pass_type,
            "confidences": confidences,
            "avg": round(sum(confidences) / len(confidences), 2) if confidences else 0,
        })

    # Build pass comparison data
    for pass_num, states in sorted(all_pass_states.items()):
        pass_type = "primitive" if pass_num % 2 == 1 else "intent"

        if pass_type == "primitive":
            unique = set(s.get("primitive_state") for s in states)
            refined = sum(1 for s in states if s.get("refined_from"))
        else:
            unique = set(s.get("hidden_intent") for s in states)
            refined = sum(1 for s in states if s.get("refined_from"))

        viz_data["pass_comparison"].append({
            "pass": pass_num,
            "type": pass_type,
            "unique_count": len(unique),
            "refined_count": refined,
            "frame_count": len(states),
        })

    return viz_data
