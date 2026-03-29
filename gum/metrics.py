# metrics.py
# Clustering and analysis metrics for multi-pass activity/intent extraction
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


# Activity state metrics

def compute_activity_metrics(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metrics for activity state extraction.

    Args:
        states: List of activity state dicts from a single pass

    Returns:
        Dict containing various activity metrics
    """
    if not states:
        return {"error": "No states provided"}

    # Extract activity states (field name remains 'primitive_state' in JSON data)
    activities = [s.get("primitive_state", "unknown") for s in states]
    confidences = [s.get("confidence", 5) for s in states]

    # Frequency distribution
    freq = Counter(activities)
    total = len(activities)

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
        "unique_activities": len(freq),
        "activity_frequency": dict(freq.most_common()),
        "entropy": round(entropy, 3),
        "normalized_entropy": round(normalized_entropy, 3),
        "avg_confidence": round(avg_confidence, 2),
        "high_confidence_ratio": round(high_conf_ratio, 3),
        "low_confidence_ratio": round(low_conf_ratio, 3),
        "refinement_rate": round(refinement_rate, 3),
        "body_position_frequency": dict(body_freq.most_common()),
        "object_frequency": dict(object_freq.most_common(20)),
        "top_activities": freq.most_common(10),
    }


def compute_activity_transitions(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute state transition matrix for activities.

    Args:
        states: List of activity state dicts (chronological order)

    Returns:
        Transition matrix and related metrics
    """
    if len(states) < 2:
        return {"error": "Need at least 2 states for transitions"}

    activities = [s.get("primitive_state", "unknown") for s in states]

    # Build transition counts using primary states
    transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(activities) - 1):
        if activities[i] != activities[i + 1]:
            transitions[activities[i]][activities[i + 1]] += 1

    # Convert to regular dict for JSON serialization
    transition_matrix = {k: dict(v) for k, v in transitions.items()}

    # Compute self-transition rate (same primary state between frames)
    self_transitions = sum(
        1 for i in range(len(activities) - 1) if activities[i] == activities[i + 1]
    )
    total_transitions = len(activities) - 1
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


# Hidden intent metrics

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

    # Supporting activities analysis
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
        "avg_supporting_activities": round(avg_supporting, 2),
        "top_intents": freq.most_common(10),
    }


# Cross-pass metrics

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
        "activity_stability": [],
        "intent_stability": [],
        "unique_count_trend": {"activities": [], "intents": []},
    }

    for i in range(1, len(pass_nums)):
        prev_pass = pass_nums[i - 1]
        curr_pass = pass_nums[i]

        prev_states = all_pass_states[prev_pass]
        curr_states = all_pass_states[curr_pass]

        if not prev_states or not curr_states:
            continue

        # Determine pass type
        is_activity = curr_pass % 2 == 1

        if is_activity:
            prev_acts = [s.get("primitive_state", "unknown") for s in prev_states if "primitive_state" in s]
            curr_acts = [s.get("primitive_state", "unknown") for s in curr_states]

            if prev_acts and curr_acts and len(prev_acts) == len(curr_acts):
                matches = sum(1 for p, c in zip(prev_acts, curr_acts) if p == c)
                stability = matches / len(curr_acts)
                metrics["activity_stability"].append({
                    "from_pass": prev_pass,
                    "to_pass": curr_pass,
                    "stability": round(stability, 3)
                })

            metrics["unique_count_trend"]["activities"].append({
                "pass": curr_pass,
                "unique_count": len(set(curr_acts))
            })
        else:
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
    if metrics["activity_stability"]:
        act_stabilities = [m["stability"] for m in metrics["activity_stability"]]
        metrics["avg_activity_stability"] = round(sum(act_stabilities) / len(act_stabilities), 3)

    if metrics["intent_stability"]:
        intent_stabilities = [m["stability"] for m in metrics["intent_stability"]]
        metrics["avg_intent_stability"] = round(sum(intent_stabilities) / len(intent_stabilities), 3)

    return metrics


def compute_activity_intent_alignment(
    activity_states: List[Dict[str, Any]],
    intent_states: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute alignment between activities and inferred intents.

    Args:
        activity_states: List of activity states
        intent_states: List of intent states (same frames)

    Returns:
        Alignment metrics
    """
    if len(activity_states) != len(intent_states):
        return {"error": "Mismatched state counts"}

    total = len(activity_states)
    if total == 0:
        return {"error": "No states provided"}

    # Build activity-to-intent mapping
    act_to_intents: Dict[str, List[str]] = defaultdict(list)
    intent_to_acts: Dict[str, List[str]] = defaultdict(list)

    for act_s, intent_s in zip(activity_states, intent_states):
        act = act_s.get("primitive_state", "unknown")
        intent = intent_s.get("hidden_intent", "unknown")
        act_to_intents[act].append(intent)
        intent_to_acts[intent].append(act)

    # Compute activity-intent consistency
    # (Do similar activities map to similar intents?)
    act_consistency = {}
    for act, intents in act_to_intents.items():
        intent_freq = Counter(intents)
        if len(intent_freq) > 0:
            top_intent, top_count = intent_freq.most_common(1)[0]
            consistency = top_count / len(intents)
            act_consistency[act] = {
                "top_intent": top_intent,
                "consistency": round(consistency, 3),
                "intent_distribution": dict(intent_freq)
            }

    # Compute intent-activity consistency
    intent_consistency = {}
    for intent, acts in intent_to_acts.items():
        act_freq = Counter(acts)
        if len(act_freq) > 0:
            top_act, top_count = act_freq.most_common(1)[0]
            consistency = top_count / len(acts)
            intent_consistency[intent] = {
                "top_activity": top_act,
                "consistency": round(consistency, 3),
                "activity_distribution": dict(act_freq)
            }

    # Average consistency scores
    avg_act_consistency = sum(
        v["consistency"] for v in act_consistency.values()
    ) / len(act_consistency) if act_consistency else 0

    avg_intent_consistency = sum(
        v["consistency"] for v in intent_consistency.values()
    ) / len(intent_consistency) if intent_consistency else 0

    return {
        "activity_to_intent_mapping": act_consistency,
        "intent_to_activity_mapping": intent_consistency,
        "avg_activity_consistency": round(avg_act_consistency, 3),
        "avg_intent_consistency": round(avg_intent_consistency, 3),
    }


# Aggregate metrics

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
        pass_type = "activity" if pass_num % 2 == 1 else "intent"

        if pass_type == "activity":
            metrics["passes"][f"pass_{pass_num}"] = {
                "type": "activity",
                "metrics": compute_activity_metrics(states),
                "transitions": compute_activity_transitions(states),
            }
        else:
            metrics["passes"][f"pass_{pass_num}"] = {
                "type": "intent",
                "metrics": compute_intent_metrics(states),
            }

    # Convergence metrics
    metrics["convergence"] = compute_pass_convergence(all_pass_states)

    # Alignment metrics (compare latest activity and intent passes)
    activity_passes = [p for p in all_pass_states.keys() if p % 2 == 1]
    intent_passes = [p for p in all_pass_states.keys() if p % 2 == 0]

    if activity_passes and intent_passes:
        latest_act = max(activity_passes)
        latest_intent = max(intent_passes)
        metrics["alignment"] = compute_activity_intent_alignment(
            all_pass_states[latest_act],
            all_pass_states[latest_intent]
        )

    return metrics


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics to JSON file."""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


# Visualization data export

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
        pass_type = "activity" if pass_num % 2 == 1 else "intent"

        for idx, state in enumerate(states):
            if pass_type == "activity":
                entry = {
                    "frame": idx,
                    "pass": pass_num,
                    "type": "activity",
                    "value": state.get("primitive_state", "unknown"),
                    "confidence": state.get("confidence", 5),
                    "body_position": state.get("body_position"),
                    "objects": state.get("objects_interacted", []),
                }
                viz_data["timeline"].append(entry)
            else:
                entry = {
                    "frame": idx,
                    "pass": pass_num,
                    "type": "intent",
                    "value": state.get("hidden_intent", "unknown"),
                    "confidence": state.get("intent_confidence", 5),
                    "supporting": state.get("supporting_primitives", []),
                    "alternatives": state.get("alternative_intents", []),
                }
                viz_data["timeline"].append(entry)

    # Build Sankey diagram data for transitions (using state-sets)
    # Use the latest activity pass
    activity_passes = [p for p in all_pass_states.keys() if p % 2 == 1]
    if activity_passes:
        latest_act = max(activity_passes)
        states = all_pass_states[latest_act]

        # Build nodes (unique states)
        all_state_names = set()
        for s in states:
            all_state_names.add(s.get("primitive_state", "unknown"))
        unique_states = sorted(all_state_names)
        viz_data["sankey"]["nodes"] = [{"name": s} for s in unique_states]

        # Build links (transitions between primary states)
        state_to_idx = {s: i for i, s in enumerate(unique_states)}
        transitions: Dict[Tuple[int, int], int] = defaultdict(int)

        for i in range(len(states) - 1):
            from_s = states[i].get("primitive_state", "unknown")
            to_s = states[i + 1].get("primitive_state", "unknown")
            if from_s != to_s:
                transitions[(state_to_idx[from_s], state_to_idx[to_s])] += 1

        viz_data["sankey"]["links"] = [
            {"source": src, "target": tgt, "value": count}
            for (src, tgt), count in transitions.items()
        ]

    # Build confidence matrix (pass x frame)
    for pass_num, states in sorted(all_pass_states.items()):
        pass_type = "activity" if pass_num % 2 == 1 else "intent"
        conf_key = "confidence" if pass_type == "activity" else "intent_confidence"

        confidences = [s.get(conf_key, 5) for s in states]
        viz_data["confidence_matrix"].append({
            "pass": pass_num,
            "type": pass_type,
            "confidences": confidences,
            "avg": round(sum(confidences) / len(confidences), 2) if confidences else 0,
        })

    # Build pass comparison data
    for pass_num, states in sorted(all_pass_states.items()):
        pass_type = "activity" if pass_num % 2 == 1 else "intent"

        if pass_type == "activity":
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
