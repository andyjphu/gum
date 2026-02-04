# schemas.py

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# =============================================================================
# STATE EXTRACTION SCHEMAS (for egocentric footage)
# =============================================================================

# Pass type enumeration for alternating primitive/intent passes
PassType = Literal["primitive", "intent"]


class PrimitiveStateSchema(BaseModel):
    """
    Output from primitive state extraction (odd passes: 1, 3, 5).
    Captures observable physical/behavioral actions.
    """
    primitive_state: str = Field(
        ...,
        description="Observable action (e.g., 'walking', 'typing', 'pointing_at_screen')"
    )
    body_position: Optional[str] = Field(
        None,
        description="Body posture/position if observable (e.g., 'standing', 'sitting', 'leaning')"
    )
    objects_interacted: Optional[List[str]] = Field(
        default_factory=list,
        description="Objects the user is interacting with (e.g., ['phone', 'coffee_cup'])"
    )
    confidence: int = Field(..., ge=1, le=10, description="Confidence score from 1 (low) to 10 (high)")

    model_config = ConfigDict(extra="allow")


class HiddenIntentSchema(BaseModel):
    """
    Output from hidden intent inference (even passes: 2, 4, 6).
    Infers underlying goals/intentions from primitive states.
    """
    hidden_intent: str = Field(
        ...,
        description="Inferred intent/goal (e.g., 'commuting_to_work', 'preparing_meal_for_family')"
    )
    supporting_primitives: List[str] = Field(
        default_factory=list,
        description="Primitive states that support this intent inference"
    )
    intent_confidence: int = Field(
        ..., ge=1, le=10,
        description="Confidence in the intent inference (1=speculative, 10=highly certain)"
    )
    alternative_intents: Optional[List[str]] = Field(
        default_factory=list,
        description="Other possible intents if ambiguous"
    )

    model_config = ConfigDict(extra="allow")


class RefinedPrimitiveSchema(BaseModel):
    """
    Output from refined primitive state extraction (passes 3, 5, ...).
    Refines primitives with context from prior passes.
    """
    primitive_state: str = Field(
        ...,
        description="Refined observable action"
    )
    body_position: Optional[str] = Field(None, description="Body posture/position")
    objects_interacted: Optional[List[str]] = Field(default_factory=list)
    confidence: int = Field(..., ge=1, le=10)
    refined_from: Optional[str] = Field(
        None,
        description="Original primitive if this was refined/collapsed"
    )
    refinement_reason: Optional[str] = Field(
        None,
        description="Why this primitive was refined (e.g., 'collapsed with walking_slowly')"
    )
    informed_by_intent: Optional[str] = Field(
        None,
        description="Hidden intent that informed this refinement"
    )

    model_config = ConfigDict(extra="allow")


class RefinedIntentSchema(BaseModel):
    """
    Output from refined intent inference (passes 4, 6, ...).
    Refines intents with broader context.
    """
    hidden_intent: str = Field(..., description="Refined inferred intent/goal")
    supporting_primitives: List[str] = Field(default_factory=list)
    intent_confidence: int = Field(..., ge=1, le=10)
    alternative_intents: Optional[List[str]] = Field(default_factory=list)
    refined_from: Optional[str] = Field(
        None,
        description="Original intent if refined"
    )
    refinement_reason: Optional[str] = Field(None)
    validated_by_outcome: Optional[bool] = Field(
        None,
        description="Whether subsequent frames validated or invalidated this intent"
    )

    model_config = ConfigDict(extra="allow")


# Legacy schemas for backward compatibility
class StateExtractionSchema(BaseModel):
    """
    Output from state extraction prompt (Pass 1).
    Single state per frame.
    DEPRECATED: Use PrimitiveStateSchema for new code.
    """
    state: str = Field(..., description="Compound state string (e.g., 'walking_while_on_phone')")
    confidence: int = Field(..., ge=1, le=10, description="Confidence score from 1 (low) to 10 (high)")

    model_config = ConfigDict(extra="allow")  # Extensible


class RefinedStateSchema(BaseModel):
    """
    Output from re-analysis prompt (Pass 2+).
    Single state per frame with optional collapse tracking.
    DEPRECATED: Use RefinedPrimitiveSchema or RefinedIntentSchema for new code.
    """
    state: str = Field(..., description="State string (may be collapsed from prior state)")
    confidence: int = Field(..., ge=1, le=10, description="Confidence score from 1 (low) to 10 (high)")
    collapsed_from: Optional[str] = Field(None, description="Original state if this was collapsed")
    collapse_reason: Optional[str] = Field(None, description="Reason for collapse if applicable")

    model_config = ConfigDict(extra="allow")  # Extensible for future fields


class PassSummary(BaseModel):
    """
    Summary of a single analysis pass.
    """
    pass_number: int = Field(..., description="Pass number (1, 2, ...)")
    num_states: int = Field(..., description="Number of unique states in this pass")
    num_collapsed: int = Field(0, description="Number of states collapsed in this pass")
    summary_text: str = Field(..., description="Text summary of this pass")
    state_counts: Dict[str, int] = Field(default_factory=dict, description="State occurrence counts")

    model_config = ConfigDict(extra="allow")


# =============================================================================
# AUDIT SCHEMA
# =============================================================================

class AuditSchema(BaseModel):
    """
    Output produced by the privacy-audit LLM call.
    """
    is_new_information: bool = Field(..., description="Whether the message reveals anything not seen before")
    data_type:          str  = Field(..., description="Category of data being disclosed")
    subject:            str  = Field(..., description="Who the data is about")
    recipient:          str  = Field(..., description="Who receives the data")
    transmit_data:      bool = Field(..., description="Should downstream processing continue")

    model_config = ConfigDict(extra="forbid")

class PropositionItem(BaseModel):
    reasoning: str = Field(..., description="The reasoning for the proposition")
    proposition: str = Field(..., description="The proposition string")
    confidence: Optional[int] = Field(
        ...,
        description="Confidence score from 1 (low) to 10 (high)"
    )
    decay: Optional[int] = Field(
        ...,
        description="Decay score from 1 (low) to 10 (high)"
    )

    model_config = ConfigDict(extra="forbid")

class PropositionSchema(BaseModel):
    propositions: List[PropositionItem] = Field(
        ...,
        description="Up to K propositions"
    )
    model_config = ConfigDict(extra="forbid")

class Update(BaseModel):
    content: str = Field(..., description="The content of the update")
    content_type: Literal["input_text", "input_image", "state_json", "collapsed_state_json"] = Field(..., description="The type of the update")

RelationLabel = Literal["IDENTICAL", "SIMILAR", "UNRELATED"]

class RelationItem(BaseModel):
    source: int                     = Field(description="Proposition ID")
    label:  RelationLabel           = Field(description="Relationship label")

    # give target a default_factory so the JSON‐schema default is [] (allowed)
    target: List[int] = Field(
        default_factory=list,
        description="IDs of other propositions (empty if none)"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["source", "label", "target"]
        }
    )


class RelationSchema(BaseModel):
    relations: List[RelationItem]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["relations"]
        }
    )

def get_schema(json_schema):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "json_output",
            "schema": json_schema,
        },
    }

UPDATE_MAP = {
    "input_text": "text",
    "input_image": "image_url",
}
