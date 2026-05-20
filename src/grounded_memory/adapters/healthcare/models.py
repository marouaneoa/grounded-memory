"""
Healthcare Extraction Models

Defines the expected structured output schema from the LLM for the
clinical domain.
"""

from pydantic import BaseModel, Field


class ExtractedMedicationLLM(BaseModel):
    """Medication information extracted by LLM."""

    name: str = Field(..., description="Exact medication name as mentioned in text")
    dosage: str | None = Field(
        None, description="Dosage amount if mentioned (e.g., '500mg', '10mg')"
    )
    frequency: str | None = Field(
        None, description="Frequency if mentioned (e.g., 'twice daily', 'every 8 hours')"
    )
    route: str | None = Field(
        None, description="Route of administration (e.g., 'oral', 'IV', 'topical')"
    )
    duration: str | None = Field(
        None, description="Duration if mentioned (e.g., '7 days', '2 weeks')"
    )
    action: str = Field(
        default="prescribe",
        description="Clinical action: 'prescribe', 'discontinue', 'continue', 'adjust', 'hold'",
    )
    confidence: float = Field(
        default=0.9,
        description="Extraction confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


class ExtractedPatientLLM(BaseModel):
    """Patient information extracted by LLM."""

    name: str = Field(..., description="Patient's name as mentioned in text")
    identifier: str | None = Field(
        None, description="Patient identifier if mentioned (MRN, room number)"
    )
    age: str | None = Field(None, description="Patient age if mentioned")
    gender: str | None = Field(None, description="Patient gender if mentioned")


class ExtractedAllergyLLM(BaseModel):
    """Allergy information extracted by LLM."""

    allergen: str = Field(..., description="Name of the allergen (medication, food, substance)")
    reaction: str | None = Field(None, description="Type of allergic reaction if mentioned")
    severity: str | None = Field(
        None, description="Severity if mentioned (mild, moderate, severe, anaphylaxis)"
    )


class ExtractedConditionLLM(BaseModel):
    """Medical condition extracted by LLM."""

    name: str = Field(..., description="Name of the condition or diagnosis")
    status: str | None = Field(None, description="Status if mentioned (active, resolved, chronic)")
    diagnosed_date: str | None = Field(None, description="When diagnosed if mentioned")


class ClinicalExtractionResult(BaseModel):
    """Complete extraction result from clinical text."""

    patient: ExtractedPatientLLM | None = Field(
        None, description="Patient information if identifiable"
    )
    medications: list[ExtractedMedicationLLM] = Field(
        default_factory=list, description="All medications mentioned with their actions"
    )
    allergies: list[ExtractedAllergyLLM] = Field(
        default_factory=list, description="Any allergies mentioned"
    )
    conditions: list[ExtractedConditionLLM] = Field(
        default_factory=list, description="Medical conditions or diagnoses mentioned"
    )
    clinical_intent: str | None = Field(None, description="The primary clinical intent of the text")
    extraction_notes: str | None = Field(
        None, description="Any notes about the extraction quality or ambiguity"
    )
