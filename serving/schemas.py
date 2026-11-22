"""Pydantic schemas for the /predict endpoint.

Two kinds of checks:
* Hard validation (request rejected with HTTP 422): wrong types, impossible values (negative counts, absurd sizes),
  values outside closed vocabularies (medication status, gender, test results), unknown field names.
* Soft handling (request accepted, warning returned): open-vocabulary fields such as medical specialty or ICD-9
  codes may carry values never seen in training; numeric values above the training range are clipped.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

MedStatus = Literal["No", "Steady", "Up", "Down"]


class Encounter(BaseModel):
    """One inpatient encounter at the moment the discharge order is signed."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    encounter_id: Optional[int] = Field(None, description="Hospital encounter id, echoed back for logging")
    race: Optional[str] = Field(None, max_length=40)
    gender: Literal["Female", "Male", "Unknown/Invalid"]
    age: str = Field(..., pattern=r"^\[\d{1,3}-\d{1,3}\)$", description="Ten-year bracket, e.g. '[70-80)'")
    admission_type_id: int = Field(..., ge=1, le=99)
    discharge_disposition_id: int = Field(..., ge=1, le=99)
    admission_source_id: int = Field(..., ge=1, le=99)
    time_in_hospital: int = Field(..., ge=1, le=60, description="Days")
    medical_specialty: Optional[str] = Field(None, max_length=60)
    num_lab_procedures: int = Field(..., ge=0, le=1000)
    num_procedures: int = Field(..., ge=0, le=100)
    num_medications: int = Field(..., ge=0, le=500)
    number_outpatient: int = Field(0, ge=0, le=500)
    number_emergency: int = Field(0, ge=0, le=500)
    number_inpatient: int = Field(0, ge=0, le=500)
    diag_1: Optional[str] = Field(None, max_length=10)
    diag_2: Optional[str] = Field(None, max_length=10)
    diag_3: Optional[str] = Field(None, max_length=10)
    number_diagnoses: int = Field(..., ge=0, le=100)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    metformin: MedStatus = "No"
    repaglinide: MedStatus = "No"
    nateglinide: MedStatus = "No"
    chlorpropamide: MedStatus = "No"
    glimepiride: MedStatus = "No"
    acetohexamide: MedStatus = "No"
    glipizide: MedStatus = "No"
    glyburide: MedStatus = "No"
    tolbutamide: MedStatus = "No"
    pioglitazone: MedStatus = "No"
    rosiglitazone: MedStatus = "No"
    acarbose: MedStatus = "No"
    miglitol: MedStatus = "No"
    troglitazone: MedStatus = "No"
    tolazamide: MedStatus = "No"
    insulin: MedStatus = "No"
    glyburide_metformin: MedStatus = Field("No", alias="glyburide-metformin")
    glipizide_metformin: MedStatus = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: MedStatus = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedStatus = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedStatus = Field("No", alias="metformin-pioglitazone")
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "No"


class Prediction(BaseModel):
    encounter_id: Optional[int]
    eligible: bool = Field(..., description="False when the patient died or went to hospice: no follow-up call applies")
    risk_score: Optional[float] = Field(None, description="Model score; higher means higher 30-day readmission risk")
    flag_for_follow_up: Optional[bool]
    threshold: float
    capacity: float
    source: Literal["model", "fallback_rule", "not_applicable"]
    model_version: str
    warnings: List[str] = []


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    threshold: float
    capacity: float
