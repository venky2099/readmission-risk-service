"""Turn dataset rows into API payloads (used by the benchmark, the robustness checks and the tests)."""
import math

DROP = {"patient_nbr", "weight", "payer_code", "readmitted", "examide", "citoglipton", "y", "split"}
INT_FIELDS = {"encounter_id", "admission_type_id", "discharge_disposition_id", "admission_source_id", "time_in_hospital",
              "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency",
              "number_inpatient", "number_diagnoses"}


def row_to_payload(row):
    out = {}
    for k, v in dict(row).items():
        if k in DROP:
            continue
        if isinstance(v, float) and math.isnan(v):
            v = None
        if k in INT_FIELDS and v is not None:
            v = int(v)
        out[k] = v
    return out
