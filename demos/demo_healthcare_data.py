import os
from datetime import datetime, timezone

# =============================================================================
# Shared Demo Scope & Data Configuration
# =============================================================================

# Shared Run ID to keep memory consistent across both scripts if they run sequentially.
# It defaults to a fixed prefix with today's date if not specified.
DEMO_RUN_ID = os.getenv("GM_HEALTHCARE_DEMO_RUN_ID") or (
    f"healthcare-multi-patient-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
)

BASE_SCOPE = {
    "tenant_id": os.getenv("GM_SCOPE_TENANT_ID", "demo-tenant"),
    "app_id": os.getenv("GM_SCOPE_APP_ID", "ground-memory-core"),
    "agent_id": os.getenv("GM_SCOPE_AGENT_ID", "healthcare-demo-agent"),
    "run_id": DEMO_RUN_ID,
    "space_type": os.getenv("GM_SCOPE_SPACE_TYPE", "user"),
}

PATIENTS = [
    {
        "name": "John Doe",
        "mrn": "JD-001",
        "interactions": [
            "Patient John Doe, MRN JD-001, has a severe Penicillin allergy with anaphylaxis.",
            "Prescribe Lisinopril 10mg daily for patient John Doe, MRN JD-001.",
            "Adjust Lisinopril to 20mg daily for patient John Doe, MRN JD-001.",
            "Continue Warfarin 5mg daily for patient John Doe, MRN JD-001.",
            "Prescribe Amiodarone 200mg daily for patient John Doe, MRN JD-001.",
            "Prescribe Penicillin 500mg daily for patient John Doe, MRN JD-001.",
            "Discontinue Lisinopril for patient John Doe, MRN JD-001.",
        ],
    },
    {
        "name": "Alice Johnson",
        "mrn": "AJ-002",
        "interactions": [
            "Patient Alice Johnson, MRN AJ-002, is allergic to Sulfa drugs.",
            "Prescribe Metformin 500mg twice daily for patient Alice Johnson, MRN AJ-002.",
            "Prescribe Furosemide 40mg daily for patient Alice Johnson, MRN AJ-002.",
            "Prescribe Trimethoprim-sulfamethoxazole 800/160mg daily for patient Alice Johnson, MRN AJ-002.",
            "Adjust Metformin to 1000mg twice daily for patient Alice Johnson, MRN AJ-002.",
        ],
    },
    {
        "name": "Robert Chen",
        "mrn": "RC-003",
        "interactions": [
            "Patient Robert Chen, MRN RC-003, has Type 2 diabetes.",
            "Prescribe Insulin glargine 20 units at bedtime for patient Robert Chen, MRN RC-003.",
            "Prescribe Simvastatin 20mg daily for patient Robert Chen, MRN RC-003.",
            "Prescribe Clarithromycin 500mg twice daily for patient Robert Chen, MRN RC-003.",
            "Discontinue Simvastatin for patient Robert Chen, MRN RC-003.",
        ],
    },
    {
        "name": "Maria Garcia",
        "mrn": "MG-004",
        "interactions": [
            "Patient Maria Garcia, MRN MG-004, is allergic to NSAIDs.",
            "Prescribe Losartan 50mg daily for patient Maria Garcia, MRN MG-004.",
            "Prescribe Ibuprofen 400mg every 6 hours for patient Maria Garcia, MRN MG-004.",
            "Prescribe Amlodipine 5mg daily for patient Maria Garcia, MRN MG-004.",
            "Adjust Losartan to 100mg daily for patient Maria Garcia, MRN MG-004.",
        ],
    },
    {
        "name": "James Taylor",
        "mrn": "JT-005",
        "interactions": [
            "Patient James Taylor, MRN JT-005, has a severe Penicillin allergy with anaphylaxis.",
            "Patient James Taylor, MRN JT-005, has atrial fibrillation.",
            "Prescribe Digoxin 0.125mg daily for patient James Taylor, MRN JT-005.",
            "Prescribe Verapamil 120mg daily for patient James Taylor, MRN JT-005.",
            "Discontinue Verapamil for patient James Taylor, MRN JT-005.",
        ],
    },
    {
        "name": "Emma Davis",
        "mrn": "ED-006",
        "interactions": [
            "Patient Emma Davis, MRN ED-006, suffers from chronic pain and depression.",
            "Prescribe Metformin 500mg twice daily for patient Emma Davis, MRN ED-006.",
            "Prescribe Tramadol 50mg every 6 hours as needed for patient Emma Davis, MRN ED-006.",
            "Prescribe Sertraline 50mg daily for patient Emma Davis, MRN ED-006.",
            "Discontinue Tramadol for patient Emma Davis, MRN ED-006.",
        ],
    },
    {
        "name": "Michael Smith",
        "mrn": "MS-007",
        "interactions": [
            "Patient Michael Smith, MRN MS-007, recently had a stent placed.",
            "Prescribe Clopidogrel 75mg daily for patient Michael Smith, MRN MS-007.",
            "Prescribe Omeprazole 20mg daily for reflux for patient Michael Smith, MRN MS-007.",
            "Switch Omeprazole to Pantoprazole 40mg daily for patient Michael Smith, MRN MS-007.",
        ],
    },
    {
        "name": "Sarah Wilson",
        "mrn": "SW-008",
        "interactions": [
            "Patient Sarah Wilson, MRN SW-008, presents with pulmonary hypertension.",
            "Prescribe Sildenafil 20mg three times daily for patient Sarah Wilson, MRN SW-008.",
            "Prescribe Nitroglycerin 0.4mg sublingual PRN chest pain for patient Sarah Wilson, MRN SW-008.",
        ],
    },
    {
        "name": "David Miller",
        "mrn": "DM-009",
        "interactions": [
            "Patient David Miller, MRN DM-009, has high cholesterol.",
            "Prescribe Amlodipine 5mg daily for patient David Miller, MRN DM-009.",
            "Prescribe Simvastatin 40mg daily for patient David Miller, MRN DM-009.",
            "Prescribe Amiodarone 200mg daily for patient David Miller, MRN DM-009.",
        ],
    },
    {
        "name": "Linda Martinez",
        "mrn": "LM-010",
        "interactions": [
            "Patient Linda Martinez, MRN LM-010, is severely allergic to cephalosporins.",
            "Prescribe Ceftriaxone 1g IV daily for patient Linda Martinez, MRN LM-010.",
            "Change Ceftriaxone to Azithromycin 500mg daily for patient Linda Martinez, MRN LM-010.",
        ],
    },
]
