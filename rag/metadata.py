def infer_metadata(doc_name: str) -> dict:
    """
    General metadata inferred from filename.
    You can extend this anytime (manual mapping, better rules, etc.).
    """
    base = doc_name.lower().replace(".pdf", "")
    base = base.replace("-", "_")

    # Year (optional)
    year = int(doc_name[:4]) if len(doc_name) >= 4 and doc_name[:4].isdigit() else None

    # Topics (general tags)
    topics = []
    rules = {
        # --- Telemedicine / Telehealth ---
        "telemedicine": "telemedicine",
        "telehealth": "telemedicine",
        "tele_med": "telemedicine",
        "tele_health": "telemedicine",
        "virtual_care": "telemedicine",
        "telecare": "telemedicine",

        # --- Prior authorization / utilization management ---
        "priorauthorization": "prior-authorization",
        "prior_authorization": "prior-authorization",
        "utilizationmanagement": "utilization-management",
        "utilization_management": "utilization-management",

        # --- HIS / informatics / interoperability ---
        "interoperability": "interoperability",
        "informatics": "medical-informatics",
        "healthinformationsystems": "health-information-systems",
        "health_information_systems": "health-information-systems",
        "healthis": "health-information-systems",
        "ehr": "ehr",
        "electronic_health_record": "ehr",
        "electronic_health_records": "ehr",

        # --- Public health ---
        "public_health": "public-health",
        "surveillance": "public-health-surveillance",

        # --- Other ---
        "robotics": "robotics",
        "robot": "robotics",
        "aging": "aging-in-place",
        "patient_journey": "patient-journey",
        "journey_mapping": "patient-journey",
        "person_centered": "person-centered-care",
        "patient_experience": "patient-experience",
        "service_design": "service-design",
        "dataquality": "data-quality",
        "data_quality": "data-quality",
        "quality": "quality",
        "ebm": "evidence-based-medicine",
        "sweden": "sweden",
    }

    for key, tag in rules.items():
        if key in base:
            topics.append(tag)

    topics = sorted(set(topics))

    # Category (coarse grouping)
    category = "general"
    if any(t in topics for t in ["telemedicine", "robotics", "aging-in-place"]):
        category = "technology"
    if any(t in topics for t in ["health-information-systems", "medical-informatics", "interoperability", "data-quality", "ehr"]):
        category = "health-informatics"
    if any(t in topics for t in ["prior-authorization", "utilization-management", "public-health-surveillance", "public-health"]):
        category = "policy"
    if any(t in topics for t in ["person-centered-care", "patient-experience", "patient-journey", "service-design"]):
        category = "service-design"

    return {
        "year": year,
        "topics": topics,
        "category": category,
    }

