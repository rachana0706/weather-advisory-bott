import yaml
from typing import List, Dict, Any, Tuple, Optional


def load_sops(filepath: str = "sops.yaml") -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def evaluate_condition(operator: str, threshold: Any, actual_value: Any) -> bool:
    if actual_value is None:
        return False

    if operator == ">":
        return actual_value > threshold
    elif operator == "<":
        return actual_value < threshold
    elif operator == ">=":
        return actual_value >= threshold
    elif operator == "<=":
        return actual_value <= threshold
    elif operator == "==":
        return actual_value == threshold
    elif operator == "between":
        return threshold[0] <= actual_value <= threshold[1]

    return False


def _matches_conditions(
    sop: Dict[str, Any],
    activity: str,
    vulnerable_group: str,
    weather_data: Dict[str, Any]
) -> bool:
    """
    Generic SOP condition evaluator.

    It evaluates activity, vulnerable-group, and weather
    conditions without knowing any specific SOP ID.
    """

    conditions = sop.get("conditions", {})

    # Activity matching
    allowed_activities = conditions.get("activities", ["any"])

    if (
        "any" not in allowed_activities
        and activity not in allowed_activities
    ):
        return False

    # Vulnerable-group matching
    if "vulnerable_groups" in conditions:
        allowed_groups = conditions["vulnerable_groups"]

        if (
            "any" not in allowed_groups
            and vulnerable_group not in allowed_groups
        ):
            return False

    # Weather matching
    weather_conditions = conditions.get("weather", {})

    for metric, criteria in weather_conditions.items():

        if metric not in weather_data:
            return False

        operator = criteria.get("operator")
        value = criteria.get("value")
        actual_value = weather_data.get(metric)

        if not evaluate_condition(operator, value, actual_value):
            return False

    return True


def match_sops(
    sops: List[Dict[str, Any]],
    intent: Dict[str, Any],
    weather_data: Dict[str, Any]
) -> Tuple[List[str], Optional[str], Optional[str]]:

    normal_matches = []
    fallback_matches = []

    activity = intent.get("activity") or "any"
    vulnerable_group = intent.get("vulnerable_group") or "none"

    # Evaluate every SOP
    for sop in sops:

        if not _matches_conditions(
            sop,
            activity,
            vulnerable_group,
            weather_data
        ):
            continue

        conditions = sop.get("conditions", {})

        # Fallback policies are kept separate.
        if conditions.get("fallback", False):
            fallback_matches.append(sop)
        else:
            normal_matches.append(sop)

    # Use specific policies whenever at least one applies.
    # Only use fallback policies when no specific policy applies.
    if normal_matches:
        matched_sops = normal_matches
    else:
        matched_sops = fallback_matches

    # No applicable policy
    if not matched_sops:
        return (
            [],
            None,
            "No SOP conditions were met by current weather and intent."
        )

    # Conflict resolution:
    # 1. Highest severity
    # 2. Highest priority
    matched_sops.sort(
        key=lambda x: (
            x.get("severity", 0),
            x.get("priority", 0)
        ),
        reverse=True
    )

    selected_sop = matched_sops[0]

    matched_ids = [s["id"] for s in matched_sops]
    selected_id = selected_sop["id"]

    reason = (
        f"Selected {selected_id} due to highest severity "
        f"({selected_sop.get('severity')}) and priority "
        f"({selected_sop.get('priority')}) among "
        f"{len(matched_ids)} applicable matches."
    )

    return matched_ids, selected_id, reason