from src.matcher import evaluate_condition, match_sops


def test_evaluate_condition():
    assert evaluate_condition(">", 40, 50) is True
    assert evaluate_condition(">", 40, 30) is False
    assert evaluate_condition(">=", 70, 70) is True
    assert evaluate_condition("between", [18, 28], 22) is True
    assert evaluate_condition("between", [18, 28], 30) is False


def test_match_sops_basic():
    sops = [
        {
            "id": "SOP-TEST-1",
            "severity": 2,
            "priority": 50,
            "conditions": {
                "activities": ["cycling"],
                "weather": {
                    "wind_speed_10m": {
                        "operator": ">",
                        "value": 40
                    }
                }
            }
        }
    ]

    intent = {"activity": "cycling"}
    weather = {"wind_speed_10m": 50.0}

    matched, selected, reason = match_sops(
        sops,
        intent,
        weather
    )

    assert "SOP-TEST-1" in matched
    assert selected == "SOP-TEST-1"


def test_match_sops_resolution():
    sops = [
        {
            "id": "SOP-LOW",
            "severity": 1,
            "priority": 10,
            "conditions": {
                "activities": ["any"],
                "weather": {
                    "wind_speed_10m": {
                        "operator": ">",
                        "value": 20
                    }
                }
            }
        },
        {
            "id": "SOP-HIGH",
            "severity": 3,
            "priority": 80,
            "conditions": {
                "activities": ["cycling"],
                "weather": {
                    "wind_speed_10m": {
                        "operator": ">",
                        "value": 40
                    }
                }
            }
        }
    ]

    intent = {"activity": "cycling"}
    weather = {"wind_speed_10m": 50.0}

    matched, selected, reason = match_sops(
        sops,
        intent,
        weather
    )

    assert "SOP-LOW" in matched
    assert "SOP-HIGH" in matched
    assert selected == "SOP-HIGH"


def test_match_sops_no_match():
    sops = [
        {
            "id": "SOP-RAIN",
            "severity": 2,
            "priority": 50,
            "conditions": {
                "activities": ["any"],
                "weather": {
                    "precipitation": {
                        "operator": ">",
                        "value": 50
                    }
                }
            }
        }
    ]

    intent = {"activity": "cycling"}
    weather = {"precipitation": 0.0}

    matched, selected, reason = match_sops(
        sops,
        intent,
        weather
    )

    assert not matched
    assert selected is None


def test_match_sops_qualitative_fallback():
    """
    A qualitative activity policy with fallback=True should
    apply when no specific weather policy matches.
    """

    sops = [
        {
            "id": "SOP-QUALITATIVE",
            "severity": 1,
            "priority": 10,
            "conditions": {
                "activities": ["picnic"],
                "fallback": True
            }
        }
    ]

    intent = {"activity": "picnic"}

    weather = {
        "temperature_2m": 30.0,
        "wind_speed_10m": 15.0,
        "precipitation": 0.0,
        "precipitation_probability": 50,
        "uv_index": 6.0,
        "cloud_cover": 50.0
    }

    matched, selected, reason = match_sops(
        sops,
        intent,
        weather
    )

    assert "SOP-QUALITATIVE" in matched
    assert selected == "SOP-QUALITATIVE"


def test_specific_weather_policy_overrides_fallback():
    """
    A specific weather safety policy must take precedence over
    a qualitative fallback policy.
    """

    sops = [
        {
            "id": "SOP-QUALITATIVE",
            "severity": 1,
            "priority": 10,
            "conditions": {
                "activities": ["picnic"],
                "fallback": True
            }
        },
        {
            "id": "SOP-HIGH-WIND",
            "severity": 3,
            "priority": 80,
            "conditions": {
                "activities": ["any"],
                "weather": {
                    "wind_speed_10m": {
                        "operator": ">",
                        "value": 40
                    }
                }
            }
        }
    ]

    intent = {"activity": "picnic"}

    weather = {
        "temperature_2m": 30.0,
        "wind_speed_10m": 45.0,
        "precipitation": 0.0,
        "precipitation_probability": 10,
        "uv_index": 5.0,
        "cloud_cover": 30.0
    }

    matched, selected, reason = match_sops(
        sops,
        intent,
        weather
    )

    assert "SOP-HIGH-WIND" in matched
    assert "SOP-QUALITATIVE" not in matched
    assert selected == "SOP-HIGH-WIND"


def test_specific_picnic_policy_overrides_fallback():
    """
    When the dedicated picnic weather policy applies, the
    fallback picnic policy should not be selected.
    """

    sops = [
        {
            "id": "SOP-PICNIC-GOOD",
            "severity": 1,
            "priority": 90,
            "conditions": {
                "activities": ["picnic"],
                "weather": {
                    "precipitation_probability": {
                        "operator": "<",
                        "value": 20
                    },
                    "temperature_2m": {
                        "operator": "between",
                        "value": [18, 28]
                    },
                    "wind_speed_10m": {
                        "operator": "<",
                        "value": 20
                    }
                }
            }
        },
        {
            "id": "SOP-PICNIC-FALLBACK",
            "severity": 1,
            "priority": 10,
            "conditions": {
                "activities": ["picnic"],
                "fallback": True
            }
        }
    ]

    intent = {"activity": "picnic"}

    weather = {
        "temperature_2m": 24.0,
        "wind_speed_10m": 8.0,
        "precipitation": 0.0,
        "precipitation_probability": 10,
        "uv_index": 5.0,
        "cloud_cover": 30.0
    }

    matched, selected, reason = match_sops(
        sops,
        intent,
        weather
    )

    assert "SOP-PICNIC-GOOD" in matched
    assert "SOP-PICNIC-FALLBACK" not in matched
    assert selected == "SOP-PICNIC-GOOD"