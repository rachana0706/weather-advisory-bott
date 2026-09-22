
import json
from typing import Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

from src.state import AgentState
from src.weather import get_weather_for_location
from src.matcher import load_sops, match_sops


class ExtractionSchema(BaseModel):
    location: Optional[str] = Field(default=None, description="The city or location mentioned by the user. If no location is mentioned but one is contextually implied, extract it. Leave null if absolutely no location is present.")
    activity: Optional[str] = Field(default=None, description="The outdoor activity the user is asking about (e.g., cycling, travel, picnic, walking, running). Leave null if none.")
    time_reference: Optional[str] = Field(default=None, description="The time the user is asking about (e.g., today, tomorrow, this evening). Leave null if none.")
    vulnerable_group: Optional[str] = Field(default=None, description="Any vulnerable group mentioned (e.g., elderly, children, pets). Leave null if none.")

def get_llm():
    return ChatOllama(
        model="llama2:latest",
        temperature=0,
    )

def extract_intent(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()

    messages = state.get("messages", [])

    if messages:
        latest_message = messages[-1]
    else:
        latest_message = HumanMessage(content="")

    system_msg = SystemMessage(
    content=(
        "You extract information from a user's weather question.\n\n"
        "Extract these four fields:\n"
        "1. location: the city or place mentioned by the user.\n"
        "2. activity: the outdoor activity mentioned by the user.\n"
        "3. time_reference: when the user wants the weather, such as today or tomorrow.\n"
        "4. vulnerable_group: elderly, children, pets, or null if none.\n\n"
        "IMPORTANT:\n"
        "- Extract explicit words from the user's message.\n"
        "- If the user says 'Bangalore', location MUST be 'Bangalore'.\n"
        "- If the user says 'tomorrow', time_reference MUST be 'tomorrow'.\n"
        "- If the user says 'cycling', activity MUST be 'cycling'.\n"
        "- Do not invent information.\n"
        "- Return ONLY valid JSON.\n"
        "- Do not include explanations or markdown.\n\n"
        "Example:\n"
        "User: Can I go cycling in Bangalore tomorrow?\n"
        'Output: {"location":"Bangalore","activity":"cycling",'
        '"time_reference":"tomorrow","vulnerable_group":null}'
    )
)

    full_messages = [system_msg, latest_message]

    response = llm.invoke(full_messages)
    response_text = response.content.strip()

    # Remove Markdown code fences if Llama adds them.
    if response_text.startswith("```"):
        lines = response_text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response_text = "\n".join(lines).strip()

    # Extract the JSON object even if Llama adds extra text.
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in Llama response.")

        json_text = response_text[start:end + 1]
        parsed_data = json.loads(json_text)

    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Llama intent extraction returned invalid JSON. "
            f"Raw response: {response_text!r}"
        ) from exc

    expected_fields = {
        "location",
        "activity",
        "time_reference",
        "vulnerable_group",
    }

    if set(parsed_data) != expected_fields:
        raise ValueError(
            "Llama intent extraction returned invalid JSON: "
            "the object must contain exactly the fields "
            "location, activity, time_reference, and vulnerable_group."
        )

    extracted = ExtractionSchema.model_validate(parsed_data)

    return {
        "current_location": extracted.location or state.get("current_location"),
        "current_activity": extracted.activity or state.get("current_activity"),
        "time_reference": extracted.time_reference or state.get("time_reference"),
        "vulnerable_group": extracted.vulnerable_group or state.get("vulnerable_group"),
    }
def fetch_weather_node(state: AgentState) -> Dict[str, Any]:
    location = state.get("current_location")
    time_reference = state.get("time_reference")
    if not location:
        return {"weather_error": "No location provided."}
        
    weather_data, err = get_weather_for_location(location, time_reference)
    if err:
        return {"weather_error": err, "weather_data": None}
        
    return {"weather_data": weather_data, "weather_error": None}

def ask_for_location(state: AgentState) -> Dict[str, Any]:
    msg = "I need a location to check the weather. Where are you asking about?"
    return {"messages": [AIMessage(content=msg)]}

def api_failure_response(state: AgentState) -> Dict[str, Any]:
    err = state.get("weather_error")
    msg = f"I'm sorry, but I couldn't fetch the weather data: {err}"
    return {"messages": [AIMessage(content=msg)]}

def match_sop_node(state: AgentState) -> Dict[str, Any]:
    sops = load_sops()
    intent = {
        "activity": state.get("current_activity"),
        "vulnerable_group": state.get("vulnerable_group")
    }
    weather_data = state.get("weather_data")
    
    if not weather_data:
         return {"matched_sops": [], "selected_sop": None, "decision_reason": "No weather data"}
         
    matched_ids, selected_id, reason = match_sops(sops, intent, weather_data)
    
    return {
        "matched_sops": matched_ids,
        "selected_sop": selected_id,
        "decision_reason": reason
    }

def generate_fact_based_advice(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    selected_id = state.get("selected_sop")
    sops = load_sops()
    sop = next((s for s in sops if s["id"] == selected_id), None)
    
    weather_data = state.get("weather_data")
    
    system_prompt = f"""You are a safety advisory bot. You MUST compose your response using ONLY the provided facts.
Do NOT invent weather values. Do NOT invent policies or safety thresholds.

FACTS:
- Weather data: {weather_data}
- Applicable Policy ID: {selected_id}
- Policy Guidance: {sop['guidance'] if sop else 'None'}

Instructions:
Write a helpful, empathetic response to the user.
You MUST explicitly state the weather values from the FACTS that triggered this policy.
You MUST explicitly provide the guidance from the policy.
You MUST explicitly cite the Policy ID.
"""
    messages = state.get("messages", [])
    full_messages = [SystemMessage(content=system_prompt)] + messages
    
    response = llm.invoke(full_messages)
    return {"messages": [response]}

def no_guidance_response(state: AgentState) -> Dict[str, Any]:
    weather_data = state.get("weather_data", {})

    location = state.get("current_location") or "your location"
    activity = state.get("current_activity") or "this activity"
    time_reference = state.get("time_reference")

    if time_reference:
        forecast_context = f"{location} {time_reference}"
    else:
        forecast_context = location

    temperature = weather_data.get("temperature_2m")
    wind_speed = weather_data.get("wind_speed_10m")
    precipitation = weather_data.get("precipitation")
    precipitation_probability = weather_data.get(
        "precipitation_probability"
    )
    uv_index = weather_data.get("uv_index")

    lines = [
        f"Based on the forecast for {forecast_context}, "
        f"there is no specific weather safety warning triggered "
        f"for {activity}."
    ]

    if temperature is not None:
        lines.append(f"🌡️ Temperature: {temperature}°C")

    if wind_speed is not None:
        lines.append(f"💨 Wind speed: {wind_speed} km/h")

    if precipitation is not None:
        lines.append(f"🌧️ Precipitation: {precipitation} mm")

    if precipitation_probability is not None:
        lines.append(
            f"🌧️ Precipitation probability: "
            f"{precipitation_probability}%"
        )

    if uv_index is not None:
        lines.append(f"☀️ UV index: {uv_index}")

    lines.append(
        f"\nThese conditions do not currently trigger any of "
        f"the bot's specific safety advisories for {activity}."
    )

    response = "\n\n".join(lines)

    return {
        "messages": [AIMessage(content=response)]
    }