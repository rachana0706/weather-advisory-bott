from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
import operator
from typing import Annotated

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_location: Optional[str]
    current_activity: Optional[str]
    time_reference: Optional[str]
    vulnerable_group: Optional[str]
    weather_data: Optional[Dict[str, Any]]
    weather_error: Optional[str]
    matched_sops: List[str]
    selected_sop: Optional[str]
    decision_reason: Optional[str]
