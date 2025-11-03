# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Event Ledger: Context-invariant event tracking for reward computation.

The event ledger maintains a complete history of environment interactions
that is independent of context management strategies (sliding window, compression, etc.).
This ensures that reward computation is based on the true state of the world,
not on what the policy can currently observe.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class EventType(Enum):
    """Types of events that can be recorded"""
    SEARCH = "search"
    INFORMATION = "information"  # Environment-provided information
    ANSWER = "answer"
    THINK = "think"
    THINK_SUMMARY = "think_summary"
    INFORMATION_SUMMARY = "information_summary"


@dataclass
class Event:
    """Single event in the trajectory"""
    turn_id: int  # Which turn this event occurred in (0-indexed)
    event_type: EventType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'turn_id': self.turn_id,
            'event_type': self.event_type.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        return cls(
            turn_id=data['turn_id'],
            event_type=EventType(data['event_type']),
            content=data['content'],
            timestamp=data.get('timestamp', time.time()),
            metadata=data.get('metadata', {})
        )


class EventLedger:
    """
    Event ledger maintains a complete, untruncated history of events.
    
    This is the source of truth for reward computation, independent of
    what context the policy actually sees during rollout.
    """
    
    def __init__(self, trajectory_id: Optional[str] = None):
        self.trajectory_id = trajectory_id or f"traj_{time.time()}"
        self.events: List[Event] = []
        self._turn_to_events: Dict[int, List[Event]] = {}
    
    def record_event(self, turn_id: int, event_type: EventType, 
                     content: str, metadata: Optional[Dict] = None):
        """Record a new event"""
        event = Event(
            turn_id=turn_id,
            event_type=event_type,
            content=content,
            metadata=metadata or {}
        )
        self.events.append(event)
        
        # Update index
        if turn_id not in self._turn_to_events:
            self._turn_to_events[turn_id] = []
        self._turn_to_events[turn_id].append(event)
    
    def record_search(self, turn_id: int, query: str, metadata: Optional[Dict] = None):
        """Record a search action"""
        self.record_event(turn_id, EventType.SEARCH, query, metadata)
    
    def record_information(self, turn_id: int, content: str, 
                          source: str = "environment", metadata: Optional[Dict] = None):
        """Record information received (from environment or model)"""
        meta = metadata or {}
        meta['source'] = source
        self.record_event(turn_id, EventType.INFORMATION, content, meta)
    
    def record_answer(self, turn_id: int, content: str, metadata: Optional[Dict] = None):
        """Record an answer action"""
        self.record_event(turn_id, EventType.ANSWER, content, metadata)
    
    def record_summary(self, turn_id: int, summary_type: str, content: str, 
                      metadata: Optional[Dict] = None):
        """Record a summary action (think_summary or information_summary)"""
        if summary_type == "think_summary":
            event_type = EventType.THINK_SUMMARY
        elif summary_type == "information_summary":
            event_type = EventType.INFORMATION_SUMMARY
        else:
            raise ValueError(f"Unknown summary type: {summary_type}")
        self.record_event(turn_id, event_type, content, metadata)
    
    def get_events_before_turn(self, turn_id: int, 
                               event_type: Optional[EventType] = None) -> List[Event]:
        """Get all events that occurred before a given turn"""
        events = [e for e in self.events if e.turn_id < turn_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events
    
    def get_events_at_turn(self, turn_id: int, 
                          event_type: Optional[EventType] = None) -> List[Event]:
        """Get all events at a specific turn"""
        events = self._turn_to_events.get(turn_id, [])
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events
    
    def get_environment_information_before_turn(self, turn_id: int) -> List[Event]:
        """Get all environment-provided information before a turn"""
        info_events = self.get_events_before_turn(turn_id, EventType.INFORMATION)
        # Filter to only environment-provided information
        return [e for e in info_events 
                if e.metadata.get('source') == 'environment']
    
    def has_evidence_before_turn(self, turn_id: int) -> bool:
        """Check if there is any evidence available before a turn"""
        return len(self.get_environment_information_before_turn(turn_id)) > 0
    
    def get_all_evidence(self) -> List[str]:
        """Get all evidence content from the trajectory"""
        info_events = [e for e in self.events 
                      if e.event_type == EventType.INFORMATION
                      and e.metadata.get('source') == 'environment']
        return [e.content for e in info_events]
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'trajectory_id': self.trajectory_id,
            'events': [e.to_dict() for e in self.events]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EventLedger':
        """Deserialize from dictionary"""
        ledger = cls(trajectory_id=data.get('trajectory_id'))
        for event_data in data.get('events', []):
            event = Event.from_dict(event_data)
            ledger.events.append(event)
            # Rebuild index
            if event.turn_id not in ledger._turn_to_events:
                ledger._turn_to_events[event.turn_id] = []
            ledger._turn_to_events[event.turn_id].append(event)
        return ledger
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __repr__(self) -> str:
        return f"EventLedger(trajectory_id={self.trajectory_id}, n_events={len(self.events)})"


