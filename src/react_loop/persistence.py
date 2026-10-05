import sqlite3
import json
from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

def message_to_dict(msg):
    if isinstance(msg, HumanMessage):
        return {"type": "human", "content": msg.content}
    if isinstance(msg, AIMessage):
        return {"type": "ai", "content": msg.content, "tool_calls": msg.tool_calls}
    if isinstance(msg, ToolMessage):
        return {"type": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id}
    if isinstance(msg, SystemMessage):
        return {"type": "system", "content": msg.content}
    return {"type": "unknown", "content": str(msg)}

def dict_to_message(d):
    t = d["type"]
    c = d["content"]
    if t == "human": return HumanMessage(content=c)
    if t == "ai": return AIMessage(content=c, tool_calls=d.get("tool_calls"))
    if t == "tool": return ToolMessage(content=c, tool_call_id=d.get("tool_call_id"))
    if t == "system": return SystemMessage(content=c)
    return HumanMessage(content=c)

class SessionManager:
    """Saves and loads conversation history to SQLite."""
    
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions "
                "(thread_id TEXT PRIMARY KEY, messages TEXT)"
            )
            conn.commit()

    def save_session(self, thread_id: str, messages: List):
        serialized = json.dumps([message_to_dict(m) for m in messages])
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (thread_id, messages) VALUES (?, ?)",
                (thread_id, serialized)
            )
            conn.commit()

    def load_session(self, thread_id: str) -> Optional[List]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT messages FROM sessions WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if row:
                return [dict_to_message(d) for d in json.loads(row[0])]
            return None
