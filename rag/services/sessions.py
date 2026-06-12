import datetime
import json
from pathlib import Path


class SessionManager:
    def __init__(self, persist_dir: str = "./rag_storage"):
        self.persist_dir = Path(persist_dir)
        self.sessions_dir = self.persist_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, title: str = "New Chat", session_id: str | None = None) -> dict:
        if not session_id:
            session_id = f"session_{int(datetime.datetime.now().timestamp() * 1000)}"

        session_data = {
            "session_id": session_id,
            "title": title,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [],
        }
        self._write_session(session_id, session_data)
        return session_data

    def _write_session(self, session_id: str, session_data: dict) -> None:
        file_path = self.sessions_dir / f"{session_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing session {session_id}: {e}")

    def save_session(self, session_id: str, title: str, messages: list) -> None:
        session_data = self.load_session(session_id)
        if not session_data:
            session_data = {
                "session_id": session_id,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        
        session_data["title"] = title
        session_data["messages"] = messages
        self._write_session(session_id, session_data)

    def load_session(self, session_id: str) -> dict | None:
        file_path = self.sessions_dir / f"{session_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def list_sessions(self) -> list[dict]:
        sessions = []
        if not self.sessions_dir.exists():
            return sessions

        for file in self.sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "title": data["title"],
                        "created_at": data["created_at"],
                    })
            except Exception as e:
                print(f"Error reading session file {file.name}: {e}")

        # Sort newest first
        sessions.sort(key=lambda x: x["session_id"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> None:
        file_path = self.sessions_dir / f"{session_id}.json"
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error deleting session file {session_id}: {e}")

    def clear_session(self, session_id: str) -> None:
        session_data = self.load_session(session_id)
        if session_data:
            session_data["messages"] = []
            self._write_session(session_id, session_data)
