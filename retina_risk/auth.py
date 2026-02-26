import os
import json
import base64
import hashlib
import time

def storage_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")

def users_file() -> str:
    return os.path.join(storage_dir(), "users.json")

def ensure_storage():
    os.makedirs(storage_dir(), exist_ok=True)
    if not os.path.exists(users_file()):
        with open(users_file(), "w", encoding="utf-8") as f:
            json.dump({}, f)

def load_users() -> dict:
    ensure_storage()
    with open(users_file(), "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_users(users: dict):
    ensure_storage()
    with open(users_file(), "w", encoding="utf-8") as f:
        json.dump(users, f)

def users_exist() -> bool:
    u = load_users()
    return len(u) > 0

def _hash_password(password: str, salt: bytes) -> str:
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return base64.b64encode(h).decode("ascii")

def register_user(username: str, password: str) -> bool:
    username = username.strip()
    if not username or not password:
        return False
    users = load_users()
    if username in users:
        return False
    salt = os.urandom(16)
    users[username] = {
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": _hash_password(password, salt),
    }
    save_users(users)
    return True

def authenticate_user(username: str, password: str) -> bool:
    users = load_users()
    info = users.get(username.strip())
    if not info:
        return False
    try:
        salt = base64.b64decode(info["salt"].encode("ascii"))
    except Exception:
        return False
    return _hash_password(password, salt) == info.get("hash")

def session_file() -> str:
    return os.path.join(storage_dir(), "session.json")

def load_session() -> dict | None:
    ensure_storage()
    if not os.path.exists(session_file()):
        return None
    try:
        with open(session_file(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_session(username: str, days: int = 30):
    ensure_storage()
    exp = int(time.time()) + days * 86400
    with open(session_file(), "w", encoding="utf-8") as f:
        json.dump({"user": username, "expires": exp}, f)

def clear_session():
    try:
        os.remove(session_file())
    except Exception:
        pass

def seed_from_bootstrap_file(remove_after: bool = True):
    ensure_storage()
    path = os.path.join(storage_dir(), "bootstrap_users.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for u, p in data.items():
                register_user(str(u), str(p))
    except Exception:
        pass
    finally:
        if remove_after:
            try:
                os.remove(path)
            except Exception:
                pass
