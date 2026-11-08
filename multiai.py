#!/usr/bin/env python3
"""multiai - one terminal app for all the AIs on your laptop.

Supports: Ollama, LM Studio, llama.cpp server (any OpenAI-compatible server),
OpenAI, Anthropic. Python 3.8+, standard library only.

Run:  python3 multiai.py
Config is created at ~/.multiai.json - edit it to add more AIs.
"""
import json
import os
import sys
import threading
import urllib.request
import urllib.error

CONFIG = os.path.expanduser("~/.multiai.json")

DEFAULT_CONFIG = {
    "providers": {
        "ollama":    {"type": "ollama",    "url": "http://localhost:11434", "model": ""},
        "lmstudio":  {"type": "openai",    "url": "http://localhost:1234/v1", "model": "", "key_env": ""},
        "llamacpp":  {"type": "openai",    "url": "http://localhost:8080/v1", "model": "", "key_env": ""},
        "openai":    {"type": "openai",    "url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
        "anthropic": {"type": "anthropic", "url": "https://api.anthropic.com", "model": "claude-sonnet-5-5", "key_env": "ANTHROPIC_API_KEY"},
    }
}


# ---------- config ----------
def load_config():
    if not os.path.exists(CONFIG):
        with open(CONFIG, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    with open(CONFIG) as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------- http ----------
def http(url, payload=None, headers=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot connect ({e.reason})")


def api_key(p):
    env = p.get("key_env")
    return os.environ.get(env, "") if env else ""


def auth_headers(p):
    key = api_key(p)
    if p["type"] == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"} if key else {}


# ---------- providers ----------
def list_models(p, timeout=3):
    t = p["type"]
    if t == "ollama":
        d = http(p["url"] + "/api/tags", timeout=timeout)
        return [m["name"] for m in d.get("models", [])]
    if t == "openai":
        d = http(p["url"] + "/models", headers=auth_headers(p), timeout=timeout)
        return [m["id"] for m in d.get("data", [])]
    if t == "anthropic":
        d = http(p["url"] + "/v1/models", headers=auth_headers(p), timeout=timeout)
        return [m["id"] for m in d.get("data", [])]
    return []


def is_available(p):
    if p.get("key_env") and not api_key(p):
        return False
    if p["type"] == "anthropic":
        return True
    try:
        list_models(p, timeout=2)
        return True
    except Exception:
        return False


def chat(p, messages):
    t = p["type"]
    model = p.get("model", "")
    if t == "ollama":
        d = http(p["url"] + "/api/chat",
                 {"model": model, "messages": messages, "stream": False})
        return d["message"]["content"]
    if t == "openai":
        d = http(p["url"] + "/chat/completions",
                 {"model": model, "messages": messages},
                 headers=auth_headers(p))
        return d["choices"][0]["message"]["content"]
    if t == "anthropic":
        d = http(p["url"] + "/v1/messages",
                 {"model": model, "max_tokens": 2048, "messages": messages},
                 headers=auth_headers(p))
        return "".join(b.get("text", "") for b in d["content"])
    raise RuntimeError(f"unknown provider type: {t}")


def ensure_model(name, p):
    """Pick the first installed model if none is set."""
    if p.get("model"):
        return True
    try:
        models = list_models(p)
        if models:
            p["model"] = models[0]
            return True
    except Exception:
        pass
    return False


# ---------- app ----------
HELP = """Commands:
  /list                 show all AIs and whether they are available
  /use <ai> [model]     switch to one AI (optionally pick its model)
  /models [ai]          show models of an AI (default: current)
  /all                  toggle asking ALL available AIs at once
  /clear                clear conversation history
  /add <name> <type> <url> [model] [KEY_ENV]
                        add a new AI (type: ollama | openai | anthropic)
  /help                 show this help
  /quit                 exit
"""


def main():
    cfg = load_config()
    providers = cfg["providers"]
    history = {name: [] for name in providers}
    available = {}

    print("Checking which AIs are available...")
    for name, p in providers.items():
        ok = is_available(p)
        if ok:
            ensure_model(name, p)
        available[name] = ok
        print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({p.get('model')})" if ok else ""))

    ready = [n for n, ok in available.items() if ok]
    if not ready:
        print("\nNo AI found. Start Ollama / LM Studio, or set OPENAI_API_KEY / ANTHROPIC_API_KEY.")
        print("You can still use /add to configure one. Type /help.")
    current = ready[0] if ready else None
    broadcast = False
    print("\nType /help for commands. Ctrl+C or /quit to exit.\n")

    while True:
        try:
            label = "ALL" if broadcast else (current or "none")
            line = input(f"[{label}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd, args = parts[0].lower(), parts[1:]

            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/list":
                for n, p in providers.items():
                    mark = "✓" if available.get(n) else "✗"
                    cur = " <- current" if n == current and not broadcast else ""
                    print(f"  {mark} {n:10} {p['type']:9} {p.get('model') or '-'}{cur}")
            elif cmd == "/use" and args:
                if args[0] not in providers:
                    print("Unknown AI. Use /list.")
                    continue
                current, broadcast = args[0], False
                if len(args) > 1:
                    providers[current]["model"] = args[1]
                    save_config(cfg)
                available[current] = True
                print(f"Now using {current} ({providers[current].get('model') or 'no model set'})")
            elif cmd == "/models":
                n = args[0] if args else current
                if n not in providers:
                    print("Unknown AI.")
                    continue
                try:
                    for m in list_models(providers[n], timeout=10):
                        print("  ", m)
                except Exception as e:
                    print("Error:", e)
            elif cmd == "/all":
                broadcast = not broadcast
                print("Asking all available AIs." if broadcast else f"Back to {current}.")
            elif cmd == "/clear":
                history = {name: [] for name in providers}
                print("History cleared.")
            elif cmd == "/add" and len(args) >= 3:
                name, typ, url = args[0], args[1], args[2]
                providers[name] = {
                    "type": typ, "url": url,
                    "model": args[3] if len(args) > 3 else "",
                    "key_env": args[4] if len(args) > 4 else "",
                }
                history[name] = []
                available[name] = is_available(providers[name])
                if available[name]:
                    ensure_model(name, providers[name])
                save_config(cfg)
                print(f"Added {name}. Available: {available[name]}")
            else:
                print("Unknown command or missing arguments. /help")
            continue

        # ----- send a message -----
        targets = [n for n in providers if available.get(n)] if broadcast else [current]
        if not targets or targets == [None]:
            print("No AI selected. Use /list and /use <ai>.")
            continue

        results = {}

        def worker(n):
            history[n].append({"role": "user", "content": line})
            try:
                reply = chat(providers[n], history[n])
                history[n].append({"role": "assistant", "content": reply})
                results[n] = reply
            except Exception as e:
                history[n].pop()
                results[n] = f"[error] {e}"

        threads = [threading.Thread(target=worker, args=(n,)) for n in targets]
        print("thinking..." if len(targets) == 1 else f"asking {len(targets)} AIs...")
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue

        for n in targets:
            if len(targets) > 1 or broadcast:
                print(f"\n━━ {n} ({providers[n].get('model')}) ━━")
            else:
                print()
            print(results.get(n, ""))
        print()


if __name__ == "__main__":
    main()
