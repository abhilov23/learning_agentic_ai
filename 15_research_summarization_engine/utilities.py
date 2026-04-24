import json 


def to_obj(s):
    if not isinstance(s, str):
        return {}
    cleaned = s.strip().replace("```json", "").replace("```", "").strip()
    for candidate in (cleaned, s.strip()):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return {}
