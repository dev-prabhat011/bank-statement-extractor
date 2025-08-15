import re
from datetime import datetime


def parse_amount_safe(raw_value):
    """Parse currency-like strings to float. Returns None if not parseable."""
    if raw_value is None:
        return None
    s = str(raw_value).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None
    cleaned = re.sub(r"[\s,$€£,]", "", s)

    is_negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
        is_negative = True
    elif cleaned.endswith("-"):
        cleaned = cleaned[:-1]
        is_negative = True
    elif cleaned.endswith("DR"):
        cleaned = cleaned[:-2]
        is_negative = True
    elif cleaned.endswith("CR"):
        cleaned = cleaned[:-2]

    if cleaned.startswith("-"):
        cleaned = cleaned[1:]
        is_negative = True

    if cleaned in {"", "."}:
        return None
    try:
        value = float(cleaned)
    except Exception:
        return None
    return -value if is_negative else value


def parse_date_flexible(raw_value):
    """Try multiple date formats; return YYYY-MM-DD or None."""
    if raw_value is None:
        return None
    s = str(raw_value).strip()
    fmts = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b, %Y", "%d-%b-%Y", "%d/%b/%Y")
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    # Last-attempt normalization: replace common separators and retry
    s2 = re.sub(r"[/\-. ]+", " ", s)
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(s2, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


