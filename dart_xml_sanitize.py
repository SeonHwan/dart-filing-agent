from __future__ import annotations

import re

_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);)")
_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_ATTR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_:.-]*")


def _fix_bare_ampersands(text: str) -> str:
    return _AMP_RE.sub("&amp;", text)


def _find_attr_value_end(text: str, value_start: int) -> int | None:
    search_from = value_start
    while True:
        q = text.find('"', search_from)
        if q == -1:
            return None
        after = q + 1
        k = after
        while k < len(text) and text[k] in " \t\r\n":
            k += 1
        if k < len(text) and text[k] == ">":
            return q
        if k + 1 < len(text) and text[k] == "/" and text[k + 1] == ">":
            return q
        attr_m = _ATTR_NAME_RE.match(text, k)
        if attr_m and attr_m.end() < len(text) and text[attr_m.end()] == "=":
            return q
        search_from = q + 1


def _match_tag(text: str, start: int) -> int | None:
    if text.startswith("<?", start):
        end = text.find("?>", start)
        return end + 2 if end != -1 else None
    if text.startswith("<!--", start):
        end = text.find("-->", start)
        return end + 3 if end != -1 else None

    i = start + 1
    if i < len(text) and text[i] == "/":
        i += 1
    m = _NAME_RE.match(text, i)
    if not m:
        return None
    i = m.end()

    while True:
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        if i < len(text) and text[i] == "/" and i + 1 < len(text) and text[i + 1] == ">":
            return i + 2
        if i < len(text) and text[i] == ">":
            return i + 1
        attr_m = _ATTR_NAME_RE.match(text, i)
        if not attr_m:
            return None
        i = attr_m.end()
        if not (i < len(text) and text[i] == "=" and i + 1 < len(text) and text[i + 1] == '"'):
            return None
        i += 2  # '="' 지나서 값 시작 위치
        q = _find_attr_value_end(text, i)
        if q is None:
            return None
        i = q + 1


def _fix_tags(text: str) -> str:
    out: list[str] = []
    i = 0
    while True:
        j = text.find("<", i)
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        end = _match_tag(text, j)
        if end is None:
            out.append("&lt;")
            i = j + 1
            continue
        tag_text = text[j:end]
        out.append(_requote_tag(tag_text))
        i = end
    return "".join(out)


def _requote_tag(tag_text: str) -> str:
    if tag_text.startswith("<?") or tag_text.startswith("<!--"):
        return tag_text

    out: list[str] = []
    i = 0
    n = len(tag_text)
    # '<', 선택적 '/', 태그명까지는 그대로 복사
    i = 1
    if i < n and tag_text[i] == "/":
        i += 1
    m = _NAME_RE.match(tag_text, i)
    i = m.end()
    out.append(tag_text[:i])

    while i < n:
        start_ws = i
        while i < n and tag_text[i] in " \t\r\n":
            i += 1
        out.append(tag_text[start_ws:i])
        if i < n and tag_text[i] in ">/":
            out.append(tag_text[i:])
            break
        attr_m = _ATTR_NAME_RE.match(tag_text, i)
        name_end = attr_m.end()
        out.append(tag_text[i:name_end])
        out.append('="')
        value_start = name_end + 2
        value_end = _find_attr_value_end(tag_text, value_start)
        raw_value = tag_text[value_start:value_end]
        out.append(raw_value.replace('"', "&quot;"))
        out.append('"')
        i = value_end + 1

    return "".join(out)


def sanitize_dart_xml(raw: str) -> str:
    """DART 사업보고서 원본 XML 문자열을 표준 xml.etree가 파싱 가능한 형태로 정제."""
    fixed = _fix_bare_ampersands(raw)
    return _fix_tags(fixed)
