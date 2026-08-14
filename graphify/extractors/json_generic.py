"""Generic structural JSON extractor.

OSAC-4050 -- the same universal-coverage direction change applied to JSON:
``extract_json()`` (graphify/extractors/json_config.py) already recognizes
config/manifest JSON (package.json, tsconfig.json, ...) via
``_is_config_json`` and gives it rich dependency/extends/$ref edges: data
JSON that doesn't match previously returned an empty result outright
(#1224 -- AST-walking arbitrary data JSON produced hundreds of orphan
key-nodes). Per the user's explicit, confirmed decision, that data JSON now
gets a genuinely generic structural walk instead (no domain semantics, one
node per key/list item) rather than staying invisible -- mirroring
``graphify/extractors/yaml_generic.py``'s design exactly, just walking
tree-sitter-JSON's simpler node vocabulary (``object``/``pair``/``array``)
instead of YAML's block/flow-wrapped one.
"""
from __future__ import annotations

from graphify.extractors.base import _make_id, _read_text

MAX_NODES_PER_DOCUMENT = 2000

_MAX_LABEL_LEN = 80


def _truncate_label(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_LABEL_LEN:
        return text
    return text[: _MAX_LABEL_LEN - 1] + "…"


def _key_text(pair_node, source: bytes) -> str:
    key_node = pair_node.child_by_field_name("key")
    if key_node is None:
        return ""
    if key_node.type == "string":
        content = key_node.child_by_field_name("string_content")
        if content:
            return _read_text(content, source)
        return _read_text(key_node, source).strip('"\'')
    return _read_text(key_node, source)


def _scalar_text(node, source: bytes) -> str:
    if node.type == "string":
        content = node.child_by_field_name("string_content")
        if content:
            return _read_text(content, source)
        return _read_text(node, source).strip('"\'')
    return _read_text(node, source)


def extract_generic_structure(root_value, source: bytes, str_path: str, file_nid: str) -> tuple[list[dict], list[dict], bool]:
    """Walk a JSON document's root value (object, array, or scalar) and emit
    structural nodes/edges with no domain semantics -- one node per object
    key, one node per array item, no dependency/extends/$ref interpretation
    (that stays exclusive to recognized config/manifest JSON in
    ``extract_json``).

    Returns (nodes, edges, truncated) -- see yaml_generic.py's
    extract_generic_structure for the identical truncation contract.
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    truncated = False

    def _mint(parts: tuple[str, ...], label: str, line: int) -> str:
        nid = _make_id(str_path, *parts)
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": _truncate_label(label), "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}"})
        return nid

    def _add_contains(parent_nid: str, child_nid: str, line: int) -> None:
        edges.append({"source": parent_nid, "target": child_nid, "relation": "contains",
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line}", "weight": 1.0})

    def _walk(node, parent_nid: str, parts: tuple[str, ...]) -> None:
        nonlocal truncated
        if truncated or node is None:
            return
        if node.type == "object":
            for child in node.children:
                if child.type != "pair":
                    continue
                if len(nodes) >= MAX_NODES_PER_DOCUMENT:
                    truncated = True
                    return
                key = _key_text(child, source)
                if not key:
                    continue
                value = child.child_by_field_name("value")
                line = child.start_point[0] + 1
                child_parts = parts + (key,)
                child_nid = _mint(child_parts, key, line)
                _add_contains(parent_nid, child_nid, line)
                _walk(value, child_nid, child_parts)
            return
        if node.type == "array":
            items = [c for c in node.children if c.is_named]
            for i, item in enumerate(items):
                if len(nodes) >= MAX_NODES_PER_DOCUMENT:
                    truncated = True
                    return
                text = _scalar_text(item, source) if item.type in ("string", "number", "true", "false", "null") else ""
                label = text if text else f"[{i}]"
                line = item.start_point[0] + 1
                child_parts = parts + (f"[{i}]",)
                child_nid = _mint(child_parts, label, line)
                _add_contains(parent_nid, child_nid, line)
                _walk(item, child_nid, child_parts)
            return
        # Scalar leaf: nothing further to mint.

    _walk(root_value, file_nid, ("doc0",))
    return nodes, edges, truncated
