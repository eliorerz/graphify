"""Generic structural YAML extractor.

OSAC-4050 -- a direction change from that ticket's original k8s-only scope,
confirmed explicitly by the user mid-implementation: universal coverage for
ANY YAML that doesn't match a known, recognized schema (like a k8s
manifest), rather than leaving it invisible to the graph. Emits raw
structural nodes (one per mapping key, one per sequence item) with NO
domain semantics -- accepted, on the user's explicit instruction, to be
lower-value/noisier than a schema-aware extractor (the same failure mode
already confirmed empirically for unrecognized JSON, #1224/OSAC-4050
investigation) in exchange for nothing being silently invisible.

Malformed/templated YAML (a Helm chart's `{{ include ... }}` etc., which
breaks the tree-sitter-yaml grammar outright -- confirmed empirically
against a real chart in this fork's OSAC-4050 investigation, producing a
root ERROR node, not a best-effort partial tree) is NOT this module's
concern: the caller (graphify/extractors/yaml_dispatch.py) checks
`node.has_error` per document before ever calling into this walker, and
warns + skips instead. This module only ever sees already-known-clean
document values.
"""
from __future__ import annotations

from graphify.extractors.base import _make_id
from graphify.extractors._yaml_cst import (
    item_value as _item_value,
    mapping as _mapping,
    pairs as _pairs,
    scalar_text as _scalar_text,
    sequence_items as _sequence_items,
)

# Safety valve: a single pathological file (deeply nested, huge sequences)
# must not single-handedly blow up a corpus-wide extraction. This is a
# ceiling, not a target -- logged explicitly when hit (no silent caps), so
# it's visible rather than read as "covered everything" when it didn't. See
# OSAC-4050's empirical corpus-scale measurement for whether real files ever
# approach it.
MAX_NODES_PER_DOCUMENT = 2000

_MAX_LABEL_LEN = 80


def _truncate_label(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_LABEL_LEN:
        return text
    return text[: _MAX_LABEL_LEN - 1] + "…"


def extract_generic_structure(doc_value, str_path: str, file_nid: str, doc_index: int) -> tuple[list[dict], list[dict], bool]:
    """Walk one YAML document's top-level value (mapping, sequence, or
    scalar -- a generic document isn't required to be a mapping the way a
    k8s resource is) and emit structural nodes/edges with no domain
    semantics.

    IDs are hierarchical and file+document+path scoped
    (`_make_id(str_path, "doc{N}", key1, key2, ...)`) -- unlike a k8s
    resource's globally-scoped (kind, namespace, name) id, a raw structural
    position has no meaningful identity outside its own file, so there is
    nothing to collapse across files here.

    Returns (nodes, edges, truncated) -- `truncated` is True if
    MAX_NODES_PER_DOCUMENT was hit, so the caller can log it once per file
    rather than the walker doing so per node.
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
        mapping = _mapping(node)
        if mapping is not None:
            for key, value, line in _pairs(mapping):
                if len(nodes) >= MAX_NODES_PER_DOCUMENT:
                    truncated = True
                    return
                child_parts = parts + (key,)
                child_nid = _mint(child_parts, key, line)
                _add_contains(parent_nid, child_nid, line)
                _walk(value, child_nid, child_parts)
            return
        seq_items = list(_sequence_items(node))
        if seq_items:
            for i, item in enumerate(seq_items):
                if len(nodes) >= MAX_NODES_PER_DOCUMENT:
                    truncated = True
                    return
                item_value = _item_value(item)
                text = _scalar_text(item_value)
                label = text if text else f"[{i}]"
                line = item.start_point[0] + 1
                child_parts = parts + (f"[{i}]",)
                child_nid = _mint(child_parts, label, line)
                _add_contains(parent_nid, child_nid, line)
                _walk(item_value, child_nid, child_parts)
            return
        # Scalar leaf: nothing further to mint -- the key/item node the
        # caller already minted for this position represents it.

    doc_parts = (f"doc{doc_index}",)
    _walk(doc_value, file_nid, doc_parts)
    return nodes, edges, truncated
