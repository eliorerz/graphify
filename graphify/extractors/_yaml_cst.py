"""Shared tree-sitter-yaml CST traversal helpers.

Used by both ``graphify/extractors/k8s_manifest.py`` and
``graphify/extractors/yaml_generic.py`` (OSAC-4050). Originally written
once inline in ``k8s_manifest.py``, adapted with attribution from the
tree-sitter-yaml traversal in the unmerged Graphify-Labs/graphify PR #2541;
factored out here once a second real consumer (the generic structural
walker) needed the exact same helpers within the same branch. Note:
``graphify/extractors/github_actions.py`` (OSAC-4049, unmerged as of this
writing) still carries its own independent copy of an earlier version of
these same helpers, since its branch predates this module and should not be
made to depend on this one landing first -- worth deduping further once
both have merged.
"""
from __future__ import annotations

_MAPPING_TYPES = frozenset({"block_mapping", "flow_mapping"})
_SEQUENCE_TYPES = frozenset({"block_sequence", "flow_sequence"})


def descend(node, wanted: frozenset[str]):
    """Return the first descendant of *node* whose type is in *wanted*.

    YAML wraps every value in `block_node`/`flow_node` before the actual
    collection, and a document adds another layer, so callers would otherwise
    repeat the same two-or-three-step unwrap everywhere.
    """
    if node is None:
        return None
    if node.type in wanted:
        return node
    for child in node.children:
        if not child.is_named:
            continue
        if child.type in ("block_node", "flow_node", "document"):
            found = descend(child, wanted)
            if found is not None:
                return found
        elif child.type in wanted:
            return child
    return None


def mapping(node):
    return descend(node, _MAPPING_TYPES)


def scalar_text(node) -> str:
    """Text of the scalar at *node*, with one layer of quotes stripped."""
    if node is None:
        return ""
    text = node.text.decode("utf-8", errors="replace").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    return text.strip()


def pairs(node):
    """Yield `(key, value_node, line)` for each pair of the mapping at *node*.

    *node* may be the mapping itself or any wrapper around it. Pairs whose key
    is not a plain scalar are skipped rather than stringified, so they never
    mint a garbage node.
    """
    m = mapping(node)
    if m is None:
        return
    for pair in m.children:
        if pair.type not in ("block_mapping_pair", "flow_pair"):
            continue
        key_node = pair.child_by_field_name("key")
        if key_node is None:
            continue
        key = scalar_text(key_node)
        if not key:
            continue
        yield key, pair.child_by_field_name("value"), key_node.start_point[0] + 1


def item_value(item):
    """The value inside a `block_sequence_item`, without the `- ` marker."""
    if item.type != "block_sequence_item":
        return item
    for child in item.children:
        if child.is_named:
            return child
    return item


def string_items(node) -> list[tuple[str, int]]:
    """Scalars reachable from *node* as `(text, line)` -- a bare scalar, a
    sequence (block or flow), or (defensively) a mapping's keys."""
    if node is None:
        return []
    seq = descend(node, _SEQUENCE_TYPES)
    if seq is not None:
        items = []
        for item in seq.children:
            if item.type not in ("block_sequence_item", "flow_node"):
                continue
            text = scalar_text(item_value(item))
            if text and "\n" not in text and ":" not in text:
                items.append((text, item.start_point[0] + 1))
        return items
    m = mapping(node)
    if m is not None:
        return [(key, line) for key, _value, line in pairs(m)]
    text = scalar_text(node)
    return [(text, node.start_point[0] + 1)] if text else []


def sequence_items(node):
    """Yield the item nodes of the sequence at *node*."""
    seq = descend(node, _SEQUENCE_TYPES)
    if seq is None:
        return
    for item in seq.children:
        if item.type in ("block_sequence_item", "flow_node"):
            yield item


def all_documents(root):
    """Yield the raw top-level node of every document in the file (whatever
    its type -- mapping, sequence, scalar, or ERROR; callers decide what to
    do with each).

    Root type is `stream` with one `document` child per resource for a
    multi-document file (confirmed against a real file in this repo,
    ``osac-operator/config/manager/manager.yaml``, which holds a Namespace
    and a Deployment separated by `---`), vs a bare `document` (or, for a
    malformed/templated file, an `ERROR` node) for a single-document file.
    """
    if root.type == "stream":
        docs = [c for c in root.children if c.type == "document"]
    elif root.type == "document":
        docs = [root]
    else:
        docs = [root]
    for doc in docs:
        yield doc


def all_top_level_mappings(root):
    """Yield the top-level MAPPING of every document in the file (documents
    that aren't mapping-shaped, or are malformed, are silently skipped --
    for callers that only care about mapping-shaped resources, e.g. a k8s
    manifest). See ``all_documents`` for the version that yields every
    document regardless of shape.
    """
    for doc in all_documents(root):
        m = mapping(doc)
        if m is not None:
            yield m
