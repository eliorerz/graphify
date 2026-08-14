"""Combined YAML dispatcher -- the actual `.yaml`/`.yml` entry point.

OSAC-4050. Layered per document (a single file can bundle multiple
`---`-separated documents, confirmed against a real file in this repo --
see graphify/extractors/_yaml_cst.py's ``all_documents`` docstring):

1. If a document parses cleanly and looks like a real k8s resource
   (``is_k8s_manifest_shape``), extract its rich ownership/reference/
   selector relationships (``graphify.extractors.k8s_manifest``).
2. Otherwise, if it parses cleanly, fall back to a generic structural walk
   with no domain semantics (``graphify.extractors.yaml_generic``) --
   universal coverage, per the user's explicit, confirmed direction change
   partway through this ticket's implementation: even YAML with no
   recognized schema should get SOME representation in the graph, rather
   than staying invisible the way pre-OSAC-4050 graphify left ALL YAML.
3. If a document doesn't parse cleanly at all (`node.has_error` -- e.g. a
   Helm chart's Go-templated `{{ include ... }}` syntax, confirmed
   empirically during this ticket's investigation to break the
   tree-sitter-yaml grammar outright, producing an ERROR node rather than a
   best-effort partial tree), it is skipped with a one-line warning naming
   the file -- never crashed on, never walked for "structure" that would
   really just be gibberish extracted from a broken parse.
"""
from __future__ import annotations

import sys
from pathlib import Path

from graphify.extractors._yaml_cst import all_documents, mapping as _mapping
from graphify.extractors.base import _make_id
from graphify.extractors.k8s_manifest import extract_k8s_resources, is_k8s_manifest_shape
from graphify.extractors.yaml_generic import extract_generic_structure

_YAML_MAX_BYTES = 1_048_576  # 1 MiB -- matches every other extractor's cap in this fork


def extract_yaml(path: Path) -> dict:
    """Extract structure from a .yaml/.yml file: rich k8s relationships for
    recognized manifests, generic structural nodes for everything else that
    parses cleanly, a skip-with-warning for anything that doesn't.
    """
    try:
        import tree_sitter_yaml as tsyaml
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_yaml not installed. Run: pip install tree-sitter-yaml"}

    try:
        with path.open("rb") as fh:
            source = fh.read(_YAML_MAX_BYTES + 1)
        if len(source) > _YAML_MAX_BYTES:
            return {"nodes": [], "edges": [], "error": "yaml file too large to index"}
        language = Language(tsyaml.language())
        parser = Parser(language)
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    file_nid = _make_id(str_path)
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    file_node_added = False
    skipped_docs = 0
    truncated_docs = 0

    def _ensure_file_node() -> None:
        nonlocal file_node_added
        if not file_node_added:
            all_nodes.append({"id": file_nid, "label": path.name, "file_type": "code",
                              "source_file": str_path, "source_location": None})
            file_node_added = True

    for doc_index, doc in enumerate(all_documents(root)):
        if doc.has_error:
            skipped_docs += 1
            continue
        m = _mapping(doc)
        if m is not None and is_k8s_manifest_shape(m):
            _ensure_file_node()
            k8s_nodes, k8s_edges = extract_k8s_resources([m], str_path, file_nid)
            all_nodes.extend(k8s_nodes)
            all_edges.extend(k8s_edges)
            continue
        # Not k8s-shaped (or not even a mapping at the top level) but parses
        # cleanly -- generic structural fallback. Walk from the raw doc
        # value (not `m`, which is None for a non-mapping document).
        generic_nodes, generic_edges, truncated = extract_generic_structure(doc, str_path, file_nid, doc_index)
        if generic_nodes:
            _ensure_file_node()
            all_nodes.extend(generic_nodes)
            all_edges.extend(generic_edges)
        if truncated:
            truncated_docs += 1

    if skipped_docs:
        # One-line warning naming the file, printed immediately rather than
        # folded into extract.py's end-of-run aggregate warnings (#1666/
        # #1745 are keyed on "produced zero nodes"/"dependency missing",
        # neither of which fits "this specific document didn't parse") --
        # the ticket's explicit ask was a clear warning naming the file, not
        # a silent skip.
        suffix = "s" if skipped_docs > 1 else ""
        print(
            f"  warning: {path.name}: {skipped_docs} document{suffix} did not "
            f"parse as valid YAML (commonly Go/Helm template syntax breaking "
            f"the grammar) -- skipped, not extracted.",
            file=sys.stderr,
        )
    if truncated_docs:
        print(
            f"  warning: {path.name}: generic structural walk hit its node "
            f"cap in {truncated_docs} document(s) -- truncated, not "
            f"exhaustive.",
            file=sys.stderr,
        )

    return {"nodes": all_nodes, "edges": all_edges}
