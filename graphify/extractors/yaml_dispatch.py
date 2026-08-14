"""Combined YAML dispatcher -- the actual `.yaml`/`.yml` entry point.

OSAC-4050. Two phases (a single file can bundle multiple `---`-separated
documents, confirmed against a real file in this repo -- see
graphify/extractors/_yaml_cst.py's ``all_documents`` docstring):

1. Classify every document in the file: unparseable-for-our-purposes
   (skip + warn), k8s-manifest-shaped, or generic.
2. Extract ALL k8s-shaped documents in ONE call to
   ``extract_k8s_resources`` (not one call per document -- a real,
   reviewer-caught bug in an earlier version of this file: calling it
   per-document defeats the two-pass same-file design
   ``graphify.extractors.k8s_manifest`` itself relies on for resolving a
   same-file forward/cross-document reference, e.g. a Deployment owned by
   a Namespace declared earlier in the same multi-document file -- each
   document got its own empty ``local_nids`` scope, so the real Namespace
   definition was never visible when resolving the Deployment's owner
   reference, minting a duplicate stub instead of linking to it). Every
   other (non-k8s-shaped) document falls back to a generic structural walk
   with no domain semantics (``graphify.extractors.yaml_generic``) --
   universal coverage, per the user's explicit, confirmed direction change
   partway through this ticket's implementation: even YAML with no
   recognized schema should get SOME representation in the graph, rather
   than staying invisible the way pre-OSAC-4050 graphify left ALL YAML.

"Unparseable-for-our-purposes" is decided by
`graphify.extractors._yaml_cst.is_unparseable` -- shared with
`k8s_manifest.all_top_level_mappings` (used by the standalone
`extract_k8s_manifest` entry point) so both paths apply the identical
gate; see that function's docstring for why `has_error` alone is not
reliable in either direction for real Helm template syntax.
"""
from __future__ import annotations

import sys
from pathlib import Path

from graphify.extractors._yaml_cst import all_documents, is_unparseable, mapping as _mapping
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

    # Phase 1: classify every document before extracting anything, so all
    # k8s-shaped documents in this file can be extracted together in ONE
    # call (see module docstring for why per-document calls are a bug, not
    # just a style choice).
    k8s_shaped_tops: list = []
    generic_docs: list[tuple[int, object]] = []
    for doc_index, doc in enumerate(all_documents(root)):
        if is_unparseable(doc):
            skipped_docs += 1
            continue
        m = _mapping(doc)
        if m is not None and is_k8s_manifest_shape(m):
            k8s_shaped_tops.append(m)
        else:
            generic_docs.append((doc_index, doc))

    # Phase 2: extract.
    if k8s_shaped_tops:
        _ensure_file_node()
        k8s_nodes, k8s_edges = extract_k8s_resources(k8s_shaped_tops, str_path, file_nid)
        all_nodes.extend(k8s_nodes)
        all_edges.extend(k8s_edges)

    for doc_index, doc in generic_docs:
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
            f"  warning: {path.name}: {skipped_docs} document{suffix} not "
            f"treated as real YAML (parse error, or Go/Helm template syntax "
            f"detected) -- skipped, not extracted.",
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
