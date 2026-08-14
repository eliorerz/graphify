"""Kubernetes manifest extractor.

Recognizes files shaped like a real Kubernetes resource (apiVersion + kind +
metadata present and value-validated, not just key names) -- OSAC-4050,
originally the real follow-up from the generic-YAML discussion on
https://github.com/eliorerz/graphify/pull/2#issuecomment-5294807827: k8s
manifests have a known-enough schema with genuinely useful relationships,
which is exactly why a scoped AST traversal is worthwhile here the same way
it was for GitHub Actions (OSAC-4049).

This module is now the RICH, PRIORITY layer of a two-layer design (the
user explicitly broadened this ticket's scope mid-implementation to
universal YAML/JSON coverage): recognized k8s manifests get these
relationships; anything else, including Helm ``values.yaml`` (still no
fixed schema of its own, still can't get THIS treatment specifically), gets
a generic structural fallback instead of being skipped entirely -- see
``graphify/extractors/yaml_dispatch.py`` for the combined entry point that
layers the two, and ``graphify/extractors/yaml_generic.py`` for the
fallback itself.

The tree-sitter-yaml CST traversal is shared with
``graphify/extractors/yaml_generic.py`` via ``graphify/extractors/_yaml_cst.py``.
That helper module's own origin is ``graphify/extractors/github_actions.py``
(OSAC-4049) -- both ultimately adapted, with attribution, from the
tree-sitter-yaml traversal in the unmerged Graphify-Labs/graphify PR #2541.
``github_actions.py`` still carries its own independent copy rather than
importing ``_yaml_cst`` (its branch predates this module and should not
depend on this one landing first) -- worth deduping further once both have
merged.

Relationships extracted (edges are all ``EXTRACTED`` -- every one is a
literal field read directly from the manifest, not inferred):

- ``metadata.ownerReferences`` (standard k8s): owner -> owned, relation
  ``owns``.
- This repo's own annotation-based convention
  (``osac.openshift.io/owner-reference`` = the parent's ID,
  ``osac.openshift.io/tenant`` = tenant scoping) -- see
  ``osac/.claude/rules/architecture-patterns.md``. Confirmed directly
  against the operator's own Go source (``subnet_type.pb.go`` et al.) that
  the annotation's value is the parent's ID, not its name, so the minted
  stub for it is keyed by that raw ID string; it will only collapse onto a
  real resource node if something else in the corpus exposes that same ID
  as an alias (a known, documented limitation, not a bug -- the same class
  of "stub that may never resolve to a local definition" GitHub Actions'
  ``actions/checkout@v4`` stubs already accept).
- ConfigMap / Secret references (``configMapKeyRef``/``secretKeyRef``,
  ``configMapRef``/``secretRef``, volume ``configMap``/``secret``), found via
  a generic recursive walk of ``spec`` rather than hardcoded exact paths
  (these appear at varying depths: ``containers[].env[].valueFrom.*``,
  ``containers[].envFrom[].*``, ``volumes[].*``) -- relation ``uses``.
- CRD cross-references via the ``*Ref``/``*Refs`` field-naming convention
  (e.g. ``subnetRef``, ``securityGroupRefs``) -- a real, live convention
  confirmed directly against this repo's own CRD samples
  (``osac-operator/config/samples/osac_v1alpha1_computeinstance.yaml``), not
  invented for this extractor. The referenced kind is inferred by stripping
  the ``Ref``/``Refs`` suffix and capitalizing (``subnetRef`` -> ``Subnet``).
  Deliberately NOT attempted: a bare field naming a resource with no
  ``Ref``/``Refs`` suffix (e.g. ``Subnet.spec.virtualNetwork``, which this
  repo's own samples set to a UUID) -- indistinguishable from an arbitrary
  opaque config value without hardcoding per-CRD-kind field knowledge, which
  would make this extractor fragile and high-maintenance for one more field
  per CRD change. A known, deliberate scope limit, not an oversight.
- Label-selector matches (Service -> pods/Deployments via ``spec.selector``)
  -- approximated via a shared "label hub" stub node per distinct
  ``key=value`` pair: the selecting resource gets a ``selects`` edge to the
  hub, the labeled resource (``metadata.labels`` or
  ``spec.template.metadata.labels``) gets a ``has_label`` edge to the same
  hub, so they connect through it without needing a bespoke cross-file
  matching pass (reuses the exact same stub-collapsing mechanism GitHub
  Actions relies on for shared actions). This is a genuine approximation
  for a selector with MORE THAN ONE key=value pair: Kubernetes requires ALL
  of a selector's pairs to match (AND semantics), but this hub-per-pair
  design would still show a connection through any ONE shared pair even if
  the others don't match. Confirmed against a REAL single-key example in
  this repo (``osac-operator/config/console-proxy/service.yaml``'s
  ``selector: {app: osac-console-proxy}`` matching
  ``deployment.yaml``'s pod template labels exactly) where this is exact,
  not approximate. Real Helm-templated Services in this repo (e.g.
  ``charts/operator/templates/metrics-service.yaml``) have their selector
  as a template ``include``, not literal YAML, so they parse to a tree-sitter
  ERROR node and are excluded automatically (see ``is_k8s_manifest_shape``
  and the module-level test coverage) -- confirmed directly, not assumed.

Not modelled BY THIS MODULE specifically: Helm ``values.yaml`` (no fixed
schema at all -- confirmed in the PR #2 discussion this cannot get THIS
rich, schema-aware treatment) and any YAML that doesn't validate as
apiVersion+kind+metadata-shaped. This function (``extract_k8s_resources``)
and the standalone ``extract_k8s_manifest`` wrapper both return an empty
result for such input -- it's ``yaml_dispatch.py``'s job to route it to the
generic fallback instead, not this module's.
"""
from __future__ import annotations

import re
from pathlib import Path

from graphify.extractors.base import _make_id
from graphify.extractors._yaml_cst import (
    all_top_level_mappings as _all_top_level_mappings,
    item_value as _item_value,
    mapping as _mapping,
    pairs as _pairs,
    scalar_text as _scalar_text,
    sequence_items as _sequence_items,
    string_items as _string_items,
)

# ---------------------------------------------------------------------------
# Real, value-validated shape check (used inside extract_k8s_manifest, which
# has already parsed the file with tree-sitter-yaml). Was paired with an
# is_k8s_manifest_path() cheap pre-filter for a classify_file() carve-out in
# an earlier version of this ticket; superseded once the direction changed
# to universal YAML/JSON coverage (all .yaml/.yml is unconditionally CODE
# now, matching .json's existing precedent -- see graphify/detect.py).
# ---------------------------------------------------------------------------

# apiVersion is either bare ("v1") or "<group>/<version>"
# ("apps/v1", "osac.openshift.io/v1alpha1", "apiextensions.k8s.io/v1").
_API_VERSION_RE = re.compile(r"^([a-zA-Z0-9.\-]+/)?v[0-9]+((alpha|beta)[0-9]*)?$")


def is_k8s_manifest_shape(top) -> bool:
    """True if *top* (a document's top-level mapping) is a real,
    value-validated k8s resource: apiVersion looks like a real k8s
    apiVersion string, kind looks like a real PascalCase type name, and
    metadata is present and is itself a mapping.

    Key presence alone is not enough (the ticket's own explicit warning,
    confirmed to matter in practice): a coincidental `kind: foo` in some
    unrelated config must not pass. All three checks must hold together.
    """
    if top is None:
        return False
    pairs = {key: value for key, value, _line in _pairs(top)}
    if not ({"apiVersion", "kind", "metadata"} <= pairs.keys()):
        return False
    api_version = _scalar_text(pairs["apiVersion"])
    kind = _scalar_text(pairs["kind"])
    if not _API_VERSION_RE.match(api_version):
        return False
    if not kind or not kind[0].isupper() or not kind.isalnum():
        return False
    return _mapping(pairs["metadata"]) is not None


# ---------------------------------------------------------------------------
# Relationship extraction.
# ---------------------------------------------------------------------------

_OWNER_ANNOTATION = "osac.openshift.io/owner-reference"
_TENANT_ANNOTATION = "osac.openshift.io/tenant"

# Field-name conventions for ConfigMap/Secret references, found at varying
# depths under spec (containers[].env[].valueFrom.*, containers[].envFrom[].*,
# volumes[].*) -- walked generically rather than hardcoding each exact path.
_CONFIGMAP_REF_KEYS = frozenset({"configMapKeyRef", "configMapRef", "configMap"})
_SECRET_REF_KEYS = frozenset({"secretKeyRef", "secretRef", "secret"})
# volumes' `secret:` block names the Secret via `secretName`, not `name`.
_REF_NAME_FIELDS = ("name", "secretName")


def _resource_id(kind: str, namespace: str, name: str) -> str:
    # namespace is folded in when present; make_id already drops empty
    # parts, so an unnamespaced resource just falls back to (kind, name) --
    # a documented simplification (this repo's own samples rarely set
    # namespace explicitly), not a false-collision risk in the common case.
    return _make_id(kind, namespace, name)


def _walk_configmap_secret_refs(node, owner_nid, namespace, add_edge, ref_stub):
    mapping = _mapping(node)
    if mapping is not None:
        for key, value, line in _pairs(mapping):
            kind = "ConfigMap" if key in _CONFIGMAP_REF_KEYS else "Secret" if key in _SECRET_REF_KEYS else None
            if kind:
                sub = _mapping(value)
                if sub is not None:
                    sub_pairs = {k: v for k, v, _l in _pairs(sub)}
                    ref_name = ""
                    for field in _REF_NAME_FIELDS:
                        if field in sub_pairs:
                            ref_name = _scalar_text(sub_pairs[field])
                            if ref_name:
                                break
                    if ref_name:
                        tgt = ref_stub(_resource_id(kind, namespace, ref_name), f"{kind}/{ref_name}")
                        add_edge(owner_nid, tgt, "uses", line)
                continue
            _walk_configmap_secret_refs(value, owner_nid, namespace, add_edge, ref_stub)
        return
    for item in _sequence_items(node):
        _walk_configmap_secret_refs(_item_value(item), owner_nid, namespace, add_edge, ref_stub)


def _infer_ref_kind(field_key: str, suffix: str) -> str:
    base = field_key[: -len(suffix)]
    return base[0].upper() + base[1:] if base else ""


def _walk_ref_convention(node, owner_nid, namespace, add_edge, ref_stub):
    mapping = _mapping(node)
    if mapping is not None:
        for key, value, line in _pairs(mapping):
            if key in _CONFIGMAP_REF_KEYS or key in _SECRET_REF_KEYS:
                continue  # handled by _walk_configmap_secret_refs; do not double-classify as a generic *Ref
            if key.endswith("Refs") and len(key) > len("Refs"):
                kind = _infer_ref_kind(key, "Refs")
                for item_text, item_line in _string_items(value):
                    tgt = ref_stub(_resource_id(kind, namespace, item_text), f"{kind}/{item_text}")
                    add_edge(owner_nid, tgt, "references", item_line)
            elif key.endswith("Ref") and len(key) > len("Ref"):
                kind = _infer_ref_kind(key, "Ref")
                ref_text = _scalar_text(value)
                if ref_text:
                    tgt = ref_stub(_resource_id(kind, namespace, ref_text), f"{kind}/{ref_text}")
                    add_edge(owner_nid, tgt, "references", line)
            else:
                _walk_ref_convention(value, owner_nid, namespace, add_edge, ref_stub)
        return
    for item in _sequence_items(node):
        _walk_ref_convention(_item_value(item), owner_nid, namespace, add_edge, ref_stub)


def _emit_label_hub_edges(labels_node, owner_nid, relation, add_edge, ref_stub):
    mapping = _mapping(labels_node)
    if mapping is None:
        return
    for key, value, line in _pairs(mapping):
        value_text = _scalar_text(value)
        if value_text:
            hub = ref_stub(_make_id("label", key, value_text), f"{key}={value_text}")
            add_edge(owner_nid, hub, relation, line)


def _handle_selectors_and_labels(spec_pairs, meta_pairs, owner_nid, add_edge, ref_stub):
    if "selector" in spec_pairs:
        selector_value, _l = spec_pairs["selector"]
        selector_mapping = _mapping(selector_value)
        if selector_mapping is not None:
            for key, value, line in _pairs(selector_mapping):
                if key == "matchLabels":
                    ml = _mapping(value)
                    if ml is not None:
                        for lk, lv, lline in _pairs(ml):
                            lv_text = _scalar_text(lv)
                            if lv_text:
                                hub = ref_stub(_make_id("label", lk, lv_text), f"{lk}={lv_text}")
                                add_edge(owner_nid, hub, "selects", lline)
                    continue
                if key == "matchExpressions":
                    # Operator-based selectors (In/NotIn/Exists/DoesNotExist)
                    # have no single value to hub on -- skip rather than guess.
                    continue
                value_text = _scalar_text(value)
                if value_text:
                    hub = ref_stub(_make_id("label", key, value_text), f"{key}={value_text}")
                    add_edge(owner_nid, hub, "selects", line)

    if "labels" in meta_pairs:
        _emit_label_hub_edges(meta_pairs["labels"][0], owner_nid, "has_label", add_edge, ref_stub)

    if "template" in spec_pairs:
        template_mapping = _mapping(spec_pairs["template"][0])
        if template_mapping is not None:
            template_pairs = {k: (v, l) for k, v, l in _pairs(template_mapping)}
            if "metadata" in template_pairs:
                tmpl_meta = _mapping(template_pairs["metadata"][0])
                if tmpl_meta is not None:
                    tmpl_meta_pairs = {k: (v, l) for k, v, l in _pairs(tmpl_meta)}
                    if "labels" in tmpl_meta_pairs:
                        _emit_label_hub_edges(tmpl_meta_pairs["labels"][0], owner_nid, "has_label", add_edge, ref_stub)


def extract_k8s_resources(resource_tops: list, str_path: str, file_nid: str) -> tuple[list[dict], list[dict]]:
    """Extract resource nodes and ownership/reference/selector edges for a
    list of already-shape-validated top-level mappings (one per k8s
    resource -- callers filter with is_k8s_manifest_shape() first).

    Split out from extract_k8s_manifest() so a combined dispatcher handling
    a mixed file (some documents k8s-shaped, some not) can call this for
    just the matching subset while owning the file node itself centrally --
    see graphify/extractors/yaml_dispatch.py.

    Nodes: one per resource (keyed globally by (kind, namespace, name), not
    file-scoped -- unlike a GitHub Actions job, the same resource can
    legitimately be defined in one file and referenced from many others).
    Sourceless stub nodes (type=module, same hub-collapsing exemption
    GitHub Actions' shared-action stubs use, #1327) stand in for anything
    referenced but not locally defined in the same file, so cross-file
    ownership/reference edges collapse onto the real definition when it
    exists elsewhere in the same extraction batch, and survive as a
    portable node when it doesn't.

    See the module docstring for the full list of relationships modelled
    and their known, deliberate limitations. Returns (nodes, edges) -- does
    NOT include the file node/contains-edge-from-file, which is the
    caller's responsibility (shared across k8s and generic-structural
    resources in the same file).
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = {file_nid}
    seen_edges: set[tuple[str, str, str]] = set()
    local_nids: dict[tuple[str, str, str], str] = {}

    def _ref_stub(nid: str, label: str) -> str:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": "", "source_location": "",
                          "origin_file": str_path, "type": "module"})
        return nid

    def _add_edge(src: str, tgt: str, relation: str, line: int) -> None:
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, relation)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line}", "weight": 1.0})

    # Parse each resource's top-level fields once, for the two-pass process
    # below (definitions first, so a same-file forward reference -- e.g. the
    # Namespace and Deployment in manager.yaml -- binds locally).
    parsed = []
    for top in resource_tops:
        pairs = {key: (value, line) for key, value, line in _pairs(top)}
        kind = _scalar_text(pairs["kind"][0])
        metadata = _mapping(pairs["metadata"][0])
        meta_pairs = {key: (value, line) for key, value, line in _pairs(metadata)} if metadata is not None else {}
        name = _scalar_text(meta_pairs["name"][0]) if "name" in meta_pairs else ""
        if not name:
            continue
        namespace = _scalar_text(meta_pairs["namespace"][0]) if "namespace" in meta_pairs else ""
        spec_entry = pairs.get("spec")
        parsed.append({
            "kind": kind, "name": name, "namespace": namespace,
            "meta_pairs": meta_pairs,
            "spec": spec_entry[0] if spec_entry else None,
            "line": pairs["kind"][1],
        })

    if not parsed:
        return nodes, edges

    for r in parsed:
        nid = _resource_id(r["kind"], r["namespace"], r["name"])
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": f"{r['kind']}/{r['name']}", "file_type": "code",
                          "source_file": str_path, "source_location": f"L{r['line']}"})
            edges.append({"source": file_nid, "target": nid, "relation": "contains",
                          "confidence": "EXTRACTED", "source_file": str_path,
                          "source_location": f"L{r['line']}", "weight": 1.0})
        local_nids[(r["kind"], r["namespace"], r["name"])] = nid
        r["nid"] = nid

    for r in parsed:
        owner_nid = r["nid"]
        namespace = r["namespace"]
        meta_pairs = r["meta_pairs"]

        # -- standard metadata.ownerReferences --
        if "ownerReferences" in meta_pairs:
            owner_refs_node, _l = meta_pairs["ownerReferences"]
            for item in _sequence_items(owner_refs_node):
                ref_mapping = _mapping(_item_value(item))
                if ref_mapping is None:
                    continue
                ref_pairs = {k: v for k, v, _ln in _pairs(ref_mapping)}
                ref_kind = _scalar_text(ref_pairs.get("kind"))
                ref_name = _scalar_text(ref_pairs.get("name"))
                if not ref_kind or not ref_name:
                    continue
                parent_nid = local_nids.get((ref_kind, namespace, ref_name)) or _ref_stub(
                    _resource_id(ref_kind, namespace, ref_name), f"{ref_kind}/{ref_name}")
                _add_edge(parent_nid, owner_nid, "owns", r["line"])

        # -- custom annotation-based owner-reference + tenant scoping --
        if "annotations" in meta_pairs:
            ann_mapping = _mapping(meta_pairs["annotations"][0])
            if ann_mapping is not None:
                for key, value, line in _pairs(ann_mapping):
                    if key == _OWNER_ANNOTATION:
                        owner_ref_value = _scalar_text(value)
                        if owner_ref_value:
                            parent_nid = _ref_stub(_make_id("owner-ref", owner_ref_value), owner_ref_value)
                            _add_edge(parent_nid, owner_nid, "owns", line)
                    elif key == _TENANT_ANNOTATION:
                        tenant_value = _scalar_text(value)
                        if tenant_value:
                            tenant_nid = _ref_stub(_make_id("tenant", tenant_value), tenant_value)
                            _add_edge(owner_nid, tenant_nid, "scoped_to", line)

        if r["spec"] is not None:
            _walk_configmap_secret_refs(r["spec"], owner_nid, namespace, _add_edge, _ref_stub)
            _walk_ref_convention(r["spec"], owner_nid, namespace, _add_edge, _ref_stub)

        spec_pairs = {}
        if r["spec"] is not None:
            spec_mapping = _mapping(r["spec"])
            if spec_mapping is not None:
                spec_pairs = {k: (v, l) for k, v, l in _pairs(spec_mapping)}
        _handle_selectors_and_labels(spec_pairs, meta_pairs, owner_nid, _add_edge, _ref_stub)

    return nodes, edges


def extract_k8s_manifest(path: Path) -> dict:
    """Standalone entry point: parse *path* and extract k8s resources from
    it directly (used for direct testing/backward-compat; the combined
    dispatcher in yaml_dispatch.py calls extract_k8s_resources() directly
    on a file it has already parsed, to avoid re-parsing).

    Any YAML that doesn't validate as a real k8s resource (data YAML, an
    unrelated config that happens to share a key name, or a
    Helm-templated file whose Go template syntax breaks the YAML grammar)
    returns an empty result and is left to the semantic pass.
    """
    _YAML_MAX_BYTES = 1_048_576  # 1 MiB -- manifests are small; this rejects junk

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

    resource_tops = [top for top in _all_top_level_mappings(root) if is_k8s_manifest_shape(top)]
    if not resource_tops:
        return {"nodes": [], "edges": []}

    str_path = str(path)
    file_nid = _make_id(str_path)
    resource_nodes, resource_edges = extract_k8s_resources(resource_tops, str_path, file_nid)
    nodes = [{"id": file_nid, "label": path.name, "file_type": "code",
              "source_file": str_path, "source_location": None}] + resource_nodes
    return {"nodes": nodes, "edges": resource_edges}
