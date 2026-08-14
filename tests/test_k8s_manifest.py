"""Tests for the Kubernetes manifest extractor
(graphify/extractors/k8s_manifest.py) and the combined YAML dispatcher
(graphify/extractors/yaml_dispatch.py) that layers it over the generic
structural fallback (graphify/extractors/yaml_generic.py) -- OSAC-4050.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from graphify.build import build_from_json
from graphify.extract import extract, extract_k8s_manifest, extract_yaml


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _labels(r) -> list[str]:
    return [n["label"] for n in r["nodes"]]


def _rel_pairs(r, relation: str) -> set[tuple[str, str]]:
    lab = {n["id"]: n["label"] for n in r["nodes"]}
    return {
        (lab.get(e["source"], e["source"]), lab.get(e["target"], e["target"]))
        for e in r["edges"]
        if e["relation"] == relation
    }


@pytest.fixture(autouse=True)
def _require_grammar():
    pytest.importorskip("tree_sitter_yaml")


# ── ownerReferences (standard k8s) ──────────────────────────────────────────

OWNER_REF_CHILD = """\
apiVersion: v1
kind: Pod
metadata:
  name: worker-pod
  ownerReferences:
    - apiVersion: apps/v1
      kind: ReplicaSet
      name: worker-rs
"""


def test_owner_references_become_owns_edges(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "pod.yaml", OWNER_REF_CHILD))
    assert r.get("error") is None
    assert ("ReplicaSet/worker-rs", "Pod/worker-pod") in _rel_pairs(r, "owns")


def test_owner_reference_resolves_to_real_definition_across_files(tmp_path):
    parent = _write(tmp_path, "rs.yaml", "apiVersion: apps/v1\nkind: ReplicaSet\nmetadata:\n  name: worker-rs\n")
    child = _write(tmp_path, "pod.yaml", OWNER_REF_CHILD)
    r = extract([parent.resolve(), child.resolve()], root=tmp_path)
    rs_ids = {n["id"] for n in r["nodes"] if n["label"] == "ReplicaSet/worker-rs"}
    assert len(rs_ids) == 1, f"expected one ReplicaSet node, got {rs_ids}"
    assert rs_ids.pop() in {e["source"] for e in r["edges"] if e["relation"] == "owns"}


def test_different_api_groups_same_kind_namespace_name_do_not_collide(tmp_path):
    """Regression test for a real reviewer-caught bug: resource identity
    was (kind, namespace, name) with no API group, so two genuinely
    different resources from different groups sharing a kind/namespace/name
    (a legitimate real k8s scenario -- apiVersion exists precisely to allow
    this, e.g. NetworkPolicy historically existed in both extensions/v1beta1
    and networking.k8s.io/v1) would silently merge into one node."""
    body = (
        "apiVersion: apps/v1\nkind: NetworkPolicy\nmetadata:\n  name: np\n  namespace: osac\n"
        "---\n"
        "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: np\n  namespace: osac\n"
    )
    r = extract_k8s_manifest(_write(tmp_path, "networkpolicies.yaml", body))
    np_nodes = [n for n in r["nodes"] if n["label"] == "NetworkPolicy/np"]
    assert len(np_nodes) == 2, f"expected two distinct NetworkPolicy/np nodes (different groups), got {np_nodes}"
    assert len({n["id"] for n in np_nodes}) == 2, "the two resources must have distinct ids"


def test_owner_reference_apiversion_group_disambiguates_cross_group_owner(tmp_path):
    """An ownerReference names its owner's group via its own apiVersion
    field -- confirms that field is actually used to resolve to the
    correctly-grouped owner, not just any resource sharing the kind/name."""
    body = (
        "apiVersion: apps/v1\nkind: Foo\nmetadata:\n  name: shared-name\n"
        "---\n"
        "apiVersion: osac.openshift.io/v1alpha1\nkind: Foo\nmetadata:\n  name: shared-name\n"
        "---\n"
        "apiVersion: v1\nkind: Bar\nmetadata:\n  name: child\n"
        "  ownerReferences:\n    - apiVersion: osac.openshift.io/v1alpha1\n      kind: Foo\n      name: shared-name\n"
    )
    r = extract_k8s_manifest(_write(tmp_path, "mixed-groups.yaml", body))
    foo_nodes = {n["id"]: n for n in r["nodes"] if n["label"] == "Foo/shared-name"}
    assert len(foo_nodes) == 2
    owns_edges = [e for e in r["edges"] if e["relation"] == "owns"]
    assert len(owns_edges) == 1
    owner_id = owns_edges[0]["source"]
    assert foo_nodes[owner_id]["source_file"], "must resolve to a real definition, not a fresh stub"
    # The resolved owner must be the osac.openshift.io one, not the apps one.
    assert "osac" in owner_id or "openshift" in owner_id, f"resolved to the wrong group's Foo: {owner_id}"


# ── custom annotation-based owner-reference + tenant (architecture-patterns.md) ──

ANNOTATED_CHILD = """\
apiVersion: osac.openshift.io/v1alpha1
kind: Subnet
metadata:
  name: my-subnet
  annotations:
    osac.openshift.io/owner-reference: 00000000-0000-0000-0000-000000000000
    osac.openshift.io/tenant: my-tenant
"""


def test_custom_owner_reference_annotation_becomes_owns_edge(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "subnet.yaml", ANNOTATED_CHILD))
    owns = _rel_pairs(r, "owns")
    assert ("00000000-0000-0000-0000-000000000000", "Subnet/my-subnet") in owns


def test_tenant_annotation_becomes_scoped_to_edge(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "subnet.yaml", ANNOTATED_CHILD))
    assert ("Subnet/my-subnet", "my-tenant") in _rel_pairs(r, "scoped_to")


# ── ConfigMap / Secret references ───────────────────────────────────────────

WORKLOAD_WITH_REFS = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  template:
    spec:
      containers:
        - name: app
          envFrom:
            - configMapRef:
                name: app-config
            - secretRef:
                name: app-secrets
          env:
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: password
          volumeMounts:
            - name: cfg
              mountPath: /etc/cfg
      volumes:
        - name: cfg
          configMap:
            name: shared-config
"""


def test_configmap_ref_from_envfrom(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "deploy.yaml", WORKLOAD_WITH_REFS))
    uses = _rel_pairs(r, "uses")
    assert ("Deployment/api", "ConfigMap/app-config") in uses


def test_secret_ref_from_envfrom(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "deploy.yaml", WORKLOAD_WITH_REFS))
    assert ("Deployment/api", "Secret/app-secrets") in _rel_pairs(r, "uses")


def test_secret_key_ref_from_env_valuefrom(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "deploy.yaml", WORKLOAD_WITH_REFS))
    assert ("Deployment/api", "Secret/db-secret") in _rel_pairs(r, "uses")


def test_configmap_ref_from_volume(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "deploy.yaml", WORKLOAD_WITH_REFS))
    assert ("Deployment/api", "ConfigMap/shared-config") in _rel_pairs(r, "uses")


def test_configmap_secret_ref_resolves_across_files(tmp_path):
    cm = _write(tmp_path, "cm.yaml", "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: app-config\n")
    dep = _write(tmp_path, "deploy.yaml", WORKLOAD_WITH_REFS)
    r = extract([cm.resolve(), dep.resolve()], root=tmp_path)
    cm_ids = {n["id"] for n in r["nodes"] if n["label"] == "ConfigMap/app-config"}
    assert len(cm_ids) == 1


# ── *Ref / *Refs CRD cross-reference convention ─────────────────────────────

COMPUTE_INSTANCE = """\
apiVersion: osac.openshift.io/v1alpha1
kind: ComputeInstance
metadata:
  name: computeinstance-sample
spec:
  networkAttachments:
    - subnetRef: my-subnet
      securityGroupRefs:
        - web-sg
        - monitoring-sg
"""


def test_singular_ref_convention(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "ci.yaml", COMPUTE_INSTANCE))
    refs = _rel_pairs(r, "references")
    assert ("ComputeInstance/computeinstance-sample", "Subnet/my-subnet") in refs


def test_plural_refs_convention(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "ci.yaml", COMPUTE_INSTANCE))
    refs = _rel_pairs(r, "references")
    assert ("ComputeInstance/computeinstance-sample", "SecurityGroup/web-sg") in refs
    assert ("ComputeInstance/computeinstance-sample", "SecurityGroup/monitoring-sg") in refs


def test_bare_field_without_ref_suffix_is_not_treated_as_a_reference(tmp_path):
    # Subnet.spec.virtualNetwork (this repo's own real sample) is a UUID with
    # no Ref/Refs suffix -- deliberately not modelled (module docstring).
    body = ("apiVersion: osac.openshift.io/v1alpha1\nkind: Subnet\nmetadata:\n"
            "  name: subnet-sample\nspec:\n  virtualNetwork: 00000000-0000-0000-0000-000000000000\n")
    r = extract_k8s_manifest(_write(tmp_path, "subnet.yaml", body))
    assert _rel_pairs(r, "references") == set()


# ── label-selector matching ──────────────────────────────────────────────────

SERVICE = "apiVersion: v1\nkind: Service\nmetadata:\n  name: osac-console-proxy\nspec:\n  selector:\n    app: osac-console-proxy\n"
DEPLOYMENT = (
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: osac-console-proxy\nspec:\n"
    "  template:\n    metadata:\n      labels:\n        app: osac-console-proxy\n"
)


def test_service_selector_and_deployment_labels_share_a_hub(tmp_path):
    """Real example from this repo: osac-operator/config/console-proxy/
    service.yaml's selector matches deployment.yaml's pod template labels
    exactly (single key=value pair -- the exact case, not the approximate
    multi-key one)."""
    svc = _write(tmp_path, "service.yaml", SERVICE)
    dep = _write(tmp_path, "deployment.yaml", DEPLOYMENT)
    r = extract([svc.resolve(), dep.resolve()], root=tmp_path)

    hub_ids = {n["id"] for n in r["nodes"] if n["label"] == "app=osac-console-proxy"}
    assert len(hub_ids) == 1, f"expected one shared label hub, got {hub_ids}"
    hub_id = hub_ids.pop()

    selects_sources = {e["source"] for e in r["edges"] if e["relation"] == "selects" and e["target"] == hub_id}
    has_label_sources = {e["source"] for e in r["edges"] if e["relation"] == "has_label" and e["target"] == hub_id}
    assert len(selects_sources) == 1
    assert len(has_label_sources) == 1

    G = build_from_json({"nodes": r["nodes"], "edges": r["edges"]})
    assert G.has_node(hub_id)


def test_metadata_labels_also_emit_has_label(tmp_path):
    body = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: worker\n  labels:\n    tier: backend\n"
    r = extract_k8s_manifest(_write(tmp_path, "pod.yaml", body))
    assert ("Pod/worker", "tier=backend") in _rel_pairs(r, "has_label")


def test_match_expressions_are_skipped_not_guessed(tmp_path):
    body = (
        "apiVersion: apps/v1\nkind: NetworkPolicy\nmetadata:\n  name: np\nspec:\n"
        "  selector:\n    matchExpressions:\n      - key: tier\n        operator: In\n        values: [backend]\n"
    )
    r = extract_k8s_manifest(_write(tmp_path, "np.yaml", body))
    assert _rel_pairs(r, "selects") == set()


# ── shape validation: real vs coincidental key names ────────────────────────

def test_key_presence_alone_is_not_enough(tmp_path):
    # Has apiVersion/kind/metadata as keys, but kind isn't a real k8s-style
    # PascalCase type and apiVersion isn't a real k8s apiVersion string.
    body = "apiVersion: yes\nkind: not-a-real-kind\nmetadata:\n  name: x\n"
    r = extract_k8s_manifest(_write(tmp_path, "coincidence.yaml", body))
    assert r["nodes"] == []


def test_data_yaml_returns_empty(tmp_path):
    body = "openapi: 3.0.0\npaths:\n  /users:\n    get:\n      summary: list users\n"
    r = extract_k8s_manifest(_write(tmp_path, "openapi.yaml", body))
    assert r["nodes"] == []
    assert r["edges"] == []


def test_docker_compose_is_out_of_scope(tmp_path):
    body = "services:\n  api:\n    image: api:latest\n"
    r = extract_k8s_manifest(_write(tmp_path, "docker-compose.yml", body))
    assert r["nodes"] == []


def test_standalone_entry_point_also_rejects_templated_yaml(tmp_path):
    """Regression test for a real reviewer-caught bypass: extract_k8s_manifest()
    is a separate, directly-callable, re-exported entry point (not wired
    into the real _DISPATCH pipeline, but used directly by most of this
    test file and importable via graphify.extract) that previously lacked
    the has_error/template-marker gate yaml_dispatch.py has -- a templated
    file routed through THIS function instead of extract_yaml() would have
    been silently mis-extracted as a valid resource."""
    body = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\nspec:\n  replicas: {{ .Values.replicaCount }}\n"
    r = extract_k8s_manifest(_write(tmp_path, "deployment.yaml", body))
    assert r == {"nodes": [], "edges": []}


# ── multi-document files ────────────────────────────────────────────────────

def test_multi_document_file_extracts_all_resources(tmp_path):
    """Real shape from osac-operator/config/manager/manager.yaml: a
    Namespace and a Deployment bundled in one file via `---`."""
    body = (
        "apiVersion: v1\nkind: Namespace\nmetadata:\n  name: osac\n"
        "---\n"
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: osac-controller-manager\n"
        "  namespace: osac\n  ownerReferences:\n    - apiVersion: v1\n      kind: Namespace\n      name: osac\n"
    )
    r = extract_k8s_manifest(_write(tmp_path, "manager.yaml", body))
    labels = set(_labels(r))
    assert "Namespace/osac" in labels
    assert "Deployment/osac-controller-manager" in labels
    assert ("Namespace/osac", "Deployment/osac-controller-manager") in _rel_pairs(r, "owns")


def test_no_dangling_edge_endpoints(tmp_path):
    r = extract_k8s_manifest(_write(tmp_path, "deploy.yaml", WORKLOAD_WITH_REFS))
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids
        assert e["target"] in node_ids


def test_cluster_scoped_owner_resolves_to_one_real_node_not_a_duplicate_stub(tmp_path):
    """Regression test for a real reviewer-caught bug: a namespaced child
    (Deployment, namespace "osac") owned by a cluster-scoped resource
    (Namespace "osac" itself, which has no metadata.namespace of its own)
    previously had its owner lookup keyed by the CHILD's namespace, missing
    the real Namespace node (indexed with an empty namespace) and minting a
    duplicate stub instead.

    Deliberately asserts on real node ID uniqueness/identity, not just
    label pairs via _rel_pairs/_labels -- those helpers are blind to two
    distinct node dicts that happen to share the same label, which is
    exactly the shape this bug produced.
    """
    body = (
        "apiVersion: v1\nkind: Namespace\nmetadata:\n  name: osac\n"
        "---\n"
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: osac-controller-manager\n"
        "  namespace: osac\n  ownerReferences:\n    - apiVersion: v1\n      kind: Namespace\n      name: osac\n"
    )
    r = extract_yaml(_write(tmp_path, "manager.yaml", body))

    namespace_nodes = [n for n in r["nodes"] if n["label"] == "Namespace/osac"]
    assert len(namespace_nodes) == 1, f"expected exactly one Namespace/osac node, got {namespace_nodes}"
    real_namespace_id = namespace_nodes[0]["id"]
    # The real definition has a source_file/source_location; a stub does not.
    assert namespace_nodes[0]["source_file"], "the surviving node must be the real definition, not a sourceless stub"

    owns_edges = [e for e in r["edges"] if e["relation"] == "owns"]
    assert len(owns_edges) == 1
    assert owns_edges[0]["source"] == real_namespace_id, (
        f"owns edge must bind to the real Namespace node ({real_namespace_id}), "
        f"not a duplicate stub (got {owns_edges[0]['source']!r})"
    )


def test_cluster_scoped_owner_across_separate_files_still_resolves(tmp_path):
    """Same scenario as above, but the Namespace and Deployment are two
    separate files fed through extract() together -- confirms the fix
    holds for genuine cross-file resolution too, not just same-file
    same-batch resolution."""
    ns = _write(tmp_path, "namespace.yaml", "apiVersion: v1\nkind: Namespace\nmetadata:\n  name: osac\n")
    dep = _write(tmp_path, "deployment.yaml", (
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: osac-controller-manager\n"
        "  namespace: osac\n  ownerReferences:\n    - apiVersion: v1\n      kind: Namespace\n      name: osac\n"
    ))
    r = extract([ns.resolve(), dep.resolve()], root=tmp_path)
    # Two independent per-file extractions can each mint their own raw dict
    # for the same id (one real, one stub) -- true collapsing across files
    # happens when the graph itself is built (same precedent as
    # test_shared_action_merges_across_workflows in test_github_actions.py),
    # so assert on the unique ID set and the built graph, not the raw list.
    namespace_ids = {n["id"] for n in r["nodes"] if n["label"] == "Namespace/osac"}
    assert len(namespace_ids) == 1, f"expected one shared Namespace/osac id, got {namespace_ids}"
    namespace_id = namespace_ids.pop()

    G = build_from_json({"nodes": r["nodes"], "edges": r["edges"]})
    assert G.has_node(namespace_id)

    owns_edges = [e for e in r["edges"] if e["relation"] == "owns"]
    assert len(owns_edges) == 1
    assert owns_edges[0]["source"] == namespace_id


# ── combined dispatcher: layering + templated-YAML safety ──────────────────

def test_dispatcher_routes_k8s_shaped_yaml_to_rich_extraction(tmp_path):
    r = extract_yaml(_write(tmp_path, "service.yaml", SERVICE))
    assert "selects" in {e["relation"] for e in r["edges"]}


def test_dispatcher_falls_back_to_generic_structure_for_non_k8s_yaml(tmp_path):
    body = "replicaCount: 3\nimage:\n  repository: myapp\n"
    r = extract_yaml(_write(tmp_path, "values.yaml", body))
    labels = set(_labels(r))
    assert "replicaCount" in labels
    assert "image" in labels
    assert "repository" in labels
    # No k8s-specific relations should appear for a plain values file.
    assert not ({"owns", "uses", "selects", "has_label", "references"} & {e["relation"] for e in r["edges"]})


def test_dispatcher_skips_templated_yaml_without_crashing(tmp_path, capsys):
    """Real Go-templated shape from osac-operator/charts/operator/templates/
    metrics-service.yaml -- confirmed empirically (during this ticket's
    investigation) to produce a tree-sitter ERROR node, not a best-effort
    partial parse."""
    body = (
        "apiVersion: v1\nkind: Service\nmetadata:\n"
        '  name: {{ include "osac-operator.fullname" . }}-metrics\n'
        "spec:\n  selector:\n    {{- include \"osac-operator.selectorLabels\" . | nindent 4 }}\n"
    )
    p = _write(tmp_path, "metrics-service.yaml", body)
    r = extract_yaml(p)
    assert r == {"nodes": [], "edges": []}
    captured = capsys.readouterr()
    assert "metrics-service.yaml" in captured.err
    assert "not treated as real YAML" in captured.err


def test_dispatcher_handles_mixed_multi_doc_file(tmp_path):
    """One k8s-shaped document and one plain document in the same file --
    each should be routed to the correct layer independently."""
    body = (
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cm\ndata:\n  key: value\n"
        "---\n"
        "plainKey: plainValue\n"
    )
    r = extract_yaml(_write(tmp_path, "mixed.yaml", body))
    labels = set(_labels(r))
    assert "ConfigMap/cm" in labels
    assert "plainKey" in labels


def test_dispatcher_skips_bare_inline_template_value(tmp_path, capsys):
    """Regression test for a real reviewer-caught gap: `replicas: {{ .Values.x }}`
    (arguably the single most common Helm templating idiom -- more common
    than the `{{ include ... }}` pattern in test_dispatcher_skips_templated_yaml_without_crashing)
    parses with has_error=False on the specific document node the dispatcher
    checks (confirmed empirically) -- `{{` looks like valid, if bogus,
    nested flow-mapping syntax to the YAML grammar, not a parse error. Only
    the additional raw-text `{{` marker check catches this; has_error alone
    does not."""
    body = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\nspec:\n  replicas: {{ .Values.replicaCount }}\n"
    p = _write(tmp_path, "deployment.yaml", body)
    r = extract_yaml(p)
    assert r == {"nodes": [], "edges": []}
    assert "not treated as real YAML" in capsys.readouterr().err


def test_dispatcher_skips_concatenated_template_blocks(tmp_path, capsys):
    """Regression test for the opposite direction of the same reviewer-caught
    gap: `image: {{ .Values.x }}:{{ .Values.y }}` sets has_error=True on the
    STREAM root but NOT on the specific per-document node the dispatcher
    checks (confirmed empirically) -- has_error alone misses an error that
    exists elsewhere in the same parse tree. Only the additional raw-text
    `{{` marker check catches this."""
    body = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: x\nspec:\n  containers:\n    - image: {{ .Values.x }}:{{ .Values.y }}\n"
    p = _write(tmp_path, "pod.yaml", body)
    r = extract_yaml(p)
    assert r == {"nodes": [], "edges": []}
    assert "not treated as real YAML" in capsys.readouterr().err
