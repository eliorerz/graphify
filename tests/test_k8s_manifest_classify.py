"""classify_file() coverage for OSAC-4050's universal YAML/JSON direction
change (graphify/detect.py).

Deliberately separate from tests/test_k8s_manifest.py: these tests only
call classify_file(), never parse YAML/JSON content, so they don't need
tree_sitter_yaml installed -- but that module's `_require_grammar` autouse
fixture would skip them anyway if they lived there (a real reviewer-caught
gap: reduced test coverage when the optional [yaml] extra is absent, for
tests that never needed it in the first place).
"""
from __future__ import annotations

from pathlib import Path

from graphify.detect import FileType, classify_file


def test_all_yaml_classified_as_code():
    # OSAC-4050: matches .json's existing precedent -- every .yaml/.yml is
    # CODE unconditionally now, regardless of shape (k8s-shaped or not).
    assert classify_file(Path("charts/myapp/values.yaml")) == FileType.CODE
    assert classify_file(Path("k8s/deployment.yaml")) == FileType.CODE
    assert classify_file(Path("openapi.yaml")) == FileType.CODE
    assert classify_file(Path("docker-compose.yml")) == FileType.CODE
    assert classify_file(Path(".github/actions/setup/action.yml")) == FileType.CODE


def test_all_json_still_classified_as_code():
    # Unaffected by this ticket -- .json was already unconditionally CODE.
    assert classify_file(Path("data.json")) == FileType.CODE
    assert classify_file(Path("package.json")) == FileType.CODE
