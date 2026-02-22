from __future__ import annotations

from core.template_pipeline import load_manifest


def test_scaffold_templates_are_dev_only_until_curated() -> None:
    manifest = load_manifest()
    scaffold_specs = [spec for spec in manifest.templates if spec.origin == "scaffold_only"]
    assert scaffold_specs, "Expected at least one scaffold template while curation is in progress"
    for spec in scaffold_specs:
        assert spec.operational_status == "dev_only"
        assert spec.production_ready is False
