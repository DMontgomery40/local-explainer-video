from __future__ import annotations

from core.template_pipeline import TEMPLATE_OPERATIONAL_STATUSES, TEMPLATE_ORIGINS, load_manifest


def test_manifest_entries_have_required_release_metadata() -> None:
    manifest = load_manifest()
    for spec in manifest.templates:
        assert spec.origin in TEMPLATE_ORIGINS
        assert spec.operational_status in TEMPLATE_OPERATIONAL_STATUSES
        assert isinstance(spec.production_ready, bool)
        assert isinstance(spec.selector_metadata, dict)


def test_generic_panel_is_dev_only_and_not_production_ready() -> None:
    manifest = load_manifest()
    generic = next(spec for spec in manifest.templates if spec.template_id == "generic_data_panel_v1")
    assert generic.operational_status == "dev_only"
    assert generic.production_ready is False

