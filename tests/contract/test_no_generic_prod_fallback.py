from __future__ import annotations

import pytest

from core.template_pipeline import TemplateSelectionError, load_manifest, select_template


def test_production_mode_never_uses_generic_fallback() -> None:
    manifest = load_manifest()
    scene = {
        "id": 4,
        "uid": "x4",
        "title": "Coherence map",
        "narration": "Coherence map",
        "scene_type": "coherence_network_map",
        "structured_data": {"nodes": [], "edges": []},
    }
    with pytest.raises(TemplateSelectionError):
        select_template(scene, manifest, production_only=True)

