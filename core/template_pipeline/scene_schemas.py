"""Typed scene schemas for deterministic template rendering."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_VERSION = "4.0.0"

# Friendly descriptive scene types (LLM-facing) aligned to the 22 data-bearing
# archetypes plus atmospheric templates.
DATA_SCENE_TYPES: tuple[str, ...] = (
    "split_opposing_trends",
    "multi_session_trend",
    "verdict_summary",
    "roadmap_agenda",
    "session_timeline",
    "gauge_ratio_meter",
    "coherence_network_map",
    "hemispheric_compare",
    "state_flexibility_rest_task",
    "baseline_target_split",
    "measurement_primer",
    "future_projection",
    "line_trajectory",
    "bar_volume_chart",
    "coherence_progression_sequence",
    "dotplot_variability",
    "pathway_hub_synthesis",
    "regional_frequency_map",
    "table_dashboard",
    "quality_alert",
    "waveform_voltage_panel",
    "radial_kpi_ring",
)

ATMOSPHERIC_SCENE_TYPES: tuple[str, ...] = (
    "atmospheric_title_card",
    "atmospheric_metaphor_scene",
    "atmospheric_mechanism_scene",
)

ALL_SCENE_TYPES: tuple[str, ...] = DATA_SCENE_TYPES + ATMOSPHERIC_SCENE_TYPES


# Backward compatibility for taxonomy codes seen in manual archetype CSV.
LEGACY_ARCHETYPE_TO_SCENE_TYPE: dict[str, str] = {
    "SPLIT": "split_opposing_trends",
    "MULTI_TREND": "multi_session_trend",
    "SUMMARY": "verdict_summary",
    "ROADMAP": "roadmap_agenda",
    "TIMELINE": "session_timeline",
    "GAUGE": "gauge_ratio_meter",
    "COH_MAP": "coherence_network_map",
    "HEMI": "hemispheric_compare",
    "STATE": "state_flexibility_rest_task",
    "BASELINE_SPLIT": "baseline_target_split",
    "MEASURE": "measurement_primer",
    "FUTURE": "future_projection",
    "LINE_TRAJECTORY": "line_trajectory",
    "BAR": "bar_volume_chart",
    "COH_SEQ": "coherence_progression_sequence",
    "DOTPLOT": "dotplot_variability",
    "PATHWAY": "pathway_hub_synthesis",
    "REGIONAL": "regional_frequency_map",
    "TABLE": "table_dashboard",
    "QC": "quality_alert",
    "WAVEFORM": "waveform_voltage_panel",
    "RADIAL": "radial_kpi_ring",
    "AT_TITLE": "atmospheric_title_card",
    "AT_METAPHOR": "atmospheric_metaphor_scene",
    "AT_MECHANISM": "atmospheric_mechanism_scene",
}


def normalize_scene_type(raw_scene_type: str | None) -> str | None:
    value = str(raw_scene_type or "").strip()
    if not value:
        return None
    if value in ALL_SCENE_TYPES:
        return value
    upper = value.upper()
    if upper in LEGACY_ARCHETYPE_TO_SCENE_TYPE:
        return LEGACY_ARCHETYPE_TO_SCENE_TYPE[upper]
    return value


class StrictModel(BaseModel):
    """Base model with strict field handling."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class NumericRange(StrictModel):
    min: float = Field(validation_alias=AliasChoices("min", "low"))
    max: float = Field(validation_alias=AliasChoices("max", "high"))
    unit: str | None = None


class MetricDelta(StrictModel):
    metric: str = ""
    from_value: float = Field(default=0.0, alias="from")
    to_value: float = Field(default=0.0, alias="to")
    unit: str | None = None
    direction: str | None = None
    pct_change: float | None = None


class LabeledValue(StrictModel):
    label: str = ""
    value: float = 0.0
    unit: str | None = None
    color_key: str | None = None


class SessionValue(StrictModel):
    session: str | int | None = None
    label: str = ""
    date: str | None = None
    value: float = 0.0


class CoherenceNode(StrictModel):
    id: str = ""
    x: float | None = None
    y: float | None = None


class CoherenceEdge(StrictModel):
    from_node: str = Field(default="", alias="from")
    to_node: str = Field(default="", alias="to")
    pair: str | None = None
    value: float | None = None
    label: str | None = None
    style_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _split_pair_when_from_to_missing(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        has_from = bool(str(data.get("from") or data.get("from_node") or "").strip())
        has_to = bool(str(data.get("to") or data.get("to_node") or "").strip())
        if has_from and has_to:
            return data
        pair = str(data.get("pair") or "").strip()
        if not pair:
            return data
        normalized = pair.replace("-", "_").replace(" ", "")
        if "_" not in normalized:
            return data
        left, right = normalized.split("_", 1)
        if not has_from:
            data["from"] = left
        if not has_to:
            data["to"] = right
        return data


class SplitOpposingTrendsData(StrictModel):
    left: MetricDelta = Field(default_factory=MetricDelta)
    right: MetricDelta = Field(default_factory=MetricDelta)
    headline: str | None = None
    takeaway: str | None = None


class MultiSessionTrendData(StrictModel):
    metric: str = ""
    unit: str | None = None
    points: list[SessionValue] = Field(default_factory=list)
    trend_direction: str | None = Field(default=None, alias="trend")
    target_band: NumericRange | None = None


class VerdictBullet(StrictModel):
    label: str = ""
    status: str | None = None
    from_value: float | None = Field(default=None, alias="from")
    to_value: float | None = Field(default=None, alias="to")
    delta: float | str | None = None


class VerdictSummaryData(StrictModel):
    verdict_title: str = ""
    bullets: list[VerdictBullet] = Field(default_factory=list)
    recommendation: str | None = None


class RoadmapItem(StrictModel):
    order: int | None = None
    label: str = ""


class RoadmapAgendaData(StrictModel):
    title: str = ""
    items: list[RoadmapItem] = Field(default_factory=list)


class TimelineSession(StrictModel):
    label: str = ""
    date: str | None = None
    days_to_next: int | None = None


class SessionTimelineData(StrictModel):
    sessions: list[TimelineSession] = Field(default_factory=list)
    total_span_days: int | None = None


class GaugeZone(StrictModel):
    label: str = ""
    min: float = 0.0
    max: float = 0.0
    color: str | None = None


class GaugeRatioMeterData(StrictModel):
    metric: str = ""
    unit: str | None = None
    readings: list[LabeledValue] = Field(default_factory=list)
    target_band: NumericRange | None = None
    zones: list[GaugeZone] | None = None


class CoherenceNetworkMapData(StrictModel):
    session_label: str | None = None
    nodes: list[CoherenceNode] = Field(default_factory=list)
    edges: list[CoherenceEdge] = Field(default_factory=list)


class HemisphereValue(StrictModel):
    label: str = ""
    value: float = 0.0
    unit: str | None = None


class HemisphericCompareData(StrictModel):
    left: HemisphereValue = Field(default_factory=HemisphereValue)
    right: HemisphereValue = Field(default_factory=HemisphereValue)
    session_pair: str | None = None
    delta: float | None = None


class StateSide(StrictModel):
    summary: str = ""
    metrics: list[LabeledValue] | None = None
    edges: list[CoherenceEdge] | None = None


class StateFlexibilityData(StrictModel):
    rest: StateSide = Field(default_factory=StateSide)
    task: StateSide = Field(default_factory=StateSide)
    interpretation: str = ""


class BaselineValue(StrictModel):
    label: str = ""
    value: float = 0.0
    unit: str | None = None


class BaselineTargetSplitData(StrictModel):
    baseline: BaselineValue = Field(default_factory=BaselineValue)
    target: NumericRange = Field(default_factory=lambda: NumericRange(min=0.0, max=0.0))
    status: str = ""


class MeasurementModality(StrictModel):
    label: str = ""
    description: str = ""


class MeasurementPrimerData(StrictModel):
    modalities: list[MeasurementModality] = Field(default_factory=list)
    electrode_sites: list[str] | None = None
    definitions: list[str] | None = None


class ProjectionCheckpoint(StrictModel):
    label: str = ""
    goal: str = ""


class FutureProjectionData(StrictModel):
    current_state: str = ""
    checkpoints: list[ProjectionCheckpoint] = Field(default_factory=list)
    monitoring_plan: list[str] | None = None


class LineSeries(StrictModel):
    name: str = ""
    values: list[float] = Field(default_factory=list)


class LineAnnotation(StrictModel):
    x: int | float | str
    text: str = ""
    severity: str | None = None


class LineTrajectoryData(StrictModel):
    x_labels: list[str] = Field(default_factory=list)
    series: list[LineSeries] = Field(default_factory=list)
    annotations: list[LineAnnotation] | None = None


class BarVolumeChartData(StrictModel):
    metric: str = ""
    unit: str | None = None
    bars: list[LabeledValue] = Field(default_factory=list)
    target_band: NumericRange | None = None
    trend: str | None = None


class CoherenceEdgePair(StrictModel):
    from_node: str = Field(default="", alias="from")
    to_node: str = Field(default="", alias="to")


class CoherenceSessionValues(StrictModel):
    label: str = ""
    values: list[float] | dict[str, float] = Field(default_factory=list)


class CoherenceProgressionSequenceData(StrictModel):
    edges: list[CoherenceEdgePair] = Field(default_factory=list)
    sessions: list[CoherenceSessionValues] = Field(default_factory=list)


class DotplotSession(StrictModel):
    label: str = ""
    mean: float = 0.0
    spread: float = 0.0


class DotplotVariabilityData(StrictModel):
    metric: str = ""
    unit: str | None = None
    sessions: list[DotplotSession] = Field(default_factory=list)
    target_band: NumericRange | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_sessions_dict(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            return data

        converted: list[dict[str, Any]] = []
        for label, payload in sessions.items():
            row: dict[str, Any] = {"label": str(label)}
            if isinstance(payload, dict):
                if "mean" in payload:
                    row["mean"] = payload.get("mean")
                if "spread" in payload:
                    row["spread"] = payload.get("spread")
                latencies = payload.get("latencies")
                if isinstance(latencies, list):
                    values = [float(v) for v in latencies if isinstance(v, (int, float))]
                    if values:
                        row.setdefault("mean", sum(values) / len(values))
                        row.setdefault("spread", max(values) - min(values))
            converted.append(row)

        data["sessions"] = converted
        return data


class PathwayNode(StrictModel):
    id: str = ""
    label: str = ""
    type: str | None = None


class PathwayEdge(StrictModel):
    from_node: str = Field(default="", alias="from")
    to_node: str = Field(default="", alias="to")
    label: str | None = None


class PathwayHubSynthesisData(StrictModel):
    nodes: list[PathwayNode] = Field(default_factory=list)
    edges: list[PathwayEdge] = Field(default_factory=list)
    conclusion: str = ""


class RegionalShift(StrictModel):
    name: str = ""
    from_value: float = Field(default=0.0, alias="from")
    to_value: float = Field(default=0.0, alias="to")
    unit: str | None = None
    target: str | None = None


class RegionalFrequencyMapData(StrictModel):
    regions: list[RegionalShift] = Field(default_factory=list)
    interpretation: str = ""


TableCellScalar = str | int | float
TableCellValue = TableCellScalar | list[TableCellScalar]


class TableDashboardData(StrictModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[TableCellValue] | dict[str, TableCellValue]] = Field(default_factory=list)
    highlights: list[str] | None = None


class QualityFlag(StrictModel):
    kind: str = ""
    severity: str = ""
    text: str = ""


class QualityAlertData(StrictModel):
    flags: list[QualityFlag] = Field(default_factory=list)
    impacted_metrics: list[str] = Field(default_factory=list)
    confidence_note: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_string_flags(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        flags = data.get("flags")
        if not isinstance(flags, list):
            return data
        normalized: list[dict[str, str] | Any] = []
        for item in flags:
            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue
                normalized.append(
                    {
                        "kind": text,
                        "severity": "warning",
                        "text": text.replace("_", " "),
                    }
                )
            else:
                normalized.append(item)
        data["flags"] = normalized
        return data


class WaveTrace(StrictModel):
    label: str = ""
    values: list[float] = Field(default_factory=list)
    color_key: str | None = None


class WaveformVoltagePanelData(StrictModel):
    metric: str = ""
    unit: str | None = None
    traces: list[WaveTrace] = Field(default_factory=list)
    y_range: NumericRange | None = None


class RadialRing(StrictModel):
    label: str = ""
    value: float = 0.0
    min: float | None = None
    max: float | None = None
    unit: str | None = None
    color_key: str | None = None


class RadialKpiRingData(StrictModel):
    title: str = ""
    rings: list[RadialRing] = Field(default_factory=list)
    center_label: str | None = None


class AtmosphericTitleData(StrictModel):
    title: str = ""
    subtitle: str | None = None


class AtmosphericMetaphorData(StrictModel):
    headline: str = ""
    caption: str | None = None


class AtmosphericMechanismData(StrictModel):
    title: str = ""
    callouts: list[str] | None = None


STRUCTURED_DATA_MODELS: dict[str, type[BaseModel]] = {
    "split_opposing_trends": SplitOpposingTrendsData,
    "multi_session_trend": MultiSessionTrendData,
    "verdict_summary": VerdictSummaryData,
    "roadmap_agenda": RoadmapAgendaData,
    "session_timeline": SessionTimelineData,
    "gauge_ratio_meter": GaugeRatioMeterData,
    "coherence_network_map": CoherenceNetworkMapData,
    "hemispheric_compare": HemisphericCompareData,
    "state_flexibility_rest_task": StateFlexibilityData,
    "baseline_target_split": BaselineTargetSplitData,
    "measurement_primer": MeasurementPrimerData,
    "future_projection": FutureProjectionData,
    "line_trajectory": LineTrajectoryData,
    "bar_volume_chart": BarVolumeChartData,
    "coherence_progression_sequence": CoherenceProgressionSequenceData,
    "dotplot_variability": DotplotVariabilityData,
    "pathway_hub_synthesis": PathwayHubSynthesisData,
    "regional_frequency_map": RegionalFrequencyMapData,
    "table_dashboard": TableDashboardData,
    "quality_alert": QualityAlertData,
    "waveform_voltage_panel": WaveformVoltagePanelData,
    "radial_kpi_ring": RadialKpiRingData,
    "atmospheric_title_card": AtmosphericTitleData,
    "atmospheric_metaphor_scene": AtmosphericMetaphorData,
    "atmospheric_mechanism_scene": AtmosphericMechanismData,
}


class SceneValidationError(ValueError):
    """Raised when a scene fails deterministic schema validation."""


def validate_structured_data(scene_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = STRUCTURED_DATA_MODELS.get(scene_type)
    if model is None:
        raise SceneValidationError(f"Unknown scene_type: {scene_type}")
    try:
        parsed = model.model_validate(payload)
    except ValidationError as exc:
        raise SceneValidationError(
            f"structured_data failed validation for scene_type={scene_type}: {exc}"
        ) from exc
    return parsed.model_dump(by_alias=True, exclude_none=True)


def validate_scene(scene: dict[str, Any]) -> dict[str, Any]:
    scene_type = normalize_scene_type(scene.get("scene_type"))
    if not scene_type:
        raise SceneValidationError(f"Scene {scene.get('id')} missing scene_type")
    if scene_type not in ALL_SCENE_TYPES:
        raise SceneValidationError(
            f"Scene {scene.get('id')} has unsupported scene_type={scene_type}"
        )
    raw_data = scene.get("structured_data")
    if not isinstance(raw_data, dict):
        raise SceneValidationError(
            f"Scene {scene.get('id')} has missing/invalid structured_data"
        )
    validated_data = validate_structured_data(scene_type, raw_data)
    out = dict(scene)
    out["scene_type"] = scene_type
    out["structured_data"] = validated_data
    return out


def validate_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            raise SceneValidationError("Scene entry must be an object")
        validated.append(validate_scene(scene))
    return validated


def is_data_bearing_scene(scene_type: str | None) -> bool:
    normalized = normalize_scene_type(scene_type)
    return bool(normalized and normalized in DATA_SCENE_TYPES)


def is_atmospheric_scene(scene_type: str | None) -> bool:
    normalized = normalize_scene_type(scene_type)
    return bool(normalized and normalized in ATMOSPHERIC_SCENE_TYPES)


ArchetypeCode = Literal[
    "SPLIT",
    "MULTI_TREND",
    "SUMMARY",
    "ROADMAP",
    "TIMELINE",
    "GAUGE",
    "COH_MAP",
    "HEMI",
    "STATE",
    "BASELINE_SPLIT",
    "MEASURE",
    "FUTURE",
    "LINE_TRAJECTORY",
    "BAR",
    "COH_SEQ",
    "DOTPLOT",
    "PATHWAY",
    "REGIONAL",
    "TABLE",
    "QC",
    "WAVEFORM",
    "RADIAL",
    "AT_TITLE",
    "AT_METAPHOR",
    "AT_MECHANISM",
]
