"""Structured append-first learning ledger for deterministic TDD loops.

The ledger lives at `.codex/LEARNING.md` and is rewritten from structured
entries to guarantee schema consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Iterable


SEVERITIES = {"blocker", "high", "medium", "low"}
LEARNING_TYPES = {
    "test_failure",
    "runtime_error",
    "policy_drift",
    "quality_regression",
    "timeout",
    "external",
}
STATUSES = {"OPEN", "RESOLVED", "REGRESSION"}
ENTRY_ID_PATTERN = re.compile(r"^L-\d{8}-\d{6}-[0-9a-f]{6}$")
REGRESSION_TEST_PATTERN = re.compile(r"^/.+::[\w\-\[\]\.:/]+$")

OPEN_SECTION = "Open Learnings"
RESOLVED_SECTION = "Resolved Learnings"
REGRESSION_SECTION = "Regression Learnings"
SECTION_ORDER = (OPEN_SECTION, RESOLVED_SECTION, REGRESSION_SECTION)
SECTION_TO_STATUS = {
    OPEN_SECTION: "OPEN",
    RESOLVED_SECTION: "RESOLVED",
    REGRESSION_SECTION: "REGRESSION",
}
STATUS_TO_SECTION = {v: k for k, v in SECTION_TO_STATUS.items()}

FIELD_KEYS = (
    "Timestamp (UTC)",
    "Phase",
    "Severity",
    "Type",
    "Signal",
    "Observed Error",
    "Root Cause",
    "Fix Applied",
    "Prevention Rule",
    "Regression Test Added",
    "Artifacts",
    "Status",
    "Linked Commit/Change",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp_string(ts: datetime | None = None) -> str:
    value = ts or _utc_now()
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _section_title_for_status(status: str) -> str:
    section = STATUS_TO_SECTION.get(status.upper())
    if not section:
        raise ValueError(f"Unsupported status: {status}")
    return section


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


@dataclass(slots=True)
class LearningEntry:
    """One structured learning row in the markdown ledger."""

    entry_id: str
    timestamp_utc: str
    phase: int
    severity: str
    learning_type: str
    signal: str
    observed_error: str
    root_cause: str
    fix_applied: str
    prevention_rule: str
    regression_test_added: str
    artifacts: list[str] = field(default_factory=list)
    status: str = "OPEN"
    linked_change: str = "none"

    def validate(self) -> None:
        if not ENTRY_ID_PATTERN.match(self.entry_id):
            raise ValueError(f"Invalid learning ID format: {self.entry_id}")
        if not (1 <= int(self.phase) <= 5):
            raise ValueError(f"Phase must be in [1, 5], got {self.phase}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Invalid severity: {self.severity}")
        if self.learning_type not in LEARNING_TYPES:
            raise ValueError(f"Invalid learning type: {self.learning_type}")
        if self.status not in STATUSES:
            raise ValueError(f"Invalid status: {self.status}")
        if not self.timestamp_utc.endswith("Z"):
            raise ValueError(f"Timestamp must be UTC with Z suffix: {self.timestamp_utc}")
        if not _normalize_text(self.signal):
            raise ValueError("Signal cannot be empty")
        if not _normalize_text(self.observed_error):
            raise ValueError("Observed error cannot be empty")
        if self.status == "RESOLVED":
            reg = _normalize_text(self.regression_test_added)
            if not reg or reg.lower() == "pending":
                raise ValueError("Resolved entries must include a regression test reference")
            if not REGRESSION_TEST_PATTERN.match(reg):
                raise ValueError(
                    "Regression test reference must use absolute path + ::test_id format"
                )

    @property
    def section(self) -> str:
        return _section_title_for_status(self.status)

    def to_markdown(self) -> str:
        artifacts = " | ".join(self.artifacts) if self.artifacts else "(none)"
        lines = [
            f"### {self.entry_id}",
            f"- Timestamp (UTC): {self.timestamp_utc}",
            f"- Phase: {self.phase}",
            f"- Severity: {self.severity}",
            f"- Type: {self.learning_type}",
            f"- Signal: {self.signal}",
            f"- Observed Error: {self.observed_error}",
            f"- Root Cause: {self.root_cause}",
            f"- Fix Applied: {self.fix_applied}",
            f"- Prevention Rule: {self.prevention_rule}",
            f"- Regression Test Added: {self.regression_test_added}",
            f"- Artifacts: {artifacts}",
            f"- Status: {self.status}",
            f"- Linked Commit/Change: {self.linked_change}",
            "",
        ]
        return "\n".join(lines)

    @classmethod
    def from_markdown_block(cls, entry_id: str, lines: list[str]) -> "LearningEntry":
        kv: dict[str, str] = {}
        for line in lines:
            if not line.startswith("- "):
                continue
            key, sep, value = line[2:].partition(":")
            if not sep:
                continue
            kv[key.strip()] = value.strip()
        artifacts_raw = kv.get("Artifacts", "")
        artifacts = []
        if artifacts_raw and artifacts_raw != "(none)":
            artifacts = [p.strip() for p in artifacts_raw.split("|") if p.strip()]
        entry = cls(
            entry_id=entry_id.strip(),
            timestamp_utc=kv.get("Timestamp (UTC)", ""),
            phase=int(kv.get("Phase", "0") or 0),
            severity=kv.get("Severity", ""),
            learning_type=kv.get("Type", ""),
            signal=kv.get("Signal", ""),
            observed_error=kv.get("Observed Error", ""),
            root_cause=kv.get("Root Cause", ""),
            fix_applied=kv.get("Fix Applied", ""),
            prevention_rule=kv.get("Prevention Rule", ""),
            regression_test_added=kv.get("Regression Test Added", ""),
            artifacts=artifacts,
            status=kv.get("Status", "OPEN"),
            linked_change=kv.get("Linked Commit/Change", "none"),
        )
        entry.validate()
        return entry


@dataclass(slots=True)
class LearningDocument:
    guardrails: list[str]
    entries: list[LearningEntry]

    def entries_by_status(self, status: str) -> list[LearningEntry]:
        return [e for e in self.entries if e.status == status]

    def entry_by_id(self, entry_id: str) -> LearningEntry | None:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None


def _default_markdown() -> str:
    return "\n".join(
        [
            "# Continuous Learning Ledger (`.codex/LEARNING.md`)",
            "",
            "Purpose: Persistent, append-first learning ledger for deterministic TDD execution.",
            "",
            "Hard rules:",
            "- Every failure event must be logged before attempting a fix.",
            "- Every resolved entry must include a regression test reference in absolute-path `::test_id` format.",
            "",
            "## Entry Schema (Required Fields)",
            "- ID (`L-YYYYMMDD-HHMMSS-<shorthash>`)",
            "- Timestamp (UTC)",
            "- Phase (`1..5`)",
            "- Severity (`blocker|high|medium|low`)",
            "- Type (`test_failure|runtime_error|policy_drift|quality_regression|timeout|external`)",
            "- Signal",
            "- Observed Error",
            "- Root Cause",
            "- Fix Applied",
            "- Prevention Rule",
            "- Regression Test Added (absolute path + test id)",
            "- Artifacts",
            "- Status (`OPEN|RESOLVED|REGRESSION`)",
            "- Linked Commit/Change",
            "",
            "## Active Guardrails",
            "<!-- AUTO-GENERATED: updated from resolved high-severity learnings -->",
            "- No high-severity resolved learnings yet.",
            "",
            "## Open Learnings",
            "<!-- Entries with Status: OPEN -->",
            "",
            "## Resolved Learnings",
            "<!-- Entries with Status: RESOLVED -->",
            "",
            "## Regression Learnings",
            "<!-- Entries with Status: REGRESSION -->",
            "",
        ]
    )


def ensure_learning_markdown(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_default_markdown(), encoding="utf-8")


def _extract_section(lines: list[str], section_title: str) -> list[str]:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"## {section_title}":
            start = i + 1
            break
    if start is None:
        return []
    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].startswith("## ") and i > start:
            end = i
            break
    return lines[start:end]


def _parse_guardrails(section_lines: list[str]) -> list[str]:
    out = []
    for line in section_lines:
        if line.startswith("- "):
            item = line[2:].strip()
            if item:
                out.append(item)
    return out


def _parse_entries_from_section(section_lines: list[str], status: str) -> list[LearningEntry]:
    entries: list[LearningEntry] = []
    current_id = ""
    block: list[str] = []
    for line in section_lines + ["### __END__"]:
        if line.startswith("### "):
            if current_id:
                entry = LearningEntry.from_markdown_block(current_id, block)
                if entry.status != status:
                    raise ValueError(
                        f"Entry {entry.entry_id} status {entry.status} mismatches section status {status}"
                    )
                entries.append(entry)
            current_id = line[4:].strip()
            block = []
            continue
        if current_id:
            block.append(line)
    return entries


def parse_learning_markdown(path: Path) -> LearningDocument:
    ensure_learning_markdown(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    guardrails = _parse_guardrails(_extract_section(lines, "Active Guardrails"))
    entries: list[LearningEntry] = []
    for section in SECTION_ORDER:
        section_lines = _extract_section(lines, section)
        entries.extend(
            _parse_entries_from_section(
                section_lines=section_lines,
                status=SECTION_TO_STATUS[section],
            )
        )
    return LearningDocument(guardrails=guardrails, entries=entries)


def render_learning_markdown(doc: LearningDocument) -> str:
    for entry in doc.entries:
        entry.validate()

    guardrails = doc.guardrails or ["No high-severity resolved learnings yet."]
    lines = [
        "# Continuous Learning Ledger (`.codex/LEARNING.md`)",
        "",
        "Purpose: Persistent, append-first learning ledger for deterministic TDD execution.",
        "",
        "Hard rules:",
        "- Every failure event must be logged before attempting a fix.",
        "- Every resolved entry must include a regression test reference in absolute-path `::test_id` format.",
        "",
        "## Entry Schema (Required Fields)",
        "- ID (`L-YYYYMMDD-HHMMSS-<shorthash>`)",
        "- Timestamp (UTC)",
        "- Phase (`1..5`)",
        "- Severity (`blocker|high|medium|low`)",
        "- Type (`test_failure|runtime_error|policy_drift|quality_regression|timeout|external`)",
        "- Signal",
        "- Observed Error",
        "- Root Cause",
        "- Fix Applied",
        "- Prevention Rule",
        "- Regression Test Added (absolute path + test id)",
        "- Artifacts",
        "- Status (`OPEN|RESOLVED|REGRESSION`)",
        "- Linked Commit/Change",
        "",
        "## Active Guardrails",
        "<!-- AUTO-GENERATED: updated from resolved high-severity learnings -->",
    ]
    lines.extend([f"- {rule}" for rule in guardrails])
    lines.extend(
        [
            "",
            "## Open Learnings",
            "<!-- Entries with Status: OPEN -->",
            "",
        ]
    )
    for entry in [e for e in doc.entries if e.status == "OPEN"]:
        lines.append(entry.to_markdown().rstrip("\n"))
    lines.extend(
        [
            "## Resolved Learnings",
            "<!-- Entries with Status: RESOLVED -->",
            "",
        ]
    )
    for entry in [e for e in doc.entries if e.status == "RESOLVED"]:
        lines.append(entry.to_markdown().rstrip("\n"))
    lines.extend(
        [
            "## Regression Learnings",
            "<!-- Entries with Status: REGRESSION -->",
            "",
        ]
    )
    for entry in [e for e in doc.entries if e.status == "REGRESSION"]:
        lines.append(entry.to_markdown().rstrip("\n"))
    return "\n".join(lines).rstrip() + "\n"


def generate_learning_id(signal: str, observed_error: str, ts: datetime | None = None) -> str:
    now = (ts or _utc_now()).astimezone(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(
        f"{stamp}|{signal.strip()}|{observed_error.strip()}".encode("utf-8")
    ).hexdigest()[:6]
    return f"L-{stamp}-{digest}"


class LearningLog:
    """Read/write helper for `.codex/LEARNING.md`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_learning_markdown(path)

    def load(self) -> LearningDocument:
        return parse_learning_markdown(self.path)

    def save(self, doc: LearningDocument) -> None:
        self.path.write_text(render_learning_markdown(doc), encoding="utf-8")

    def create_open_entry(
        self,
        *,
        phase: int,
        severity: str,
        learning_type: str,
        signal: str,
        observed_error: str,
        artifacts: Iterable[str] | None = None,
        root_cause: str = "pending",
        fix_applied: str = "pending",
        prevention_rule: str = "pending",
        regression_test_added: str = "pending",
        linked_change: str = "none",
    ) -> LearningEntry:
        entry = LearningEntry(
            entry_id=generate_learning_id(signal, observed_error),
            timestamp_utc=utc_timestamp_string(),
            phase=phase,
            severity=severity,
            learning_type=learning_type,
            signal=_normalize_text(signal),
            observed_error=_normalize_text(observed_error),
            root_cause=root_cause.strip() or "pending",
            fix_applied=fix_applied.strip() or "pending",
            prevention_rule=prevention_rule.strip() or "pending",
            regression_test_added=regression_test_added.strip() or "pending",
            artifacts=[str(a) for a in (artifacts or []) if str(a).strip()],
            status="OPEN",
            linked_change=linked_change.strip() or "none",
        )
        entry.validate()
        doc = self.load()
        doc.entries.append(entry)
        self.save(doc)
        return entry

    def resolve_entry(
        self,
        entry_id: str,
        *,
        root_cause: str,
        fix_applied: str,
        prevention_rule: str,
        regression_test_added: str,
        artifacts: Iterable[str] | None = None,
        linked_change: str | None = None,
    ) -> LearningEntry:
        doc = self.load()
        entry = doc.entry_by_id(entry_id)
        if not entry:
            raise ValueError(f"Learning entry not found: {entry_id}")
        entry.root_cause = _normalize_text(root_cause)
        entry.fix_applied = _normalize_text(fix_applied)
        entry.prevention_rule = _normalize_text(prevention_rule)
        entry.regression_test_added = _normalize_text(regression_test_added)
        if artifacts is not None:
            entry.artifacts = [str(a) for a in artifacts if str(a).strip()]
        if linked_change is not None:
            entry.linked_change = linked_change.strip() or "none"
        entry.status = "RESOLVED"
        entry.validate()
        self._refresh_guardrails(doc)
        self.save(doc)
        return entry

    def mark_regression(
        self,
        entry_id: str,
        *,
        linked_change: str | None = None,
    ) -> LearningEntry:
        doc = self.load()
        entry = doc.entry_by_id(entry_id)
        if not entry:
            raise ValueError(f"Learning entry not found: {entry_id}")
        entry.status = "REGRESSION"
        if linked_change:
            entry.linked_change = linked_change
        entry.validate()
        self.save(doc)
        return entry

    def record_recurrence(
        self,
        previous_entry_id: str,
        *,
        phase: int,
        severity: str,
        learning_type: str,
        signal: str,
        observed_error: str,
        artifacts: Iterable[str] | None = None,
    ) -> LearningEntry:
        self.mark_regression(
            previous_entry_id,
            linked_change=f"recurrence_reopened_as_new_issue({previous_entry_id})",
        )
        return self.create_open_entry(
            phase=phase,
            severity=severity,
            learning_type=learning_type,
            signal=signal,
            observed_error=observed_error,
            artifacts=artifacts,
            linked_change=f"recurrence_of={previous_entry_id}",
        )

    def has_open_blockers(self) -> bool:
        doc = self.load()
        return any(e.status == "OPEN" and e.severity == "blocker" for e in doc.entries)

    def unresolved_entry_ids(self) -> list[str]:
        doc = self.load()
        return [e.entry_id for e in doc.entries if e.status == "OPEN"]

    def resolved_count(self) -> int:
        doc = self.load()
        return len([e for e in doc.entries if e.status == "RESOLVED"])

    def maybe_refresh_guardrails(self) -> bool:
        doc = self.load()
        resolved = [e for e in doc.entries if e.status == "RESOLVED"]
        if not resolved:
            return False
        if len(resolved) % 5 != 0:
            return False
        self._refresh_guardrails(doc)
        self.save(doc)
        return True

    def _refresh_guardrails(self, doc: LearningDocument) -> None:
        resolved_high = [
            e for e in doc.entries if e.status == "RESOLVED" and e.severity in {"blocker", "high"}
        ]
        resolved_high.sort(key=lambda e: e.timestamp_utc, reverse=True)
        picked = resolved_high[:5]
        if not picked:
            doc.guardrails = ["No high-severity resolved learnings yet."]
            return
        rules = []
        for entry in picked:
            rule = _normalize_text(entry.prevention_rule)
            if not rule:
                continue
            rules.append(f"{rule} (from {entry.entry_id})")
        doc.guardrails = rules or ["No high-severity resolved learnings yet."]


def validate_learning_markdown(path: Path) -> None:
    """Schema-check helper used by contract tests and loop guards."""
    doc = parse_learning_markdown(path)
    # Enforce section parsing and required entry fields by calling validate().
    for entry in doc.entries:
        entry.validate()

