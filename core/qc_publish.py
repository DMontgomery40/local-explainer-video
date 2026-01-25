"""QC + publish pipeline for qEEG explainer video projects.

Design goals:
- Use qEEG Council Stage 4 consolidation as narrative ground truth.
- Use Stage 1 _data_pack.json as numeric ground truth.
- Be liberal about ELI5 analogies (don't nitpick style).
- NEVER regenerate images as a QC fix; use Qwen Image Edit for surgical text fixes.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sqlite3
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import anthropic
import numpy as np
import requests
from PIL import Image

from core.image_gen import edit_image
from core.voice_gen import generate_scene_audio
from core.video_assembly import assemble_video


_PATIENT_ID_RE = re.compile(r"^(?P<mm>\d{2})-(?P<dd>\d{2})-(?P<yyyy>\d{4})-(?P<n>\d+)$")
_PATIENT_ID_PREFIX_RE = re.compile(r"^(?P<pid>\d{2}-\d{2}-\d{4}-\d+)(?:__\d+)?$")


def infer_patient_id(project_name: str) -> str | None:
    """Infer MM-DD-YYYY-N from a project folder name (supports __02 suffix)."""
    raw = (project_name or "").strip()
    m = _PATIENT_ID_PREFIX_RE.match(raw)
    if not m:
        return None
    pid = m.group("pid")
    return pid if _PATIENT_ID_RE.match(pid) else None


def extract_quoted_texts(prompt: str) -> list[str]:
    """Extract double-quoted strings from a visual prompt."""
    if not isinstance(prompt, str) or not prompt:
        return []
    # Intentionally simple: prompts are plain text.
    return [m.group(1) for m in re.finditer(r"\"([^\"]+)\"", prompt)]


def extract_numbers(text: str) -> list[str]:
    """Extract numeric tokens from text (e.g., 3.5, 200%, 1,200)."""
    if not isinstance(text, str) or not text:
        return []
    tokens = []
    for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?\b", text):
        tokens.append(m.group(0))
    # Preserve order but unique.
    seen = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def stable_seed(value: str) -> int:
    """Return a stable non-negative 31-bit seed from a string."""
    raw = (value or "").encode("utf-8", errors="ignore")
    return zlib.crc32(raw) & 0x7FFFFFFF


def image_change_metrics(
    original_path: Path,
    edited_path: Path,
    *,
    sample_size: tuple[int, int] = (128, 72),
    per_pixel_threshold: float = 18.0,
) -> tuple[float, float]:
    """
    Compute rough image-drift metrics between two PNGs.

    Returns:
      (mean_abs_diff_0_255, changed_pixel_ratio_0_1)
    """
    a = Image.open(original_path).convert("RGB").resize(sample_size)
    b = Image.open(edited_path).convert("RGB").resize(sample_size)
    arr_a = np.asarray(a, dtype=np.int16)
    arr_b = np.asarray(b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b).astype(np.float32)
    mean_abs = float(diff.mean())
    per_pixel = diff.mean(axis=2)
    changed_ratio = float((per_pixel > float(per_pixel_threshold)).mean())
    return mean_abs, changed_ratio


def _repo_root() -> Path:
    # core/qc_publish.py -> core/ -> repo root
    return Path(__file__).resolve().parents[1]


def default_qeeg_analysis_dir() -> Path:
    env = os.getenv("QEEG_ANALYSIS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root().parent / "qEEG-analysis").resolve()


def default_qeeg_backend_url() -> str:
    return os.getenv("QEEG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def default_cliproxy_url() -> str:
    return os.getenv("CLIPROXY_BASE_URL", "http://127.0.0.1:8317").rstrip("/")


def default_cliproxy_api_key() -> str:
    return os.getenv("CLIPROXY_API_KEY", "").strip()


def _load_prompt_text(name: str) -> str:
    path = _repo_root() / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


class QCPublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class QEGGroundTruth:
    patient_uuid: str
    run_id: str
    consolidation_path: Path
    data_pack_path: Path
    consolidated_md: str
    data_pack: dict[str, Any]


def _resolve_qeeg_path(qeeg_dir: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    return p if p.is_absolute() else (qeeg_dir / p)


def load_qeeg_ground_truth(*, qeeg_dir: Path, patient_label: str) -> QEGGroundTruth:
    db_path = qeeg_dir / "data" / "app.db"
    if not db_path.exists():
        raise QCPublishError(f"qEEG Council DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        # Patients may be duplicated (case-insensitive label collisions). Choose the newest run across all matches.
        cur.execute(
            """
            SELECT
              runs.id AS run_id,
              runs.patient_id AS patient_uuid,
              runs.report_id AS report_id,
              runs.status AS status,
              runs.error_message AS error_message,
              runs.created_at AS created_at,
              runs.consolidator_model_id AS consolidator_model_id
            FROM runs
            JOIN patients ON patients.id = runs.patient_id
            WHERE lower(patients.label) = lower(?)
            ORDER BY runs.created_at DESC
            """,
            (patient_label,),
        )
        runs = cur.fetchall() or []
        if not runs:
            # Different UIs may have created a patient record without any run yet.
            cur.execute("SELECT 1 FROM patients WHERE lower(label) = lower(?) LIMIT 1", (patient_label,))
            if cur.fetchone() is None:
                raise QCPublishError(f"No patient found in qEEG Council with label: {patient_label}")
            raise QCPublishError(f"No runs found in qEEG Council for patient label: {patient_label}")

        best: QEGGroundTruth | None = None
        checked: list[dict[str, Any]] = []
        for r in runs:
            run_id = str(r["run_id"])
            patient_uuid = str(r["patient_uuid"])
            report_id = str(r["report_id"] or "")
            status = str(r["status"] or "")
            error_message = str(r["error_message"] or "")
            consolidator_model_id = str(r["consolidator_model_id"] or "")

            cur.execute(
                """
                SELECT content_path, model_id, created_at
                FROM artifacts
                WHERE run_id = ? AND stage_num = 4 AND kind = 'consolidation'
                ORDER BY created_at DESC
                """,
                (run_id,),
            )
            stage4_rows = cur.fetchall() or []
            consolidation_path: Path | None = None
            if stage4_rows:
                if consolidator_model_id:
                    for ar in stage4_rows:
                        if str(ar["model_id"] or "") == consolidator_model_id:
                            consolidation_path = _resolve_qeeg_path(qeeg_dir, str(ar["content_path"]))
                            break
                if consolidation_path is None:
                    consolidation_path = _resolve_qeeg_path(qeeg_dir, str(stage4_rows[0]["content_path"]))

            cur.execute(
                """
                SELECT content_path, created_at
                FROM artifacts
                WHERE run_id = ? AND stage_num = 1 AND kind = 'data_pack' AND model_id = '_data_pack'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id,),
            )
            dp_row = cur.fetchone()
            data_pack_path = _resolve_qeeg_path(qeeg_dir, str(dp_row["content_path"])) if dp_row else None

            checked.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "report_id": report_id,
                    "has_stage4": bool(consolidation_path),
                    "has_data_pack": bool(data_pack_path),
                    "consolidation_path": str(consolidation_path) if consolidation_path else None,
                    "data_pack_path": str(data_pack_path) if data_pack_path else None,
                    "error_message": error_message.strip() or None,
                }
            )

            if not consolidation_path or not data_pack_path:
                continue
            if not consolidation_path.exists() or not data_pack_path.exists():
                continue

            try:
                consolidated_md = consolidation_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                raise QCPublishError(f"Failed reading consolidation markdown: {consolidation_path} ({e})") from e

            try:
                data_pack = json.loads(data_pack_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise QCPublishError(f"Failed reading data pack JSON: {data_pack_path} ({e})") from e

            best = QEGGroundTruth(
                patient_uuid=patient_uuid,
                run_id=run_id,
                consolidation_path=consolidation_path,
                data_pack_path=data_pack_path,
                consolidated_md=consolidated_md,
                data_pack=data_pack,
            )
            break

        if best is None:
            lines: list[str] = []
            for item in checked[:5]:
                err = item.get("error_message") or ""
                if isinstance(err, str) and len(err) > 160:
                    err = err[:160] + "…"
                lines.append(
                    "- "
                    + f"{item.get('run_id')} status={item.get('status')!r} "
                    + f"stage4={bool(item.get('has_stage4'))} data_pack={bool(item.get('has_data_pack'))} "
                    + (f"report_id={item.get('report_id')!r} " if item.get("report_id") else "")
                    + (f"error={err!r}" if err else "")
                )
            hint = (
                "Hint: In qEEG-analysis, ensure the report was re-extracted (OCR) and a successful run exists "
                "with Stage 1 `_data_pack.json` and Stage 4 consolidation."
            )
            raise QCPublishError(
                "Could not find a recent run with BOTH Stage 4 consolidation and Stage 1 _data_pack.json.\n"
                f"Patient label: {patient_label}\n"
                f"Checked runs: {len(runs)}\n"
                + ("Recent runs:\n" + "\n".join(lines) + "\n" if lines else "")
                + hint
            )
        return best
    finally:
        con.close()


def _anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise QCPublishError("Model returned empty output")
    if "```" in raw:
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if m:
            raw = m.group(1).strip()
        else:
            m2 = re.search(r"```(?:\w+)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
            if m2:
                raw = m2.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise QCPublishError("Model did not return a JSON object")
    return json.loads(raw[start : end + 1])


@dataclass
class NarrativeQCResult:
    passed: bool
    critical_issues: list[dict[str, Any]]
    safe_autofixes: list[dict[str, Any]]
    raw: dict[str, Any]


def run_opus_narrative_qc(
    *,
    consolidated_md: str,
    data_pack: dict[str, Any],
    plan: dict[str, Any],
    model: str = "claude-opus-4-5",
    max_tokens: int = 3000,
) -> NarrativeQCResult:
    system = _load_prompt_text("qc_opus_system")

    dp_for_prompt = {
        "schema_version": data_pack.get("schema_version"),
        "meta": data_pack.get("meta"),
        "facts": data_pack.get("facts"),
        "derived": data_pack.get("derived"),
    }

    scenes = plan.get("scenes") if isinstance(plan, dict) else None
    prompt_plan = {
        "meta": plan.get("meta") if isinstance(plan, dict) else None,
        "scenes": [
            {
                "id": s.get("id"),
                "title": s.get("title"),
                "narration": s.get("narration"),
                "visual_prompt": s.get("visual_prompt"),
            }
            for s in scenes
            if isinstance(s, dict)
        ]
        if isinstance(scenes, list)
        else [],
    }

    user = (
        "You are reviewing a patient-friendly explainer video script and slide prompts.\n\n"
        "GROUND TRUTH SOURCES:\n"
        "- Stage 4 CONSOLIDATED report (narrative truth)\n"
        "- Stage 1 DATA PACK (numeric truth)\n\n"
        "STAGE 4 CONSOLIDATED REPORT (markdown):\n"
        "-----\n"
        f"{consolidated_md}\n"
        "-----\n\n"
        "DATA PACK (JSON):\n"
        "-----\n"
        f"{json.dumps(dp_for_prompt, indent=2, sort_keys=True)}\n"
        "-----\n\n"
        "VIDEO PLAN (JSON):\n"
        "-----\n"
        f"{json.dumps(prompt_plan, indent=2)}\n"
        "-----\n"
    )

    client = _anthropic_client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break
    parsed = _extract_json_object(text)
    passed = bool(parsed.get("pass"))
    critical_issues = parsed.get("critical_issues") if isinstance(parsed.get("critical_issues"), list) else []
    safe_autofixes = parsed.get("safe_autofixes") if isinstance(parsed.get("safe_autofixes"), list) else []
    return NarrativeQCResult(
        passed=passed,
        critical_issues=[x for x in critical_issues if isinstance(x, dict)],
        safe_autofixes=[x for x in safe_autofixes if isinstance(x, dict)],
        raw=parsed,
    )


def apply_safe_autofixes(
    *,
    plan: dict[str, Any],
    fixes: list[dict[str, Any]],
    only_confidence: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply safe string-replacement fixes to plan scenes."""
    only_confidence = only_confidence or {"high"}
    scenes = plan.get("scenes")
    if not isinstance(scenes, list):
        return plan, []

    applied: list[str] = []
    by_id: dict[int, dict[str, Any]] = {}
    for s in scenes:
        if isinstance(s, dict) and isinstance(s.get("id"), int):
            by_id[int(s["id"])] = s

    for f in fixes:
        scene_id = f.get("scene_id")
        field = f.get("field")
        find = f.get("find")
        replace = f.get("replace")
        confidence = str(f.get("confidence") or "").lower().strip() or "low"

        if confidence not in only_confidence:
            continue
        if not isinstance(scene_id, int) or scene_id not in by_id:
            continue
        if field not in {"narration", "visual_prompt"}:
            continue
        if not isinstance(find, str) or not find:
            continue
        if not isinstance(replace, str) or replace == "":
            continue

        scene = by_id[scene_id]
        current = scene.get(field)
        if not isinstance(current, str) or not current:
            continue
        if find not in current:
            continue

        scene[field] = current.replace(find, replace)
        applied.append(f"Scene {scene_id} {field}: replace {find!r} -> {replace!r}")

    return plan, applied


class CLIProxyError(RuntimeError):
    pass


def _parse_openai_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if not isinstance(err, dict):
        return None
    msg = err.get("message")
    return msg.strip() if isinstance(msg, str) and msg.strip() else None


def _looks_like_chat_unsupported(status_code: int, message: str) -> bool:
    if status_code not in {400, 404, 405}:
        return False
    text = (message or "").lower()
    return (
        "responses" in text
        or "response endpoint" in text
        or ("chat completions" in text and "not supported" in text)
        or ("not support chat" in text)
    )


class CLIProxyClient:
    def __init__(self, *, base_url: str, api_key: str = "", timeout_s: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/v1/models"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=(5, 30))
        except Exception as e:
            raise CLIProxyError(f"CLIProxyAPI request failed: {e}") from e
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = None
            msg = _parse_openai_error(payload) if payload else None
            raise CLIProxyError(msg or f"CLIProxyAPI /v1/models failed (HTTP {resp.status_code})")
        data = resp.json().get("data", [])
        ids: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.append(item["id"])
        return ids

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=(10, self.timeout_s))
        except Exception as e:
            raise CLIProxyError(f"CLIProxyAPI request failed: {e}") from e
        if resp.status_code >= 400:
            try:
                err_payload = resp.json()
            except Exception:
                err_payload = None
            msg = _parse_openai_error(err_payload) if err_payload else None
            msg2 = msg or f"CLIProxyAPI /v1/chat/completions failed (HTTP {resp.status_code})"
            if _looks_like_chat_unsupported(resp.status_code, msg2):
                input_text = json.dumps(messages)
                return self.responses(model=model, input_text=input_text)
            raise CLIProxyError(msg2)
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise CLIProxyError(f"CLIProxyAPI returned unexpected shape: {e}") from e
        if not isinstance(content, str):
            raise CLIProxyError("CLIProxyAPI returned non-text content")
        return content

    def responses(self, *, model: str, input_text: str) -> str:
        url = f"{self.base_url}/v1/responses"
        payload = {"model": model, "input": input_text, "stream": False}
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=(10, self.timeout_s))
        except Exception as e:
            raise CLIProxyError(f"CLIProxyAPI request failed: {e}") from e
        if resp.status_code >= 400:
            try:
                err_payload = resp.json()
            except Exception:
                err_payload = None
            msg = _parse_openai_error(err_payload) if err_payload else None
            raise CLIProxyError(msg or f"CLIProxyAPI /v1/responses failed (HTTP {resp.status_code})")
        data = resp.json()
        out = data.get("output_text")
        if isinstance(out, str) and out.strip():
            return out.strip()
        try:
            blocks = data.get("output", [])
            texts: list[str] = []
            for item in blocks:
                for c in item.get("content", []) if isinstance(item, dict) else []:
                    if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        texts.append(c["text"])
            return "\n".join(texts).strip()
        except Exception:
            raise CLIProxyError("CLIProxyAPI /v1/responses returned unexpected shape")


def select_discovered_model_id(preferred: str, discovered: list[str]) -> str | None:
    pref = (preferred or "").strip()
    if not pref:
        return None
    ids = [x for x in discovered if isinstance(x, str)]
    if pref in ids:
        return pref
    pref_lower = pref.lower()
    for mid in ids:
        if mid.lower() == pref_lower:
            return mid
    matches = [mid for mid in ids if pref_lower in mid.lower()]
    if not matches:
        return None

    def rank(mid: str) -> tuple[int, int, str]:
        lower = mid.lower()
        preview_penalty = 1 if "preview" in lower else 0
        return (preview_penalty, len(mid), mid)

    return sorted(matches, key=rank)[0]


@dataclass
class VisualQCResult:
    passed: bool
    replacements: list[dict[str, str]]
    notes: str
    raw: dict[str, Any]


def run_gemini_visual_qc(
    *,
    cliproxy: CLIProxyClient,
    image_path: Path,
    scene_context: dict[str, Any],
    data_pack: dict[str, Any] | None,
    model: str = "gemini-3-flash",
    max_tokens: int = 1200,
) -> VisualQCResult:
    system = _load_prompt_text("qc_gemini_visual_system")

    dp_for_prompt: dict[str, Any] | None = None
    if isinstance(data_pack, dict):
        dp_for_prompt = {
            "schema_version": data_pack.get("schema_version"),
            "meta": data_pack.get("meta"),
            "facts": data_pack.get("facts"),
            "derived": data_pack.get("derived"),
        }

    input_payload = {
        "scene_context": scene_context,
        "data_pack": dp_for_prompt,
    }

    png_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": system},
        {
            "type": "text",
            "text": (
                "INPUT_JSON:\n"
                f"{json.dumps(input_payload, ensure_ascii=False, indent=2, sort_keys=True)}\n"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{png_b64}", "detail": "high"},
        },
    ]

    text = cliproxy.chat_completions(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    parsed = _extract_json_object(text)
    replacements = parsed.get("replacements") if isinstance(parsed.get("replacements"), list) else []
    cleaned: list[dict[str, str]] = []
    for r in replacements:
        if not isinstance(r, dict):
            continue
        frm = r.get("from")
        to = r.get("to")
        if not (isinstance(frm, str) and frm and isinstance(to, str) and to and frm != to):
            continue
        item: dict[str, str] = {"from": frm, "to": to}
        why = r.get("why")
        where = r.get("where")
        if isinstance(why, str) and why.strip():
            item["why"] = why.strip()
        if isinstance(where, str) and where.strip():
            item["where"] = where.strip()
        cleaned.append(item)
    notes = parsed.get("notes") if isinstance(parsed.get("notes"), str) else ""
    raw_pass = bool(parsed.get("pass"))
    passed = raw_pass and not cleaned
    return VisualQCResult(passed=passed, replacements=cleaned, notes=notes, raw=parsed)


def _build_image_edit_instruction(replacements: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for r in replacements:
        frm = r.get("from")
        to = r.get("to")
        where = (r.get("where") or "").strip()
        if not frm or not to:
            continue
        if where:
            parts.append(f'On {where.upper()} side: change "{frm}" to "{to}"')
        else:
            parts.append(f'change "{frm}" to "{to}"')
    if not parts:
        return ""
    # Keep it exactly in the style the UI uses; verbose instructions reduce reliability.
    return "\n".join(parts)


@dataclass
class QCPublishConfig:
    qeeg_dir: Path
    backend_url: str
    cliproxy_url: str
    cliproxy_api_key: str
    opus_model: str = "claude-opus-4-5"
    gemini_model: str = "gemini-3-flash"
    max_visual_passes: int = 5
    max_image_edit_attempts: int = 3
    # If false, visual QC only reports issues and blocks publish (no image edits).
    auto_fix_images: bool = False
    # Guardrails to prevent "edit" drift into a brand-new image.
    max_image_mean_abs_diff: float = 30.0
    max_image_changed_ratio: float = 0.25
    fps: int = 24
    output_filename: str = "final_video.mp4"
    tts_voice: str = "af_bella"
    tts_speed: float = 1.1
    image_edit_seed: int | None = None


@dataclass
class QCPublishSummary:
    patient_id: str
    run_id: str
    applied_plan_fixes: list[str]
    narrative_critical_issues: list[dict[str, Any]]
    visual_edits_applied: list[str]
    video_path: Path
    portal_copy_path: Path | None
    backend_upload_ok: bool
    backend_upload_response: dict[str, Any] | None


def qc_and_publish_project(
    *,
    project_dir: Path,
    plan: dict[str, Any],
    patient_id: str,
    config: QCPublishConfig,
    log: Callable[[str], None] | None = None,
    set_phase: Callable[[str], None] | None = None,
    set_progress: Callable[[float], None] | None = None,
) -> tuple[dict[str, Any], QCPublishSummary]:
    """Run QC loop and publish artifacts to qEEG Council + portal sync folder."""

    def _log(msg: str) -> None:
        if log:
            log(msg)

    def _phase(p: str) -> None:
        if set_phase:
            set_phase(p)
        _log(p)

    def _prog(v: float) -> None:
        if set_progress:
            set_progress(v)

    _phase("Loading qEEG ground truth (Stage 4 consolidation + _data_pack.json)…")
    gt = load_qeeg_ground_truth(qeeg_dir=config.qeeg_dir, patient_label=patient_id)

    _phase(f"Narrative QC (Opus: {config.opus_model})…")
    narrative = run_opus_narrative_qc(consolidated_md=gt.consolidated_md, data_pack=gt.data_pack, plan=plan, model=config.opus_model)
    _log(f"Narrative pass={narrative.passed}, critical_issues={len(narrative.critical_issues)}, safe_fixes={len(narrative.safe_autofixes)}")

    plan_before = json.dumps(
        [(s.get("id"), s.get("narration"), s.get("visual_prompt")) for s in (plan.get("scenes") or []) if isinstance(s, dict)],
        ensure_ascii=False,
    )

    applied_fixes: list[str] = []
    # Apply high-confidence safe fixes and re-check once (or twice) to ensure we end in a clean state.
    for _ in range(3):
        plan, applied = apply_safe_autofixes(plan=plan, fixes=narrative.safe_autofixes, only_confidence={"high"})
        applied_fixes.extend(applied)
        if applied:
            _log(f"Applied {len(applied)} safe fix(es); re-running narrative QC…")
            narrative = run_opus_narrative_qc(
                consolidated_md=gt.consolidated_md,
                data_pack=gt.data_pack,
                plan=plan,
                model=config.opus_model,
            )
            _log(
                f"Narrative pass={narrative.passed}, critical_issues={len(narrative.critical_issues)}, safe_fixes={len(narrative.safe_autofixes)}"
            )
            continue
        break

    plan_after = json.dumps(
        [(s.get("id"), s.get("narration"), s.get("visual_prompt")) for s in (plan.get("scenes") or []) if isinstance(s, dict)],
        ensure_ascii=False,
    )
    narration_changed_scene_ids: set[int] = set()
    if plan_before != plan_after:
        try:
            before_list = json.loads(plan_before)
            after_list = json.loads(plan_after)
            before_map = {int(x[0]): x[1] for x in before_list if isinstance(x, list) and len(x) > 1}
            after_map = {int(x[0]): x[1] for x in after_list if isinstance(x, list) and len(x) > 1}
            for sid, before_n in before_map.items():
                if sid in after_map and before_n != after_map[sid]:
                    narration_changed_scene_ids.add(sid)
        except Exception:
            narration_changed_scene_ids = set()

    for line in applied_fixes:
        _log(f"Applied plan fix: {line}")

    if narrative.critical_issues:
        _log("Narrative critical issues found (will block publish unless resolved):")
        for issue in narrative.critical_issues[:25]:
            _log(f"- {issue}")
        raise QCPublishError("Narrative QC found critical issues. Fix them, then re-run QC.")

    _phase(f"Visual QC (Gemini: {config.gemini_model})…")
    cliproxy = CLIProxyClient(base_url=config.cliproxy_url, api_key=config.cliproxy_api_key)
    try:
        models = cliproxy.list_models()
    except Exception as e:
        raise QCPublishError(f"CLIProxyAPI not reachable for Gemini visual QC: {e}") from e
    gemini_model_id = select_discovered_model_id(config.gemini_model, models)
    if not gemini_model_id:
        raise QCPublishError(f"Gemini model not found in CLIProxy /v1/models (preferred={config.gemini_model!r}).")
    if gemini_model_id != config.gemini_model:
        _log(f"Gemini model resolved: {config.gemini_model!r} -> {gemini_model_id!r}")

    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise QCPublishError("plan.json has no scenes")

    missing_images: list[int] = []
    missing_audio: list[int] = []
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue
        sid = int(scene.get("id", i))
        img = scene.get("image_path")
        aud = scene.get("audio_path")
        if not img or not Path(str(img)).exists():
            missing_images.append(sid)
        if not aud or not Path(str(aud)).exists():
            missing_audio.append(sid)
    if missing_images or missing_audio:
        parts: list[str] = []
        if missing_images:
            parts.append(f"missing images: {sorted(set(missing_images))}")
        if missing_audio:
            parts.append(f"missing audio: {sorted(set(missing_audio))}")
        raise QCPublishError("QC requires all scene assets to exist (" + "; ".join(parts) + ").")

    visual_edits_applied: list[str] = []
    visual_issues: list[dict[str, Any]] = []

    max_passes = 1 if not config.auto_fix_images else max(1, config.max_visual_passes)
    for pass_idx in range(1, max_passes + 1):
        _phase(f"Visual check pass {pass_idx}/{max_passes}…")
        any_edits = False
        for i, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue
            scene_id = scene.get("id", i)
            image_path = scene.get("image_path")
            if not image_path:
                continue
            img_path = Path(str(image_path))
            if not img_path.exists():
                continue

            visual_prompt = str(scene.get("visual_prompt") or "")
            narration = str(scene.get("narration") or "")

            text_hints = extract_quoted_texts(visual_prompt)

            expected_numbers = []
            expected_numbers.extend(extract_numbers(visual_prompt))
            expected_numbers.extend(extract_numbers(narration))
            seen_nums = set()
            expected_numbers2: list[str] = []
            for n in expected_numbers:
                if n in seen_nums:
                    continue
                seen_nums.add(n)
                expected_numbers2.append(n)

            scene_context = {
                "patient_id": patient_id,
                "scene_id": scene_id,
                "title": scene.get("title"),
                "narration": narration,
                "visual_prompt": visual_prompt,
                "expected_text_hints": text_hints,
                "expected_numbers": expected_numbers2,
            }

            _prog((i + 1) / max(1, len(scenes)) * 0.6)

            res = run_gemini_visual_qc(
                cliproxy=cliproxy,
                image_path=img_path,
                scene_context=scene_context,
                data_pack=gt.data_pack,
                model=gemini_model_id,
            )
            if not res.passed and not res.replacements:
                # The model is uncertain and provided no deterministic replacements.
                visual_issues.append(
                    {
                        "scene_id": scene_id,
                        "slide_num": int(scene_id) + 1 if isinstance(scene_id, int) else None,
                        "image_path": str(img_path),
                        "replacements": [],
                        "notes": res.notes,
                        "manual_review_required": True,
                    }
                )
                continue
            if not res.replacements:
                continue

            if not config.auto_fix_images:
                visual_issues.append(
                    {
                        "scene_id": scene_id,
                        "slide_num": int(scene_id) + 1 if isinstance(scene_id, int) else None,
                        "image_path": str(img_path),
                        "replacements": res.replacements,
                        "notes": res.notes,
                        "manual_review_required": False,
                    }
                )
                continue

            instr = _build_image_edit_instruction(res.replacements)
            if not instr:
                continue
            _log(f"Scene {scene_id}: proposed image edits: {res.replacements}")

            fixed = False
            for attempt_idx in range(1, max(1, int(config.max_image_edit_attempts)) + 1):
                seed = (
                    int(config.image_edit_seed)
                    if config.image_edit_seed is not None
                    else stable_seed(f"{patient_id}:{scene_id}:pass{pass_idx}:try{attempt_idx}")
                )

                edited_path = img_path.with_name(
                    f"{img_path.stem}__qc_edit_p{pass_idx}_t{attempt_idx}{img_path.suffix}"
                )
                _log(f"Scene {scene_id}: applying edit (try {attempt_idx}/{config.max_image_edit_attempts}, seed={seed})")
                edit_image(instr, img_path, edited_path, seed=seed)

                try:
                    mean_abs, changed_ratio = image_change_metrics(img_path, edited_path)
                except Exception as e:
                    try:
                        edited_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise QCPublishError(f"Failed computing image drift metrics for scene {scene_id}: {e}") from e

                _log(
                    f"Scene {scene_id}: drift metrics mean_abs={mean_abs:.1f}, changed_ratio={changed_ratio:.2f}"
                )

                res2 = run_gemini_visual_qc(
                    cliproxy=cliproxy,
                    image_path=edited_path,
                    scene_context=scene_context,
                    data_pack=gt.data_pack,
                    model=gemini_model_id,
                )

                text_ok = bool(res2.passed)
                drift_ok = bool(
                    mean_abs <= float(config.max_image_mean_abs_diff)
                    and changed_ratio <= float(config.max_image_changed_ratio)
                )

                if text_ok and drift_ok:
                    try:
                        backup = img_path.with_name(f"{img_path.stem}__qc_backup_p{pass_idx}{img_path.suffix}")
                        if backup.exists():
                            backup = img_path.with_name(
                                f"{img_path.stem}__qc_backup_p{pass_idx}_t{attempt_idx}{img_path.suffix}"
                            )
                        if not backup.exists():
                            try:
                                os.link(img_path, backup)
                            except Exception:
                                shutil.copy2(img_path, backup)
                    except Exception as e:
                        try:
                            edited_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise QCPublishError(f"Failed to create image backup for scene {scene_id}: {e}") from e

                    edited_path.replace(img_path)
                    visual_edits_applied.append(f"Scene {scene_id}: {res.replacements}")
                    any_edits = True
                    fixed = True
                    _log(f"Scene {scene_id}: edit accepted.")
                    break

                _log(
                    f"Scene {scene_id}: edit rejected (text_ok={text_ok}, drift_ok={drift_ok}). "
                    f"Remaining replacements: {res2.replacements}"
                )
                # Do not overwrite the original on a failed attempt.
                try:
                    edited_path.unlink(missing_ok=True)
                except Exception:
                    pass

            if not fixed:
                raise QCPublishError(
                    f"Visual QC could not safely fix scene {scene_id} (edit did not converge without drift). "
                    "Use the UI's Edit Image button, then re-run QC."
                )
            time.sleep(0.2)

        if not any_edits:
            _log("Visual QC: all clear.")
            break
        if pass_idx == config.max_visual_passes:
            raise QCPublishError("Visual QC did not reach all-clear within max passes.")

    if visual_issues:
        report_path = project_dir / "qc_visual_issues.json"
        payload = {
            "patient_id": patient_id,
            "run_id": gt.run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issues": visual_issues,
        }
        try:
            report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            raise QCPublishError(
                f"Visual QC found {len(visual_issues)} issue(s), but failed to write report: {e}"
            ) from e

        preview = visual_issues[0]
        raise QCPublishError(
            "Visual QC found issues and is running in check-only mode (no auto edits).\n"
            f"- Issues: {len(visual_issues)}\n"
            f"- Report: {report_path}\n"
            f"- Example: scene_id={preview.get('scene_id')} (slide {preview.get('slide_num')}), replacements={preview.get('replacements')}\n"
            "Fix the slides in the UI, then re-run QC."
        )

    if narration_changed_scene_ids:
        _phase(f"Regenerating audio for {len(narration_changed_scene_ids)} changed scene(s)…")
        for i, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue
            sid = scene.get("id")
            if sid not in narration_changed_scene_ids:
                continue
            try:
                path = generate_scene_audio(
                    scene,
                    project_dir,
                    voice=config.tts_voice,
                    speed=config.tts_speed,
                )
                scene["audio_path"] = str(path)
                _log(f"Audio regenerated: scene {sid} -> {path}")
            except Exception as e:
                raise QCPublishError(f"Failed regenerating audio for scene {sid}: {e}") from e
            _prog(0.6 + (i + 1) / max(1, len(scenes)) * 0.1)

    _phase("Rendering final video…")
    video_path = assemble_video(
        scenes,
        project_dir,
        output_filename=config.output_filename,
        fps=config.fps,
    )
    _log(f"Video rendered: {video_path}")
    _prog(0.8)

    _phase("Publishing to qEEG Council portal folder…")
    portal_copy_path: Path | None = None
    try:
        out_dir = config.qeeg_dir / "data" / "portal_patients" / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{patient_id}.mp4"
        tmp = dest.with_name(f".{dest.name}.partial")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if dest.exists():
            try:
                dest.unlink()
            except Exception:
                pass
        try:
            os.link(video_path, dest)
        except Exception:
            import shutil

            shutil.copy2(video_path, tmp)
            tmp.replace(dest)
        portal_copy_path = dest
        _log(f"Portal folder updated: {dest}")
    except Exception as e:
        _log(f"Portal folder publish failed (non-fatal): {e}")

    _prog(0.9)
    _phase("Uploading MP4 via qEEG Council backend (DB-tracked)…")
    backend_upload_ok = False
    backend_upload_response: dict[str, Any] | None = None
    try:
        url = f"{config.backend_url}/api/patients/{gt.patient_uuid}/files"
        with open(video_path, "rb") as f:
            files = {"file": (f"{patient_id}.mp4", f, "video/mp4")}
            resp = requests.post(url, files=files, timeout=(10, 180))
        if resp.status_code >= 400:
            raise QCPublishError(f"Backend upload failed (HTTP {resp.status_code}): {resp.text[:500]}")
        backend_upload_response = resp.json()
        backend_upload_ok = bool(backend_upload_response.get("file"))
        _log(f"Backend upload ok={backend_upload_ok}")
    except Exception as e:
        _log(f"Backend upload failed (non-fatal): {e}")

    _prog(1.0)
    _phase("QC + Publish complete.")

    summary = QCPublishSummary(
        patient_id=patient_id,
        run_id=gt.run_id,
        applied_plan_fixes=applied_fixes,
        narrative_critical_issues=narrative.critical_issues,
        visual_edits_applied=visual_edits_applied,
        video_path=video_path,
        portal_copy_path=portal_copy_path,
        backend_upload_ok=backend_upload_ok,
        backend_upload_response=backend_upload_response,
    )
    return plan, summary
