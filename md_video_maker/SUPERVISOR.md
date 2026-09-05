# Detached renderer attempts

Run the supervisor with the versioned application's locked Python environment.
Keep the application and lock immutable for the life of its admitted attempts.
Keep project data and attempt state outside that release directory. Codex CLI is
resolved and version-checked per executing job; its executable and version are
provenance and remain independently updateable.

```python
from pathlib import Path
from md_video_maker.supervisor import (
    prepare_attempt, start_attempt, attempt_snapshot, project_write_lock,
)

attempt = prepare_attempt(
    Path('/data/projects/example'),
    'stable-workbench-authorization-id',
    state_dir=Path('/data/render-state'),
    force_images=False,
    force_audio=False,
)
start_attempt(attempt)
status = attempt_snapshot(attempt)
```

`prepare_attempt` persists immutable admission without launching a process.
One operation ID within a state root addresses one attempt; reuse with identical
inputs rejoins it, including while it is running. Changed plan bytes, local
source bytes, force flags, effective non-secret settings, application bytes or
Python/lock identity raise `ReceiptConflict`. Different authorized operation IDs
remain distinct even when their content matches. Persist/reuse the operation ID
and returned attempt path across workbench restarts. A fresh operation ID is a
fresh authorization; use the original attempt when recovering.

The CLI prints JSON. `prepare PROJECT --operation-id OP --state-dir STATE`
also accepts `--force-images` and `--force-audio`. `start ATTEMPT`,
`snapshot ATTEMPT` and the detached child entrypoint `run ATTEMPT` accept the
saved attempt directory. Invoke as `python -m md_video_maker.supervisor`.
A handled admission/read/launch error writes JSON with `error_type`/`error` to
stderr and exits 2. A completed run/snapshot command exits 0 with the actual
state in its JSON; exit 0 alone does not establish render success.

`start_attempt` detaches stdin/session and redirects both streams to persistent
`stdout.log`/`stderr.log`. It has no parent deadline. It may return `pending`
before the child acquires its lock. It can be called again if the parent dies
around launch; competing children only observe unless they own `owner.lock`.
Only that OS lock owner writes started/terminal/output receipts or calls the
renderer. `owner_active` comes from the lock, never elapsed time or PID alone.
`owner` includes PID, an optional OS process-start diagnostic, an unguessable
owner token, attempt token and start time. Restricted process diagnostics are
represented by null and do not disable execution ownership.

States:

- `pending`: admission exists and no supervisor owns it. `recovery_required`
  indicates an earlier owner; call `start_attempt` with the same path. A busy
  project leaves the attempt pending with no provider dispatch.
- `running`: an OS owner is active, including the brief pre-acknowledgment window.
- `complete`: terminal, admission, renderer manifest, exact owned output SHA256
  and probed media facts agree. Deliver `output.path`, not the canonical MP4.
- `reconciliation_required`: an asset's paid result is unknown or immutable
  evidence conflicts. `failures` maps each affected asset (or `attempt`) to
  `state` and `error_type`. Independent successful assets retain their receipts.
- `local_failure`: a local/provider error stopped completion. The same failure
  mapping is present. Retrying the same attempt reuses asset receipts and the
  original force flags; it never invents fresh paid intent from missing output.

`AssetBusy` is a transient lock conflict for admission/editing, not permission
to steal ownership. `ReceiptConflict` from a snapshot is invalid evidence, not
success. An asset `UnknownDispatch` remains reconciliation-required on recovery.
No blanket paid retry loop exists in the supervisor.

All state is below `STATE/attempts/SHA256(operation_id)/`. Admission contains a
random attempt token, exact immutable inputs, expected output and hashes of
pre-existing top-level MP4s. `assets/ATTEMPT_ID/` holds the existing renderer's
immutable complete prepared asset set and paid-boundary receipts. A nonempty
asset attempt always enters renderer recovery, even if its manifest is missing.
An empty/unstarted attempt can prepare normally under its original admission.

The existing renderer project lock remains
`PROJECT/.render-operations/project.lock`, irrespective of attempt-state path.
All workbench plan/local-source edits must acquire `project_write_lock(PROJECT)`
and hold it across the whole read/modify/atomic-write. All other callers of this
renderer use the same lock. Do not hold it around `render_project`: that function
acquires it internally. The optional `on_locked()` callback verifies admitted
inputs before rendering; `on_output(new_mp4_path)` runs after media validation
and before canonical replacement while the lock is still held.

The output callback copies the exact new MP4 into `output.mp4` and atomically
writes `output.json`, binding admission digest, renderer manifest digest,
attempt/owner tokens, SHA256 and actual duration/dimensions/codec/size. Only then
can the canonical project MP4 be replaced. `terminal.json` binds that evidence.
An interrupted output receipt causes only local reassembly from existing asset
receipts; an interrupted terminal receipt can complete directly from the saved
output without invoking the renderer. This recovery validates the admitted
application, Python/lock and effective runtime config, then the immutable output
evidence. It does not read current project plan/local-source bytes; lawful later
edits or removal cannot invalidate already registered output. The complete
current-input check runs only when rendering/reassembly is required, including
under the project lock. Rejoin recovery by the saved attempt path rather than
re-preparing the old operation against newer project inputs. A later render cannot change historical
attempt-owned bytes. File writes use the existing flush/fsync/atomic-replace/
parent-fsync helpers; no daemon service or wall-clock lease is involved.

Completed output also includes a complete `audio` list, sorted by numeric scene
ID. Each record has `scene_id`, an absolute owned `path`, `sha256`,
`duration_seconds`, and `size_bytes`. Paths are exactly
`ATTEMPT/audio/scene_NNN.wav`, with one record per `audio-*` asset in the hashed
renderer manifest. Noncontiguous scene IDs retain their actual IDs.

During the same locked output callback, every final canonical narration WAV is
copied into that attempt-owned directory, including narration reused from cache.
The copied WAV is validated and probed before the output receipt is registered.
Only an independent regular file at the exact owned path can satisfy audio
evidence; symlinks, indirect directories, hardlinks, missing/truncated files,
changed hashes/sizes/durations, and incomplete/duplicate inventories fail.
`output.json` is written only after the MP4 and all narration copies are durable,
and `terminal.json` binds that entire object. An interrupted unregistered copy
uses the original-input asset recovery path; registered output recovers without
reading current project audio or invoking providers. Preserve the whole bundle
for as long as its deliverable/quality evidence is retained.

Consumers match audio records to the original plan by scene ID. For quality
checks, use copied in-memory scenes whose `audio_path` points to the validated
owned WAV. Original plan bytes remain frozen; an old explicit absolute
`audio_path` must not override the owned evidence. Existing duration checks can
then probe actual narration files and retain their full timing rules.

This renderer release requires the complete audio field. An older release's
video-only output remains unchanged at its historical path and is read through
that saved original release. It cannot supply the new full narration contract.
Missing audio evidence is never repaired by attaching today's mutable cache or
by automatically spending on generation. No historical receipt is migrated or
enriched by this extension.
