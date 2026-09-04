"""Private, atomic receipts at paid dispatch boundaries. No ambiguous replay."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable


class ReceiptConflict(RuntimeError):
    """Saved evidence is incompatible or corrupt; operator action is required."""


class SavedSizeRejection(RuntimeError):
    """Reuse a documented size rejection when resuming its accepted fallback."""
    def __init__(self, code: int):
        self.status_code = code
        self.body = {"param": "size"}
        super().__init__("Requested image size was explicitly rejected")


class UnknownDispatch(RuntimeError):
    """The provider may have run; automatic paid replay is prohibited."""


class AssetFailures(RuntimeError):
    """Independent assets failed; successful receipts remain reusable."""
    def __init__(self, failures: dict[str, Exception]):
        self.failures = failures
        super().__init__("; ".join(f"{key}: {type(exc).__name__}: {exc}" for key, exc in failures.items()))


class AssetBusy(RuntimeError):
    """Another live process owns this asset or project."""


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def request_digest(value: dict) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, json.dumps(value, sort_keys=True).encode())


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('a+b') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AssetBusy(str(path)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


_current: ContextVar = ContextVar('generation_asset', default=None)


class AssetOperation:
    def __init__(self, operation_dir: Path, attempt_id: str, asset_id: str):
        for value in (attempt_id, asset_id):
            if not value or Path(value).name != value or value in {'.', '..'}:
                raise ValueError('Attempt and asset identities must be single path components')
        self.directory = Path(operation_dir) / attempt_id / asset_id
        self.identity = {'attempt_id': attempt_id, 'asset_id': asset_id}

    def __enter__(self):
        self._lock = exclusive_lock(self.directory / 'asset.lock')
        self._lock.__enter__()
        self._token = _current.set(self)
        return self

    def __exit__(self, *args):
        _current.reset(self._token)
        return self._lock.__exit__(*args)


def status_code(exc: Exception):
    code = getattr(exc, 'status_code', None)
    if code is None:
        code = getattr(getattr(exc, 'response', None), 'status_code', None)
    if code is None:
        code = getattr(exc, 'status', None)
    return code


def rejected_request(exc: Exception) -> bool:
    return status_code(exc) in {400, 401, 403, 404, 422, 429}


def paid_bytes(request: dict, dispatch: Callable[[], bytes], *, output_path: Path | None = None,
               key: str = 'paid', provenance: dict | None = None,
               recover: Callable[[], bytes | None] | None = None) -> bytes:
    """Return saved raw response bytes, or dispatch once after durable intent.

    The dispatch callback includes response body collection, never conversion.
    A lost response/body is unknown. Explicit rejected requests can be retried.
    Outside mdvm, output-bound receipts use the exact request as attempt identity.
    """
    operation = _current.get()
    if operation is None:
        if output_path is None:
            raise ValueError('Paid dispatch requires an asset context or output_path')
        with AssetOperation(output_path.parent / '.generation', request_digest(request), output_path.name):
            return paid_bytes(request, dispatch, key=key, provenance=provenance, recover=recover)
    path = operation.directory / (key + '.json')
    digest = request_digest(request)
    marker = operation.directory / (key + '.dispatch.json')
    if marker.exists():
        if json.loads(marker.read_text()).get('request_digest') != digest:
            raise ReceiptConflict(f'Incompatible dispatch marker: {marker}')
        if not path.exists():
            raise UnknownDispatch(f'Paid acknowledgement is missing after dispatch intent: {path}')
    record = json.loads(path.read_text()) if path.exists() else {}
    if record and record.get('request_digest') != digest:
        raise ReceiptConflict(f'Incompatible request: {path}')
    if record.get('state') == 'rejected' and record.get('invalid_size'):
        raise SavedSizeRejection(record['status_code'])
    if record.get('state') == 'acknowledged':
        try:
            raw = base64.b64decode(record['raw_base64'], validate=True)
        except (KeyError, ValueError) as exc:
            raise ReceiptConflict(f'Corrupt raw response: {path}') from exc
        if digest_bytes(raw) != record.get('raw_sha256'):
            raise ReceiptConflict(f'Raw response hash mismatch: {path}')
        return raw
    if record.get('state') in {'dispatched', 'unknown'}:
        raw = recover() if recover is not None else None
        if raw:
            record.update(state='acknowledged', raw_base64=base64.b64encode(raw).decode(), raw_sha256=digest_bytes(raw))
            atomic_json(path, record)
            return raw
        raise UnknownDispatch(f'Unacknowledged paid call: {path}')
    record = {**operation.identity, 'request': request, 'request_digest': digest,
              'provenance': provenance or {}, 'state': 'dispatched',
              'dispatch_count': int(record.get('dispatch_count', 0)) + 1}
    if not marker.exists():
        atomic_json(marker, {**operation.identity, 'request_digest': digest})
    atomic_json(path, record)
    try:
        raw = dispatch()
        if not isinstance(raw, bytes) or not raw:
            raise ValueError('Provider returned no complete response bytes')
    except Exception as exc:
        record['state'] = 'rejected' if rejected_request(exc) else 'unknown'
        record['error_type'] = type(exc).__name__
        record['status_code'] = status_code(exc)
        body = getattr(exc, 'body', {})
        error = body.get('error', body) if isinstance(body, dict) else {}
        record['invalid_size'] = status_code(exc) in {400, 422} and error.get('param') == 'size'
        atomic_json(path, record)
        if record['state'] == 'unknown':
            raise UnknownDispatch(f'Paid response unknown: {path}') from exc
        raise
    # One atomic envelope contains the complete acknowledged bytes and hash. A
    # crash during this write leaves intent unknown, never permission to repay.
    record.update(state='acknowledged', raw_base64=base64.b64encode(raw).decode(), raw_sha256=digest_bytes(raw))
    atomic_json(path, record)
    return raw
