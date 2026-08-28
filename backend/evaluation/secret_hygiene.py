"""Scan source files, runtime logs, and Git snapshots for secret literals.

The scanner reports only detector names and file locations. It never returns
the matched value or source line, so its own JSON output is safe to retain.
The ignored root ``.env`` file is intentionally excluded because it is the
documented local secret store.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".conf",
    ".env",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
SKIPPED_PARTS = {".git", ".venv", "node_modules", "__pycache__"}
PROVIDER_PATTERNS = (
    (
        "evren_api_key",
        re.compile(r"sk-evren-[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    ),
    (
        "qdrant_api_key",
        re.compile(r"qdr-[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    ),
    (
        "database_url_credential",
        re.compile(
            r"postgres(?:ql)?://[^\s/:@<>{}]+:[^\s/@<>{}]{8,}@",
            re.IGNORECASE,
        ),
    ),
    (
        "bearer_token",
        re.compile(r"\bbearer\s+[A-Za-z0-9._~-]{20,}", re.IGNORECASE),
    ),
)
SENSITIVE_LITERAL_PATTERN = re.compile(
    r"""
    (?:
        ["']?[a-z0-9_]*
        (?:
            password|passwd|parola|api[_-]?key|secret|anahtar[iı]?
            |(?:access|auth|bearer|owner|refresh|session)[_-]?token
            |token[_-]?(?:key|secret)
        )
        [a-z0-9_]*["']?
    )
    \s*[:=]\s*
    (?P<quote>["'])
    (?P<value>[^"'\r\n]{12,})
    (?P=quote)
    """,
    re.IGNORECASE | re.VERBOSE,
)
SENSITIVE_ENV_PATTERN = re.compile(
    r"""
    ^[ \t]*(?:(?i:export)[ \t]+|(?i:\$env:))?
    [A-Z0-9_]*
    (?:
        PASSWORD|PASSWD|PAROLA|API[_-]?KEY|SECRET|ANAHTAR
        |(?:ACCESS|AUTH|OWNER|REFRESH|SESSION)[_-]?TOKEN
        |TOKEN[_-]?(?:KEY|SECRET)
    )
    [A-Z0-9_]*[ \t]*=[ \t]*
    (?P<value>[^\s#;"']{12,})
    """,
    re.MULTILINE | re.VERBOSE,
)
SAFE_LITERAL_MARKERS = (
    "change_me",
    "dummy",
    "example",
    "fake",
    "not-a-real",
    "placeholder",
    "redacted",
    "sentinel",
    "test",
)
HISTORY_PREFILTER = (
    r"sk-evren-[A-Za-z0-9_-]{20,}"
    r"|qdr-[A-Za-z0-9_-]{20,}"
    r"|postgres(ql)?://[^[:space:]/:@]+:[^[:space:]/@]{8,}@"
    r"|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[A-Za-z0-9._~-]{20,}"
    r"|password|passwd|parola|api[_-]?key|secret|anahtar"
    r"|(access|auth|owner|refresh|session)[_-]?token"
    r"|token[_-]?(key|secret)"
)


@dataclass(frozen=True, slots=True)
class SecretFinding:
    detector: str
    location: str
    snapshot: str | None = None


class SecretScanError(RuntimeError):
    """Raised when the repository cannot be scanned safely."""


def _run_git(
    arguments: Sequence[str],
    *,
    project_dir: Path,
    text: bool = False,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=project_dir,
            check=False,
            capture_output=True,
            text=text,
            encoding="utf-8" if text else None,
            errors="replace" if text else None,
        )
    except OSError as error:
        raise SecretScanError("git_command_unavailable") from error


def _is_safe_literal(value: str) -> bool:
    lowered = value.casefold()
    if re.fullmatch(r"<[^>\r\n]+>|\$\{[A-Z][A-Z0-9_]*\}", value):
        return True
    if any(marker in lowered for marker in SAFE_LITERAL_MARKERS):
        return True
    if re.fullmatch(
        r"[A-Z][A-Z0-9_]*(?:_API_KEY|_PASSWORD|_TOKEN|_SECRET|_ENV)",
        value,
    ):
        return True
    return re.fullmatch(r"[a-z]+-password-long", lowered) is not None


def scan_text(text: str, location: str, snapshot: str | None = None) -> list[SecretFinding]:
    """Return redacted findings for one decoded text payload."""

    findings: list[SecretFinding] = []
    provider_spans: list[tuple[int, int]] = []
    for detector, pattern in PROVIDER_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            findings.append(SecretFinding(detector, location, snapshot))
            provider_spans.extend(match.span() for match in matches)

    def overlaps_provider(span: tuple[int, int]) -> bool:
        return any(
            span[0] < provider_span[1] and provider_span[0] < span[1]
            for provider_span in provider_spans
        )

    for match in SENSITIVE_LITERAL_PATTERN.finditer(text):
        if (
            not overlaps_provider(match.span("value"))
            and not _is_safe_literal(match.group("value"))
        ):
            findings.append(
                SecretFinding("sensitive_literal", location, snapshot)
            )
            break
    for match in SENSITIVE_ENV_PATTERN.finditer(text):
        if (
            not overlaps_provider(match.span("value"))
            and not _is_safe_literal(match.group("value"))
        ):
            findings.append(
                SecretFinding("sensitive_env_literal", location, snapshot)
            )
            break
    return findings


def _decode(payload: bytes) -> str | None:
    if b"\x00" in payload:
        return None
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _eligible_path(path: Path, project_dir: Path) -> bool:
    try:
        relative = path.relative_to(project_dir)
    except ValueError:
        return False
    if relative.as_posix() == ".env" or any(
        part in SKIPPED_PARTS for part in relative.parts
    ):
        return False
    return (
        path.suffix.casefold() in TEXT_SUFFIXES
        or path.name.startswith(".env.")
    )


def _unique_findings(findings: Iterable[SecretFinding]) -> list[SecretFinding]:
    return sorted(
        set(findings),
        key=lambda item: (item.snapshot or "", item.location, item.detector),
    )


def source_paths(project_dir: Path = PROJECT_DIR) -> list[Path]:
    """List tracked and non-ignored untracked source paths."""

    result = _run_git(
        ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        project_dir=project_dir,
    )
    if result.returncode != 0:
        raise SecretScanError("git_source_listing_failed")
    paths = []
    for raw_path in result.stdout.split(b"\x00"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = project_dir / relative
        if candidate.is_file() and _eligible_path(candidate, project_dir):
            paths.append(candidate)
    return sorted(set(paths))


def scan_sources(project_dir: Path = PROJECT_DIR) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for path in source_paths(project_dir):
        text = _decode(path.read_bytes())
        if text is None:
            continue
        location = path.relative_to(project_dir).as_posix()
        findings.extend(scan_text(text, location))
    return _unique_findings(findings)


def scan_runtime_logs(project_dir: Path = PROJECT_DIR) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for path in project_dir.rglob("*.log"):
        if not _eligible_path(path, project_dir):
            continue
        text = _decode(path.read_bytes())
        if text is None:
            continue
        location = path.relative_to(project_dir).as_posix()
        findings.extend(scan_text(text, location))
    return _unique_findings(findings)


def _history_candidates(project_dir: Path) -> list[tuple[str, str]]:
    revisions = _run_git(
        ["rev-list", "--all"],
        project_dir=project_dir,
        text=True,
    )
    if revisions.returncode != 0:
        raise SecretScanError("git_history_listing_failed")
    candidates: set[tuple[str, str]] = set()
    for revision in revisions.stdout.splitlines():
        revision = revision.strip()
        if not revision:
            continue
        # --full-name: backend/ artik deponun alt dizini oldugu icin git grep
        # yollari cwd'ye gore veriyordu; "git show <rev>:<yol>" ise depo
        # kokune gore yol bekler ve okuma basarisiz oluyordu.
        matches = _run_git(
            ["grep", "--full-name", "-I", "-l", "-E", HISTORY_PREFILTER, revision],
            project_dir=project_dir,
            text=True,
        )
        if matches.returncode not in {0, 1}:
            raise SecretScanError("git_history_search_failed")
        for line in matches.stdout.splitlines():
            prefix = f"{revision}:"
            if line.startswith(prefix):
                path = line[len(prefix) :]
                if path:
                    candidates.add((revision, path))
    return sorted(candidates)


def scan_history(project_dir: Path = PROJECT_DIR) -> list[SecretFinding]:
    """Scan every reachable Git snapshot without exposing matched values."""

    findings: list[SecretFinding] = []
    for revision, location in _history_candidates(project_dir):
        payload = _run_git(
            ["show", f"{revision}:{location}"],
            project_dir=project_dir,
        )
        if payload.returncode != 0:
            raise SecretScanError("git_history_object_read_failed")
        text = _decode(payload.stdout)
        if text is not None:
            findings.extend(scan_text(text, location, revision))
    return _unique_findings(findings)


def build_report(project_dir: Path = PROJECT_DIR) -> dict[str, object]:
    source_findings = scan_sources(project_dir)
    log_findings = scan_runtime_logs(project_dir)
    history_findings = scan_history(project_dir)
    all_findings = _unique_findings(
        [*source_findings, *log_findings, *history_findings]
    )
    return {
        "schema_version": "1.0",
        "status": "passed" if not all_findings else "failed",
        "policy": {
            "matched_values_are_never_reported": True,
            "root_dotenv_excluded": True,
            "git_history_scanned": True,
        },
        "counts": {
            "source": len(source_findings),
            "runtime_logs": len(log_findings),
            "git_history": len(history_findings),
            "total": len(all_findings),
        },
        "findings": [asdict(item) for item in all_findings],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan source files, logs, and Git history for credential literals "
            "without printing matched values."
        )
    )
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.project_dir.resolve())
    except SecretScanError as error:
        report = {
            "schema_version": "1.0",
            "status": "error",
            "reason": str(error),
        }
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
