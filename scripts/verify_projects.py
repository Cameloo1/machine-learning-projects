from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".verify",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "ENV",
}


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    seconds: float = 0.0
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "seconds": round(self.seconds, 3),
        }
        if self.details:
            data["details"] = self.details
        return data


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema_version: {manifest.get('schema_version')}")
    if not isinstance(manifest.get("projects"), list):
        raise ValueError("Manifest must contain a projects list")
    return manifest


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def has_files(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(child.is_file() and child.stat().st_size > 0 for child in path.rglob("*"))
    return False


def iter_python_files(project_root: Path) -> Iterable[Path]:
    for path in project_root.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def check_expected_paths(project_root: Path, paths: list[str]) -> CheckResult:
    start = time.perf_counter()
    missing: list[str] = []
    empty: list[str] = []
    for item in paths:
        path = project_root / item
        if not path.exists():
            missing.append(item)
        elif path.is_file() and path.stat().st_size == 0:
            empty.append(item)

    if missing or empty:
        message_parts = []
        if missing:
            message_parts.append(f"missing: {', '.join(missing)}")
        if empty:
            message_parts.append(f"empty: {', '.join(empty)}")
        return CheckResult(
            "expected paths",
            "fail",
            "; ".join(message_parts),
            time.perf_counter() - start,
            {"missing": missing, "empty": empty},
        )

    return CheckResult(
        "expected paths",
        "pass",
        f"{len(paths)} required paths present",
        time.perf_counter() - start,
    )


def check_dependency_files(project_root: Path, files: list[str]) -> CheckResult:
    start = time.perf_counter()
    if not files:
        return CheckResult(
            "dependency files",
            "skip",
            "no dependency file declared; install command documents packages",
            time.perf_counter() - start,
        )
    missing = [item for item in files if not nonempty_file(project_root / item)]
    status = "fail" if missing else "pass"
    message = f"missing/non-empty dependency files: {', '.join(missing)}" if missing else f"{len(files)} dependency file(s) present"
    return CheckResult(
        "dependency files",
        status,
        message,
        time.perf_counter() - start,
        {"missing": missing} if missing else None,
    )


def check_python_syntax(project_root: Path, repo_root: Path) -> CheckResult:
    start = time.perf_counter()
    failures: list[dict[str, str]] = []
    count = 0
    for path in iter_python_files(project_root):
        count += 1
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except Exception as exc:  # SyntaxError plus encoding/OS edge cases.
            failures.append({"path": rel(path, repo_root), "error": str(exc)})

    if failures:
        return CheckResult(
            "python syntax",
            "fail",
            f"{len(failures)} of {count} Python files failed syntax compilation",
            time.perf_counter() - start,
            {"failures": failures},
        )

    return CheckResult(
        "python syntax",
        "pass",
        f"{count} Python files compile",
        time.perf_counter() - start,
    )


def check_notebooks(project_root: Path, repo_root: Path) -> CheckResult:
    start = time.perf_counter()
    notebooks = [path for path in project_root.rglob("*.ipynb") if not any(part in DEFAULT_EXCLUDED_DIRS for part in path.parts)]
    failures: list[dict[str, str]] = []
    for path in notebooks:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "cells" not in data or "metadata" not in data:
                failures.append({"path": rel(path, repo_root), "error": "missing cells or metadata"})
        except Exception as exc:
            failures.append({"path": rel(path, repo_root), "error": str(exc)})

    if failures:
        return CheckResult(
            "notebook metadata",
            "fail",
            f"{len(failures)} of {len(notebooks)} notebooks failed JSON/schema check",
            time.perf_counter() - start,
            {"failures": failures},
        )
    if not notebooks:
        return CheckResult("notebook metadata", "skip", "no notebooks present", time.perf_counter() - start)
    return CheckResult(
        "notebook metadata",
        "pass",
        f"{len(notebooks)} notebook(s) parse as JSON",
        time.perf_counter() - start,
    )


def expand_globs(project_root: Path, patterns: list[str]) -> dict[str, list[Path]]:
    return {pattern: sorted(project_root.glob(pattern)) for pattern in patterns}


def check_artifacts(project_root: Path, patterns: list[str]) -> CheckResult:
    start = time.perf_counter()
    if not patterns:
        return CheckResult("artifact inventory", "skip", "no artifact globs declared", time.perf_counter() - start)

    matched = expand_globs(project_root, patterns)
    missing = [pattern for pattern, paths in matched.items() if not paths]
    empty = [
        str(path.relative_to(project_root).as_posix())
        for paths in matched.values()
        for path in paths
        if not has_files(path)
    ]
    total = sum(len(paths) for paths in matched.values())
    if missing or empty:
        return CheckResult(
            "artifact inventory",
            "fail",
            f"{len(missing)} missing glob(s), {len(empty)} empty match(es)",
            time.perf_counter() - start,
            {"missing_globs": missing, "empty_matches": empty},
        )
    return CheckResult(
        "artifact inventory",
        "pass",
        f"{total} artifact match(es) found across {len(patterns)} glob(s)",
        time.perf_counter() - start,
    )


def check_json_files(project_root: Path, repo_root: Path, patterns: list[str]) -> CheckResult:
    start = time.perf_counter()
    if not patterns:
        return CheckResult("json artifacts", "skip", "no JSON artifact globs declared", time.perf_counter() - start)

    matches = [path for paths in expand_globs(project_root, patterns).values() for path in paths if path.is_file()]
    failures: list[dict[str, str]] = []
    for path in matches:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({"path": rel(path, repo_root), "error": str(exc)})

    if failures:
        return CheckResult(
            "json artifacts",
            "fail",
            f"{len(failures)} of {len(matches)} JSON files failed parsing",
            time.perf_counter() - start,
            {"failures": failures},
        )
    if not matches:
        return CheckResult("json artifacts", "skip", "no JSON files matched declared globs", time.perf_counter() - start)
    return CheckResult("json artifacts", "pass", f"{len(matches)} JSON file(s) parse", time.perf_counter() - start)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_scripts_dir(venv_dir: Path) -> Path:
    return venv_python(venv_dir).parent


def rewrite_command(cmd: list[str], python_exe: str) -> list[str]:
    if not cmd:
        raise ValueError("empty command")
    if cmd[0].lower() == "python":
        return [python_exe, *cmd[1:]]
    if cmd[0].lower() == "pip":
        return [python_exe, "-m", "pip", *cmd[1:]]
    return cmd


def run_subprocess(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    python_exe: str,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    completed = subprocess.run(
        rewrite_command(cmd, python_exe),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    return completed.returncode, completed.stdout, completed.stderr, time.perf_counter() - start


def ensure_project_venv(repo_root: Path, project_id: str, python_exe: str) -> tuple[Path, CheckResult]:
    start = time.perf_counter()
    venv_dir = repo_root / ".verify" / "venvs" / project_id
    py = venv_python(venv_dir)
    if py.exists():
        return py, CheckResult("venv", "pass", f"using existing {rel(venv_dir, repo_root)}", time.perf_counter() - start)

    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)
    return py, CheckResult("venv", "pass", f"created {rel(venv_dir, repo_root)}", time.perf_counter() - start)


def command_env(venv_py: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if venv_py:
        scripts = str(venv_py.parent)
        env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(venv_py.parent.parent)
    return env


def run_install_steps(
    project: dict[str, Any],
    command_root: Path,
    repo_root: Path,
    args: argparse.Namespace,
    project_python: Path,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for index, step in enumerate(project.get("install", []), start=1):
        name = f"install {index}"
        if step.get("requires_network") and not args.allow_network:
            results.append(CheckResult(name, "skip", "requires --allow-network"))
            continue
        try:
            returncode, stdout, stderr, seconds = run_subprocess(
                step["cmd"],
                command_root,
                args.install_timeout,
                str(project_python),
                command_env(project_python),
            )
        except subprocess.TimeoutExpired:
            results.append(CheckResult(name, "fail", f"timed out after {args.install_timeout}s", args.install_timeout))
            continue
        status = "pass" if returncode == 0 else "fail"
        message = "install command completed" if status == "pass" else f"exit {returncode}"
        results.append(
            CheckResult(
                name,
                status,
                message,
                seconds,
                {"stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:], "cmd": step["cmd"]},
            )
        )
    return results


def command_in_level(command: dict[str, Any], level: str) -> bool:
    if level == "full":
        return command.get("level") in {"quick", "full"}
    return command.get("level") == "quick"


def run_commands(
    project: dict[str, Any],
    command_root: Path,
    repo_root: Path,
    args: argparse.Namespace,
    project_python: Path | None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    commands = [cmd for cmd in project.get("commands", []) if command_in_level(cmd, args.level)]
    if not commands:
        return [CheckResult("commands", "skip", f"no {args.level} commands declared")]

    if not args.run_commands:
        return [CheckResult("commands", "skip", "command execution disabled; pass --run-commands")]

    python_exe = str(project_python) if project_python else args.python
    env = command_env(project_python)
    for command in commands:
        name = f"command: {command['name']}"
        if command.get("requires_network") and not args.allow_network:
            results.append(CheckResult(name, "skip", "requires --allow-network"))
            continue
        timeout = int(command.get("timeout_seconds", args.command_timeout))
        try:
            returncode, stdout, stderr, seconds = run_subprocess(
                command["cmd"],
                command_root,
                timeout,
                python_exe,
                env,
            )
        except subprocess.TimeoutExpired:
            results.append(CheckResult(name, "fail", f"timed out after {timeout}s", timeout, {"cmd": command["cmd"]}))
            if args.fail_fast:
                break
            continue

        status = "pass" if returncode == 0 else "fail"
        details: dict[str, Any] = {
            "cmd": command["cmd"],
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
        if status == "pass":
            expected = command.get("expected_globs", [])
            missing = [pattern for pattern, paths in expand_globs(command_root, expected).items() if not paths]
            if missing:
                status = "fail"
                details["missing_expected_globs"] = missing
                message = f"command ran, but missing expected outputs: {', '.join(missing)}"
            else:
                message = "command completed"
        else:
            message = f"exit {returncode}"

        results.append(CheckResult(name, status, message, seconds, details))
        if args.fail_fast and status == "fail":
            break
    return results


def copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in DEFAULT_EXCLUDED_DIRS:
            ignored.add(name)
    return ignored


def prepare_command_workspace(
    project: dict[str, Any],
    project_root: Path,
    repo_root: Path,
    args: argparse.Namespace,
) -> tuple[Path, CheckResult | None]:
    if not (args.run_commands or args.install):
        return project_root, None
    if args.workspace_mode == "in-place":
        return project_root, CheckResult("command workspace", "pass", "using project directory in place")

    run_root = repo_root / ".verify" / "runs" / args.run_id / project["id"]
    if run_root.exists():
        shutil.rmtree(run_root)
    shutil.copytree(project_root, run_root, ignore=copy_ignore)
    message = f"copied to {rel(run_root, repo_root)}"
    return run_root, CheckResult("command workspace", "pass", message)


def verify_project(
    project: dict[str, Any],
    repo_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    project_root = repo_root / project["path"]
    results: list[CheckResult] = []
    if not project_root.exists():
        results.append(CheckResult("project path", "fail", f"missing {project['path']}"))
        return {
            "id": project["id"],
            "name": project["name"],
            "path": project["path"],
            "status": "fail",
            "checks": [item.as_dict() for item in results],
        }

    results.append(CheckResult("project path", "pass", project["path"]))
    results.append(check_expected_paths(project_root, project.get("expected_paths", [])))
    results.append(check_dependency_files(project_root, project.get("dependency_files", [])))
    results.append(check_python_syntax(project_root, repo_root))
    results.append(check_notebooks(project_root, repo_root))
    results.append(check_artifacts(project_root, project.get("artifact_globs", [])))
    results.append(check_json_files(project_root, repo_root, project.get("json_globs", [])))

    command_root, workspace_result = prepare_command_workspace(project, project_root, repo_root, args)
    if workspace_result:
        results.append(workspace_result)

    project_python: Path | None = None
    if args.install:
        project_python, venv_result = ensure_project_venv(repo_root, project["id"], args.python)
        results.append(venv_result)
        install_results = run_install_steps(project, command_root, repo_root, args, project_python)
        results.extend(install_results)
        if any(result.status == "fail" for result in install_results) and args.fail_fast:
            return project_result(project, results)

    results.extend(run_commands(project, command_root, repo_root, args, project_python))
    return project_result(project, results)


def project_result(project: dict[str, Any], results: list[CheckResult]) -> dict[str, Any]:
    if any(result.status == "fail" for result in results):
        status = "fail"
    elif any(result.status == "skip" for result in results):
        status = "pass_with_skips"
    else:
        status = "pass"
    return {
        "id": project["id"],
        "name": project["name"],
        "path": project["path"],
        "status": status,
        "checks": [item.as_dict() for item in results],
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Reproducibility Verification Report",
        "",
        f"- Started: `{report['started_at']}`",
        f"- Finished: `{report['finished_at']}`",
        f"- Level: `{report['options']['level']}`",
        f"- Commands run: `{report['options']['run_commands']}`",
        f"- Install enabled: `{report['options']['install']}`",
        f"- Network allowed: `{report['options']['allow_network']}`",
        "",
        "| Project | Status | Failed | Skipped |",
        "| --- | --- | ---: | ---: |",
    ]
    for project in report["projects"]:
        failed = sum(1 for check in project["checks"] if check["status"] == "fail")
        skipped = sum(1 for check in project["checks"] if check["status"] == "skip")
        lines.append(f"| {project['id']} - {project['name']} | `{project['status']}` | {failed} | {skipped} |")

    lines.extend(["", "## Failures", ""])
    failures = []
    for project in report["projects"]:
        for check in project["checks"]:
            if check["status"] == "fail":
                failures.append(f"- `{project['id']}` {check['name']}: {check['message']}")
    lines.extend(failures if failures else ["No failing checks."])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify reproducibility contracts for each ML project.")
    parser.add_argument("--manifest", default="projects.json", help="Path to the project manifest.")
    parser.add_argument("--level", choices=["quick", "full"], default="quick", help="Check level to run.")
    parser.add_argument("--project", action="append", help="Project id to verify. Can be repeated.")
    parser.add_argument("--run-commands", action="store_true", help="Run declared project commands.")
    parser.add_argument("--install", action="store_true", help="Create per-project venvs and run install commands.")
    parser.add_argument("--allow-network", action="store_true", help="Allow network-tagged install or run commands.")
    parser.add_argument("--report-dir", default="reports/reproducibility", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--workspace-mode", choices=["copy", "in-place"], default="copy", help="Where install and command checks run.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for commands and venv creation.")
    parser.add_argument("--command-timeout", type=int, default=600, help="Default command timeout in seconds.")
    parser.add_argument("--install-timeout", type=int, default=1200, help="Install command timeout in seconds.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop command execution after the first failure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_id = now_stamp()
    repo_root = Path.cwd()
    manifest_path = (repo_root / args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    selected = set(args.project or [])

    projects = [
        project
        for project in manifest["projects"]
        if not selected or project["id"] in selected
    ]
    unknown = selected - {project["id"] for project in manifest["projects"]}
    if unknown:
        print(f"Unknown project id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    report_projects = [verify_project(project, repo_root, args) for project in projects]
    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()

    report = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": finished_at,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "manifest": rel(manifest_path, repo_root),
        "options": {
            "level": args.level,
            "run_commands": args.run_commands,
            "install": args.install,
            "allow_network": args.allow_network,
            "workspace_mode": args.workspace_mode,
            "projects": sorted(selected),
        },
        "projects": report_projects,
    }

    report_dir = repo_root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    json_path = report_dir / f"{stamp}-{args.level}.json"
    md_path = report_dir / f"{stamp}-{args.level}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, md_path)

    for project in report_projects:
        print(f"{project['id']}: {project['status']} - {project['name']}")
        for check in project["checks"]:
            if check["status"] != "pass":
                print(f"  {check['status']}: {check['name']} - {check['message']}")
    print(f"\nWrote {rel(json_path, repo_root)}")
    print(f"Wrote {rel(md_path, repo_root)}")

    return 1 if any(project["status"] == "fail" for project in report_projects) else 0


if __name__ == "__main__":
    raise SystemExit(main())
