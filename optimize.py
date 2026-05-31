#!/usr/bin/env python3
"""Hermes Agent Optimizer -- TF-IDF based selective memory retrieval.

Patches an existing hermes-agent installation to add selective memory
retrieval. Zero external dependencies, pure Python TF-IDF ranking.

Usage:
    python optimize.py                    # auto-detect + patch
    python optimize.py --path /path/to/hermes-agent
    python optimize.py --rollback         # restore from last backup
    python optimize.py --dry-run          # show what would change
"""

import argparse
import difflib
import filecmp
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Patch definitions -- each entry references a full modified file
# in patches/modified/ so the source is visible on GitHub.
# ---------------------------------------------------------------------------

PATCHES_DIR = Path(__file__).resolve().parent / "patches"
MODIFIED_DIR = PATCHES_DIR / "modified"

# (id, rel_path_in_hermes, modified_filename, label)
PATCHES: List[Tuple[str, str, str, str]] = [
    (
        "hermes_cli_config",
        "hermes_cli/config.py",
        "hermes_cli_config.py",
        "Config: default selective_retrieval settings",
    ),
    (
        "agent_init_defaults_router",
        "agent/agent_init.py",
        "agent_agent_init.py",
        "Agent init: _selective_retrieval + MemoryRouter",
    ),
    (
        "memory_tool",
        "tools/memory_tool.py",
        "tools_memory_tool.py",
        "Memory tool: select_for_query() + get_target_descriptions()",
    ),
    (
        "system_prompt",
        "agent/system_prompt.py",
        "agent_system_prompt.py",
        "System prompt: remove built-in memory from volatile tier",
    ),
    (
        "conversation_loop",
        "agent/conversation_loop.py",
        "agent_conversation_loop.py",
        "Conversation loop: per-turn TF-IDF memory injection",
    ),
    (
        "banner",
        "hermes_cli/banner.py",
        "hermes_cli_banner.py",
        "Banner: show 'Optimized with <3'",
    ),
]

NEW_FILES: List[Tuple[str, str]] = [
    ("agent/memory_router.py", "memory_router.py"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BACKUP_ROOT = Path.home() / ".hermes-optimizer" / "backups"


def _find_hermes_agent() -> Optional[Path]:
    """Locate the hermes-agent installation directory."""
    env_path = os.environ.get("HERMES_AGENT_PATH")
    if env_path:
        candidate = Path(env_path).resolve()
        if _is_hermes_root(candidate):
            return candidate

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "hermes-agent"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("location:"):
                loc = line.split(":", 1)[1].strip()
                candidate = Path(loc).resolve()
                for p in [candidate, candidate.parent]:
                    if _is_hermes_root(p):
                        return p
    except Exception:
        pass

    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    common_locations = [
        hermes_home / "hermes-agent",
        Path.home() / ".hermes" / "hermes-agent",
        Path("/usr/local/lib/hermes-agent"),
    ]
    for loc in common_locations:
        if _is_hermes_root(loc):
            return loc

    for start in [Path.cwd(), *sys.path]:
        try:
            p = Path(start).resolve()
            if _is_hermes_root(p):
                return p
        except Exception:
            pass

    return None


def _is_hermes_root(path: Path) -> bool:
    markers = [
        path / "run_agent.py",
        path / "cli.py",
        path / "agent" / "system_prompt.py",
        path / "tools" / "memory_tool.py",
    ]
    return all(m.is_file() for m in markers)


def _find_hermes_via_python() -> Optional[Path]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import run_agent; print(run_agent.__file__)"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            path = Path(result.stdout.strip()).resolve()
            return path.parent if _is_hermes_root(path.parent) else None
    except Exception:
        pass
    return None


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _backup_path(base: Path, rel: str) -> Path:
    return base / rel.replace("/", os.sep)


# ---------------------------------------------------------------------------
# Merge-based patch application
# ---------------------------------------------------------------------------

PatchResult = Dict[str, bool]


def _merge_with_modified(original: str, modified: str) -> str:
    """Line-by-line merge using SequenceMatcher opcodes.

    Walks through both files in order and emits the modified version's
    lines for changed regions, preserving equal regions from either.
    """
    old = original.splitlines(keepends=True)
    new = modified.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, old, new)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(old[i1:i2])
        elif tag == "replace":
            out.extend(new[j1:j2])
        elif tag == "delete":
            pass
        elif tag == "insert":
            out.extend(new[j1:j2])
    return "".join(out)


def apply_patches(hermes_root: Path, dry_run: bool = False) -> PatchResult:
    """Apply all patches by reading modified files from patches/modified/.

    Uses difflib.SequenceMatcher to merge the installed file with the
    post-patch source, preserving local changes outside the modified
    regions.
    """
    results: PatchResult = {}

    for pid, rel, modified_name, label in PATCHES:
        target = (hermes_root / rel).resolve()
        modified_src = (MODIFIED_DIR / modified_name).resolve()

        if not target.exists():
            print(f"  [SKIP] {label}: {rel} not found")
            results[pid] = False
            continue

        if not modified_src.exists():
            print(f"  [FAIL] {label}: patches/modified/{modified_name} not found")
            results[pid] = False
            continue

        original_content = _read_file(target)
        modified_content = _read_file(modified_src)

        if original_content == modified_content:
            print(f"  [OK]   {label}: already applied (files identical)")
            results[pid] = True
            continue

        patched = _merge_with_modified(original_content, modified_content)

        if patched == original_content:
            print(f"  [SKIP] {label}: no changes detected")
            results[pid] = False
            continue

        if dry_run:
            print(f"  [DRY]  {label}")
            results[pid] = True
            continue

        _write_file(target, patched)
        print(f"  [OK]   {label}")
        results[pid] = True

    return results


def copy_new_files(hermes_root: Path, dry_run: bool = False) -> PatchResult:
    """Copy new files from patches/ into the hermes-agent tree."""
    results: PatchResult = {}

    for rel, src_name in NEW_FILES:
        src = (PATCHES_DIR / src_name).resolve()
        dst = (hermes_root / rel).resolve()

        if not src.exists():
            print(f"  [FAIL] new file {src_name}: patch file not found")
            results[rel] = False
            continue

        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            print(f"  [OK]   {rel}: already exists (no change)")
            results[rel] = True
            continue

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  [OK]   {rel} -> copied")
        else:
            print(f"  [DRY]  {rel} -> will copy")

        results[rel] = True

    return results


# ---------------------------------------------------------------------------
# Backup / rollback
# ---------------------------------------------------------------------------


def create_backup(hermes_root: Path) -> Optional[Path]:
    """Backup all files that will be patched."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    all_rel_paths: set[str] = set()
    for _pid, rel, _modified_name, _label in PATCHES:
        all_rel_paths.add(rel)
    for rel, _src_name in NEW_FILES:
        all_rel_paths.add(rel)

    for rel in sorted(all_rel_paths):
        src = (hermes_root / rel).resolve()
        if not src.exists():
            continue
        dst = _backup_path(backup_dir, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    meta = {
        "timestamp": timestamp,
        "hermes_root": str(hermes_root),
        "files": sorted(all_rel_paths),
    }
    (backup_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  Backup: {backup_dir}")
    return backup_dir


def list_backups() -> List[Path]:
    if not BACKUP_ROOT.exists():
        return []
    backups = sorted(
        [d for d in BACKUP_ROOT.iterdir() if d.is_dir() and (d / "meta.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    return backups


def rollback(backup_dir: Optional[Path] = None) -> bool:
    """Restore files from *backup_dir* (or latest if None)."""
    if backup_dir is None:
        backups = list_backups()
        if not backups:
            print("  No backup found to restore.")
            return False
        backup_dir = backups[0]
        print(f"  Using latest backup: {backup_dir}")

    meta_path = backup_dir / "meta.json"
    if not meta_path.exists():
        print(f"  ERROR: {meta_path} not found, invalid backup.")
        return False

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hermes_root = Path(meta["hermes_root"])

    for rel in meta.get("files", []):
        src = _backup_path(backup_dir, rel)
        dst = (hermes_root / rel).resolve()
        if not src.exists():
            print(f"  [SKIP] {rel}: not in backup")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  [OK]   {rel} -> restored")

    router_file = hermes_root / "agent" / "memory_router.py"
    if router_file.exists():
        router_file.unlink()
        print(f"  [OK]   agent/memory_router.py -> removed")

    print(f"  Restore complete: {backup_dir}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Hermes Agent Optimizer -- TF-IDF selective memory retrieval",
    )
    parser.add_argument(
        "--path", "-p",
        default=None,
        help="Hermes-agent root directory (use if auto-detection fails)",
    )
    parser.add_argument(
        "--rollback", "-r",
        action="store_true",
        help="Restore from latest backup",
    )
    parser.add_argument(
        "--backup-dir", "-b",
        default=None,
        help="Restore from a specific backup (use with --rollback)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would change, don't apply",
    )
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="List available backups",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Hermes Agent Optimizer -- Selective Memory Retrieval")
    print("=" * 60)

    if args.list_backups:
        backups = list_backups()
        if not backups:
            print("  No backups found.")
            return
        print(f"\n  Available backups ({len(backups)}):")
        for b in backups:
            meta = json.loads((b / "meta.json").read_text(encoding="utf-8"))
            print(f"    {b.name}  ({meta.get('hermes_root', '?')})")
        return

    if args.rollback:
        backup_dir = None
        if args.backup_dir:
            backup_dir = BACKUP_ROOT / args.backup_dir
            if not backup_dir.exists():
                print(f"  ERROR: {backup_dir} not found")
                sys.exit(1)
        print("\n  Restoring from backup...")
        rollback(backup_dir)
        return

    # Find hermes-agent
    hermes_root = None
    if args.path:
        hermes_root = Path(args.path).resolve()
        if not _is_hermes_root(hermes_root):
            print(f"  ERROR: {hermes_root} is not a valid hermes-agent directory")
            print(f"  (checked: run_agent.py, cli.py, agent/system_prompt.py)")
            sys.exit(1)
    else:
        hermes_root = _find_hermes_agent()
        if hermes_root is None:
            hermes_root = _find_hermes_via_python()

    if hermes_root is None:
        print("\n  Hermes-agent not found. Use --path:")
        print("    python optimize.py --path /path/to/hermes-agent")
        sys.exit(1)

    hermes_root = hermes_root.resolve()
    print(f"\n  Hermes-agent: {hermes_root}")

    if not PATCHES_DIR.exists():
        print(f"  ERROR: patches/ directory not found ({PATCHES_DIR})")
        sys.exit(1)

    if not MODIFIED_DIR.exists():
        print(f"  ERROR: patches/modified/ directory not found ({MODIFIED_DIR})")
        sys.exit(1)

    if args.dry_run:
        print("\n  --- DRY RUN (no changes made) ---\n")
        apply_patches(hermes_root, dry_run=True)
        copy_new_files(hermes_root, dry_run=True)
        print("\n  Dry run complete. Run without --dry-run to apply.")
        return

    # Backup
    print("\n  Backing up...")
    backup_dir = create_backup(hermes_root)
    if backup_dir is None:
        print("  ERROR: Backup failed")
        sys.exit(1)

    # Apply patches
    print("\n  Applying patches...")
    patch_results = apply_patches(hermes_root)

    # Copy new files
    print("\n  Copying new files...")
    copy_results = copy_new_files(hermes_root)

    # Summary
    all_ok = True
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    for pid, rel, _modified_name, label in PATCHES:
        ok = patch_results.get(pid, False)
        icon = "+" if ok else "-"
        print(f"  [{icon}] {rel}")
        if not ok:
            all_ok = False

    for rel, _src_name in NEW_FILES:
        ok = copy_results.get(rel, False)
        icon = "+" if ok else "-"
        print(f"  [{icon}] {rel} (new file)")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  Success! Hermes-agent optimized with TF-IDF selective memory.")
        print(f"  To restore: python optimize.py --rollback")
        print(f"  Backup: {backup_dir}")
    else:
        print("\n  Some patches could not be applied. Check [FAIL] lines above.")
        print(f"  To restore: python optimize.py --rollback")

    print()


if __name__ == "__main__":
    main()
