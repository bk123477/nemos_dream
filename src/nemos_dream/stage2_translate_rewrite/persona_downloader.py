"""Download the public Google Drive persona folder into ``data/``.

The stage-2 pipeline expects the persona bank to live under
``data/persona_age_gender`` relative to the repository root.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/1SGrM4gG4kz155nJTqV7aFZgkSehhGXgF?usp=sharing"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"
DEFAULT_FOLDER_NAME = "persona_age_gender"


def _extract_folder_id(url: str) -> str | None:
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def _ensure_gdown_installed() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "gdown is required to download the Google Drive persona folder. "
            "Run `uv sync` in the repo root first."
        ) from exc


def download_persona_folder(
    *,
    url: str = DEFAULT_DRIVE_URL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    folder_name: str = DEFAULT_FOLDER_NAME,
    force: bool = False,
) -> Path:
    """Download the shared Google Drive folder to ``output_dir/folder_name``."""

    _ensure_gdown_installed()
    import gdown

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    target_dir = output_root / folder_name

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        return target_dir

    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    before_children = {child.name for child in output_root.iterdir()}
    downloaded = gdown.download_folder(url=url, output=str(target_dir), quiet=False)
    if downloaded is None:
        folder_id = _extract_folder_id(url) or "<unknown>"
        raise RuntimeError(
            "Google Drive folder download failed. "
            f"Check that folder {folder_id} is shared publicly and retry."
        )

    resolved_dir = target_dir
    if not resolved_dir.exists():
        after_children = {child.name for child in output_root.iterdir()}
        new_dirs = [
            output_root / name
            for name in sorted(after_children - before_children)
            if (output_root / name).is_dir()
        ]
        if len(new_dirs) == 1:
            new_dirs[0].rename(target_dir)
            resolved_dir = target_dir
        else:
            raise RuntimeError(
                "Downloaded persona folder, but could not determine its local directory. "
                f"Expected {target_dir}."
            )

    source_note = resolved_dir / ".download_source.txt"
    source_note.write_text(url + "\n", encoding="utf-8")
    return resolved_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the public persona Google Drive folder into data/persona_age_gender."
    )
    parser.add_argument("--url", default=DEFAULT_DRIVE_URL, help="Public Google Drive folder URL.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Parent directory where the persona folder will be created.",
    )
    parser.add_argument(
        "--folder-name",
        default=DEFAULT_FOLDER_NAME,
        help="Local folder name to create under --output-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing target folder and re-download it from scratch.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    target = download_persona_folder(
        url=args.url,
        output_dir=args.output_dir,
        folder_name=args.folder_name,
        force=args.force,
    )
    print(f"Downloaded persona folder to {target}")


if __name__ == "__main__":
    main()
