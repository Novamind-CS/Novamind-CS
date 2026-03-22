"""
Simple version bump helper for Novamind-CS release assets.

Updates:
- README.md
- train.py
- core/met_controller.py

Replaces:
- v1.0-RC
- v1.0

With:
- v1.1

It also refreshes the README header so the landing section reflects the
"Sniper and Heavyweight" branding used for the v1.1 paper launch.

Usage:
    python3 bump_version.py
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGETS = [
    ROOT / "README.md",
    ROOT / "train.py",
    ROOT / "core" / "met_controller.py",
]


def replace_version_markers(text: str) -> str:
    text = text.replace("v1.0-RC", "v1.1")
    text = text.replace("v1.0", "v1.1")
    return text


def update_readme_branding(text: str) -> str:
    marker = "# NovaMind"
    branded = "# NovaMind\n\n**v1.1: The Sniper and The Heavyweight.**"
    if marker in text and branded not in text:
        text = text.replace(marker, branded, 1)
    return text


def main():
    updated = []
    for path in TARGETS:
        original = path.read_text(encoding="utf-8")
        text = replace_version_markers(original)
        if path.name == "README.md":
            text = update_readme_branding(text)
        if text != original:
            path.write_text(text, encoding="utf-8")
            updated.append(path)

    if updated:
        print("[VersionBump] Updated:")
        for path in updated:
            print(f"  - {path}")
    else:
        print("[VersionBump] No changes were necessary.")


if __name__ == "__main__":
    main()
