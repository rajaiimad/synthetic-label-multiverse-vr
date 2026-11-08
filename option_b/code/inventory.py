"""Inventory of the original research (read-only) with SHA-256 hashes, so that later runs can prove
nothing original was modified.  usage:
  python inventory.py snapshot   -> option_b/reports/inventory_hashes.json + INVENTORY_FILES.md
  python inventory.py verify     -> compares current files with the snapshot (exit 1 on any change)
"""
import hashlib
import json
import sys
from pathlib import Path

J7 = Path(__file__).resolve().parents[2]
OB = J7 / "option_b"
ROOTS = {
    "code_and_plans": (J7 / "rigorous_rerun", ["*.py", "*.md", "*.sh"], False),
    "results_json": (J7 / "rigorous_rerun" / "outputs_v2", ["*.json", "*.md"], True),
    "predictions": (J7 / "rigorous_rerun" / "outputs_v2", ["*.npy", "*.npz"], True),
    "manuscripts": (J7 / "manuscript", ["main_revised*.tex", "*.py"], False),
    "submission_packages": (J7, ["final_submission_v1*/*"], False),
    "generator_code": (J7 / "coper_vr", ["*.py"], True),
}


def sha(p, big=False):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def collect():
    out = {}
    for group, (root, pats, rec) in ROOTS.items():
        files = set()
        for pat in pats:
            files.update(root.rglob(pat) if rec else root.glob(pat))
        for f in sorted(x for x in files if x.is_file()):
            if "cache" in f.parts and group != "predictions":
                continue
            out[str(f.relative_to(J7))] = {"group": group, "bytes": f.stat().st_size, "sha256": sha(f)}
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    snap = OB / "reports" / "inventory_hashes.json"
    cur = collect()
    if mode == "snapshot":
        snap.write_text(json.dumps(cur, indent=0))
        by = {}
        for k, v in cur.items():
            by.setdefault(v["group"], []).append((k, v["bytes"]))
        lines = ["# Inventory of original research files (hashed snapshot)", "",
                 f"Total files: {len(cur)}. Hashes: `option_b/reports/inventory_hashes.json`.", ""]
        for g, L in by.items():
            lines += [f"## {g} ({len(L)} files, {sum(b for _, b in L) / 1e6:.1f} MB)", ""]
            if len(L) > 60:
                lines += [f"* {k}" for k, _ in L[:15]] + [f"* ... ({len(L) - 15} more; see hash file)"]
            else:
                lines += [f"* {k}" for k, _ in L]
            lines.append("")
        (OB / "reports" / "INVENTORY_FILES.md").write_text("\n".join(lines), encoding="utf-8")
        print(len(cur), "files hashed")
    else:
        old = json.loads(snap.read_text())
        changed = [k for k in old if k not in cur or cur[k]["sha256"] != old[k]["sha256"]]
        print("original files changed or missing:", changed or "none", f"({len(old)} checked)")
        sys.exit(1 if changed else 0)
