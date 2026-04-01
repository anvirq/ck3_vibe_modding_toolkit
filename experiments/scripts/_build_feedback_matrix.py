import json
from pathlib import Path

base = Path("experiments/results")
cfgs = [
    "feedback_profile_a",
    "feedback_profile_a_ast256",
    "feedback_profile_a_ast512",
    "feedback_profile_a_ast768",
    "feedback_profile_b",
    "feedback_profile_b_ast256",
    "feedback_profile_b_ast512",
    "feedback_profile_b_ast768",
]
modes = [
    ("none", "No rerank"),
    ("heuristic", "Heuristic rerank"),
    ("ce_l12_k20", "Cross-encoder L12 k20"),
]

lines = [
    "# Feedback Suite Matrix (A/B)",
    "",
    "| Config | Mode | Top1 | Top3 | Noise@5 | Actionability |",
    "|---|---:|---:|---:|---:|---:|",
]

for cfg in cfgs:
    for suffix, label in modes:
        path = base / cfg / f"feedback_eval_{suffix}.json"
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        noise = "-" if summary["noise_top5_mean"] is None else f"{summary['noise_top5_mean']:.0%}"
        lines.append(
            f"| {cfg} | {label} | {summary['top1']:.0%} | {summary['top3']:.0%} | {noise} | {summary['actionability']:.0%} |"
        )

out = base / "feedback_suite_matrix_ab.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(out)
