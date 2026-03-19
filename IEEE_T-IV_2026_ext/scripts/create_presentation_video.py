#!/usr/bin/env python3
"""
create_presentation_video.py — comprehensive RECTOR evidence presentation

This script builds a long-form presentation video for the RECTOR paper using
artifacts that actually exist in the current workspace. The output is an
evidence-led presentation rather than a short teaser: it mixes quantitative
slides, protocol explanations, ablations, test-set evidence, and a broad set
of closed-loop and replayed scenario clips.

Design goals:
  - Use live artifacts from /workspace/output/evaluation rather than stale paths.
  - Show a broader range of scenarios automatically instead of a few hard-coded clips.
  - Keep the narrative convincing by combining headline results, audit protocol,
    oracle-evaluation evidence, and many diverse qualitative examples.
  - Degrade gracefully when optional files are missing.

Output:
  /workspace/IEEE_T-IV_2026_extpresentation/rector_presentation.mp4

Requirements:
  ffmpeg, matplotlib, numpy, opencv-python, pillow
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/workspace")
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPT_DIR.parent
FIGURES = PAPER_ROOT / "Figures"
EVAL_DIR = ROOT / "output/evaluation"
BEV_DIR = ROOT / "output/closedloop/bev_frames"
M2I_DIR = ROOT / "models/RECTOR/movies/m2i_rector"
OUT_DIR = PAPER_ROOT / "presentation"
FINAL = OUT_DIR / "rector_presentation.mp4"
TMP = OUT_DIR / "_tmp"

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
RESAMPLING = getattr(Image, "Resampling", Image)


# ── Video specs ────────────────────────────────────────────────────────────────
W, H = 1920, 1080
BEV_W = 1280
RULE_W = W - BEV_W
FPS = 30

STILL_SECONDS = 8.0
SECTION_SECONDS = 5.0
BEV_CLIP_SECONDS = 10.0
M2I_CLIP_SECONDS = 10.0
SCENARIO_CONTEXT_SECONDS = 4.5
SCENARIO_DISCUSSION_SECONDS = 5.5
MAX_M2I_SCENARIOS = 14


# ── Look and feel ──────────────────────────────────────────────────────────────
DARK = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
WHITE = "#f0f6fc"
GRAY = "#8b949e"
RED = "#f85149"
GREEN = "#3fb950"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#ff9800"
BLUE = "#1565c0"

STRAT_COLORS = {
    "confidence": "#e53935",
    "weighted_sum": "#f57c00",
    "lexicographic": "#1565c0",
}
STRAT_LABELS = {
    "confidence": "Confidence-only",
    "weighted_sum": "Weighted-sum",
    "lexicographic": "RECTOR (Lex.)",
}

matplotlib.rcParams.update(
    {
        "figure.facecolor": DARK,
        "axes.facecolor": PANEL,
        "text.color": WHITE,
        "axes.labelcolor": WHITE,
        "xtick.color": WHITE,
        "ytick.color": WHITE,
        "axes.edgecolor": BORDER,
        "grid.color": BORDER,
        "legend.facecolor": PANEL,
        "legend.edgecolor": BORDER,
        "axes.titlecolor": WHITE,
        "font.size": 12,
    }
)


# ══════════════════════════════════════════════════════════════════════════════
# Generic helpers
# ══════════════════════════════════════════════════════════════════════════════


def safe_load_json(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    with path.open() as handle:
        return json.load(handle)


def coerce_value(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_csv_indexed(path: Path, index_field: str | None = None) -> dict[int, dict]:
    if not path.exists():
        return {}
    indexed = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            parsed = {key: coerce_value(val) for key, val in row.items()}
            idx = parsed.get(index_field, row_idx) if index_field else row_idx
            indexed[int(idx)] = parsed
    return indexed


def fmtn(value, digits: int = 2, default: str = "N/A") -> str:
    if value is None:
        return default
    return f"{float(value):.{digits}f}"


def fmt_bool(flag: bool) -> tuple[str, str]:
    return ("VIOL", RED) if flag else ("OK", GREEN)


def figure_path(name: str) -> Path:
    return FIGURES / name


def draw_image(ax, path: Path):
    ax.set_facecolor(DARK)
    ax.axis("off")
    if path.exists():
        ax.imshow(plt.imread(str(path)))
    else:
        ax.text(
            0.5,
            0.5,
            f"Missing figure\n{path.name}",
            ha="center",
            va="center",
            fontsize=18,
            color=GRAY,
            transform=ax.transAxes,
        )
        ax.add_patch(
            mpatches.Rectangle(
                (0.05, 0.05),
                0.90,
                0.90,
                fill=False,
                edgecolor=BORDER,
                lw=2,
                transform=ax.transAxes,
            )
        )


def add_slide_header(fig, title: str, subtitle: str = "", accent: str = ACCENT):
    ax = fig.add_axes([0, 0.89, 1, 0.11], facecolor=DARK)
    ax.axis("off")
    ax.axhline(0.10, color=accent, lw=3, xmin=0.04, xmax=0.96)
    ax.text(
        0.5,
        0.62,
        title,
        ha="center",
        va="center",
        fontsize=34,
        color=WHITE,
        fontweight="bold",
        transform=ax.transAxes,
    )
    if subtitle:
        ax.text(
            0.5,
            0.20,
            subtitle,
            ha="center",
            va="center",
            fontsize=17,
            color=GRAY,
            style="italic",
            transform=ax.transAxes,
        )
    return ax


def add_footer_box(fig, headline: str, subline: str = "", edge: str = ACCENT):
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.12], facecolor=DARK)
    ax.axis("off")
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.02",
            facecolor="#1f2d3d",
            edgecolor=edge,
            lw=2.0,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        0.66,
        headline,
        ha="center",
        va="center",
        fontsize=15,
        color=WHITE,
        fontweight="bold",
        transform=ax.transAxes,
    )
    if subline:
        ax.text(
            0.5,
            0.25,
            subline,
            ha="center",
            va="center",
            fontsize=12,
            color=GRAY,
            style="italic",
            transform=ax.transAxes,
        )
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# Data loading and scenario selection
# ══════════════════════════════════════════════════════════════════════════════


def load_eval():
    canon = safe_load_json(EVAL_DIR / "canonical_results.json", {})
    significance = safe_load_json(EVAL_DIR / "significance_tests.json", {})
    proxy = safe_load_json(EVAL_DIR / "proxy_correlations_all24.json", {})
    val_test = safe_load_json(EVAL_DIR / "val_test_distribution.json", {})
    divergence = safe_load_json(EVAL_DIR / "natural_divergence_test.json", {})
    misspec = safe_load_json(EVAL_DIR / "weight_grid_misspecification.json", {})
    adversarial = safe_load_json(EVAL_DIR / "adversarial_injection_results.json", {})

    return {
        "canonical": canon,
        "significance": significance,
        "proxy": proxy,
        "val_test": val_test,
        "divergence": divergence,
        "misspec": misspec,
        "adversarial": adversarial,
        "summary": load_csv_indexed(
            EVAL_DIR / "per_scenario_metrics.csv", "scenario_idx"
        ),
        "confidence": load_csv_indexed(EVAL_DIR / "per_scenario_confidence.csv"),
        "weighted_sum": load_csv_indexed(EVAL_DIR / "per_scenario_weighted_sum.csv"),
        "lexicographic": load_csv_indexed(EVAL_DIR / "per_scenario_lexicographic.csv"),
        "protB_lexicographic": load_csv_indexed(
            EVAL_DIR / "per_scenario_protB_lexicographic.csv"
        ),
    }


def bundle_for_idx(idx: int, data: dict) -> dict:
    return {
        "idx": idx,
        "summary": data["summary"].get(idx, {}),
        "confidence": data["confidence"].get(idx, {}),
        "weighted_sum": data["weighted_sum"].get(idx, {}),
        "lexicographic": data["lexicographic"].get(idx, {}),
        "protB_lexicographic": data["protB_lexicographic"].get(idx, {}),
    }


def available_bev_items(data: dict) -> list[dict]:
    items = []
    for path in sorted(BEV_DIR.glob("scenario_*")):
        try:
            idx = int(path.name.split("_")[-1])
        except ValueError:
            continue
        if idx == 45:
            continue
        item = bundle_for_idx(idx, data)
        item["path"] = path
        item["kind"] = "bev"
        items.append(item)
    return items


def available_m2i_items(data: dict) -> list[dict]:
    items = []
    for path in sorted(M2I_DIR.glob("m2i_rector_*.mp4")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[2])
        except ValueError:
            continue
        item = bundle_for_idx(idx, data)
        item["path"] = path
        item["kind"] = "m2i"
        item["bytes"] = path.stat().st_size
        items.append(item)
    return items


def scenario_tags(bundle: dict) -> list[str]:
    tags = []
    summary = bundle.get("summary", {})
    lex = bundle.get("lexicographic", {})
    protb = bundle.get("protB_lexicographic", {})

    tags.append("MISS" if summary.get("miss") else "HIT")
    if protb.get("tier_0_violated") or lex.get("tier_0_violated"):
        tags.append("Safety")
    if protb.get("tier_1_violated") or lex.get("tier_1_violated"):
        tags.append("Legal")
    if protb.get("tier_2_violated") or lex.get("tier_2_violated"):
        tags.append("Road")
    if protb.get("tier_3_violated") or lex.get("tier_3_violated"):
        tags.append("Comfort")
    if lex.get("infeasible_selected"):
        tags.append("Infeasible")
    if not protb.get("total_violated", True):
        tags.append("Oracle-clean")
    if summary.get("minADE", 10.0) < 0.5:
        tags.append("High-precision")
    if summary.get("minADE", 0.0) > 3.0:
        tags.append("Hard-case")
    return tags


def scenario_score(bundle: dict) -> float:
    summary = bundle.get("summary", {})
    lex = bundle.get("lexicographic", {})
    protb = bundle.get("protB_lexicographic", {})

    score = 0.0
    score += min(float(summary.get("minADE", 0.0) or 0.0), 8.0) * 0.8
    score += min(float(summary.get("minFDE", 0.0) or 0.0), 20.0) * 0.35
    score += 5.0 if summary.get("miss") else 0.0
    score += 2.0 if lex.get("infeasible_selected") else 0.0

    weights = {
        "tier_0_violated": 5.0,
        "tier_1_violated": 3.5,
        "tier_2_violated": 2.5,
        "tier_3_violated": 1.5,
    }
    for key, weight in weights.items():
        if lex.get(key):
            score += weight
        if protb.get(key):
            score += weight * 1.25

    if not protb.get("total_violated", True):
        score += 1.0
    return score


def select_diverse_m2i_scenarios(
    data: dict, limit: int = MAX_M2I_SCENARIOS
) -> list[dict]:
    candidates = available_m2i_items(data)
    for item in candidates:
        item["tags"] = scenario_tags(item)
        item["score"] = scenario_score(item)

    selected = []
    used = set()
    covered = set()

    category_order = [
        "Safety",
        "Legal",
        "Road",
        "Comfort",
        "MISS",
        "Oracle-clean",
        "Infeasible",
        "High-precision",
        "Hard-case",
    ]

    for category in category_order:
        matches = [
            item
            for item in candidates
            if category in item["tags"] and item["idx"] not in used
        ]
        matches.sort(
            key=lambda item: (item["score"], item.get("bytes", 0)), reverse=True
        )
        if matches:
            pick = matches[0]
            selected.append(pick)
            used.add(pick["idx"])
            covered.update(pick["tags"])
            if len(selected) >= limit:
                return selected

    remaining = [item for item in candidates if item["idx"] not in used]
    while remaining and len(selected) < limit:
        remaining.sort(
            key=lambda item: (
                item["score"] + 0.85 * len(set(item["tags"]) - covered),
                item.get("bytes", 0),
            ),
            reverse=True,
        )
        pick = remaining.pop(0)
        selected.append(pick)
        used.add(pick["idx"])
        covered.update(pick["tags"])
        remaining = [item for item in remaining if item["idx"] not in used]

    return selected


def classify_scenario(bundle: dict) -> tuple[str, str]:
    tags = scenario_tags(bundle)
    idx = bundle["idx"]
    source = (
        "Waymax closed-loop replay"
        if bundle["kind"] == "bev"
        else "M2I + RECTOR replay"
    )

    if "Safety" in tags and "MISS" in tags:
        headline = f"Scenario {idx:03d} — safety-critical miss case"
    elif "Safety" in tags and "Legal" in tags:
        headline = f"Scenario {idx:03d} — safety and legal conflict"
    elif "Road" in tags and "Comfort" in tags:
        headline = f"Scenario {idx:03d} — road geometry and comfort stress"
    elif "Oracle-clean" in tags and "High-precision" in tags:
        headline = f"Scenario {idx:03d} — clean compliant execution"
    elif "MISS" in tags:
        headline = f"Scenario {idx:03d} — hard multi-agent miss"
    elif "Legal" in tags:
        headline = f"Scenario {idx:03d} — rule-sensitive interaction"
    else:
        headline = f"Scenario {idx:03d} — diverse interaction replay"

    subtitle = f"{source} · {' · '.join(tags[:5])}"
    return headline, subtitle


def scenario_takeaways(bundle: dict) -> list[str]:
    conf = bundle.get("confidence", {})
    ws = bundle.get("weighted_sum", {})
    lex = bundle.get("lexicographic", {})
    protb = bundle.get("protB_lexicographic", {})
    summary = bundle.get("summary", {})

    takeaways = []
    if conf.get("tier_0_violated") and not lex.get("tier_0_violated"):
        takeaways.append(
            "RECTOR removes a Safety-tier violation present under confidence-only selection."
        )
    if conf.get("tier_1_violated") and not lex.get("tier_1_violated"):
        takeaways.append(
            "Legal-tier failure under confidence-only is removed by lexicographic selection."
        )
    if conf.get("total_violated") and not lex.get("total_violated"):
        takeaways.append(
            "RECTOR converts this case from violating to compliant under the training-time protocol."
        )
    if not protb.get("total_violated", True):
        takeaways.append(
            "The selected trajectory also passes the full oracle-audit configuration."
        )
    elif protb.get("tier_0_violated"):
        takeaways.append(
            "Oracle auditing still flags a residual Safety issue, showing the current frontier rather than a cherry-picked success."
        )
    elif protb.get("tier_3_violated"):
        takeaways.append(
            "Oracle auditing preserves the main safety/legal outcome but still exposes lower-tier comfort stress."
        )
    if ws.get("total_violated") == lex.get("total_violated") and ws.get(
        "tier_0_violated"
    ) == lex.get("tier_0_violated"):
        takeaways.append(
            "Weighted-sum matches the final violation status here, but RECTOR reaches the same outcome through explicit priority ordering."
        )
    if summary.get("miss"):
        takeaways.append(
            "Prediction miss remains present, which is exactly why the video includes hard failures as well as wins."
        )
    if lex.get("infeasible_selected"):
        takeaways.append(
            "RECTOR raises an infeasibility condition rather than hiding the absence of a clean candidate."
        )

    if not takeaways:
        takeaways.append(
            "This replay illustrates stable rule-aware behaviour on a non-trivial interaction without relying on manual curation."
        )
    return takeaways[:3]


def scenario_context_lines(bundle: dict) -> list[str]:
    summary = bundle.get("summary", {})
    conf = bundle.get("confidence", {})
    lex = bundle.get("lexicographic", {})
    protb = bundle.get("protB_lexicographic", {})
    tags = scenario_tags(bundle)

    lines = []
    lines.append(
        f"Source: {'Waymax closed-loop replay' if bundle['kind'] == 'bev' else 'M2I + RECTOR replay'} · tags: {' · '.join(tags[:5])}."
    )
    lines.append(
        f"Geometric difficulty: minADE {fmtn(summary.get('minADE'), 2)} m, minFDE {fmtn(summary.get('minFDE'), 2)} m, outcome {'MISS' if summary.get('miss') else 'HIT'}."
    )

    conflicts = []
    if conf.get("tier_0_violated"):
        conflicts.append("confidence hits Safety")
    if conf.get("tier_1_violated"):
        conflicts.append("confidence hits Legal")
    if conf.get("tier_2_violated"):
        conflicts.append("confidence hits Road")
    if conf.get("tier_3_violated"):
        conflicts.append("confidence hits Comfort")
    if not conflicts:
        conflicts.append("confidence has no tier violation in Protocol A")

    lines.append(f"What to watch for: {'; '.join(conflicts)}.")
    lines.append(
        f"RECTOR enters this scene with selADE {fmtn(lex.get('selADE'), 2)} m and oracle-audit status {'clean' if not protb.get('total_violated', True) else 'still stressed'} after selection."
    )
    return lines


def scenario_discussion_lines(bundle: dict) -> list[str]:
    conf = bundle.get("confidence", {})
    ws = bundle.get("weighted_sum", {})
    lex = bundle.get("lexicographic", {})
    protb = bundle.get("protB_lexicographic", {})

    lines = []
    lines.extend(scenario_takeaways(bundle))
    lines.append(
        f"Strategy comparison: confidence selADE {fmtn(conf.get('selADE'), 2)} m, weighted-sum {fmtn(ws.get('selADE'), 2)} m, RECTOR {fmtn(lex.get('selADE'), 2)} m."
    )

    audit_summary = []
    for label, key in [
        ("Safety", "tier_0_violated"),
        ("Legal", "tier_1_violated"),
        ("Road", "tier_2_violated"),
        ("Comfort", "tier_3_violated"),
    ]:
        audit_summary.append(f"{label}={'VIOL' if protb.get(key) else 'OK'}")
    lines.append("Oracle audit after the replay: " + ", ".join(audit_summary) + ".")
    return lines[:5]


# ══════════════════════════════════════════════════════════════════════════════
# ffmpeg and compositing helpers
# ══════════════════════════════════════════════════════════════════════════════


def run(cmd, label=""):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"[ffmpeg error: {label}]")
        print(result.stderr.decode(errors="ignore")[-500:])
        raise RuntimeError(f"ffmpeg failed: {label}")


def static_loop_mp4(png: Path, duration: float, out: Path, w=W, h=H) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "5",
            "-t",
            str(duration),
            "-i",
            str(png),
            "-vf",
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={DARK[1:]},setsar=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            str(out),
        ],
        label=f"static {out.name}",
    )
    if not out.exists():
        raise FileNotFoundError(f"Expected segment was not created: {out}")
    return out


def _pil_font(size: int, bold=True):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
    except Exception:
        return ImageFont.load_default()


def _letterbox(
    img_bgr: np.ndarray, target_w: int, target_h: int, bg=(13, 17, 23)
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((target_h, target_w, 3), bg, dtype=np.uint8)
    yo = (target_h - nh) // 2
    xo = (target_w - nw) // 2
    canvas[yo : yo + nh, xo : xo + nw] = resized
    return canvas


def _crop_fill(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    xo = (nw - target_w) // 2
    yo = (nh - target_h) // 2
    return resized[yo : yo + target_h, xo : xo + target_w]


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    font,
    fg=(255, 255, 255),
    bg=(13, 17, 23),
    pad=14,
):
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad // 2, bbox[2] + pad, bbox[3] + pad // 2], fill=bg
    )
    draw.text((x, y), text, font=font, fill=fg)


def composite_scenario_mp4(
    bev_dir: Path,
    panel_png: Path,
    out: Path,
    title1: str,
    title2: str,
    fps: int = 10,
    max_frames: int | None = None,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)

    panel_img = Image.open(str(panel_png)).convert("RGB").resize(
        (RULE_W, H), RESAMPLING.LANCZOS
    )
    panel_bgr = np.array(panel_img)[:, :, ::-1].copy()
    f_title = _pil_font(40, bold=True)
    f_sub = _pil_font(24, bold=False)

    frames = sorted(bev_dir.glob("frame_*.png"))
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"No frames found in {bev_dir}")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        f"fps={FPS}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin pipe")
    try:
        for frame_path in frames:
            bev = cv2.imread(str(frame_path))
            if bev is None:
                continue
            pane = _letterbox(bev, BEV_W, H)
            pil_img = Image.fromarray(pane[:, :, ::-1])
            draw = ImageDraw.Draw(pil_img)
            title_w = draw.textbbox((0, 0), title1, font=f_title)[2]
            _draw_text_box(draw, title1, max((BEV_W - title_w) // 2, 40), 18, f_title)
            if title2:
                sub_w = draw.textbbox((0, 0), title2, font=f_sub)[2]
                _draw_text_box(
                    draw,
                    title2,
                    max((BEV_W - sub_w) // 2, 40),
                    74,
                    f_sub,
                    fg=(180, 190, 200),
                )
            pane = np.array(pil_img)[:, :, ::-1]
            full = np.concatenate([pane, panel_bgr], axis=1)
            proc.stdin.write(full.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg pipe failed for {out.name}")
    if not out.exists():
        raise FileNotFoundError(f"Expected segment was not created: {out}")
    return out


def composite_video_mp4(
    src_mp4: Path,
    panel_png: Path,
    out: Path,
    title1: str,
    title2: str,
    fps_out: int = 10,
    max_seconds: float | None = None,
    skip_seconds: float = 0.0,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)

    panel_img = Image.open(str(panel_png)).convert("RGB").resize(
        (RULE_W, H), RESAMPLING.LANCZOS
    )
    panel_bgr = np.array(panel_img)[:, :, ::-1].copy()
    f_title = _pil_font(40, bold=True)
    f_sub = _pil_font(24, bold=False)

    cap = cv2.VideoCapture(str(src_mp4))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps_out
    skip_frames = int(skip_seconds * src_fps)
    max_frames = int(max_seconds * src_fps) if max_seconds else int(1e9)

    for _ in range(skip_frames):
        cap.read()

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps_out),
        "-i",
        "pipe:0",
        "-vf",
        f"fps={FPS}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin pipe")
    try:
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            pane = _crop_fill(frame, BEV_W, H)
            pil_img = Image.fromarray(pane[:, :, ::-1])
            draw = ImageDraw.Draw(pil_img)
            title_w = draw.textbbox((0, 0), title1, font=f_title)[2]
            _draw_text_box(draw, title1, max((BEV_W - title_w) // 2, 40), 18, f_title)
            if title2:
                sub_w = draw.textbbox((0, 0), title2, font=f_sub)[2]
                _draw_text_box(
                    draw,
                    title2,
                    max((BEV_W - sub_w) // 2, 40),
                    74,
                    f_sub,
                    fg=(180, 190, 200),
                )
            pane = np.array(pil_img)[:, :, ::-1]
            full = np.concatenate([pane, panel_bgr], axis=1)
            proc.stdin.write(full.tobytes())
            frame_count += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg pipe failed for {out.name}")
    if not out.exists():
        raise FileNotFoundError(f"Expected segment was not created: {out}")
    return out


def concat_segments(segments: list[Path], out: Path) -> Path:
    missing = [str(segment) for segment in segments if not segment.exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} segment files before concat:\n{preview}"
        )
    manifest = out.parent / "_concat.txt"
    with manifest.open("w") as handle:
        for segment in segments:
            handle.write(f"file '{segment.resolve()}'\n")
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c",
            "copy",
            str(out),
        ],
        label="concat",
    )
    manifest.unlink(missing_ok=True)
    return out


def fig_to_png(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100, facecolor=DARK, edgecolor="none")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Scenario dashboard panel
# ══════════════════════════════════════════════════════════════════════════════


def make_strategy_panel(title: str, subtitle: str, bundle: dict) -> plt.Figure:
    fig = plt.figure(figsize=(6.4, 10.8), facecolor=DARK)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    conf = bundle.get("confidence", {})
    ws = bundle.get("weighted_sum", {})
    lex = bundle.get("lexicographic", {})
    protb = bundle.get("protB_lexicographic", {})
    summary = bundle.get("summary", {})

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.03, 0.93),
            0.94,
            0.06,
            boxstyle="round,pad=0.01",
            facecolor="#1f6feb30",
            edgecolor=ACCENT,
            lw=1.4,
            transform=t,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        0.968,
        "Per-Scenario Selection Audit",
        ha="center",
        va="center",
        fontsize=14,
        color=ACCENT,
        fontweight="bold",
        transform=t,
    )
    ax.text(
        0.5,
        0.942,
        title,
        ha="center",
        va="center",
        fontsize=11,
        color=WHITE,
        fontweight="bold",
        transform=t,
    )
    ax.text(
        0.5,
        0.914,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.8,
        color=GRAY,
        style="italic",
        transform=t,
    )

    cards = [
        ("minADE", f"{fmtn(summary.get('minADE'), 2)} m", BLUE),
        ("minFDE", f"{fmtn(summary.get('minFDE'), 2)} m", PURPLE),
        (
            "Outcome",
            "MISS" if summary.get("miss") else "HIT",
            RED if summary.get("miss") else GREEN,
        ),
        (
            "Infeasible",
            "YES" if lex.get("infeasible_selected") else "NO",
            YELLOW if lex.get("infeasible_selected") else GREEN,
        ),
    ]
    x_positions = [0.05, 0.29, 0.53, 0.77]
    for x, (label, value, color) in zip(x_positions, cards):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.82),
                0.18,
                0.075,
                boxstyle="round,pad=0.006",
                facecolor=PANEL,
                edgecolor=color,
                lw=1.3,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            x + 0.09,
            0.872,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=GRAY,
            transform=t,
        )
        ax.text(
            x + 0.09,
            0.842,
            value,
            ha="center",
            va="center",
            fontsize=12,
            color=color,
            fontweight="bold",
            transform=t,
        )

    ax.text(
        0.5,
        0.785,
        "Selection outcomes by strategy",
        ha="center",
        va="center",
        fontsize=11.5,
        color=WHITE,
        fontweight="bold",
        transform=t,
    )

    headers = [
        ("Confidence", conf, STRAT_COLORS["confidence"]),
        ("Weighted", ws, STRAT_COLORS["weighted_sum"]),
        ("RECTOR", lex, STRAT_COLORS["lexicographic"]),
    ]
    col_x = [0.42, 0.66, 0.89]
    for (name, _, color), x in zip(headers, col_x):
        ax.text(
            x,
            0.753,
            name,
            ha="center",
            va="center",
            fontsize=9.2,
            color=color,
            fontweight="bold",
            transform=t,
        )

    rows = [
        ("selADE", "selADE"),
        ("selFDE", "selFDE"),
        ("Total", "total_violated"),
        ("Safety", "tier_0_violated"),
        ("Legal", "tier_1_violated"),
        ("Road", "tier_2_violated"),
        ("Comfort", "tier_3_violated"),
        ("Infeasible", "infeasible_selected"),
    ]
    y = 0.715
    for label, key in rows:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.05, y - 0.022),
                0.90,
                0.045,
                boxstyle="round,pad=0.003",
                facecolor="#0f141b",
                edgecolor=BORDER,
                lw=0.8,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            0.09,
            y,
            label,
            ha="left",
            va="center",
            fontsize=9.3,
            color=WHITE,
            transform=t,
        )
        for x, (_, values, color) in zip(col_x, headers):
            value = values.get(key)
            if isinstance(value, bool):
                text, fg = fmt_bool(value)
            elif value is None:
                text, fg = "N/A", GRAY
            else:
                text, fg = f"{float(value):.2f}", WHITE
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=9.2,
                color=fg,
                fontweight="bold" if fg != WHITE else None,
                transform=t,
            )
        y -= 0.053

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.05, 0.245),
            0.90,
            0.16,
            boxstyle="round,pad=0.006",
            facecolor=PANEL,
            edgecolor=ACCENT,
            lw=1.3,
            transform=t,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        0.385,
        "RECTOR under full oracle audit (Protocol B)",
        ha="center",
        va="center",
        fontsize=10.5,
        color=ACCENT,
        fontweight="bold",
        transform=t,
    )
    audit_labels = [
        ("Total", protb.get("total_violated", False)),
        ("Safety", protb.get("tier_0_violated", False)),
        ("Legal", protb.get("tier_1_violated", False)),
        ("Road", protb.get("tier_2_violated", False)),
        ("Comfort", protb.get("tier_3_violated", False)),
    ]
    audit_x = [0.12, 0.30, 0.48, 0.66, 0.84]
    for (label, flag), x in zip(audit_labels, audit_x):
        text, fg = fmt_bool(flag)
        ax.text(
            x,
            0.345,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=GRAY,
            transform=t,
        )
        ax.text(
            x,
            0.312,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            color=fg,
            fontweight="bold",
            transform=t,
        )

    ax.text(
        0.5,
        0.212,
        "Why this clip matters",
        ha="center",
        va="center",
        fontsize=11.5,
        color=WHITE,
        fontweight="bold",
        transform=t,
    )
    for idx, takeaway in enumerate(scenario_takeaways(bundle)):
        ax.text(
            0.07,
            0.178 - idx * 0.048,
            f"• {takeaway}",
            ha="left",
            va="center",
            fontsize=8.8,
            color=GRAY if idx > 0 else WHITE,
            transform=t,
            wrap=True,
        )

    ax.axhline(0.035, color=BORDER, lw=1, xmin=0.03, xmax=0.97)
    ax.text(
        0.5,
        0.018,
        "Training-time protocol, oracle audit, and geometric error shown together for transparency.",
        ha="center",
        va="center",
        fontsize=7.6,
        color=GRAY,
        style="italic",
        transform=t,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Slide builders
# ══════════════════════════════════════════════════════════════════════════════


def make_title_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    lex = canon["selection_strategies"]["lexicographic"]
    conf = canon["selection_strategies"]["confidence"]
    n = canon["metadata"]["sample_count"]

    ax.axhline(0.93, color=ACCENT, lw=5, xmin=0.04, xmax=0.96)
    ax.text(
        0.5,
        0.80,
        "RECTOR",
        ha="center",
        va="center",
        fontsize=118,
        color=ACCENT,
        fontweight="bold",
        transform=t,
        fontfamily="monospace",
    )
    ax.text(
        0.5,
        0.66,
        "Rule-Enforced Constrained Trajectory Orchestrator\n"
        "for Autonomous Driving Trajectory Selection",
        ha="center",
        va="center",
        fontsize=32,
        color=WHITE,
        linespacing=1.45,
        transform=t,
    )
    ax.text(
        0.5,
        0.53,
        "A long-form evidence video: mechanism, audit protocol, ablations, and diverse scenario replays",
        ha="center",
        va="center",
        fontsize=20,
        color=GRAY,
        style="italic",
        transform=t,
    )

    stats = [
        ("Validation scenarios", f"{n:,}", ACCENT),
        (
            "Total violations",
            f'{conf["Total_Viol_pct"]:.2f}% → {lex["Total_Viol_pct"]:.2f}%',
            GREEN,
        ),
        (
            "Safety violations",
            f'{conf["Safety_Viol_pct"]:.2f}% → {lex["Safety_Viol_pct"]:.2f}%',
            RED,
        ),
        ("selADE", f'{lex["selADE_mean"]:.3f} m', BLUE),
    ]
    for i, (label, value, color) in enumerate(stats):
        x = 0.14 + i * 0.18
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.28),
                0.15,
                0.105,
                boxstyle="round,pad=0.01",
                facecolor=PANEL,
                edgecolor=color,
                lw=2,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            x + 0.075,
            0.348,
            label,
            ha="center",
            va="center",
            fontsize=14,
            color=GRAY,
            transform=t,
        )
        ax.text(
            x + 0.075,
            0.308,
            value,
            ha="center",
            va="center",
            fontsize=18,
            color=color,
            fontweight="bold",
            transform=t,
        )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.06, 0.08),
            0.88,
            0.10,
            boxstyle="round,pad=0.012",
            facecolor="#1f2d3d",
            edgecolor=ACCENT,
            lw=1.8,
            transform=t,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        0.13,
        "This video argues the whole case: why rule-aware selection matters, what RECTOR changes, and where the limits still are.",
        ha="center",
        va="center",
        fontsize=18,
        color=WHITE,
        fontweight="bold",
        transform=t,
    )
    return fig


def make_contribution_slide(canon: dict, divergence: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "What This Video Demonstrates",
        "Contribution, evidence standard, and what is new beyond geometric ranking",
    )
    ax = fig.add_axes([0.05, 0.18, 0.90, 0.66], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    lex = canon["selection_strategies"]["lexicographic"]
    conf = canon["selection_strategies"]["confidence"]
    protb = canon["selection_strategies_protB"]["lexicographic"]
    relative_total = (
        100.0
        * (conf["Total_Viol_pct"] - lex["Total_Viol_pct"])
        / conf["Total_Viol_pct"]
    )
    relative_safety = (
        100.0
        * (conf["Safety_Viol_pct"] - lex["Safety_Viol_pct"])
        / conf["Safety_Viol_pct"]
    )

    bullets = [
        (
            "1. A rule-governed selector, not just another predictor",
            "RECTOR adds an explicit four-tier rule hierarchy over candidate trajectories and makes the selection decision auditable.",
        ),
        (
            "2. Strong gains on the full validation benchmark",
            f'Across {canon["metadata"]["sample_count"]:,} scenarios, RECTOR cuts total violations by {relative_total:.1f}% relative and Safety violations by {relative_safety:.1f}% relative versus confidence-only selection.',
        ),
        (
            "3. Oracle auditing is part of the story, not hidden in the appendix",
            f'The same selected trajectories are re-scored under the full 28-rule oracle configuration, where RECTOR reports {protb["Total_Viol_pct"]:.2f}% total and {protb["Safety_Viol_pct"]:.2f}% Safety violations.',
        ),
        (
            "4. Weighted-sum parity is handled honestly",
            f'WS and lex choose different modes in {100.0 * divergence.get("divergence_rate", 0.0):.0f}% of moderate weight configs, yet compliance outcomes are identical at w_S\u22655 (0/12,800 divergent). '
            "A 250-config misspecification sweep and adversarial injection test make this explicit rather than hiding it.",
        ),
    ]
    for i, (head, body) in enumerate(bullets):
        y = 0.88 - i * 0.22
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.00, y - 0.11),
                0.98,
                0.17,
                boxstyle="round,pad=0.008",
                facecolor=PANEL,
                edgecolor=BORDER,
                lw=1.0,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            0.03,
            y,
            head,
            ha="left",
            va="center",
            fontsize=22,
            color=ACCENT if i % 2 == 0 else GREEN,
            fontweight="bold",
            transform=t,
        )
        ax.text(
            0.03,
            y - 0.055,
            body,
            ha="left",
            va="center",
            fontsize=16,
            color=WHITE,
            transform=t,
            wrap=True,
        )

    add_footer_box(
        fig,
        "The target is not a short highlight reel. The target is a defensible technical argument backed by many views of the evidence.",
        "Slides establish the claim; scenario clips test whether the claim survives contact with diverse traffic scenes.",
        edge=GREEN,
    )
    return fig


def make_problem_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Why Selection Still Fails in Strong Predictors",
        "Good candidate sets do not guarantee safe or lawful chosen trajectories",
    )
    ax = fig.add_axes([0.05, 0.18, 0.90, 0.66], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    geo = canon["geometric_metrics"]
    conf = canon["selection_strategies"]["confidence"]
    lex = canon["selection_strategies"]["lexicographic"]

    problems = [
        (
            RED,
            "High-quality candidates still permit bad final choices",
            f'minADE = {geo["minADE_mean"]:.3f} m and minFDE = {geo["minFDE_mean"]:.3f} m do not stop confidence-only selection from landing at {conf["Total_Viol_pct"]:.2f}% total violations.',
        ),
        (
            ORANGE,
            "Flat scalar costs hide priority",
            "Without an explicit ordering, safety, legality, road geometry, and comfort compete inside one number instead of a transparent policy.",
        ),
        (
            ACCENT,
            "Auditability matters",
            "A selector must say which tier made the decision, not just output one chosen mode and one probability score.",
        ),
    ]
    for i, (color, head, body) in enumerate(problems):
        y = 0.80 - i * 0.25
        ax.add_patch(
            mpatches.Circle(
                (0.06, y),
                0.035,
                facecolor=color + "30",
                edgecolor=color,
                lw=2,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            0.06,
            y,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=20,
            color=color,
            fontweight="bold",
            transform=t,
        )
        ax.text(
            0.12,
            y + 0.035,
            head,
            ha="left",
            va="center",
            fontsize=24,
            color=color,
            fontweight="bold",
            transform=t,
        )
        ax.text(
            0.12,
            y - 0.03,
            body,
            ha="left",
            va="center",
            fontsize=17,
            color=GRAY if i == 1 else WHITE,
            transform=t,
        )

    add_footer_box(
        fig,
        f'RECTOR pushes total violations from {conf["Total_Viol_pct"]:.2f}% to {lex["Total_Viol_pct"]:.2f}% by changing the decision rule, not by changing the candidate set itself.',
        "That distinction is why the script spends so much time on selection logic, audit protocol, and per-scenario comparisons.",
        edge=RED,
    )
    return fig


def make_tiers_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Lexicographic Priority Structure",
        "Lower-tier improvements are accepted only after higher-tier constraints are settled",
    )
    ax = fig.add_axes([0.05, 0.16, 0.90, 0.70], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    lex = canon["selection_strategies"]["lexicographic"]
    conf = canon["selection_strategies"]["confidence"]
    rows = [
        (
            "T0",
            "Safety",
            RED,
            conf["Safety_Viol_pct"],
            lex["Safety_Viol_pct"],
            "Collision, overlap, VRU clearance, emergency constraints",
        ),
        (
            "T1",
            "Legal",
            ORANGE,
            conf["Legal_Viol_pct"],
            lex["Legal_Viol_pct"],
            "Signals, stop signs, right-of-way, speed-related rules",
        ),
        (
            "T2",
            "Road",
            YELLOW,
            conf["Road_Viol_pct"],
            lex["Road_Viol_pct"],
            "Lane and drivable-surface geometry",
        ),
        (
            "T3",
            "Comfort",
            GREEN,
            conf["Comfort_Viol_pct"],
            lex["Comfort_Viol_pct"],
            "Acceleration, jerk, and interaction smoothness",
        ),
    ]
    for i, (badge, name, color, base, rect, desc) in enumerate(rows):
        y = 0.83 - i * 0.19
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.02, y - 0.09),
                0.96,
                0.14,
                boxstyle="round,pad=0.006",
                facecolor=color + "18",
                edgecolor=color,
                lw=1.5,
                transform=t,
                clip_on=False,
            )
        )
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.03, y - 0.06),
                0.09,
                0.08,
                boxstyle="round,pad=0.004",
                facecolor=color + "30",
                edgecolor=color,
                lw=1.8,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            0.075,
            y - 0.005,
            badge,
            ha="center",
            va="center",
            fontsize=18,
            color=color,
            fontweight="bold",
            transform=t,
        )
        ax.text(
            0.16,
            y + 0.015,
            name,
            ha="left",
            va="center",
            fontsize=24,
            color=color,
            fontweight="bold",
            transform=t,
        )
        ax.text(
            0.16,
            y - 0.035,
            desc,
            ha="left",
            va="center",
            fontsize=15,
            color=WHITE,
            transform=t,
        )
        ax.text(
            0.77,
            y + 0.01,
            f"{base:.2f}% → {rect:.2f}%",
            ha="center",
            va="center",
            fontsize=22,
            color=WHITE,
            fontweight="bold",
            transform=t,
        )
        ax.text(
            0.77,
            y - 0.04,
            "confidence → RECTOR",
            ha="center",
            va="center",
            fontsize=12,
            color=GRAY,
            transform=t,
        )

    add_footer_box(
        fig,
        "This ordering is the core behavioural claim: RECTOR never accepts a lower-tier gain by knowingly worsening a higher-tier decision boundary.",
        "Weighted-sum can emulate some outcomes on one checkpoint; lexicographic ordering explains the policy directly.",
        edge=ORANGE,
    )
    return fig


def make_arch_slide() -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "RECTOR Architecture",
        "Scene encoding, candidate generation, rule evaluation, and lexicographic selection",
    )
    arch_path = figure_path("rector_architecture_diagram.png")
    if arch_path.exists():
        ax_img = fig.add_axes([0.02, 0.12, 0.43, 0.72], facecolor=DARK)
        draw_image(ax_img, arch_path)
        ax_txt = fig.add_axes([0.48, 0.16, 0.48, 0.66], facecolor=DARK)
    else:
        ax_txt = fig.add_axes([0.06, 0.16, 0.88, 0.66], facecolor=DARK)
    ax_txt.axis("off")
    t = ax_txt.transAxes
    steps = [
        (
            "① Candidate generator",
            "A learned trajectory model proposes multiple futures for the same scene.",
        ),
        (
            "② Applicability head",
            "Each trajectory is tested against rule-activation logic so RECTOR knows which rules matter in-context.",
        ),
        (
            "③ Proxy scoring",
            "Differentiable proxies estimate rule-severity signals during training-time selection.",
        ),
        (
            "④ Lexicographic comparator",
            "Candidates are ordered by tier: Safety before Legal, Legal before Road, Road before Comfort.",
        ),
        (
            "⑤ Audit trace",
            "The system can record active rules, per-tier scores, and the first tier that separated the chosen mode from rejected modes.",
        ),
    ]
    for i, (head, body) in enumerate(steps):
        y = 0.92 - i * 0.18
        ax_txt.text(
            0.00,
            y,
            head,
            ha="left",
            va="top",
            fontsize=20,
            color=ACCENT if i % 2 == 0 else GREEN,
            fontweight="bold",
            transform=t,
        )
        ax_txt.text(
            0.00,
            y - 0.055,
            body,
            ha="left",
            va="top",
            fontsize=15,
            color=WHITE,
            transform=t,
            wrap=True,
        )
    add_footer_box(
        fig,
        "The selector is the contribution: it changes how existing candidates are chosen and exposes why that choice was made.",
        edge=ACCENT,
    )
    return fig


def make_protocol_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Evaluation Protocol and Oracle Audit",
        "Training-time selection and full-catalog auditing are both shown because they answer different questions",
    )

    ax_left = fig.add_axes([0.04, 0.24, 0.42, 0.56], facecolor=DARK)
    draw_image(ax_left, figure_path("protocol_comparison_heatmap_a.png"))
    ax_right = fig.add_axes([0.50, 0.24, 0.42, 0.56], facecolor=DARK)
    draw_image(ax_right, figure_path("protocol_comparison_heatmap_b.png"))

    ax = fig.add_axes([0.05, 0.12, 0.90, 0.10], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes
    ax.text(
        0.00,
        0.70,
        "Protocol A: proxy-24 with predicted applicability. This is the operating selection setting used throughout the main benchmark.",
        ha="left",
        va="center",
        fontsize=15,
        color=WHITE,
        transform=t,
    )
    ax.text(
        0.00,
        0.25,
        f'Protocol B: full-28 with oracle applicability. Under this stricter audit, RECTOR reports {canon["selection_strategies_protB"]["lexicographic"]["Total_Viol_pct"]:.2f}% total and {canon["selection_strategies_protB"]["lexicographic"]["Safety_Viol_pct"]:.2f}% Safety violations.',
        ha="left",
        va="center",
        fontsize=15,
        color=GRAY,
        transform=t,
    )

    add_footer_box(
        fig,
        "The video keeps both views because a convincing claim has to show performance in the selection regime and in the stricter oracle audit regime.",
        edge=ACCENT,
    )
    return fig


def make_results_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Headline Quantitative Results",
        "Protocol A drives the operating comparison; Protocol B shows what survives oracle auditing",
    )

    strategies = canon["selection_strategies"]
    protb = canon["selection_strategies_protB"]["lexicographic"]
    names = ["confidence", "weighted_sum", "lexicographic"]

    # ── Left panel: per-strategy stat cards (replaces sparse scatter) ──────────
    ax_cards = fig.add_axes([0.04, 0.20, 0.40, 0.60], facecolor=DARK)
    ax_cards.axis("off")
    t = ax_cards.transAxes

    ax_cards.text(0.5, 0.99, "Strategy comparison — Protocol A",
                  ha="center", va="top", fontsize=15, color=WHITE,
                  fontweight="bold", transform=t)

    card_rows = [
        ("selADE (m)", "selADE_mean", ".3f"),
        ("Total viol (%)", "Total_Viol_pct", ".2f"),
        ("Safety viol (%)", "Safety_Viol_pct", ".2f"),
        ("Comfort viol (%)", "Comfort_Viol_pct", ".2f"),
    ]
    col_xs = [0.33, 0.57, 0.82]
    # column headers
    row_y = 0.88
    ax_cards.text(0.02, row_y, "", transform=t)  # row label column spacer
    for name, cx in zip(names, col_xs):
        ax_cards.text(cx, row_y, STRAT_LABELS[name],
                      ha="center", va="center", fontsize=11,
                      color=STRAT_COLORS[name], fontweight="bold", transform=t)
    ax_cards.plot([0, 1], [row_y - 0.06, row_y - 0.06],
                  color=BORDER, lw=1, transform=t, clip_on=False)
    row_y -= 0.10

    for row_label, key, fmt in card_rows:
        vals = [strategies[name][key] for name in names]
        # Highlight best (lowest) value
        best_idx = int(np.argmin(vals))
        ax_cards.text(0.02, row_y, row_label,
                      ha="left", va="center", fontsize=12,
                      color=GRAY, transform=t)
        for ni, (name, cx) in enumerate(zip(names, col_xs)):
            val = strategies[name][key]
            color = GREEN if ni == best_idx else WHITE
            weight = "bold" if ni == best_idx else None
            ax_cards.text(cx, row_y, f"{val:{fmt}}",
                          ha="center", va="center", fontsize=14,
                          color=color, fontweight=weight, transform=t)
        ax_cards.plot([0, 1], [row_y - 0.07, row_y - 0.07],
                      color=BORDER, lw=0.6, linestyle=":",
                      transform=t, clip_on=False)
        row_y -= 0.16

    # accuracy-vs-compliance prose note
    lex = strategies["lexicographic"]
    conf = strategies["confidence"]
    ws = strategies["weighted_sum"]
    note = (
        f"Rule-aware selection (WS & lex) cuts total violations to "
        f"{lex['Total_Viol_pct']:.2f}% vs {conf['Total_Viol_pct']:.2f}% for "
        f"confidence-only, at a selADE cost of "
        f"{lex['selADE_mean'] - conf['selADE_mean']:+.3f} m "
        f"(lex) / {ws['selADE_mean'] - conf['selADE_mean']:+.3f} m (WS)."
    )
    ax_cards.text(0.02, 0.06, note,
                  ha="left", va="center", fontsize=11, color=GRAY,
                  style="italic", transform=t, wrap=True)

    # ── Right panel: per-tier bar chart (unchanged, wider now) ─────────────────
    ax2 = fig.add_axes([0.50, 0.20, 0.47, 0.60], facecolor=PANEL)

    tiers = ["Safety_Viol_pct", "Legal_Viol_pct", "Road_Viol_pct", "Comfort_Viol_pct"]
    labels = ["Safety", "Legal", "Road", "Comfort"]
    x = np.arange(len(tiers))
    width = 0.36
    ax2.bar(
        x - width / 2,
        [strategies["lexicographic"][tier] for tier in tiers],
        width,
        color=BLUE,
        label="RECTOR · Protocol A",
    )
    ax2.bar(
        x + width / 2,
        [protb[tier] for tier in tiers],
        width,
        color=GREEN,
        label="RECTOR · Protocol B (oracle)",
    )
    for xpos, value in zip(
        x - width / 2, [strategies["lexicographic"][tier] for tier in tiers]
    ):
        ax2.text(xpos, value + 0.35, f"{value:.2f}", ha="center", fontsize=11)
    for xpos, value in zip(x + width / 2, [protb[tier] for tier in tiers]):
        ax2.text(xpos, value + 0.35, f"{value:.2f}", ha="center", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Violation rate (%)")
    ax2.set_title("RECTOR under training protocol vs oracle audit")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25, linestyle="--", axis="y")

    conf = strategies["confidence"]
    lex = strategies["lexicographic"]
    add_footer_box(
        fig,
        f'Total violations fall from {conf["Total_Viol_pct"]:.2f}% to {lex["Total_Viol_pct"]:.2f}% and Safety violations from {conf["Safety_Viol_pct"]:.2f}% to {lex["Safety_Viol_pct"]:.2f}% under Protocol A.',
        f'Protocol B is stricter by design and still leaves RECTOR at {protb["Total_Viol_pct"]:.2f}% total.',
        edge=GREEN,
    )
    return fig


def make_statistics_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Statistical Strength of the Main Effect",
        "The improvement over confidence-only is not a fragile average-case artifact",
    )
    ax = fig.add_axes([0.05, 0.16, 0.90, 0.68], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    stats = canon["statistical_tests"]["lexicographic_vs_confidence"]
    cards = [
        (
            "McNemar Safety",
            f"discordant = {stats['mcnemar_safety']['n_discordant']:,}",
            f"p = {stats['mcnemar_safety']['p_value']:.1e}",
        ),
        (
            "McNemar Total",
            f"discordant = {stats['mcnemar_total']['n_discordant']:,}",
            f"p = {stats['mcnemar_total']['p_value']:.1e}",
        ),
        (
            "Wilcoxon selADE",
            f"nonzero = {stats['wilcoxon_selADE']['n_nonzero']:,}",
            f"p = {stats['wilcoxon_selADE']['p_value']:.1e}",
        ),
    ]
    for i, (head, line1, line2) in enumerate(cards):
        x = 0.02 + i * 0.32
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.64),
                0.28,
                0.24,
                boxstyle="round,pad=0.008",
                facecolor=PANEL,
                edgecolor=ACCENT if i < 2 else GREEN,
                lw=1.5,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            x + 0.14,
            0.81,
            head,
            ha="center",
            va="center",
            fontsize=18,
            color=WHITE,
            fontweight="bold",
            transform=t,
        )
        ax.text(
            x + 0.14,
            0.74,
            line1,
            ha="center",
            va="center",
            fontsize=15,
            color=GRAY,
            transform=t,
        )
        ax.text(
            x + 0.14,
            0.68,
            line2,
            ha="center",
            va="center",
            fontsize=16,
            color=GREEN,
            fontweight="bold",
            transform=t,
        )

    points = [
        "The confidence baseline creates unique Safety and Total violations that RECTOR removes on thousands of scenarios.",
        "The selADE change is statistically non-random as well, which matters because RECTOR is changing the chosen mode rather than only the violation count.",
        "Against weighted-sum, aggregate violation parity holds on this checkpoint, so the differentiator is the semantics of priority ordering and auditability, not an invented accuracy win.",
    ]
    for i, point in enumerate(points):
        ax.text(
            0.02,
            0.46 - i * 0.12,
            f"• {point}",
            ha="left",
            va="center",
            fontsize=18,
            color=WHITE if i == 0 else GRAY,
            transform=t,
            wrap=True,
        )

    add_footer_box(
        fig,
        "The core empirical claim is strong against confidence-only; the weighted-sum comparison is presented as a structural argument, not exaggerated as a broad empirical separation.",
        edge=PURPLE,
    )
    return fig


def make_misspecification_slide(misspec: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Weight-Space Robustness: Misspecification Analysis",
        "How far can weights deviate from the lex-like regime before WS and lex diverge on compliance?",
    )

    all_results = misspec.get("all_results", [])
    lex_baseline = misspec.get("lexicographic_baseline", {})
    lex_safety = lex_baseline.get("SL_Viol_pct", 13.148437)

    # Aggregate WS Safety violation rate by safety_weight (max over all legal/road configs)
    from collections import defaultdict
    by_ws: dict = defaultdict(list)
    for r in all_results:
        w = r.get("weights", {})
        ws = w.get("safety", 0)
        by_ws[ws].append(r.get("Safety_Viol_pct", lex_safety))

    ws_vals = sorted(by_ws.keys())
    max_ws_safety = [max(by_ws[ws]) for ws in ws_vals]
    deltas = [v - lex_safety for v in max_ws_safety]

    # ── Left panel: bar chart of delta (WS - Lex) safety violation ──
    ax_bar = fig.add_axes([0.05, 0.22, 0.40, 0.55], facecolor=PANEL)
    colors = [RED if d > 0 else GREEN for d in deltas]
    xs = range(len(ws_vals))
    ax_bar.bar(xs, deltas, color=colors, width=0.6, zorder=3)
    ax_bar.axhline(0, color=BORDER, lw=1.5)
    ax_bar.set_xticks(list(xs))
    ax_bar.set_xticklabels([str(w) for w in ws_vals], fontsize=10, rotation=45)
    ax_bar.set_xlabel("Safety weight  w\u209B", fontsize=13)
    ax_bar.set_ylabel("\u0394 Safety violation (WS \u2212 Lex, pp)", fontsize=12)
    ax_bar.set_title("WS divergence from lex by safety weight", fontsize=14, color=WHITE)
    ax_bar.grid(True, axis="y", alpha=0.25, linestyle="--")

    # Annotate the w_S=1 bar
    if deltas:
        idx1 = ws_vals.index(1) if 1 in ws_vals else 0
        ax_bar.text(
            idx1,
            deltas[idx1] + 0.001,
            f"+{deltas[idx1]:.4f} pp\n(1/12,800)",
            ha="center",
            va="bottom",
            fontsize=10,
            color=RED,
            fontweight="bold",
        )

    # ── Right panel: summary table ──
    ax_tbl = fig.add_axes([0.52, 0.22, 0.44, 0.55], facecolor=DARK)
    ax_tbl.axis("off")
    t = ax_tbl.transAxes

    ax_tbl.text(
        0.5, 0.97, "Compliance delta per safety weight (max over all legal/road configs)",
        ha="center", va="top", fontsize=13, color=ACCENT, fontweight="bold", transform=t,
    )

    headers = ["w\u209B", "WS Safety (%)", "Lex Safety (%)", "\u0394 (pp)", "Divergent?"]
    col_xs = [0.05, 0.25, 0.50, 0.72, 0.88]
    row_y = 0.86
    for hdr, cx in zip(headers, col_xs):
        ax_tbl.text(cx, row_y, hdr, ha="left", va="center", fontsize=11,
                    color=GRAY, fontweight="bold", transform=t)
    ax_tbl.plot([0.02, 0.98], [row_y - 0.04, row_y - 0.04],
                color=BORDER, lw=1, transform=t, clip_on=False)

    show_ws = [1, 5, 10, 20, 50, 100, 500, 1000]
    row_y = 0.78
    for ws in show_ws:
        if ws not in by_ws:
            continue
        ws_s = max(by_ws[ws])
        delta = ws_s - lex_safety
        diverges = delta > 0.0001
        color = RED if diverges else GREEN
        vals = [
            str(ws),
            f"{ws_s:.4f}",
            f"{lex_safety:.4f}",
            f"{delta:+.4f}",
            "YES" if diverges else "NO",
        ]
        for val, cx in zip(vals, col_xs):
            ax_tbl.text(cx, row_y, val, ha="left", va="center", fontsize=11,
                        color=color if cx == col_xs[-1] else WHITE, transform=t)
        row_y -= 0.075

    add_footer_box(
        fig,
        "At w\u209B = 1 (flat weights), WS diverges on exactly 1/12,800 scenarios (+0.0078 pp). "
        "At w\u209B \u2265 5, compliance outcomes are identical across all 250 tested configurations.",
        "The misspecification boundary lies near w\u209B/w\u2099 \u2248 1; practical weight choices remain safely above it.",
        edge=YELLOW,
    )
    return fig


def make_adversarial_slide(adversarial: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Adversarial Confidence Injection",
        "Does rule-aware selection resist a safety-violating trajectory planted with inflated confidence?",
    )

    per_inj = adversarial.get("per_injection", {})
    inject_types = list(per_inj.keys())
    labels_map = {
        "collision_course": "Collision course",
        "clearance_violation": "Clearance violation",
        "vru_incursion": "VRU incursion",
    }

    ax = fig.add_axes([0.05, 0.20, 0.90, 0.62], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    n_cols = len(inject_types)
    col_w = 0.90 / n_cols

    strategies = [
        ("confidence", "Confidence-only", STRAT_COLORS["confidence"]),
        ("weighted_sum", "Weighted-sum", STRAT_COLORS["weighted_sum"]),
        ("lexicographic", "RECTOR (Lex.)", STRAT_COLORS["lexicographic"]),
    ]

    # Column headers
    for ci, itype in enumerate(inject_types):
        cx = 0.05 + ci * col_w + col_w / 2
        ax.text(cx, 0.97, labels_map.get(itype, itype),
                ha="center", va="top", fontsize=18, color=ACCENT,
                fontweight="bold", transform=t)

    for ci, itype in enumerate(inject_types):
        data = per_inj.get(itype, {})
        cx_left = 0.05 + ci * col_w + 0.01

        row_y = 0.82
        for strat_key, strat_label, color in strategies:
            sd = data.get(strat_key, {})
            sel_rate = sd.get("adversarial_selection_rate_pct", 0.0)
            safety_viol = sd.get("Safety_Viol_pct", 0.0)

            # Selection-rate box
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (cx_left, row_y - 0.09),
                    col_w - 0.04,
                    0.14,
                    boxstyle="round,pad=0.006",
                    facecolor=PANEL,
                    edgecolor=RED if sel_rate > 50 else GREEN,
                    lw=2.0,
                    transform=t,
                    clip_on=False,
                )
            )
            ax.text(cx_left + (col_w - 0.04) / 2, row_y + 0.01,
                    strat_label,
                    ha="center", va="center", fontsize=11, color=color,
                    fontweight="bold", transform=t)
            ax.text(cx_left + (col_w - 0.04) / 2, row_y - 0.045,
                    f"Adv. selected: {sel_rate:.1f}%",
                    ha="center", va="center", fontsize=13,
                    color=RED if sel_rate > 50 else GREEN,
                    fontweight="bold", transform=t)
            ax.text(cx_left + (col_w - 0.04) / 2, row_y - 0.075,
                    f"Safety viol: {safety_viol:.1f}%",
                    ha="center", va="center", fontsize=10,
                    color=GRAY, transform=t)
            row_y -= 0.195

    # Key finding box
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.03, -0.02), 0.94, 0.14,
            boxstyle="round,pad=0.008",
            facecolor="#1f2d3d",
            edgecolor=GREEN,
            lw=2,
            transform=t,
            clip_on=False,
        )
    )
    ax.text(0.5, 0.085,
            "Key finding: confidence-only selects the injected trajectory 100% of the time. "
            "Both WS and RECTOR reject it at near-identical rates (3.6\u20135.7%).",
            ha="center", va="center", fontsize=15, color=WHITE,
            fontweight="bold", transform=t)
    ax.text(0.5, 0.018,
            "This test separates rule-aware from confidence-only selection, not lexicographic from weighted-sum.",
            ha="center", va="center", fontsize=13, color=GRAY,
            style="italic", transform=t)

    add_footer_box(
        fig,
        "Rule-aware selection (WS and lex) is robust to confidence inflation because compliance scores gate "
        "the choice before confidence is consulted.",
        "Confidence-only has no such gate \u2014 a high-confidence violating mode is always selected.",
        edge=RED,
    )
    return fig


def make_oracle_tier_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Four-Tier Breakdown: Protocol B (Oracle Applicability)",
        "All four tiers are non-zero under oracle labels \u2014 showing the full hierarchy is exercised",
    )

    protb = canon.get("selection_strategies_protB", {})
    strategies_order = [
        ("confidence", "Confidence-only", STRAT_COLORS["confidence"]),
        ("weighted_sum", "Weighted-sum", STRAT_COLORS["weighted_sum"]),
        ("lexicographic", "RECTOR (Lex.)", STRAT_COLORS["lexicographic"]),
    ]
    tiers = [
        ("Safety_Viol_pct", "Safety", RED),
        ("Legal_Viol_pct", "Legal", ORANGE),
        ("Road_Viol_pct", "Road", YELLOW),
        ("Comfort_Viol_pct", "Comfort", GREEN),
        ("Total_Viol_pct", "Total", ACCENT),
    ]

    # Grouped bar chart
    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.97,
                           top=0.82, bottom=0.20, wspace=0.32)
    ax_bar = fig.add_subplot(gs[0])

    tier_keys = [tk for tk, _, _ in tiers[:-1]]  # exclude Total
    tier_labels = [tl for _, tl, _ in tiers[:-1]]
    n_tiers = len(tier_keys)
    width = 0.22
    x = np.arange(n_tiers)

    for si, (skey, slabel, scolor) in enumerate(strategies_order):
        sd = protb.get(skey, {})
        vals = [sd.get(tk, 0.0) for tk in tier_keys]
        offset = (si - 1) * width
        bars = ax_bar.bar(x + offset, vals, width, label=slabel,
                          color=scolor, alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0.1:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=9, color=WHITE)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tier_labels, fontsize=13)
    ax_bar.set_ylabel("Violation rate (%)", fontsize=12)
    ax_bar.set_title("Per-tier violations by strategy\n(Protocol B, oracle applicability, 43,219 scenarios)",
                     fontsize=13, color=WHITE)
    ax_bar.legend(loc="upper right", fontsize=10)
    ax_bar.grid(True, axis="y", alpha=0.25, linestyle="--")

    # Right panel: reduction table
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    t = ax_tbl.transAxes

    ax_tbl.text(0.5, 0.99,
                "Violation rates (%) — Protocol B",
                ha="center", va="top", fontsize=14, color=ACCENT,
                fontweight="bold", transform=t)

    col_headers = ["Strategy", "Safety", "Legal", "Road", "Comfort", "Total"]
    col_xs = [0.00, 0.28, 0.42, 0.55, 0.68, 0.82]
    row_y = 0.88
    for hdr, cx in zip(col_headers, col_xs):
        ax_tbl.text(cx, row_y, hdr, ha="left", va="center", fontsize=11,
                    color=GRAY, fontweight="bold", transform=t)
    ax_tbl.plot([0, 1], [row_y - 0.05, row_y - 0.05],
                color=BORDER, lw=1, transform=t, clip_on=False)

    row_y = 0.78
    for skey, slabel, scolor in strategies_order:
        sd = protb.get(skey, {})
        vals = [
            slabel,
            f'{sd.get("Safety_Viol_pct", 0.0):.2f}',
            f'{sd.get("Legal_Viol_pct", 0.0):.2f}',
            f'{sd.get("Road_Viol_pct", 0.0):.2f}',
            f'{sd.get("Comfort_Viol_pct", 0.0):.2f}',
            f'{sd.get("Total_Viol_pct", 0.0):.2f}',
        ]
        for vi, (val, cx) in enumerate(zip(vals, col_xs)):
            ax_tbl.text(cx, row_y, val, ha="left", va="center", fontsize=11,
                        color=scolor if vi == 0 else WHITE, transform=t,
                        fontweight="bold" if vi == 0 else None)
        row_y -= 0.10

    # Reduction row
    conf_sd = protb.get("confidence", {})
    lex_sd = protb.get("lexicographic", {})
    ax_tbl.plot([0, 1], [row_y + 0.05, row_y + 0.05],
                color=BORDER, lw=1, transform=t, clip_on=False)
    reduction_vals = [
        "Reduction",
        f'{lex_sd.get("Safety_Viol_pct", 0) - conf_sd.get("Safety_Viol_pct", 0):+.2f}',
        f'{lex_sd.get("Legal_Viol_pct", 0) - conf_sd.get("Legal_Viol_pct", 0):+.2f}',
        f'{lex_sd.get("Road_Viol_pct", 0) - conf_sd.get("Road_Viol_pct", 0):+.2f}',
        f'{lex_sd.get("Comfort_Viol_pct", 0) - conf_sd.get("Comfort_Viol_pct", 0):+.2f}',
        f'{lex_sd.get("Total_Viol_pct", 0) - conf_sd.get("Total_Viol_pct", 0):+.2f}',
    ]
    for vi, (val, cx) in enumerate(zip(reduction_vals, col_xs)):
        ax_tbl.text(cx, row_y, val, ha="left", va="center", fontsize=11,
                    color=GRAY if vi == 0 else (GREEN if vi > 0 else RED),
                    transform=t, style="italic" if vi == 0 else None)
    row_y -= 0.14

    # Key insight text block
    ax_tbl.text(0.0, row_y,
                "Limitation (Protocol A):",
                ha="left", va="top", fontsize=12, color=YELLOW,
                fontweight="bold", transform=t)
    ax_tbl.text(0.0, row_y - 0.08,
                "Under the learned applicability head,\n"
                "Legal and Road are 0.0% for all strategies.\n"
                "Protocol B (oracle) restores all four tiers.",
                ha="left", va="top", fontsize=11, color=GRAY,
                transform=t)
    row_y -= 0.30
    ax_tbl.text(0.0, row_y,
                "WS vs. RECTOR (lex):",
                ha="left", va="top", fontsize=12, color=ACCENT,
                fontweight="bold", transform=t)
    ax_tbl.text(0.0, row_y - 0.08,
                "Identical on Safety, Legal, Road.\n"
                "Differ by 0.002 pp on Comfort\n"
                "(<1 scenario in 43,219).",
                ha="left", va="top", fontsize=11, color=GRAY,
                transform=t)

    add_footer_box(
        fig,
        "Under oracle applicability, rule-aware selection reduces violations at every tier vs. confidence-only "
        "(-6.07 pp Safety, -0.07 pp Legal, -0.05 pp Road, -2.28 pp Comfort).",
        "The four-tier hierarchy validates as an evaluation construct; it does not empirically separate lex from WS on this distribution.",
        edge=ACCENT,
    )
    return fig


def make_proxy_slide(proxy: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Applicability and Proxy Fidelity",
        "The video includes the weak spots as well as the strong spots because those weak spots explain the oracle-audit gap",
    )
    ax1 = fig.add_axes([0.05, 0.24, 0.42, 0.54], facecolor=DARK)
    draw_image(ax1, figure_path("rule_breakdown_a.png"))
    ax2 = fig.add_axes([0.53, 0.24, 0.42, 0.54], facecolor=DARK)
    draw_image(ax2, figure_path("rule_breakdown_b.png"))

    per_rule = proxy.get("per_rule", {})
    ranked = sorted(
        [
            (name, values.get("f1", 0.0), values.get("tier", ""))
            for name, values in per_rule.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    if ranked:
        summary = " · ".join(
            f"{name} ({tier}) F1={f1:.2f}" for name, f1, tier in ranked
        )
    else:
        summary = "Proxy summary unavailable."

    add_footer_box(
        fig,
        "Rule prediction quality is uneven across the catalog, which is exactly why the oracle-audit numbers are shown alongside the main operating results.",
        summary,
        edge=YELLOW,
    )
    return fig


def make_ablation_slide() -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Component Ablations",
        "If the system is real, removing core pieces should predictably hurt the right things",
    )
    ax1 = fig.add_axes([0.05, 0.22, 0.42, 0.58], facecolor=DARK)
    draw_image(ax1, figure_path("component_impact.png"))
    ax2 = fig.add_axes([0.53, 0.22, 0.42, 0.58], facecolor=DARK)
    alt = figure_path("ablation_radar.png")
    if alt.exists():
        draw_image(ax2, alt)
    else:
        draw_image(ax2, figure_path("ablation_comparison.png"))
    add_footer_box(
        fig,
        "Removing the tiered scorer hurts Safety most; removing applicability heads and differentiable proxies reduces rule-awareness; removing structure turns the method back into plain ranking.",
        edge=GREEN,
    )
    return fig


def make_generalization_slide(val_test: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Validation-to-Test Generalization",
        "The test split is harder, but the coarse scenario statistics remain broadly similar",
    )
    ax1 = fig.add_axes([0.05, 0.24, 0.42, 0.54], facecolor=DARK)
    draw_image(ax1, figure_path("test_set_generalization_a.png"))
    ax2 = fig.add_axes([0.53, 0.24, 0.42, 0.54], facecolor=DARK)
    draw_image(ax2, figure_path("test_set_generalization_b.png"))

    ks = val_test.get("ks_tests", {})
    if ks:
        largest = max(ks.items(), key=lambda item: item[1].get("ks_statistic", 0.0))
        ks_summary = f"Largest KS gap among tracked scalar descriptors: {largest[0]} with statistic {largest[1]['ks_statistic']:.3f} and p = {largest[1]['p_value']:.3f}."
    else:
        ks_summary = "KS summary unavailable."

    add_footer_box(
        fig,
        "The aggregate descriptor distributions are close across validation and test, yet test error rises sharply in the tail. That is consistent with harder rare interactions rather than a trivial split mismatch.",
        ks_summary,
        edge=ACCENT,
    )
    return fig


def make_scenario_gallery_slide(
    bev_items: list[dict], m2i_items: list[dict]
) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Scenario Diversity in the Video",
        "The clip roster is selected to cover different failure modes, clean wins, and edge cases",
    )

    ax_gallery = fig.add_axes([0.03, 0.24, 0.94, 0.50], facecolor=DARK)
    draw_image(ax_gallery, figure_path("scenario_gallery.png"))
    ax_legend = fig.add_axes([0.18, 0.08, 0.64, 0.10], facecolor=DARK)
    draw_image(ax_legend, figure_path("scenario_gallery_legend.png"))

    bev_count = len(bev_items)
    m2i_count = len(m2i_items)
    add_footer_box(
        fig,
        f"This video uses all {bev_count} available Waymax closed-loop frame sequences and {m2i_count} automatically selected M2I replays ranked for diversity and severity.",
        "The point is breadth: compliant cases, hard misses, safety-critical scenes, road-geometry stress, and lower-tier comfort trade-offs all appear.",
        edge=PURPLE,
    )
    return fig


def make_section_slide(title: str, subtitle: str, accent: str = ACCENT) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes
    ax.axhline(0.90, color=accent, lw=5, xmin=0.05, xmax=0.95)
    ax.text(
        0.5,
        0.60,
        title,
        ha="center",
        va="center",
        fontsize=48,
        color=WHITE,
        fontweight="bold",
        transform=t,
    )
    ax.text(
        0.5,
        0.42,
        subtitle,
        ha="center",
        va="center",
        fontsize=22,
        color=GRAY,
        style="italic",
        transform=t,
    )
    return fig


def make_scenario_context_slide(bundle: dict) -> plt.Figure:
    title, subtitle = classify_scenario(bundle)
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        title,
        "Before the replay: what to watch for in this particular scenario",
        accent=YELLOW if bundle["kind"] == "bev" else GREEN,
    )
    ax = fig.add_axes([0.05, 0.18, 0.90, 0.68], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    summary = bundle.get("summary", {})
    protb = bundle.get("protB_lexicographic", {})

    cards = [
        (
            "Replay source",
            "Waymax" if bundle["kind"] == "bev" else "M2I + RECTOR",
            ACCENT,
        ),
        ("minADE", f"{fmtn(summary.get('minADE'), 2)} m", BLUE),
        (
            "Outcome",
            "MISS" if summary.get("miss") else "HIT",
            RED if summary.get("miss") else GREEN,
        ),
        (
            "Oracle audit",
            "clean" if not protb.get("total_violated", True) else "stressed",
            GREEN if not protb.get("total_violated", True) else ORANGE,
        ),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 0.02 + i * 0.24
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.76),
                0.21,
                0.14,
                boxstyle="round,pad=0.008",
                facecolor=PANEL,
                edgecolor=color,
                lw=1.5,
                transform=t,
                clip_on=False,
            )
        )
        ax.text(
            x + 0.105,
            0.84,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color=GRAY,
            transform=t,
        )
        ax.text(
            x + 0.105,
            0.79,
            value,
            ha="center",
            va="center",
            fontsize=20,
            color=color,
            fontweight="bold",
            transform=t,
        )

    ax.text(
        0.02,
        0.64,
        subtitle,
        ha="left",
        va="center",
        fontsize=20,
        color=WHITE,
        fontweight="bold",
        transform=t,
    )
    ax.text(
        0.02,
        0.57,
        "What viewers should pay attention to in the next replay",
        ha="left",
        va="center",
        fontsize=22,
        color=ACCENT,
        fontweight="bold",
        transform=t,
    )

    for idx, line in enumerate(scenario_context_lines(bundle)):
        ax.text(
            0.03,
            0.47 - idx * 0.11,
            f"• {line}",
            ha="left",
            va="center",
            fontsize=18,
            color=WHITE if idx < 2 else GRAY,
            transform=t,
            wrap=True,
        )

    add_footer_box(
        fig,
        "These context slides set expectations before the clip so the viewer knows which conflict, miss, or clean pass is being evaluated.",
        edge=YELLOW if bundle["kind"] == "bev" else GREEN,
    )
    return fig


def make_scenario_discussion_slide(bundle: dict) -> plt.Figure:
    title, subtitle = classify_scenario(bundle)
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        title,
        "After the replay: how RECTOR performed in this particular simulation",
        accent=GREEN,
    )
    ax = fig.add_axes([0.05, 0.18, 0.90, 0.68], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    conf = bundle.get("confidence", {})
    ws = bundle.get("weighted_sum", {})
    lex = bundle.get("lexicographic", {})
    protb = bundle.get("protB_lexicographic", {})

    ax.text(
        0.02,
        0.86,
        subtitle,
        ha="left",
        va="center",
        fontsize=18,
        color=GRAY,
        style="italic",
        transform=t,
    )
    ax.text(
        0.02,
        0.77,
        "Per-strategy summary",
        ha="left",
        va="center",
        fontsize=22,
        color=WHITE,
        fontweight="bold",
        transform=t,
    )

    headers = [
        ("Confidence", conf, STRAT_COLORS["confidence"]),
        ("Weighted-sum", ws, STRAT_COLORS["weighted_sum"]),
        ("RECTOR", lex, STRAT_COLORS["lexicographic"]),
    ]
    col_x = [0.42, 0.62, 0.82]
    for (name, _, color), x in zip(headers, col_x):
        ax.text(
            x,
            0.77,
            name,
            ha="center",
            va="center",
            fontsize=16,
            color=color,
            fontweight="bold",
            transform=t,
        )

    rows = [
        ("selADE", "selADE"),
        ("selFDE", "selFDE"),
        ("Total", "total_violated"),
        ("Safety", "tier_0_violated"),
        ("Legal", "tier_1_violated"),
        ("Road", "tier_2_violated"),
        ("Comfort", "tier_3_violated"),
    ]
    y = 0.69
    for label, key in rows:
        ax.text(
            0.08,
            y,
            label,
            ha="left",
            va="center",
            fontsize=16,
            color=WHITE,
            transform=t,
        )
        for x, (_, values, _) in zip(col_x, headers):
            value = values.get(key)
            if isinstance(value, bool):
                text = "VIOL" if value else "OK"
                color = RED if value else GREEN
            elif value is None:
                text = "N/A"
                color = GRAY
            else:
                text = f"{float(value):.2f}"
                color = WHITE
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=15,
                color=color,
                fontweight="bold" if color != WHITE else None,
                transform=t,
            )
        y -= 0.075

    ax.text(
        0.02,
        0.20,
        "Discussion",
        ha="left",
        va="center",
        fontsize=22,
        color=ACCENT,
        fontweight="bold",
        transform=t,
    )
    for idx, line in enumerate(scenario_discussion_lines(bundle)):
        ax.text(
            0.03,
            0.12 - idx * 0.08,
            f"• {line}",
            ha="left",
            va="center",
            fontsize=16,
            color=WHITE if idx < 2 else GRAY,
            transform=t,
            wrap=True,
        )

    add_footer_box(
        fig,
        f"Scenario-level readout: Protocol B total={'VIOL' if protb.get('total_violated') else 'OK'}, Safety={'VIOL' if protb.get('tier_0_violated') else 'OK'}, Legal={'VIOL' if protb.get('tier_1_violated') else 'OK'}.",
        edge=GREEN,
    )
    return fig


def make_takeaways_slide(canon: dict) -> plt.Figure:
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=DARK)
    add_slide_header(
        fig,
        "Final Takeaways",
        "What the full video should leave a technical audience believing",
    )
    ax = fig.add_axes([0.05, 0.16, 0.90, 0.68], facecolor=DARK)
    ax.axis("off")
    t = ax.transAxes

    conf = canon["selection_strategies"]["confidence"]
    lex = canon["selection_strategies"]["lexicographic"]
    protb = canon["selection_strategies_protB"]["lexicographic"]

    bullets = [
        f'RECTOR materially changes the chosen trajectory: total violations fall from {conf["Total_Viol_pct"]:.2f}% to {lex["Total_Viol_pct"]:.2f}% and Safety from {conf["Safety_Viol_pct"]:.2f}% to {lex["Safety_Viol_pct"]:.2f}% on the full benchmark.',
        "The system explains its decisions in a way scalar confidence selection does not: active rules, tier ordering, and first-differing tier are all explicit.",
        f'Oracle auditing remains stricter, and the video shows that openly: {protb["Total_Viol_pct"]:.2f}% total and {protb["Safety_Viol_pct"]:.2f}% Safety violations under the full 28-rule evaluator.',
        "The qualitative section is intentionally broad. It includes clean wins, persistent misses, infeasible selections, and multi-tier stress cases rather than only polished demos.",
        "The result is a stronger paper supplement: the contribution is visible, the evidence is diverse, and the remaining gap is visible instead of hidden.",
    ]

    for i, bullet in enumerate(bullets):
        y = 0.88 - i * 0.16
        ax.text(
            0.03,
            y,
            "•",
            ha="center",
            va="center",
            fontsize=28,
            color=ACCENT if i % 2 == 0 else GREEN,
            transform=t,
        )
        ax.text(
            0.06,
            y,
            bullet,
            ha="left",
            va="center",
            fontsize=18,
            color=WHITE if i < 2 else GRAY,
            transform=t,
            wrap=True,
        )

    add_footer_box(
        fig,
        "RECTOR is most convincing when the paper and the video make the same claim: explicit priority-ordered selection improves compliance on a fixed candidate set, and the evidence is broad enough to inspect that claim from multiple angles.",
        edge=ACCENT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main video build
# ══════════════════════════════════════════════════════════════════════════════


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print("Loading current evaluation artifacts...")
    data = load_eval()
    canon = data["canonical"]
    if not canon:
        raise FileNotFoundError(
            "canonical_results.json not found under /workspace/output/evaluation"
        )

    bev_items = available_bev_items(data)
    for item in bev_items:
        item["tags"] = scenario_tags(item)
        item["score"] = scenario_score(item)
    bev_items.sort(key=lambda item: item["score"], reverse=True)

    m2i_items = select_diverse_m2i_scenarios(data)

    print(
        f"  Found {len(bev_items)} BEV scenario folders and {len(m2i_items)} selected M2I replays"
    )

    segments = []
    seg_counter = [0]

    def next_seg(name: str) -> Path:
        seg_counter[0] += 1
        return TMP / f"seg_{seg_counter[0]:02d}_{name}.mp4"

    def add_still(fig, name: str, seconds: float = STILL_SECONDS):
        png = TMP / f"{name}.png"
        fig_to_png(fig, png)
        segments.append(static_loop_mp4(png, seconds, next_seg(name)))

    add_still(make_title_slide(canon), "title", 6.0)
    add_still(make_contribution_slide(canon, data["divergence"]), "contributions", 9.0)
    add_still(make_problem_slide(canon), "problem", 8.0)
    add_still(make_tiers_slide(canon), "tiers", 8.0)
    add_still(make_arch_slide(), "architecture", 8.0)
    add_still(make_protocol_slide(canon), "protocol", 8.0)
    add_still(make_results_slide(canon), "results", 9.0)
    add_still(make_statistics_slide(canon), "statistics", 8.0)
    add_still(make_misspecification_slide(data["misspec"]), "misspecification", 9.0)
    add_still(make_adversarial_slide(data["adversarial"]), "adversarial", 9.0)
    add_still(make_oracle_tier_slide(canon), "oracle_tier", 9.0)
    add_still(make_proxy_slide(data["proxy"]), "proxy", 8.0)
    add_still(make_ablation_slide(), "ablation", 8.0)
    add_still(make_generalization_slide(data["val_test"]), "generalization", 8.0)
    add_still(
        make_scenario_gallery_slide(bev_items, m2i_items), "scenario_gallery", 8.0
    )

    add_still(
        make_section_slide(
            "Closed-Loop Waymax Replays",
            "All available frame sequences are included: each clip shows a real replay pane plus a per-scenario selection audit",
            accent=YELLOW,
        ),
        "bev_section",
        SECTION_SECONDS,
    )

    for item in bev_items:
        add_still(
            make_scenario_context_slide(item),
            f'bev_context_{item["idx"]:03d}',
            SCENARIO_CONTEXT_SECONDS,
        )
        title, subtitle = classify_scenario(item)
        panel = make_strategy_panel(title, subtitle, item)
        panel_png = TMP / f'bev_panel_{item["idx"]:03d}.png'
        fig_to_png(panel, panel_png)
        max_frames = int(BEV_CLIP_SECONDS * 10)
        segments.append(
            composite_scenario_mp4(
                item["path"],
                panel_png,
                next_seg(f'bev_{item["idx"]:03d}'),
                title1=title,
                title2=subtitle,
                fps=10,
                max_frames=max_frames,
            )
        )
        add_still(
            make_scenario_discussion_slide(item),
            f'bev_discussion_{item["idx"]:03d}',
            SCENARIO_DISCUSSION_SECONDS,
        )

    add_still(
        make_section_slide(
            "M2I + RECTOR Replay Bank",
            "A larger automatically selected set ranked for severity and diversity across misses, safety conflicts, road geometry, comfort stress, and clean passes",
            accent=GREEN,
        ),
        "m2i_section",
        SECTION_SECONDS,
    )

    for item in m2i_items:
        add_still(
            make_scenario_context_slide(item),
            f'm2i_context_{item["idx"]:03d}',
            SCENARIO_CONTEXT_SECONDS,
        )
        title, subtitle = classify_scenario(item)
        panel = make_strategy_panel(title, subtitle, item)
        panel_png = TMP / f'm2i_panel_{item["idx"]:03d}.png'
        fig_to_png(panel, panel_png)
        segments.append(
            composite_video_mp4(
                item["path"],
                panel_png,
                next_seg(f'm2i_{item["idx"]:03d}'),
                title1=title,
                title2=subtitle,
                fps_out=10,
                max_seconds=M2I_CLIP_SECONDS,
                skip_seconds=0.0,
            )
        )
        add_still(
            make_scenario_discussion_slide(item),
            f'm2i_discussion_{item["idx"]:03d}',
            SCENARIO_DISCUSSION_SECONDS,
        )

    add_still(make_takeaways_slide(canon), "takeaways", 10.0)

    print(f"Concatenating {len(segments)} segments...")
    concat_segments(segments, FINAL)
    shutil.rmtree(TMP, ignore_errors=True)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(FINAL),
        ],
        capture_output=True,
        text=True,
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
    minutes, seconds = divmod(int(duration), 60)
    print(f"\nWrote {FINAL}")
    print(f"  Duration: {minutes}m {seconds}s")
    print(f"  Size    : {FINAL.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
