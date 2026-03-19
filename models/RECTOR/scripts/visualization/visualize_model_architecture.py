#!/usr/bin/env python3
"""
RECTOR Model Architecture Visualization Script

Generates multiple visualization formats:
1. Torchviz - Computational graph (autograd graph)
2. Torchinfo - Textual summary (Keras-style)
3. TensorBoard - Interactive graph exploration
4. Custom layered diagram for publication

Usage:
    python visualization/visualize_model_architecture.py [--checkpoint PATH] [--output-dir PATH]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add RECTOR source to path
sys.path.insert(0, "/workspace/models/RECTOR/scripts")

import torch
import torch.nn as nn

# Visualization imports
try:
    from torchviz import make_dot

    HAS_TORCHVIZ = True
except ImportError:
    HAS_TORCHVIZ = False
    print("Warning: torchviz not available")

try:
    from torchinfo import summary

    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("Warning: torchinfo not available")

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available")


def parse_args():
    parser = argparse.ArgumentParser(description="RECTOR Model Visualization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/models/RECTOR/models/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/models/RECTOR/output/artifacts/model_viz",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--diagram-only",
        action="store_true",
        help="Generate only the architecture diagram (no checkpoint needed)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str):
    """Load RECTOR model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    from models.rule_aware_generator import RuleAwareGenerator

    # Create model with architecture matching checkpoint
    model = RuleAwareGenerator(
        embed_dim=256,
        num_heads=8,
        num_encoder_layers=4,
        history_length=11,
        max_agents=32,
        max_lanes=64,
        trajectory_length=50,
        num_modes=6,
        latent_dim=64,
        decoder_hidden_dim=256,
        decoder_num_layers=4,
        num_rules=28,
        dropout=0.1,
        use_m2i_encoder=True,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def create_dummy_inputs(batch_size=1, device="cpu"):
    """Create dummy inputs matching RECTOR's expected format."""
    return {
        "ego_history": torch.randn(batch_size, 11, 4, device=device),  # [B, T, 4]
        "agent_states": torch.randn(
            batch_size, 32, 11, 4, device=device
        ),  # [B, A, T, 4]
        "lane_centers": torch.randn(
            batch_size, 64, 20, 2, device=device
        ),  # [B, L, P, 2]
    }


def generate_torchviz_graph(model, output_dir: Path):
    """Generate computational graph using torchviz."""
    if not HAS_TORCHVIZ:
        print("Skipping torchviz (not installed)")
        return None

    print("\n" + "=" * 60)
    print("Generating Torchviz Computational Graph")
    print("=" * 60)

    # Create inputs
    inputs = create_dummy_inputs()

    # Forward pass to build graph
    model.train()  # Need train mode to get gradients
    outputs = model(**inputs)

    # Get the main output tensor
    trajectory = outputs["trajectory"]

    # Create dot graph
    dot = make_dot(
        trajectory,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )

    # Set graph attributes for better visualization
    dot.attr(rankdir="TB")  # Top to bottom
    dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

    # Save in multiple formats
    output_path = output_dir / "torchviz_computational_graph"
    dot.render(str(output_path), format="pdf", cleanup=True)
    dot.render(str(output_path), format="png", cleanup=True)
    dot.render(str(output_path), format="svg", cleanup=True)

    print(f"  Saved: {output_path}.pdf")
    print(f"  Saved: {output_path}.png")
    print(f"  Saved: {output_path}.svg")

    model.eval()
    return output_path


def generate_torchinfo_summary(model, output_dir: Path):
    """Generate Keras-style model summary using torchinfo."""
    if not HAS_TORCHINFO:
        print("Skipping torchinfo (not installed)")
        return None

    print("\n" + "=" * 60)
    print("Generating Torchinfo Summary")
    print("=" * 60)

    # Create input shapes
    input_data = create_dummy_inputs()

    # Generate summary
    model_summary = summary(
        model,
        input_data=input_data,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        col_width=20,
        row_settings=["var_names"],
        depth=5,
        verbose=0,
    )

    # Save to file
    output_path = output_dir / "torchinfo_summary.txt"
    with open(output_path, "w") as f:
        f.write("RECTOR Model Architecture Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 100 + "\n\n")
        f.write(str(model_summary))

    print(f"  Saved: {output_path}")

    # Also print to console
    print("\n" + str(model_summary))

    return output_path


def generate_tensorboard_graph(model, output_dir: Path):
    """Generate TensorBoard graph for interactive exploration."""
    if not HAS_TENSORBOARD:
        print("Skipping TensorBoard (not installed)")
        return None

    print("\n" + "=" * 60)
    print("Generating TensorBoard Graph")
    print("=" * 60)

    log_dir = output_dir / "tensorboard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    # Create dummy inputs
    inputs = create_dummy_inputs()

    # Add graph
    # We need to wrap the model for proper graph visualization
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, ego_history, agent_states, lane_centers):
            return self.model(ego_history, agent_states, lane_centers)["trajectory"]

    wrapped_model = ModelWrapper(model)

    try:
        writer.add_graph(
            wrapped_model,
            (inputs["ego_history"], inputs["agent_states"], inputs["lane_centers"]),
        )
        writer.close()
        print(f"  Saved: {log_dir}")
        print(f"  Run: tensorboard --logdir={log_dir}")
    except Exception as e:
        print(f"  Warning: Could not create TensorBoard graph: {e}")
        writer.close()

    return log_dir


def generate_layer_diagram(model, output_dir: Path):
    """Generate a publication-ready layer diagram reflecting the actual V2 architecture."""
    print("\n" + "=" * 60)
    print("Generating Layer Diagram (Graphviz)")
    print("=" * 60)

    try:
        import graphviz
    except ImportError:
        print("Skipping layer diagram (graphviz not installed)")
        return None

    dot = graphviz.Digraph(comment="RECTOR Architecture V2")
    dot.attr(rankdir="TB", size="14,20", dpi="150")
    dot.attr(
        "node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11"
    )
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # ── Color palette ─────────────────────────────────────────────────────────
    C = {
        "input": "#E3F2FD",  # light blue
        "encoder": "#BBDEFB",  # blue
        "attn": "#90CAF9",  # mid-blue
        "app": "#FFE0B2",  # orange
        "app_dark": "#FFCC80",
        "cvae": "#E8F5E9",  # green
        "cvae_dark": "#C8E6C9",
        "proxy": "#F3E5F5",  # purple
        "proxy_dark": "#CE93D8",
        "scorer": "#FFF9C4",  # yellow
        "output": "#DCEDC8",  # lime
        "best": "#AED581",
    }

    # ── 1. INPUTS ─────────────────────────────────────────────────────────────
    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(
            label="Inputs", style="rounded,dashed", color="#90A4AE", bgcolor="#FAFAFA"
        )
        c.node(
            "inp_ego",
            "Ego History\n[B, 11, 4]\n(x, y, heading, speed)",
            fillcolor=C["input"],
        )
        c.node("inp_agents", "Agent States\n[B, 32, 11, 4]", fillcolor=C["input"])
        c.node("inp_lanes", "Lane Centers\n[B, 64, 20, 2]", fillcolor=C["input"])

    # ── 2. M2I SCENE ENCODER ──────────────────────────────────────────────────
    with dot.subgraph(name="cluster_encoder") as c:
        c.attr(
            label="M2I Scene Encoder  (324K params, fine-tuned @ 0.1× LR)",
            style="rounded",
            color="#1565C0",
            bgcolor="#E8F0FE",
        )
        c.node(
            "enc_subgraph",
            "SubGraph Encoder\n(MLP + masked max-pool, 3 layers)\n→ polyline tokens [B, N, 128]",
            fillcolor=C["encoder"],
        )
        c.node(
            "enc_global",
            "GlobalGraphRes\n(2× GlobalGraph concat)\n→ [B, N, 128]",
            fillcolor=C["attn"],
        )
        c.node(
            "enc_xattn",
            "CrossAttention\n(A2L, L2L, L2A)\noptional lane interaction",
            fillcolor=C["attn"],
        )
        c.node(
            "enc_proj",
            "Output Projection\nLinear→LayerNorm→ReLU→Linear\n128 → 256",
            fillcolor=C["encoder"],
        )
        c.node(
            "scene_embed",
            "Scene Embedding\n[B, 256]",
            fillcolor=C["encoder"],
            style="rounded,filled,bold",
        )

    dot.edge("inp_ego", "enc_subgraph")
    dot.edge("inp_agents", "enc_subgraph")
    dot.edge("inp_lanes", "enc_subgraph")
    dot.edge("enc_subgraph", "enc_global")
    dot.edge("enc_global", "enc_xattn")
    dot.edge("enc_xattn", "enc_proj")
    dot.edge("enc_proj", "scene_embed")

    # ── 3. RULE APPLICABILITY HEAD ─────────────────────────────────────────────
    with dot.subgraph(name="cluster_app") as c:
        c.attr(
            label="Rule Applicability Head  (3.33M params)",
            style="rounded",
            color="#E65100",
            bgcolor="#FFF3E0",
        )
        c.node(
            "app_proj",
            "Scene Projection\n[B, 256] → [B, 384]\nLayerNorm + GELU + Dropout",
            fillcolor=C["app"],
        )
        # Four TierAwareBlocks
        c.node(
            "app_t0",
            "TierAwareBlock: Safety (5 rules)\nrule_queries [5,256]\n"
            "→ rule self-attn → scene cross-attn → FFN → classifier\nbias init −1.0",
            fillcolor=C["app_dark"],
        )
        c.node(
            "app_t1",
            "TierAwareBlock: Legal (7 rules)\nrule_queries [7,256]\n"
            "→ rule self-attn → scene cross-attn → FFN → classifier\nbias init −1.0",
            fillcolor=C["app_dark"],
        )
        c.node(
            "app_t2",
            "TierAwareBlock: Road (2 rules)\nrule_queries [2,256]\n"
            "→ rule self-attn → scene cross-attn → FFN → classifier\nbias init −1.0",
            fillcolor=C["app_dark"],
        )
        c.node(
            "app_t3",
            "TierAwareBlock: Comfort (14 rules)\nrule_queries [14,256]\n"
            "→ rule self-attn → scene cross-attn → FFN → classifier\nbias init −1.0",
            fillcolor=C["app_dark"],
        )
        c.node(
            "app_cross", "Cross-Tier Attention + Gated Refinement", fillcolor=C["app"]
        )
        c.node(
            "app_out",
            "Concatenate logits → [B, 28]\nsigmoid → applicability probs [B, 28]",
            fillcolor=C["app"],
        )

    dot.edge("scene_embed", "app_proj")
    dot.edge("app_proj", "app_t0")
    dot.edge("app_proj", "app_t1")
    dot.edge("app_proj", "app_t2")
    dot.edge("app_proj", "app_t3")
    dot.edge("app_t0", "app_cross")
    dot.edge("app_t1", "app_cross")
    dot.edge("app_t2", "app_cross")
    dot.edge("app_t3", "app_cross")
    dot.edge("app_cross", "app_out")

    # ── 4. CVAE TRAJECTORY HEAD V2 ────────────────────────────────────────────
    with dot.subgraph(name="cluster_cvae") as c:
        c.attr(
            label="CVAE Trajectory Head V2  (5.18M params)",
            style="rounded",
            color="#2E7D32",
            bgcolor="#F1F8E9",
        )
        c.node(
            "cvae_prior",
            "Prior  p(z | scene)\nLinear → μ, log σ²\n[B, 64]",
            fillcolor=C["cvae_dark"],
        )
        c.node(
            "cvae_post",
            "Posterior  q(z | scene, traj_gt)\n[training only]\nConv1D×2 + AvgPool → μ, log σ²\n[B, 64]",
            fillcolor=C["cvae_dark"],
            style="rounded,filled,dashed",
        )
        c.node(
            "cvae_z",
            "Reparameterize\nz = μ + ε·σ,   ε ~ N(0,I)\n[B, K=6, 64]",
            fillcolor=C["cvae"],
        )
        c.node(
            "cvae_goal",
            "Goal Head\nlearnable queries → cross-attn → scene\n→ goal_positions [B, 6, 2]\n   goal_confidence [B, 6]",
            fillcolor=C["cvae"],
        )
        c.node(
            "cvae_dec",
            "Transformer Decoder\n4 layers, 8 heads\ntraj queries [50, 256] + PosEnc\ncross-attend to (z ⊕ goal ⊕ scene)",
            fillcolor=C["attn"],
        )
        c.node(
            "cvae_delta",
            "Delta Prediction\nper-dim scales [0.15, 0.15, 0.04, 0.10]\ncumsum → absolute positions\n[B, 6, 50, 4]",
            fillcolor=C["cvae"],
        )
        c.node(
            "cvae_refiner",
            "Trajectory Refiner\nresidual MLP  (gate scale 0.1)\n[B, 6, 50, 4]",
            fillcolor=C["cvae"],
        )
        c.node("cvae_conf", "Confidence Head\n[B, 6]", fillcolor=C["cvae"])

    dot.edge("scene_embed", "cvae_prior")
    dot.edge("scene_embed", "cvae_post", label="train only", style="dashed")
    dot.edge("cvae_prior", "cvae_z", label="inference")
    dot.edge("cvae_post", "cvae_z", label="training", style="dashed")
    dot.edge("cvae_z", "cvae_dec")
    dot.edge("scene_embed", "cvae_goal")
    dot.edge("cvae_goal", "cvae_dec")
    dot.edge("scene_embed", "cvae_dec")
    dot.edge("cvae_dec", "cvae_delta")
    dot.edge("cvae_delta", "cvae_refiner")
    dot.edge("cvae_dec", "cvae_conf")

    # ── 5. DIFFERENTIABLE RULE PROXIES ────────────────────────────────────────
    with dot.subgraph(name="cluster_proxies") as c:
        c.attr(
            label="DifferentiableRuleProxies  (24/28 rules, no learned params)",
            style="rounded",
            color="#6A1B9A",
            bgcolor="#F8F0FF",
        )
        c.node(
            "px_col",
            "CollisionProxy\nSAT OBB penetration depth\n→ L0.R0, L0.R1, L0.R3, L0.R4",
            fillcolor=C["proxy_dark"],
        )
        c.node(
            "px_smooth",
            "SmoothnessProxy\naccel, braking, steering,\nspeed consistency, lane-change\n→ L3.R0–R4",
            fillcolor=C["proxy"],
        )
        c.node(
            "px_lane",
            "LaneProxy + SpeedLimitProxy\nSDF to road/lane boundary\n→ L2.R0, L2.R1, L1.R2",
            fillcolor=C["proxy"],
        )
        c.node(
            "px_sig",
            "SignalProxy\nred-light crossing, stop sign,\ncrosswalk yield, signal state\n→ L1.R0, L1.R3, L1.R4, L1.R5",
            fillcolor=C["proxy"],
        )
        c.node(
            "px_int",
            "InteractionProxy\nleft-turn gap, coop lane-change,\nfollowing dist, intersection,\nped/cyclist interaction\n→ L3.R5, L3.R9–R13",
            fillcolor=C["proxy"],
        )
        c.node(
            "px_none",
            "4 rules without proxies\n(L1.R1 priority, L3.R6–R8 zones)\n→ zeros (Protocol A)",
            fillcolor="#EEEEEE",
            style="rounded,filled,dashed",
        )
        c.node(
            "px_out",
            "violations  [B, 6, 28]\n(one score per mode per rule)",
            fillcolor=C["proxy_dark"],
            style="rounded,filled,bold",
        )

    dot.edge("cvae_refiner", "px_col")
    dot.edge("cvae_refiner", "px_smooth")
    dot.edge("cvae_refiner", "px_lane")
    dot.edge("cvae_refiner", "px_sig")
    dot.edge("cvae_refiner", "px_int")
    dot.edge("px_col", "px_out")
    dot.edge("px_smooth", "px_out")
    dot.edge("px_lane", "px_out")
    dot.edge("px_sig", "px_out")
    dot.edge("px_int", "px_out")
    dot.edge("px_none", "px_out", style="dashed")

    # ── 6. TIERED RULE SCORER ─────────────────────────────────────────────────
    with dot.subgraph(name="cluster_scorer") as c:
        c.attr(
            label="TieredRuleScorer  (140 fixed constants, no learned params)",
            style="rounded",
            color="#F57F17",
            bgcolor="#FFFDE7",
        )
        c.node(
            "sc_apply",
            "Apply applicability mask\nviolations × app_probs → masked [B, 6, 28]",
            fillcolor=C["scorer"],
        )
        c.node(
            "sc_hard",
            "Hard Lexicographic Score  [inference]\nScore = 1e12·S_safety\n"
            "+ 1e9·S_legal + 1e6·S_road + 1e3·S_comfort\n(B=1000 per tier)",
            fillcolor=C["scorer"],
        )
        c.node(
            "sc_soft",
            "Soft Differentiable Selection  [training]\nDifferentiableTieredSelection\nsoftmin per tier, temp=0.1",
            fillcolor=C["scorer"],
            style="rounded,filled,dashed",
        )
        c.node("sc_idx", "best_idx  [B]\n(argmin score)", fillcolor=C["scorer"])

    dot.edge("px_out", "sc_apply")
    dot.edge("app_out", "sc_apply", label="applicability probs")
    dot.edge("sc_apply", "sc_hard")
    dot.edge("sc_apply", "sc_soft", style="dashed", label="train only")
    dot.edge("sc_hard", "sc_idx")
    dot.edge("sc_soft", "sc_idx", style="dashed")

    # ── 7. FINAL OUTPUT ───────────────────────────────────────────────────────
    dot.node(
        "out_best",
        "Best Trajectory\n[B, 50, 4]\n(rule-compliant selection)",
        shape="box",
        style="rounded,filled,bold",
        fillcolor=C["best"],
    )
    dot.node(
        "out_allk",
        "All K=6 Trajectories\n[B, 6, 50, 4]  +  confidences [B, 6]",
        shape="box",
        style="rounded,filled",
        fillcolor=C["output"],
    )

    dot.edge("sc_idx", "out_best", label="index into")
    dot.edge("cvae_refiner", "out_allk")
    dot.edge("cvae_conf", "out_allk")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    output_path = output_dir / "rector_architecture_diagram"
    dot.render(str(output_path), format="pdf", cleanup=True)
    dot.render(str(output_path), format="png", cleanup=True)

    print(f"  Saved: {output_path}.pdf")
    print(f"  Saved: {output_path}.png")

    return output_path


def generate_component_breakdown(model, output_dir: Path):
    """Generate detailed parameter breakdown by component."""
    print("\n" + "=" * 60)
    print("Generating Component Parameter Breakdown")
    print("=" * 60)

    components = {
        "scene_encoder": 0,
        "applicability_head": 0,
        "trajectory_head": 0,
        "tiered_scorer": 0,
        "differentiable_selector": 0,
        "rule_proxies": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        found = False
        for comp in components:
            if comp in name:
                components[comp] += param.numel()
                found = True
                break
        if not found:
            components["other"] += param.numel()

    total = sum(components.values())

    output_path = output_dir / "parameter_breakdown.txt"
    with open(output_path, "w") as f:
        f.write("RECTOR Model Parameter Breakdown\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Component':<30} {'Parameters':>15} {'Percentage':>12}\n")
        f.write("-" * 60 + "\n")

        for comp, count in sorted(components.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / total
                f.write(f"{comp:<30} {count:>15,} {pct:>11.2f}%\n")

        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<30} {total:>15,} {100:>11.2f}%\n")

    print(f"  Saved: {output_path}")

    # Print to console
    print(f"\n{'Component':<30} {'Parameters':>15} {'Percentage':>12}")
    print("-" * 60)
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / total
            print(f"{comp:<30} {count:>15,} {pct:>11.2f}%")
    print("-" * 60)
    print(f"{'TOTAL':<30} {total:>15,}")

    return output_path


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RECTOR Model Architecture Visualization")
    print("=" * 60)
    print(f"Output dir: {output_dir}")

    # Diagram-only mode: skip checkpoint loading
    if args.diagram_only:
        print("Diagram-only mode — skipping checkpoint load.")
        output_path = generate_layer_diagram(None, output_dir)
        print(f"\nDone. Diagram saved to: {output_path}")
        return

    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    model = load_model(args.checkpoint)

    # Generate visualizations
    results = {}

    # 1. Torchinfo summary (text)
    results["torchinfo"] = generate_torchinfo_summary(model, output_dir)

    # 2. Layer diagram (graphviz)
    results["layer_diagram"] = generate_layer_diagram(model, output_dir)

    # 3. Component breakdown
    results["breakdown"] = generate_component_breakdown(model, output_dir)

    # 4. Torchviz computational graph
    results["torchviz"] = generate_torchviz_graph(model, output_dir)

    # 5. TensorBoard graph
    results["tensorboard"] = generate_tensorboard_graph(model, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"\nGenerated files in: {output_dir}")
    print("\nFiles created:")
    for name, path in results.items():
        if path:
            print(f"  - {name}: {path}")

    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={output_dir / 'tensorboard_logs'}")


if __name__ == "__main__":
    main()
