"""
reporting.py — Visualizzazione e salvataggio risultati LOEO.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(fold_results: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Una figura per engine
    for fr in fold_results:
        esn    = fr["left_out_esn"]
        cycles = fr["info_test"]["Cycle"].values
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(cycles, fr["y_true"], "-o", ms=3, lw=2, alpha=0.85,
                label="Actual",    color="#2E86AB")
        ax.plot(cycles, fr["y_pred"], "--", lw=2.5, alpha=0.80,
                label="Predicted", color="#F18F01")
        ax.set_xlabel("Cycle", fontweight="bold")
        ax.set_ylabel("RUL to WW (cycles)", fontweight="bold")
        ax.set_title(f"Hybrid v3 – ESN {esn} | "
                     f"MAE={fr['mae']:.1f}, TWE={fr['twe']:.4f}, R²={fr['r2']:.3f}",
                     fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"loeo_esn{esn}.png"), dpi=150)
        plt.close()

    esns = [fr["left_out_esn"] for fr in fold_results]
    maes = [fr["mae"]          for fr in fold_results]
    twes = [fr["twe"]          for fr in fold_results]
    r2s  = [fr["r2"]           for fr in fold_results]
    imps = [fr["improvement"]  for fr in fold_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, vals, title, ylabel, fmt in [
        (axes[0,0], maes, "MAE per Engine",  "MAE (cycles)", "{:.1f}"),
        (axes[0,1], twes, "TWE Score (↓)",   "TWE",          "{:.4f}"),
    ]:
        bars = ax.bar(range(len(esns)), vals, edgecolor="black")
        ax.set_xticks(range(len(esns)))
        ax.set_xticklabels([f"ESN {e}" for e in esns], fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title,   fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontweight="bold")

    ax = axes[1, 0]
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
    for i, fr in enumerate(fold_results):
        ax.scatter(fr["y_true"], fr["y_pred"], s=15, alpha=0.45,
                   color=colors[i%4], label=f"ESN {fr['left_out_esn']}")
    all_y = np.concatenate([fr["y_true"] for fr in fold_results])
    lim   = [0, all_y.max()*1.05]
    ax.plot(lim, lim, "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual RUL",    fontweight="bold")
    ax.set_ylabel("Predicted RUL", fontweight="bold")
    ax.set_title("Actual vs Predicted", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]; ax.axis("off")
    txt = (f"HYBRID v3 RESULTS\n{'='*38}\n\n"
           f"Avg MAE:  {np.mean(maes):.1f} cycles\n"
           f"Avg TWE:  {np.mean(twes):.4f}\n"
           f"Avg R²:   {np.mean(r2s):.3f}\n"
           f"Avg Impr: {np.mean(imps):+.1f}%\n\nPer-fold:\n")
    for fr in fold_results:
        txt += (f"  ESN {fr['left_out_esn']}: "
                f"MAE={fr['mae']:.1f}, TWE={fr['twe']:.4f}, R²={fr['r2']:.3f}\n")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=1", facecolor="wheat", alpha=0.7))

    plt.suptitle("WW Prediction – Hybrid v3 (Residual+Periodic+Original)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(os.path.join(out_dir, "loeo_summary.png"), dpi=150)
    plt.close()
    print(f"\n  ✓ Plots saved → {out_dir}/")


def save_results(fold_results: list, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    rows = [{
        "left_out_esn": fr["left_out_esn"],
        "mae":          fr["mae"],
        "twe":          fr["twe"],
        "r2":           fr["r2"],
        "baseline_mae": fr["baseline_mae"],
        "improvement":  fr["improvement"],
    } for fr in fold_results]

    df_res = pd.DataFrame(rows)
    path   = os.path.join(out_dir, "hybrid_v3_results.csv")
    df_res.to_csv(path, index=False)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(df_res.to_string(index=False))
    print(f"\n  Avg MAE : {df_res['mae'].mean():.1f}")
    print(f"  Avg TWE : {df_res['twe'].mean():.4f}")
    print(f"  Avg R²  : {df_res['r2'].mean():.3f}")
    print(f"  Avg Impr: {df_res['improvement'].mean():+.1f}%")
    print("="*70)
    return df_res
