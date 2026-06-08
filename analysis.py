"""Analyse sentiment vs marché : corrélations et graphiques.

Logique partagée entre `report.ipynb` et les tests. Les fonctions prennent un
DataFrame avec une colonne par canal (`nlp`, `audio`, `vision`, `fusion`) plus une
colonne `market` (variation % du S&P 500), et ne dépendent d'aucun service réseau.
"""
import matplotlib
matplotlib.use("Agg")  # backend headless : fonctionne sans écran (CI, Colab, tests)
import matplotlib.pyplot as plt
import pandas as pd

CHANNELS = ["nlp", "audio", "vision", "fusion"]


def correlation_table(df, channels=CHANNELS):
    """Corrélation de Pearson entre chaque canal et la variation du marché.

    Renvoie un DataFrame indexé par canal avec une colonne `corr_with_market`.
    Les canaux absents du DataFrame sont ignorés.
    """
    present = [c for c in channels if c in df.columns]
    corr = {c: df[c].corr(df["market"]) for c in present}
    return pd.DataFrame.from_dict(corr, orient="index", columns=["corr_with_market"])


def plot_correlations(df, out_path, channels=CHANNELS):
    """Nuage de points sentiment vs marché, un sous-graphe par canal.

    Sauvegarde l'image dans `out_path` et renvoie ce chemin.
    """
    present = [c for c in channels if c in df.columns]
    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4))
    if len(present) == 1:
        axes = [axes]
    for ax, c in zip(axes, present):
        ax.scatter(df[c], df["market"])
        ax.set_title(f"{c} vs marché")
        ax.set_xlabel("sentiment [-1, 1]")
        ax.set_ylabel("variation S&P 500 (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
