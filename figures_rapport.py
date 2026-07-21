#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenere les six figures du rapport, depuis negations.parquet et les CSV des
phases. Aucune valeur n'est saisie a la main : chaque point trace est recalcule
ici ou lu dans un CSV produit par le pipeline.

    Figure 1  omission_genre.pdf        taux d'omission par genre (Wilson)
    Figure 2  forest_modelA.pdf         GLMM d'omission, forest plot
    Figure 3  omission_unites.pdf       taux par unite (chenille), n >= 20
    Figure 4  conditioning_dotplot.pdf  par negateur (pool) | par temps (payeton)
    Figure 5  heatmap_tense.pdf         residus de Haberman, temps x corpus
    Figure 6  longueur_ecdf.pdf         fonctions de repartition des longueurs

Entrees  : negations.parquet
           resultats_phase5/modele_A_chute.csv   (figure 2)
Sortie   : figures/*.pdf  (vectoriel, inclus tel quel par LaTeX)
Lancer   : python figures_rapport.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency

from expletifs import EXPLETIFS_CUE_ID

OUT = Path("figures"); OUT.mkdir(exist_ok=True)

COULEURS = {"payetoncorpus": "#B03A2E", "litbank": "#2E5A87", "sequoia": "#4E8D5B"}
ORDRE_CORPUS = ["payetoncorpus", "litbank", "sequoia"]
PAIRES_NEG = {
    "pas": ("ne_pas", "pas_seul"), "plus": ("ne_plus", "plus_seul"),
    "jamais": ("ne_jamais", "jamais_seul"), "rien": ("ne_rien", "rien_seul"),
    "personne": ("ne_personne", "personne_seul"), "aucun": ("ne_aucun", "aucun_seul"),
    "point": ("ne_point", "point_seul"), "nul": ("ne_nul", "nul_seul"),
}
ORDRE_NEG = list(PAIRES_NEG)
ORDRE_TEMPS = [("Pres", "present"), ("Past", "past"), ("Fut", "future"), ("Imp", "imperfect")]

plt.rcParams.update({
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,   # polices vectorielles, pas de bitmap
})


def wilson(k, n, z=1.96):
    """Intervalle de score de Wilson (cf. annexe du rapport)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    demi = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - demi, centre + demi


def charger():
    df = pd.read_parquet("negations.parquet")
    df["cue_id"] = df["negation_id"].str.split("::").str[-1]
    df = df[~df["cue_id"].isin(EXPLETIFS_CUE_ID)].copy()
    df["a_verbe"] = df["verbe_lemme"].fillna("") != ""
    df["unite"] = np.where(df["sous_corpus"] == df["document_set"],
                           df["oeuvre"], df["sous_corpus"])
    return df


def table_omission(df):
    """Les 4 890 negations verbales portant un des huit negateurs a deux formes."""
    formes = [f for paire in PAIRES_NEG.values() for f in paire]
    om = df[df["type_fin"].isin(formes) & df["a_verbe"]].copy()
    om["ne_absent"] = om["type_fin"].str.endswith("_seul")
    om["negateur"] = (om["type_fin"].str.replace("^ne_", "", regex=True)
                                     .str.replace("_seul", "", regex=False))
    return om


# --------------------------------------------------------------------------- 1
def fig_omission_genre(om):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, c in enumerate(ORDRE_CORPUS):
        s = om[om.document_set == c]
        k, n = int(s.ne_absent.sum()), len(s)
        taux = k / n * 100
        lo, hi = wilson(k, n)
        ax.bar(i, taux, color=COULEURS[c], width=0.6)
        ax.errorbar(i, taux, yerr=[[taux - lo * 100], [hi * 100 - taux]],
                    fmt="none", ecolor="black", capsize=4, lw=1.2)
        ax.text(i, hi * 100 + 0.6, f"{taux:.1f}%", ha="center",
                fontweight="bold", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(ORDRE_CORPUS, fontsize=9)
    ax.set_ylabel("Omission of $ne$ (%)")
    ax.set_title("Omission rate by genre (Wilson 95% CI)", fontsize=9.5)
    ax.set_ylim(0, 20); ax.grid(axis="y", ls=":", alpha=.5)
    fig.tight_layout(); fig.savefig(OUT / "omission_genre.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- 2
def fig_forest():
    """Lit les estimations du GLMM ; ne refait pas l'ajustement."""
    src = Path("resultats_phase5/modele_A_chute.csv")
    if not src.exists():
        print("   [saute] figure 2 : lancez phase5_mixtes.py d'abord")
        return
    t = pd.read_csv(src).set_index("terme")
    lignes = [
        ("Corpus: payetoncorpus (vs litbank)", "corpus[T.payetoncorpus]", "corpus"),
        ("Corpus: sequoia (vs litbank)", "corpus[T.sequoia]", "corpus"),
        ("Negator: plus (vs pas)", "negateur[T.plus]", "neg"),
        ("Negator: jamais", "negateur[T.jamais]", "neg"),
        ("Negator: rien", "negateur[T.rien]", "neg"),
        ("Negator: personne", "negateur[T.personne]", "neg"),
        ("Negator: aucun", "negateur[T.aucun]", "neg"),
        ("Negator: point", "negateur[T.point]", "neg"),
        ("Negator: nul", "negateur[T.nul]", "neg"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, (lab, terme, genre) in enumerate(lignes):
        r = t.loc[terme]
        col = "#B03A2E" if genre == "corpus" else "#4D4D4D"
        ax.plot([r.ic95_bas, r.ic95_haut], [i, i], color=col, lw=1.4)
        ax.plot(r.OR, i, "o", color=col, ms=5.5)
    ax.axvline(1, ls="--", color="black", lw=.9)
    ax.set_yticks(range(len(lignes)))
    ax.set_yticklabels([l for l, _, _ in lignes], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    graduations = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(graduations)
    # ScalarFormatter ecrirait "0.10" / "20.00" et supprimerait 0.25 : on pose
    # les etiquettes a la main, sans zeros inutiles.
    ax.set_xticklabels([("%g" % g) for g in graduations])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Odds ratio (log scale), 95% credible interval")
    ax.set_title("Omission GLMM (reference: litbank, $pas$)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / "forest_modelA.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- 3
def fig_unites(om, n_min=20):
    g = (om.groupby(["document_set", "unite"])["ne_absent"]
           .agg(n="size", k="sum").reset_index())
    g = g[g.n >= n_min].copy()
    g["taux"] = g.k / g.n
    g["ordre"] = g.document_set.map({c: i for i, c in enumerate(ORDRE_CORPUS)})   # payeton en bas
    g = g.sort_values(["ordre", "taux"])

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for i, (_, r) in enumerate(g.iterrows()):
        lo, hi = wilson(r.k, r.n)
        ax.plot([lo * 100, hi * 100], [i, i], color=COULEURS[r.document_set], lw=1.3, alpha=.75)
        ax.plot(r.taux * 100, i, "o", color=COULEURS[r.document_set], ms=4.5)
    for c in ORDRE_CORPUS:                       # taux du corpus entier
        s = om[om.document_set == c]
        ax.axvline(s.ne_absent.mean() * 100, color=COULEURS[c], ls=":", lw=1.0, alpha=.6)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.unite, fontsize=7.5)
    ax.set_xlabel("Omission of ne (%), Wilson 95% CI")
    ax.set_xlim(-1, 52)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=COULEURS[c], lw=1.3, label=c)
                       for c in ORDRE_CORPUS],
              frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "omission_unites.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- 4
def fig_conditioning(om):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))

    ax = axes[0]                                  # negateurs, tous corpus
    stats = [(neg, len(s), int(s.ne_absent.sum()))
             for neg in ORDRE_NEG
             for s in [om[om.negateur == neg]]]
    stats.sort(key=lambda x: -(x[2] / x[1]) if x[0] != "nul" else 1)
    stats = [s for s in stats if s[0] != "nul"] + [s for s in stats if s[0] == "nul"]
    for i, (neg, n, k) in enumerate(stats):
        lo, hi = wilson(k, n)
        col = "#666666" if neg == "nul" else "#2E5A87"   # nul : trop rare pour etre classe
        ax.plot([lo * 100, hi * 100], [i, i], color=col, lw=1.4)
        ax.plot(k / n * 100, i, "o", color=col, ms=5)
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([f"{neg}  (n={n})" for neg, n, _ in stats], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Omission (%), pooled"); ax.set_title("By negator", fontsize=10)

    ax = axes[1]                                  # temps, payetoncorpus seul
    omp = om[om.document_set == "payetoncorpus"]
    etiquettes = []
    for i, (code, lab) in enumerate(ORDRE_TEMPS):
        s = omp[omp.verbe_temps == code]
        k, n = int(s.ne_absent.sum()), len(s)
        lo, hi = wilson(k, n)
        ax.plot([lo * 100, hi * 100], [i, i], color="#B03A2E", lw=1.4)
        ax.plot(k / n * 100, i, "o", color="#B03A2E", ms=5)
        etiquettes.append(f"{lab}  (n={n})")
    ax.set_yticks(range(len(ORDRE_TEMPS))); ax.set_yticklabels(etiquettes, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Omission (%), payetoncorpus"); ax.set_title("By tense", fontsize=10)

    fig.tight_layout(); fig.savefig(OUT / "conditioning_dotplot.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- 5
def fig_heatmap_tense(df):
    """Residus ajustes de Haberman : (o - e) / sqrt(e (1 - ri/N)(1 - cj/N))."""
    va = df[df.a_verbe].copy()
    va["temps"] = va["verbe_temps"].fillna("(inconnu)")
    tab = pd.crosstab(va["temps"], va["document_set"])
    tab = tab.reindex(index=["Pres", "Imp", "Past", "Fut", "(inconnu)"],
                      columns=ORDRE_CORPUS)
    chi2, p, dof, att = chi2_contingency(tab.values, correction=False)
    N = tab.values.sum()
    ri = tab.values.sum(axis=1, keepdims=True)
    cj = tab.values.sum(axis=0, keepdims=True)
    d = (tab.values - att) / np.sqrt(att * (1 - ri / N) * (1 - cj / N))

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(d, cmap="RdBu_r", vmin=-20, vmax=20, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            fort = abs(d[i, j]) > 1.96
            ax.text(j, i, f"{d[i, j]:+.1f}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold" if fort else "normal",
                    color="white" if abs(d[i, j]) > 10 else "black")
    ax.set_xticks(range(len(ORDRE_CORPUS)))
    ax.set_xticklabels(ORDRE_CORPUS, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(tab.index))); ax.set_yticklabels(tab.index, fontsize=8.5)
    ax.set_title("Verb tense × corpus (Haberman residuals)", fontsize=9.5)
    fig.colorbar(im, ax=ax, label="adjusted residual")
    fig.tight_layout(); fig.savefig(OUT / "heatmap_tense.pdf"); plt.close(fig)
    print(f"   (controle : chi2={chi2:.1f}, df={dof}, N={N})")


# --------------------------------------------------------------------------- 6
def fig_ecdf(df):
    comp = df[df.a_une_portee & (df.longueur_portee_tokens > 0)]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for c in ORDRE_CORPUS:
        x = np.sort(comp.loc[comp.document_set == c, "longueur_portee_tokens"].values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post",
                color=COULEURS[c], lw=1.5,
                label=f"{c} (n={len(x):,}, med={int(np.median(x))})")
    ax.set_xscale("log"); ax.set_xlim(1, 90)
    ax.set_xlabel("Scope length (tokens, log scale)"); ax.set_ylabel("ECDF")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "longueur_ecdf.pdf"); plt.close(fig)


if __name__ == "__main__":
    df = charger()
    om = table_omission(df)
    print(f"{len(df)} negations, dont {len(om)} dans la table d'omission")
    fig_omission_genre(om);  print(" -> figures/omission_genre.pdf")
    fig_forest();            print(" -> figures/forest_modelA.pdf")
    fig_unites(om);          print(" -> figures/omission_unites.pdf")
    fig_conditioning(om);    print(" -> figures/conditioning_dotplot.pdf")
    fig_heatmap_tense(df);   print(" -> figures/heatmap_tense.pdf")
    fig_ecdf(df);            print(" -> figures/longueur_ecdf.pdf")
