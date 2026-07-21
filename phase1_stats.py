# -*- coding: utf-8 -*-
"""
PHASE 1 - Statistiques descriptives (FReND).

Produit EXACTEMENT 5 fichiers CSV (un par section), dans resultats_phase1/ :

  1. nombre_de_negations_par_corpus.csv   comptage des negations a tous les
                              niveaux (corpus > sous-corpus > oeuvre)
  2. frequence_indice.csv     frequence des formes d'indice normalisees (annexe A)
  3. densite_negation.csv     densite a tous les niveaux
  4. distribution_types.csv   types d'indice (macro + fin reunis)
  5. longueur_portee.csv      longueur des portees par type macro

Principe : chaque section = UN tableau = UN fichier. Les niveaux hierarchiques
sont fusionnes dans un seul tableau grace a la colonne 'niveau'
(TOTAL / corpus / sous_corpus / oeuvre). NE PAS sommer les lignes entre elles :
ce sont des agregations emboitees (une ligne corpus = somme de ses sous_corpus).

Chaque fonction accepte ecrire=True : passer ecrire=False pour seulement calculer
le tableau sans ecrire de CSV (utile dans un notebook, evite tout verrou de
fichier ouvert dans Excel).

Lancer :  python phase1_stats.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from expletifs import EXPLETIFS_CUE_ID  # source unique

OUT = Path("resultats_phase1"); OUT.mkdir(exist_ok=True)


def _ecrire(t, nom):
    """Ecrit le CSV ; si le fichier est ouvert ailleurs (Excel), previent sans planter."""
    try:
        t.to_csv(OUT / nom, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(f"  [!] {nom} est ouvert ailleurs (Excel ?) : ecriture ignoree, fermez-le puis relancez.")
    return t


def charger(path="negations.parquet"):
    """Charge la table maitre. Le parquet preserve les types : pas de piege 'nan'."""
    df = pd.read_parquet(path)
    # Exclusion des 'ne' expletifs confirmes (coherence avec les phases 2-4).
    df["cue_id"] = df["negation_id"].str.split("::").str[-1]
    df = df[~df["cue_id"].isin(EXPLETIFS_CUE_ID)].copy()
    print(f"{len(df)} negations (expletifs exclus), {df.shape[1]} colonnes")
    return df


def _hierarchie(df, statfn):
    """
    Applique statfn (qui renvoie un dict de mesures) a 4 niveaux emboites et
    empile le tout dans un seul tableau, repere par la colonne 'niveau' :
    TOTAL > corpus (document_set) > sous_corpus > oeuvre.
    """
    rows = [{"niveau": "TOTAL", "document_set": "", "sous_corpus": "", "oeuvre": "", **statfn(df)}]
    for ds, g1 in df.groupby("document_set"):
        rows.append({"niveau": "corpus", "document_set": ds, "sous_corpus": "", "oeuvre": "", **statfn(g1)})
        for sc, g2 in g1.groupby("sous_corpus"):
            rows.append({"niveau": "sous_corpus", "document_set": ds, "sous_corpus": sc, "oeuvre": "", **statfn(g2)})
            for oe, g3 in g2.groupby("oeuvre"):
                rows.append({"niveau": "oeuvre", "document_set": ds, "sous_corpus": sc, "oeuvre": oe, **statfn(g3)})
    return pd.DataFrame(rows)


# 1. NOMBRE DE NEGATIONS PAR CORPUS (Tableau 1 etendu)
def nombre_negations_par_corpus(df, ecrire=True):
    """Comptage des negations a chaque niveau (total, avec portee, sans portee)."""
    def cnt(sub):
        neg, avec = len(sub), int(sub["a_une_portee"].sum())
        return {"negations": neg, "avec_portee": avec, "sans_portee": neg - avec}
    t = _hierarchie(df, cnt)
    if ecrire: _ecrire(t, "nombre_de_negations_par_corpus.csv")
    return t


# 2. FREQUENCE DES FORMES
def frequence_indice(df, col="indice_norm", ecrire=True):
    """Frequence de chaque forme normalisee d'indice (annexe A), avec pourcentage."""
    s = df[col].value_counts()
    t = s.rename_axis("forme").reset_index(name="frequence")
    t["pct"] = (t["frequence"] / len(df) * 100).round(2)
    if ecrire: _ecrire(t, "frequence_indice.csv")
    return t


# 3. DENSITE
def _agg_niveaux(frame, statfn):
    """Comme _hierarchie mais sur une table quelconque (numerateurs OU denominateurs)."""
    rows = [{"niveau": "TOTAL", "document_set": "", "sous_corpus": "", "oeuvre": "", **statfn(frame)}]
    for ds, g1 in frame.groupby("document_set"):
        rows.append({"niveau": "corpus", "document_set": ds, "sous_corpus": "", "oeuvre": "", **statfn(g1)})
        for sc, g2 in g1.groupby("sous_corpus"):
            rows.append({"niveau": "sous_corpus", "document_set": ds, "sous_corpus": sc, "oeuvre": "", **statfn(g2)})
            for oe, g3 in g2.groupby("oeuvre"):
                rows.append({"niveau": "oeuvre", "document_set": ds, "sous_corpus": sc, "oeuvre": oe, **statfn(g3)})
    return pd.DataFrame(rows)


def densite(df, ecrire=True, den_path="denominateurs.parquet",
            phr_path="phrases_par_document.parquet"):
    """
    Densite a chaque niveau (negations / taille de texte).

    Deux denominateurs, deux grains (cf. denominateurs.py) :
      * TOKENS / MOTS : depuis denominateurs.parquet (grain DocumentPart). Le
        tokenizer est local -> la somme per-part egale le total direct, et la
        densite neg/1000 tokens correspond a la reference.
      * PHRASES : depuis phrases_par_document.parquet (grain Document). Sommer les
        phrases per-part sur-compterait les phrases coupees par un bord de part
        (artefact brat). Le compte per-document est le bon denominateur en phrases.
        On agrege ce fichier dans la MEME hierarchie : la colonne oeuvre y suit la
        meme regle, donc corpus = somme de ses sous_corpus = somme de ses oeuvres.

    La densite couvre TOUS les DocumentParts (y compris ceux SANS negation), d'ou
    la jointure numerateurs (negations) x denominateurs. NB : phrases exactes
    seulement si les parquets ont ete construits avec fr_core_news_md.
    """
    if not Path(den_path).exists():
        raise FileNotFoundError(
            f"{den_path} introuvable. Lancez d'abord 'python denominateurs.py' "
            "(construit les denominateurs sur TOUS les parts depuis le XML).")
    den = pd.read_parquet(den_path)
    num = _agg_niveaux(df, lambda s: {"negations": len(s)})
    # tokens / mots : grain part
    dde = _agg_niveaux(den, lambda s: {"mots": int(s["n_mots"].sum()),
                                       "tokens": int(s["n_tokens"].sum()),
                                       "parts": int(s["document_part_id"].nunique())})
    # phrases : grain document (phrases_par_document.parquet). Repli sur la somme
    # per-part si le fichier manque (md pas encore relance), avec avertissement.
    if Path(phr_path).exists():
        phr = pd.read_parquet(phr_path)
        dphr = _agg_niveaux(phr, lambda s: {"phrases": int(s["n_phrases"].sum())})
    else:
        print(f"  [!] {phr_path} introuvable : phrases comptees per-part (sur-comptage "
              "possible sur litbank). Relancez 'python denominateurs.py' avec fr_core_news_md.")
        dphr = _agg_niveaux(den, lambda s: {"phrases": int(s["n_phrases"].sum())})
    cle = ["niveau", "document_set", "sous_corpus", "oeuvre"]
    dde = dde.merge(dphr, on=cle, how="left")
    t = dde.merge(num, on=cle, how="left")
    t["negations"] = t["negations"].fillna(0).astype(int)
    # Division protegee : densite indefinie (NaN) si le denominateur est nul, au lieu
    # d'inf + RuntimeWarning. Aucun niveau n'est a 0 sur le corpus actuel, mais ceci
    # securise une oeuvre / un sous-corpus vide eventuel (et couvre tokens & phrases,
    # pas seulement mots).
    t["neg_1000_mots"] = (t["negations"] / t["mots"].replace(0, np.nan) * 1000).round(2)
    t["neg_1000_tokens"] = (t["negations"] / t["tokens"].replace(0, np.nan) * 1000).round(2)
    t["neg_par_phrase"] = (t["negations"] / t["phrases"].replace(0, np.nan)).round(3)
    t = t[cle + ["negations", "mots", "tokens", "phrases", "parts",
                 "neg_1000_mots", "neg_1000_tokens", "neg_par_phrase"]]
    if ecrire: _ecrire(t, "densite_negation.csv")
    return t


# 4. DISTRIBUTION DES TYPES
def distribution_types(df, ecrire=True):
    """
    type_macro (sous-total) + type_fin (detail) reunis dans un seul tableau.
    Le pct est calcule sur le TOTAL des negations : les pct des lignes 'fin'
    d'un meme macro se somment au pct de la ligne 'macro'.
    """
    rows = []
    for macro in df["type_macro"].value_counts(dropna=False).index:
        gm = df[df["type_macro"] == macro]
        rows.append({"niveau": "macro", "type_macro": macro, "type_fin": "",
                     "n": len(gm), "pct": round(len(gm) / len(df) * 100, 2)})
        for fin, n in gm["type_fin"].value_counts(dropna=False).items():
            rows.append({"niveau": "fin", "type_macro": macro, "type_fin": fin,
                         "n": int(n), "pct": round(n / len(df) * 100, 2)})
    t = pd.DataFrame(rows)
    if ecrire: _ecrire(t, "distribution_types.csv")
    return t


# 5. LONGUEUR DES PORTEES
def longueur_portee(df, ecrire=True):
    """
    Longueur de portee (en tokens) par type_macro, sur les seules negations
    ayant une portee. mediane ET moyenne : la distribution est asymetrique a
    droite, donc la mediane est plus representative que la moyenne.
    """
    # garde-fou : longueur 0 avec portee = incoherence d'annotation (indice ayant
    # "avale" la portee) -> exclu des stats de longueur, mais reste compte comme "avec portee".
    comp = df[df["a_une_portee"] & (df["longueur_portee_tokens"] > 0)]
    g = (comp.groupby("type_macro")["longueur_portee_tokens"]
             .agg(n="size", mediane="median", moyenne="mean",
                  q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75),
                  maxi="max").round(2).reset_index())
    s = comp["longueur_portee_tokens"]
    tous = pd.DataFrame([{"type_macro": "TOUS", "n": len(s),
                          "mediane": round(s.median(), 2), "moyenne": round(s.mean(), 2),
                          "q1": round(s.quantile(.25), 2), "q3": round(s.quantile(.75), 2),
                          "maxi": int(s.max())}])
    t = pd.concat([tous, g.sort_values("moyenne", ascending=False)], ignore_index=True)
    if ecrire: _ecrire(t, "longueur_portee.csv")
    return t


# Les 5 seuls CSV que cette phase doit produire (sert aussi au nettoyage).
ATTENDUS = ["nombre_de_negations_par_corpus.csv", "frequence_indice.csv",
            "densite_negation.csv", "distribution_types.csv", "longueur_portee.csv"]


if __name__ == "__main__":
    df = charger()
    nombre_negations_par_corpus(df)
    frequence_indice(df)
    densite(df)
    distribution_types(df)
    longueur_portee(df)

    # Nettoyage : supprime les CSV obsoletes (renommages / versions precedentes)
    # pour ne laisser QUE les 5 fichiers attendus dans resultats_phase1/.
    for f in OUT.glob("*.csv"):
        if f.name not in ATTENDUS:
            try:
                f.unlink()
                print(f"  [nettoye] fichier obsolete supprime : {f.name}")
            except OSError as e:
                print(f"  [!] impossible de supprimer {f.name} : {e}")

    print(f"\n{len(ATTENDUS)} fichiers attendus dans {OUT} :")
    for nom in ATTENDUS:
        etat = "ok" if (OUT / nom).exists() else "MANQUANT (ouvert dans Excel ?)"
        print(f"  - {nom}  [{etat}]")