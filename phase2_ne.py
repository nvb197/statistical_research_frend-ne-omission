# -*- coding: utf-8 -*-
"""
PHASE 2 - Approfondissement du 'ne' (FReND).

Tableaux croises : les 3 corpus en COLONNES + une colonne TOTAL (et une ligne
TOTAL). Sorties dans resultats_phase2/ :
  1. chute_ne.csv               taux de chute du 'ne' GENERALISE aux 8 negateurs a
                                deux formes (pas, plus, jamais, rien, personne, aucun,
                                point, nul) + detail 'pas', par corpus + TOTAL
  2. ne_seul_classe.csv         'ne' seuls par classe x corpus + TOTAL
  3. ne_seul_detail_verbe.csv   'ne' seuls par verbe x corpus + TOTAL
  4. variantes_second_element.csv  paradigme du negateur (TOUTES formes, avec/sans
                                'ne') x corpus + TOTAL (et non le seul bipartite :
                                sinon les formes sans 'ne' seraient exclues)
  5. elision.csv                n' / ne x corpus + TOTAL (elision phonologiquement
                                conditionnee par le phoneme suivant, pas un marqueur
                                de registre : a lire avec prudence)

Tout est calcule depuis negations.parquet (aucun nombre en dur). Apres correction
du XML + relance de build, il suffit de relancer ce script.

Lancer :  python phase2_ne.py
"""
import pandas as pd
from pathlib import Path

# ====================== PARAMETRES ======================
MODALES_LITTERAIRES = {"pouvoir", "savoir", "cesser", "oser"}
# negateurs du paradigme (forme avec/sans 'ne') et paires a deux formes (chute)
NEG_PARADIGME = ["pas", "plus", "jamais", "rien", "personne", "aucun", "point", "nul", "guere", "ni"]
PAIRES_OMISSION = {"pas": ("ne_pas", "pas_seul"), "plus": ("ne_plus", "plus_seul"),
                   "jamais": ("ne_jamais", "jamais_seul"), "rien": ("ne_rien", "rien_seul"),
                   "personne": ("ne_personne", "personne_seul"), "aucun": ("ne_aucun", "aucun_seul"),
                   "point": ("ne_point", "point_seul"), "nul": ("ne_nul", "nul_seul")}
FORMES_OMISSION = [x for pr in PAIRES_OMISSION.values() for x in pr]
# corrections de lemmes (md mal-lemmatise quelques formes archaiques/litteraires)
NORMALISATION_LEMME = {"ose": "oser", "pouvoit": "pouvoir", "donne": "donner", "porte": "porter"}
EXCLURE_EXPLETIFS_CONNUS = True
from expletifs import EXPLETIFS_CUE_ID  # source unique (coherence avec phases 1, 3, 4)
# Ordre d'affichage des corpus (noms reels, pas d'etiquette de registre)
ORDRE_CORPUS = ["payetoncorpus", "litbank", "sequoia"]
# ========================================================

OUT = Path("resultats_phase2"); OUT.mkdir(exist_ok=True)
ATTENDUS = ["chute_ne.csv", "ne_seul_classe.csv", "ne_seul_detail_verbe.csv",
            "variantes_second_element.csv", "elision.csv"]


def _ecrire(t, nom, ecrire):
    if not ecrire:
        return t
    try:
        t.to_csv(OUT / nom, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(f"  [!] {nom} ouvert ailleurs (Excel ?) : ecriture ignoree.")
    return t


def charger():
    df = pd.read_parquet("negations.parquet")
    df["cue_id"] = df["negation_id"].str.split("::").str[-1]
    if EXCLURE_EXPLETIFS_CONNUS:
        df = df[~df["cue_id"].isin(EXPLETIFS_CUE_ID)].copy()
    df["a_verbe"] = df["verbe_lemme"].fillna("") != ""
    return df


def _croise(sub, cat, nom, ecrire, trier=True):
    """Tableau croise : <cat> en lignes, corpus + TOTAL en colonnes (+ ligne TOTAL)."""
    t = pd.crosstab(sub[cat], sub["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    t["TOTAL"] = t.sum(axis=1)
    if trier:
        t = t.sort_values("TOTAL", ascending=False)
    t.loc["TOTAL"] = t.sum(axis=0)
    t = t.reset_index()
    return _ecrire(t, nom, ecrire)


# 1. CHUTE DU 'NE' : corpus en colonnes + TOTAL ; mesures en lignes
def chute_ne(df, ecrire=True):
    sub = df[df["type_fin"].isin(FORMES_OMISSION) & df["a_verbe"]].copy()
    sub["ne_absent"] = sub["type_fin"].str.endswith("_seul")
    subp = sub[sub["type_fin"].isin(["ne_pas", "pas_seul"])]
    cols = {}
    for corp in ORDRE_CORPUS + ["TOTAL"]:
        s = sub if corp == "TOTAL" else sub[sub["document_set"] == corp]
        sp = subp if corp == "TOTAL" else subp[subp["document_set"] == corp]
        nep = int((~s["ne_absent"]).sum()); ab = int(s["ne_absent"].sum()); tot = nep + ab
        taux = round(ab / tot * 100, 2) if tot else 0
        taux_pas = round((sp["type_fin"] == "pas_seul").sum() / len(sp) * 100, 2) if len(sp) else 0
        cols[corp] = [nep, ab, tot, taux, taux_pas]
    t = (pd.DataFrame(cols, index=["ne_present", "ne_absent", "total",
                                   "taux_chute_pct", "taux_chute_pas_pct"])
           .reset_index().rename(columns={"index": "mesure"}))
    return _ecrire(t, "chute_ne.csv", ecrire)


# 2. 'NE' SEUL par classe
def ne_seul_classe(df, ecrire=True):
    ns = df[df["type_fin"] == "ne_seul"].copy()
    ns["verbe_lemme"] = ns["verbe_lemme"].replace(NORMALISATION_LEMME)
    ns["classe"] = ns["verbe_lemme"].apply(
        lambda v: "litteraire_modal" if v in MODALES_LITTERAIRES
        else ("verbe_absent" if not isinstance(v, str) or v == "" else "autre_verbe"))
    return _croise(ns, "classe", "ne_seul_classe.csv", ecrire)


# 3. 'NE' SEUL par verbe
def ne_seul_detail_verbe(df, ecrire=True):
    ns = df[df["type_fin"] == "ne_seul"].copy()
    ns["verbe_lemme"] = ns["verbe_lemme"].replace(NORMALISATION_LEMME)
    ns["verbe_lemme"] = ns["verbe_lemme"].fillna("(absent)").replace("", "(absent)")
    return _croise(ns, "verbe_lemme", "ne_seul_detail_verbe.csv", ecrire)


# 4. VARIANTES DU SECOND ELEMENT (bipartite)
def variantes_second_element(df, ecrire=True):
    neg = df["type_fin"].astype(str).str.replace("^ne_", "", regex=True).str.replace("_seul", "", regex=False)
    sub = df.assign(second=neg.where(neg.isin(NEG_PARADIGME)))
    sub = sub[sub["second"].notna()]
    return _croise(sub, "second", "variantes_second_element.csv", ecrire)


# 5. ELISION n' / ne
def elision(df, ecrire=True):
    ne = df[df["indice_norm"].str.startswith("ne")].copy()
    brut = ne["indice_brut"].fillna("")
    elid = brut.str.contains("n'", regex=False) | brut.str.contains("n\u2019", regex=False)
    ne["forme"] = elid.map({True: "elidee (n')", False: "pleine (ne)"})
    return _croise(ne, "forme", "elision.csv", ecrire, trier=False)


if __name__ == "__main__":
    df = charger()
    print(f"{len(df)} negations retenues\n")
    print("=== 1. Chute du 'ne' ===");                  print(chute_ne(df).to_string(index=False))
    print("\n=== 2. 'ne' seul par classe ===");          print(ne_seul_classe(df).to_string(index=False))
    print("\n=== 4. Variantes du second element ===");   print(variantes_second_element(df).to_string(index=False))
    print("\n=== 5. Elision ===");                       print(elision(df).to_string(index=False))
    ne_seul_detail_verbe(df)
    for f in OUT.glob("*.csv"):
        if f.name not in ATTENDUS:
            try: f.unlink()
            except OSError: pass
    print("\nFichiers dans", OUT, ":", ", ".join(ATTENDUS))