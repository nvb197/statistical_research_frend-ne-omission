# -*- coding: utf-8 -*-
"""
PHASE 5 - Modeles mixtes (FReND)
================================

POURQUOI. Les tests de la Phase 4 supposent que chaque negation est independante.
C'est faux : les negations d'une meme oeuvre (ou d'un meme blog) sont correlees
(meme auteur, meme registre). Cette violation sous-estime les erreurs-types et
gonfle les p. Les modeles mixtes ajoutent un EFFET ALEATOIRE par oeuvre qui
absorbe cette correlation intra-groupe, redonne des erreurs-types correctes, et
separe la variation "entre corpus" (ce qui nous interesse) de la variation
"entre oeuvres" (le bruit de regroupement).

VOCABULAIRE.
- Effet fixe : facteur dont on veut estimer chaque niveau et generaliser. Ici
  `corpus` (3 niveaux, references = litbank).
- Effet aleatoire : facteur de regroupement dont les niveaux sont un echantillon ;
  on modelise sa VARIANCE, pas chaque coefficient. Ici `oeuvre` (= unite : 37
  niveaux, l'oeuvre pour litbank, le sous_corpus pour payeton/sequoia).
- Random intercept : chaque oeuvre a son niveau de base propre.
- Partial pooling / shrinkage : les oeuvres a petit effectif sont tirees vers la
  moyenne generale -> gere les cas extremes (sequoia 0 %, oeuvres litbank courtes).
- ICC : part de variance due au regroupement par oeuvre (clustering fort = Phase 4
  d'autant plus trompeuse).

LES TROIS MODELES.
- Modele A (PHARE) - chute du 'ne', GENERALISEE a tous les negateurs a deux
  formes (pas, plus, jamais, rien, personne, aucun, point, nul ; ~4883 negations
  verbales). GLMM logistique bayesien avec `corpus` ET `negateur` en effets fixes
  + random oeuvre : on separe l'effet de REGISTRE (corpus, net du negateur) de la
  hierarchie de DISCORDANCE (negateur, net du registre). Le prior faible gere la
  separation (sequoia ~0 %). Sortie : OR corpus + OR negateur + IC, SD(oeuvre), ICC.
- Modele B - longueur de portee. LMM gaussien sur log(longueur), effets fixes
  corpus + type_macro, aleatoire oeuvre. exp(coef) = rapport de moyennes
  geometriques. Attendu : effet corpus ~1 apres controle -> confirme "trivial".
  (NB-GLMM = version rigoureuse, voir bloc R/brms en bas.)
- Modele C - densite de la negation. GEE Poisson, offset = log(tokens), erreurs
  robustes en grappes (cluster = oeuvre). Premier vrai test inferentiel de la
  densite. Sortie : IRR densite + IC robustes.

OUTILS. statsmodels (pur Python, aucune compilation) :
  BinomialBayesMixedGLM (A), MixedLM (B), GEE Poisson (C). Pour publication,
  refaire A/B/C sous R (lme4 / glmmTMB / brms) ; snippets fournis en commentaire.

DONNEES. negations.parquet (A, B) ; + denominateurs.parquet (C, denominateur
tokens, tous les parts y compris 0 negation).

Lancer :  python phase5_mixtes.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# ====================== PARAMETRES ======================
ORDRE_CORPUS = ["litbank", "payetoncorpus", "sequoia"]   # litbank = reference
EXCLURE_EXPLETIFS_CONNUS = True
from expletifs import EXPLETIFS_CUE_ID
# negateurs a deux formes (avec / sans 'ne') : discordance possible
PAIRES_NEG = {"pas": ("ne_pas", "pas_seul"), "plus": ("ne_plus", "plus_seul"),
              "jamais": ("ne_jamais", "jamais_seul"), "rien": ("ne_rien", "rien_seul"),
              "personne": ("ne_personne", "personne_seul"), "aucun": ("ne_aucun", "aucun_seul"),
              "point": ("ne_point", "point_seul"), "nul": ("ne_nul", "nul_seul")}
ORDRE_NEG = ["pas", "plus", "jamais", "rien", "personne", "aucun", "point", "nul"]
CHEMIN_NEG = "negations.parquet"
CHEMIN_DENOM = "denominateurs.parquet"
# ========================================================

OUT = Path("resultats_phase5"); OUT.mkdir(exist_ok=True)
ATTENDUS = ["modele_A_chute.csv", "modele_A_chute_verbe.csv", "modele_B_portee.csv", "modele_C_densite.csv"]


def _ecrire(t, nom, ecrire=True):
    if ecrire and t is not None:
        try:
            t.to_csv(OUT / nom, index=False, encoding="utf-8-sig")
        except PermissionError:
            print(f"  [!] {nom} ouvert ailleurs : ecriture ignoree.")
    return t


def charger():
    df = pd.read_parquet(CHEMIN_NEG)
    df["cue_id"] = df["negation_id"].str.split("::").str[-1]
    if EXCLURE_EXPLETIFS_CONNUS:
        df = df[~df["cue_id"].isin(EXPLETIFS_CUE_ID)].copy()
    df["a_verbe"] = df["verbe_lemme"].fillna("") != ""
    # oeuvre = unite : oeuvre pour litbank (pas de sous-corpus), sinon sous_corpus
    df["oeuvre_u"] = np.where(df["sous_corpus"] == df["document_set"],
                              df["oeuvre"], df["sous_corpus"])
    df["corpus"] = pd.Categorical(df["document_set"], ORDRE_CORPUS)
    return df


# ==========================================================================
# MODELE A - chute du 'ne' : GLMM logistique bayesien, random intercept oeuvre
# Generalise a TOUS les negateurs a deux formes (pas, plus, jamais, rien,
# personne, aucun, point, nul) ; `negateur` en effet fixe pour separer l'effet
# de REGISTRE (corpus) de la hierarchie de discordance (quel negateur).
# Lecture : OR corpus = effet registre net du negateur ; OR negateur (ref=pas)
# = la hierarchie de retention du 'ne', nette du registre.
# ==========================================================================
def modele_A_chute(df, ecrire=True):
    formes = [x for p in PAIRES_NEG.values() for x in p]
    om = df[df["type_fin"].isin(formes) & df["a_verbe"]].copy()
    om["ne_absent"] = om["type_fin"].str.endswith("_seul").astype(int)
    neg = (om["type_fin"].str.replace("^ne_", "", regex=True)
             .str.replace("_seul", "", regex=False))
    om["negateur"] = pd.Categorical(neg, ORDRE_NEG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = BinomialBayesMixedGLM.from_formula(
            "ne_absent ~ corpus + negateur", {"oeuvre": "0 + C(oeuvre_u)"}, om)
        res = mod.fit_vb()
    rows = []
    for nom, mean, sd in zip(res.model.fep_names, res.fe_mean, res.fe_sd):
        rows.append({"terme": nom, "OR": np.exp(mean),
                     "ic95_bas": np.exp(mean - 1.96 * sd), "ic95_haut": np.exp(mean + 1.96 * sd),
                     "log_odds": mean, "sd_post": sd})
    sd_oeuvre = float(np.exp(res.vcp_mean[0]))          # ecart-type aleatoire (echelle logit)
    icc = sd_oeuvre ** 2 / (sd_oeuvre ** 2 + np.pi ** 2 / 3)
    t = pd.DataFrame(rows).round(4)
    t.loc[len(t)] = {"terme": "SD(oeuvre)_logit", "OR": round(sd_oeuvre, 4)}
    t.loc[len(t)] = {"terme": "ICC_oeuvre", "OR": round(icc, 4)}
    t.attrs["n"] = len(om)
    return _ecrire(t, "modele_A_chute.csv", ecrire)


# ==========================================================================
# MODELE A v2 - chute du 'ne' avec DEUX effets aleatoires CROISES : oeuvre ET verbe.
# Motivation : le verbe est une seconde source de regroupement (effet lexical fort,
# ex. 'falloir' chute ~45 %). 'oeuvre' est EMBOITE dans corpus (chaque oeuvre = un
# corpus -> corpus reste FIXE, oeuvre ALEATOIRE), tandis que 'verbe' est CROISE avec
# 'oeuvre' (un meme verbe traverse plusieurs oeuvres) -> effets aleatoires croises
# (1|oeuvre) + (1|verbe). On NE met PAS de pente aleatoire de corpus|oeuvre : corpus
# ne varie pas A L'INTERIEUR d'une oeuvre (emboitement), donc une telle pente est
# non identifiable. Verifie que l'effet REGISTRE (OR corpus) survit a l'ajout du
# regroupement par verbe (resultat : OR payeton ~16.5 -> ~16.8, stable ; ICC verbe
# ~0.16 > ICC oeuvre ~0.06, donc le verbe regroupe MEME plus que l'oeuvre).
# ==========================================================================
def modele_A_chute_verbe(df, ecrire=True):
    formes = [x for p in PAIRES_NEG.values() for x in p]
    om = df[df["type_fin"].isin(formes) & df["a_verbe"]].copy()
    om["ne_absent"] = om["type_fin"].str.endswith("_seul").astype(int)
    neg = (om["type_fin"].str.replace("^ne_", "", regex=True)
             .str.replace("_seul", "", regex=False))
    om["negateur"] = pd.Categorical(neg, ORDRE_NEG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = BinomialBayesMixedGLM.from_formula(
            "ne_absent ~ corpus + negateur",
            {"oeuvre": "0 + C(oeuvre_u)", "verbe": "0 + C(verbe_lemme)"}, om)
        res = mod.fit_vb()
    rows = []
    for nom, mean, sd in zip(res.model.fep_names, res.fe_mean, res.fe_sd):
        rows.append({"terme": nom, "OR": np.exp(mean),
                     "ic95_bas": np.exp(mean - 1.96 * sd), "ic95_haut": np.exp(mean + 1.96 * sd),
                     "log_odds": mean, "sd_post": sd})
    # deux composantes de variance (echelle logit) ; ICC = part de chaque grappe
    sds = {nm: float(np.exp(m)) for nm, m in zip(res.model.vcp_names, res.vcp_mean)}
    denom = sum(s**2 for s in sds.values()) + np.pi**2 / 3
    t = pd.DataFrame(rows).round(4)
    for nm, s in sds.items():
        t.loc[len(t)] = {"terme": f"SD({nm})_logit", "OR": round(s, 4)}
        t.loc[len(t)] = {"terme": f"ICC_{nm}", "OR": round(s**2 / denom, 4)}
    t.attrs["n"] = len(om)
    return _ecrire(t, "modele_A_chute_verbe.csv", ecrire)


# ==========================================================================
# MODELE B - longueur de portee : LMM gaussien sur log(longueur), random oeuvre
# ==========================================================================
def modele_B_portee(df, ecrire=True):
    comp = df[df["a_une_portee"] & (df["longueur_portee_tokens"] > 0)].copy()
    comp["loglen"] = np.log(comp["longueur_portee_tokens"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm("loglen ~ corpus + C(type_macro)", comp,
                          groups=comp["oeuvre_u"]).fit(reml=True)
    rows = []
    ci = res.conf_int()
    for k in res.params.index:
        if k in ("Group Var",):
            continue
        rows.append({"terme": k, "ratio_moy_geom": np.exp(res.params[k]),
                     "ic95_bas": np.exp(ci.loc[k, 0]), "ic95_haut": np.exp(ci.loc[k, 1]),
                     "coef_log": res.params[k], "p_value": res.pvalues[k]})
    t = pd.DataFrame(rows).round(4)
    t.loc[len(t)] = {"terme": "Var(oeuvre)", "ratio_moy_geom": round(float(res.cov_re.iloc[0, 0]), 4)}
    t.attrs["n"] = len(comp)
    return _ecrire(t, "modele_B_portee.csv", ecrire)


# ==========================================================================
# MODELE C - densite : GEE Poisson, offset log(tokens), erreurs robustes (oeuvre)
# ==========================================================================
def charger_denominateurs(chemin=CHEMIN_DENOM):
    p = Path(chemin)
    if not p.exists():
        print(f"  [!] {chemin} introuvable : Modele C ignore.")
        return None
    den = pd.read_parquet(p)
    den["oeuvre_u"] = np.where(den["sous_corpus"] == den["document_set"],
                               den["oeuvre"], den["sous_corpus"])
    return den


def modele_C_densite(df, den, ecrire=True):
    if den is None:
        return None
    nn = df.groupby("document_part_id").size().rename("n_neg")
    base = den.merge(nn, on="document_part_id", how="left")
    base["n_neg"] = base["n_neg"].fillna(0).astype(int)
    base = base[base["n_tokens"] > 0].copy()
    base["log_tokens"] = np.log(base["n_tokens"])
    base["corpus"] = pd.Categorical(base["document_set"], ORDRE_CORPUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.gee("n_neg ~ corpus", groups="oeuvre_u", data=base,
                      family=sm.families.Poisson(), offset=base["log_tokens"].values,
                      cov_struct=sm.cov_struct.Independence()).fit()
    ci = res.conf_int()
    rows = [{"terme": k, "IRR": np.exp(res.params[k]),
             "ic95_bas": np.exp(ci.loc[k, 0]), "ic95_haut": np.exp(ci.loc[k, 1]),
             "coef_log": res.params[k], "p_value": res.pvalues[k]} for k in res.params.index]
    t = pd.DataFrame(rows).round(4)
    t.attrs["n_parts"] = len(base)
    return _ecrire(t, "modele_C_densite.csv", ecrire)


def lancer_tout(ecrire=True):
    df = charger()
    res = {}
    print("MODELE A (chute du 'ne') ...");  res["A"] = modele_A_chute(df, ecrire)
    print("MODELE A v2 (chute, random oeuvre + verbe) ..."); res["A_verbe"] = modele_A_chute_verbe(df, ecrire)
    print("MODELE B (longueur de portee) ..."); res["B"] = modele_B_portee(df, ecrire)
    print("MODELE C (densite) ...")
    res["C"] = modele_C_densite(df, charger_denominateurs(), ecrire)
    return res


if __name__ == "__main__":
    r = lancer_tout()
    for nom, t in r.items():
        print(f"\n===== MODELE {nom} =====")
        print("(aucune sortie : donnees manquantes)" if t is None else t.to_string(index=False))
    for f in OUT.glob("*.csv"):
        if f.name not in ATTENDUS:
            try: f.unlink()
            except OSError: pass


# ==========================================================================
# VERSION RIGOUREUSE (R) - pour publication. A executer sous R.
# --------------------------------------------------------------------------
# library(lme4); library(glmmTMB); library(brms)
# # Modele A : chute (separation -> bayesien brms recommande)
# brm(ne_absent ~ corpus + (1 | oeuvre), family = bernoulli(),
#     prior = set_prior("normal(0, 2.5)", class = "b"), data = om)
# # Modele B : longueur (binomiale negative, vrai compte surdisperse)
# glmmTMB(longueur ~ corpus + type_macro + (1 | oeuvre), family = nbinom2, data = comp)
# # Modele C : densite (Poisson/NB avec offset)
# glmmTMB(n_neg ~ corpus + (1 | oeuvre) + offset(log(n_tokens)),
#         family = nbinom2, data = base)
# ==========================================================================