#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses citées dans le rapport et absentes de phase5_mixtes.py :

  (a) GLMM croisé + temps : ne_absent ~ corpus + negateur + temps,
      intercepts aléatoires oeuvre + verbe. Le temps indéfini (NaN, n=328)
      est conservé comme niveau propre '(inconnu)'.
      -> resultats_phase5/modele_croise_temps.csv

  (b) Sensibilité au prior : le GLMM d'omission (corpus + negateur,
      intercept oeuvre) refit avec fe_p (écart-type du prior gaussien
      des effets fixes) dans {1, 2, 3, 5}.
      -> resultats_phase5/sensibilite_prior.csv

  (c) Typologie regroupée (tableau du rapport, section « inventaire ») :
      les classes de l'annotation sont regroupées de sorte que TOUS les
      seconds éléments nus (sortie de surface de la chute du ne) forment
      une classe, distincte du ne nu (ne littéraire). La classe 'ellipse'
      d'origine confondait les deux ; 'adverbiale' mélangeait non/ni avec
      les seconds éléments nus autres que pas.
      -> resultats_phase5/typologie_regroupee.csv

  (d) LMM log-longueur refit sur cette classification regroupée
      (référence = bipartite). REMPLACE modele_B_portee.csv de phase5,
      qui utilisait type_macro brut : le rapport ne peut pas décrire une
      typologie et contrôler par une autre.
      -> resultats_phase5/modele_B_regroupe.csv

Toutes les valeurs citées dans le rapport pour ces analyses sortent d'ici.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from expletifs import EXPLETIFS_CUE_ID  # filet de sécurité, cf. phases 1-5

# Classification de surface du rapport. Source unique : utilisée par le
# tableau de l'inventaire ET par le LMM, pour qu'ils ne puissent pas diverger.
CLASSES_SURFACE = ["bipartite", "bare second element", "bare ne",
                   "negative adverb", "morphological", "prepositional",
                   "lexical", "exceptive", "other"]

_MACRO_VERS_CLASSE = {"morphologique": "morphological",
                      "prepositionnelle": "prepositional",
                      "lexicale": "lexical",
                      "exceptive": "exceptive",
                      "autre": "other"}


def classe_surface(type_fin, type_macro):
    """Classe de surface d'une négation (cf. tableau de l'inventaire)."""
    if type_macro == "bipartite":
        return "bipartite"
    if type_fin == "ne_seul":            # ne nu = ne littéraire
        return "bare ne"
    if type_fin.endswith("_seul"):       # pas/rien/jamais... nu = chute du ne
        return "bare second element"
    if type_fin in ("non", "ni"):
        return "negative adverb"
    return _MACRO_VERS_CLASSE.get(type_macro, type_macro)

PAIRES_OMISSION = {
    "pas": ("ne_pas", "pas_seul"), "plus": ("ne_plus", "plus_seul"),
    "jamais": ("ne_jamais", "jamais_seul"), "rien": ("ne_rien", "rien_seul"),
    "personne": ("ne_personne", "personne_seul"), "aucun": ("ne_aucun", "aucun_seul"),
    "point": ("ne_point", "point_seul"), "nul": ("ne_nul", "nul_seul"),
}


def _charger():
    df = pd.read_parquet("negations.parquet")
    df["cue_id"] = df["negation_id"].str.split("::").str[-1]
    df = df[~df["cue_id"].isin(EXPLETIFS_CUE_ID)]
    df["a_verbe"] = df["verbe_lemme"].fillna("") != ""
    df["oeuvre_u"] = np.where(df["sous_corpus"] == df["document_set"],
                              df["oeuvre"], df["sous_corpus"])
    df["corpus"] = pd.Categorical(df["document_set"],
                                  ["litbank", "payetoncorpus", "sequoia"])
    df["classe"] = pd.Categorical(
        [classe_surface(tf, tm) for tf, tm in zip(df["type_fin"], df["type_macro"])],
        CLASSES_SURFACE)                      # référence = bipartite
    return df


def _sous_table_omission(df):
    formes = [f for paire in PAIRES_OMISSION.values() for f in paire]
    om = df[df["type_fin"].isin(formes) & df["a_verbe"]].copy()
    om["ne_absent"] = om["type_fin"].str.endswith("_seul").astype(int)
    neg = (om["type_fin"].str.replace("^ne_", "", regex=True)
                          .str.replace("_seul", "", regex=False))
    om["negateur"] = pd.Categorical(neg, list(PAIRES_OMISSION))
    om["temps"] = pd.Categorical(om["verbe_temps"].fillna("(inconnu)"),
                                 ["Pres", "Past", "Imp", "Fut", "(inconnu)"])
    return om


def _table_or(res):
    rows = []
    for nom, mu, sd in zip(res.model.fep_names, res.fe_mean, res.fe_sd):
        rows.append({"terme": nom, "OR": np.exp(mu),
                     "ic95_bas": np.exp(mu - 1.96 * sd),
                     "ic95_haut": np.exp(mu + 1.96 * sd),
                     "log_odds": mu, "sd_post": sd})
    return pd.DataFrame(rows).round(4)


def modele_croise_temps(om):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BinomialBayesMixedGLM.from_formula(
            "ne_absent ~ corpus + negateur + temps",
            {"oeuvre": "0 + C(oeuvre_u)", "verbe": "0 + C(verbe_lemme)"}, om)
        res = m.fit_vb()
    tab = _table_or(res)
    for nom, mu in zip(res.model.vcp_names, res.vcp_mean):
        tab.loc[len(tab)] = {"terme": f"SD({nom})_logit", "OR": round(float(np.exp(mu)), 4),
                             "ic95_bas": np.nan, "ic95_haut": np.nan,
                             "log_odds": np.nan, "sd_post": np.nan}
    tab.to_csv("resultats_phase5/modele_croise_temps.csv", index=False)
    print("-> modele_croise_temps.csv")


def sensibilite_prior(om):
    lignes = []
    for fep in (1.0, 2.0, 3.0, 5.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = BinomialBayesMixedGLM.from_formula(
                "ne_absent ~ corpus + negateur",
                {"oeuvre": "0 + C(oeuvre_u)"}, om, fe_p=fep)
            res = m.fit_vb()
        d = dict(zip(res.model.fep_names, zip(res.fe_mean, res.fe_sd)))
        for terme in ("corpus[T.payetoncorpus]", "corpus[T.sequoia]"):
            mu, sd = d[terme]
            lignes.append({"prior_sd": fep, "terme": terme,
                           "OR": np.exp(mu),
                           "ic95_bas": np.exp(mu - 1.96 * sd),
                           "ic95_haut": np.exp(mu + 1.96 * sd)})
        print(f"   fe_p={fep:.0f} : OR payeton = {np.exp(d['corpus[T.payetoncorpus]'][0]):.2f}")
    pd.DataFrame(lignes).round(4).to_csv(
        "resultats_phase5/sensibilite_prior.csv", index=False)
    print("-> sensibilite_prior.csv")


def typologie_regroupee(df):
    """Tableau de l'inventaire : effectifs par classe de surface regroupée."""
    t = (df["classe"].value_counts().reindex(CLASSES_SURFACE)
         .rename_axis("classe").reset_index(name="n"))
    t["pct"] = (t["n"] / len(df) * 100).round(1)
    t.loc[len(t)] = {"classe": "TOTAL", "n": len(df), "pct": 100.0}
    t.to_csv("resultats_phase5/typologie_regroupee.csv", index=False)
    print("-> typologie_regroupee.csv")


def modele_B_regroupe(df):
    """LMM gaussien sur log(longueur), classification regroupée, réf. bipartite."""
    comp = df[df["a_une_portee"] & (df["longueur_portee_tokens"] > 0)].copy()
    comp["loglen"] = np.log(comp["longueur_portee_tokens"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm("loglen ~ corpus + classe", comp,
                          groups=comp["oeuvre_u"]).fit(reml=True)
    ci = res.conf_int()
    rows = []
    for k in res.params.index:
        if k == "Group Var":
            continue
        rows.append({"terme": k,
                     "ratio_moy_geom": np.exp(res.params[k]),
                     "ic95_bas": np.exp(ci.loc[k, 0]),
                     "ic95_haut": np.exp(ci.loc[k, 1]),
                     "coef_log": res.params[k],
                     "p_value": res.pvalues[k]})
    tab = pd.DataFrame(rows).round(4)
    tab.loc[len(tab)] = {"terme": "Var(oeuvre)",
                         "ratio_moy_geom": round(float(res.cov_re.iloc[0, 0]), 4)}
    tab.to_csv("resultats_phase5/modele_B_regroupe.csv", index=False)
    print(f"-> modele_B_regroupe.csv (N = {len(comp)})")


if __name__ == "__main__":
    df = _charger()
    typologie_regroupee(df)
    modele_B_regroupe(df)

    om = _sous_table_omission(df)
    print(f"N omission = {len(om)} (attendu : 4890)")
    modele_croise_temps(om)
    sensibilite_prior(om)