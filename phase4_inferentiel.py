# -*- coding: utf-8 -*-
"""
PHASE 4 - Inference statistique + omission approfondie (FReND), aux deux niveaux : CORPUS et UNITE
(niveau juste sous le corpus), reunis dans chaque fichier.

UNITE = sous_corpus, SAUF litbank (pas de sous-decoupage) ou l'on prend l'OEUVRE :
payetoncorpus -> 14 sous_corpus, sequoia -> 5 sous_corpus, litbank -> 18 oeuvres.
Une colonne 'corpus' (corpus parent) accompagne chaque unite pour le suivi.

Sorties dans resultats_phase4/ :
  0. densite_tests.csv      densite de la negation : taux pour 1000 tokens par
                            groupe (corpus + unite) + IC Poisson exact ; ratio de
                            taux (IRR) entre CORPUS + IC. Joint negations.parquet
                            (numerateur) et denominateurs.parquet (parts, denom).
  1. tests_homogeneite.csv   chi2 + Cramer V (IC bootstrap) + Fisher MC, par
                             (croisement x niveau)
  2. residus_significatifs.csv  residus ajustes + BH + Holm (colonnes niveau, corpus),
                             correction par famille = (croisement x niveau)
  3. chute_ne_tests.csv      omission du 'ne' : taux par groupe (corpus + unite),
                             omnibus par niveau, paires de CORPUS (OR + Fisher + Holm).
                             Colonnes 'type_ligne', 'niveau', 'corpus'.
  --- VOLET OMISSION APPROFONDI (ex-phase4b), depuis negations.parquet ---
  5. discordance_negateur.csv  chute du 'ne' par negateur (hierarchie de discordance)
                             + IC de Wilson + chi2/Cramer V.
  6. chute_corpus.csv        chute generalisee (8 negateurs) par corpus + IC Wilson,
                             avec rappel 'pas seul' pour comparaison.
  7. chute_verbe.csv         chute par verbe (effet lexical/frequence) + IC Wilson.
  8. ne_litteraire.csv       'ne' seul (ne litteraire) par corpus et par verbe porteur.
  9. proprietes_diverses.csv  ne...que, negation imbriquee (concord), negation sans verbe.

  4. kruskal_dunn.csv        longueur de portee : medianes par groupe (corpus + unite),
                             Kruskal-Wallis + eta2_H par niveau, Dunn entre CORPUS, et
                             Hodges-Lehmann (decalage median + IC distribution-free) par
                             paire de CORPUS -> rend rigoureux le verdict "effet trivial".

Les comparaisons PAR PAIRE restent au niveau CORPUS (facteur d'etude, 3 paires) :
au niveau unite elles exploseraient en dizaines de paires instables. L'unite est
couverte en DESCRIPTIF (taux / medianes) + OMNIBUS.

LIMITE : negations d'une meme oeuvre non independantes -> tests MARGINAUX ;
effets aleatoires par oeuvre = Phase 5. La valeur de retour en memoire reste au
niveau CORPUS (pour les notebooks) ; les CSV portent corpus + unite.

Lancer :  python phase4_inferentiel.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import chi2_contingency, norm, kruskal, rankdata, fisher_exact

# ====================== PARAMETRES ======================
ORDRE_CORPUS = ["payetoncorpus", "litbank", "sequoia"]
EXCLURE_EXPLETIFS_CONNUS = True
from expletifs import EXPLETIFS_CUE_ID  # source unique (voir expletifs.py)
ALPHA = 0.05
# second element = paradigme du negateur (forme avec OU sans 'ne'), aligne Phase 3/4b
NEG_PARADIGME = ["pas", "plus", "jamais", "rien", "personne", "aucun", "point", "nul", "guere", "ni"]
# chute du 'ne' generalisee : 8 negateurs a deux formes (aligne Phase 3/4b/5)
PAIRES_OMISSION = {"pas": ("ne_pas", "pas_seul"), "plus": ("ne_plus", "plus_seul"),
                   "jamais": ("ne_jamais", "jamais_seul"), "rien": ("ne_rien", "rien_seul"),
                   "personne": ("ne_personne", "personne_seul"), "aucun": ("ne_aucun", "aucun_seul"),
                   "point": ("ne_point", "point_seul"), "nul": ("ne_nul", "nul_seul")}
FORMES_OMISSION = [x for p in PAIRES_OMISSION.values() for x in p]
TOP_VERBES = 12  # nb de verbes les plus frequents pour la chute par verbe
B_BOOTSTRAP = 2000
MIN_OCC_VERBE = 5  # seuil d'occurrences par verbe (chute par verbe payeton)
MODAUX = {"pouvoir", "devoir", "vouloir", "savoir", "falloir"}  # type_verbe = modal
MORPHO_COLS = ["verbe_pos", "verbe_temps", "verbe_mode"]  # ajoutees au build (relancer avec md)
# ========================================================

OUT = Path("resultats_phase4"); OUT.mkdir(exist_ok=True)
ATTENDUS = ["densite_tests.csv", "tests_homogeneite.csv", "residus_significatifs.csv",
            "chute_ne_tests.csv", "kruskal_dunn.csv",
            # --- volet omission approfondi (ex-phase4b) ---
            "discordance_negateur.csv", "chute_corpus.csv", "chute_verbe.csv",
            "ne_litteraire.csv", "proprietes_diverses.csv",
            # --- volet verbe (ex-phase6 fusionne ici) ---
            "chute_verbe_payeton.csv", "chute_morpho_payeton.csv", "verbes_imbriquees.csv",
            "mode_detail.csv"]
CHEMIN_DENOM = "denominateurs.parquet"


def _ecrire(t, nom, ecrire=True):
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
    df["type_macro2"] = df["type_macro"]
    neg = df["type_fin"].astype(str).str.replace("^ne_", "", regex=True).str.replace("_seul", "", regex=False)
    df["second"] = neg.where(neg.isin(NEG_PARADIGME))
    df["a_verbe"] = df["verbe_lemme"].fillna("") != ""
    df["unite"] = np.where(df["sous_corpus"] == df["document_set"], df["oeuvre"], df["sous_corpus"])
    return df


def _parent_map(df):
    return df.drop_duplicates("unite").set_index("unite")["document_set"].to_dict()


# ---------- corrections multiples ----------
def p_adjust(pvals, methode):
    p = np.asarray(pvals, float); m = len(p); ordre = np.argsort(p)
    adj = np.empty(m)
    if methode == "holm":
        cummax = 0.0
        for rang, i in enumerate(ordre):
            val = (m - rang) * p[i]; cummax = max(cummax, val); adj[i] = min(cummax, 1.0)
    elif methode == "bh":
        cummin = 1.0
        for rang in range(m - 1, -1, -1):
            i = ordre[rang]; val = p[i] * m / (rang + 1); cummin = min(cummin, val); adj[i] = min(cummin, 1.0)
    return adj


def _cramer_corrige(chi2, N, r, c):
    """Cramer V corrige du biais (Bergsma 2013)."""
    phi2 = chi2 / N
    phi2t = max(0.0, phi2 - (r - 1) * (c - 1) / (N - 1))
    rt = r - (r - 1) ** 2 / (N - 1); ct = c - (c - 1) ** 2 / (N - 1)
    m = min(rt - 1, ct - 1)
    return np.sqrt(phi2t / m) if m > 0 else np.nan


def _cramer(table):
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum(); k = min(table.shape)
    return np.sqrt(chi2 / (n * (k - 1))) if k > 1 and n > 0 else np.nan


def cramer_ic(cat, corp, B=B_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    cat = np.asarray(cat); corp = np.asarray(corp); n = len(cat); vs = []
    k_plein = min(len(np.unique(cat)), len(np.unique(corp)))
    if k_plein < 2:
        return (np.nan, np.nan)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        t = pd.crosstab(pd.Series(cat[idx]), pd.Series(corp[idx])).values
        nn = t.sum()
        if min(t.shape) > 1 and nn > 0:
            chi2 = chi2_contingency(t, correction=False)[0]
            vs.append(np.sqrt(chi2 / (nn * (k_plein - 1))))
    return tuple(np.nanpercentile(vs, [2.5, 97.5]))


def _fisher_mc(serie_lig, serie_col, n_iter=10000, seed=0):
    cl, _ = pd.factorize(np.asarray(serie_lig)); cc, _ = pd.factorize(np.asarray(serie_col))
    R, C, N = cl.max() + 1, cc.max() + 1, len(cl)
    obs = np.bincount(cl * C + cc, minlength=R * C).reshape(R, C).astype(float)
    exp = np.outer(obs.sum(1), obs.sum(0)) / N
    chi0 = ((obs - exp) ** 2 / exp).sum()
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_iter):
        t = np.bincount(rng.permutation(cl) * C + cc, minlength=R * C).reshape(R, C)
        if ((t - exp) ** 2 / exp).sum() >= chi0 - 1e-9:
            cnt += 1
    return chi0, (cnt + 1) / (n_iter + 1)


# ==========================================================================
# EN MEMOIRE : niveau CORPUS (pour les notebooks ; inchange)
# ==========================================================================
def homogeneite(sub, col, nom):
    obs = pd.crosstab(sub[col], sub["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    arr = obs.values
    chi2, p, dof, exp = chi2_contingency(arr, correction=False)
    V = _cramer(arr); lo, hi = cramer_ic(sub[col].values, sub["document_set"].values)
    _, p_ffh = _fisher_mc(sub[col], sub["document_set"])
    return {"croisement": nom, "N": int(arr.sum()), "modalites": arr.shape[0],
            "chi2": round(chi2, 1), "ddl": dof, "p_chi2_pearson": p, "p_fisher_mc": p_ffh,
            "cramer_v": round(V, 3), "v_ic95_bas": round(lo, 3), "v_ic95_haut": round(hi, 3),
            "pct_cellules_attendu_lt5": round((exp < 5).mean() * 100, 1)}


def residus_significatifs(sub, col, nom):
    obs = pd.crosstab(sub[col], sub["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    arr = obs.values.astype(float)
    _, _, _, exp = chi2_contingency(arr, correction=False)
    N = arr.sum(); row = arr.sum(1, keepdims=True); colm = arr.sum(0, keepdims=True)
    adj = (arr - exp) / np.sqrt(exp * (1 - row / N) * (1 - colm / N))
    p = 2 * (1 - norm.cdf(np.abs(adj)))
    rows = []
    for i, lig in enumerate(obs.index):
        for j, cc in enumerate(ORDRE_CORPUS):
            rows.append({"croisement": nom, "modalite": lig, "corpus": cc,
                         "residu_ajuste": round(adj[i, j], 2), "p_brut": p[i, j]})
    t = pd.DataFrame(rows)
    t["p_bh"] = p_adjust(t["p_brut"], "bh"); t["p_holm"] = p_adjust(t["p_brut"], "holm")
    t["signif_bh"] = t["p_bh"] < ALPHA
    t["sens"] = np.where(t["residu_ajuste"] > 0, "sur-represente", "sous-represente")
    return t.sort_values("p_brut").reset_index(drop=True)


def _paires_or(om):
    paires = []
    for a, b in combinations(ORDRE_CORPUS, 2):
        s = om[om["document_set"].isin([a, b])]
        pa = int(((s.document_set == a) & ~s.ne_absent).sum())
        ab = int(((s.document_set == a) & s.ne_absent).sum())
        pb = int(((s.document_set == b) & ~s.ne_absent).sum())
        bb = int(((s.document_set == b) & s.ne_absent).sum())
        _, pf = fisher_exact([[ab, pa], [bb, pb]])
        a_, b_, c_, d_ = ab + .5, pa + .5, bb + .5, pb + .5
        OR = (a_ / b_) / (c_ / d_); se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
        lo, hi = np.exp(np.log(OR) - 1.96 * se), np.exp(np.log(OR) + 1.96 * se)
        paires.append({"comparaison": f"{a} vs {b}", "n": len(s),
                       "odds_ratio": round(OR, 2), "or_ic95_bas": round(lo, 2),
                       "or_ic95_haut": round(hi, 2), "p_fisher": pf})
    return paires


def chute_ne_tests(df):
    om = df[df["type_fin"].isin(FORMES_OMISSION) & df["a_verbe"]].copy()
    om["ne_absent"] = om["type_fin"].str.endswith("_seul")
    obs = pd.crosstab(om["ne_absent"], om["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    chi2, p, dof, _ = chi2_contingency(obs.values, correction=False); V = _cramer(obs.values)
    lignes = [{"comparaison": "GLOBAL (3 corpus)", "n": int(obs.values.sum()),
               "taux_absent_pct": round(om["ne_absent"].mean() * 100, 2),
               "chi2": round(chi2, 1), "ddl": dof, "p_value": p, "cramer_v": round(V, 3),
               "odds_ratio": np.nan, "or_ic95_bas": np.nan, "or_ic95_haut": np.nan, "p_fisher": np.nan}]
    paires = _paires_or(om)
    for x, pad in zip(paires, p_adjust([x["p_fisher"] for x in paires], "holm")):
        x["p_fisher_holm"] = pad
        x.update({"taux_absent_pct": np.nan, "chi2": np.nan, "ddl": np.nan,
                  "p_value": np.nan, "cramer_v": np.nan})
    return pd.DataFrame(lignes + paires)


def _kw_stats(groupes):
    H, p = kruskal(*groupes)
    allv = np.concatenate(groupes); N = len(allv); k = len(groupes)
    eta2_H = (H - k + 1) / (N - k)
    ranks = rankdata(allv)
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    sigma2 = (N * (N + 1) / 12) - tie / (12 * (N - 1))
    return H, p, eta2_H, ranks, sigma2, N, k


def kruskal_dunn(df):
    comp = df[df["a_une_portee"] & (df["longueur_portee_tokens"] > 0)]
    groupes = [comp.loc[comp.document_set == k, "longueur_portee_tokens"].values for k in ORDRE_CORPUS]
    H, p, eta2_H, ranks, sigma2, N, k = _kw_stats(groupes)
    idx = 0; Rbar = {}; n = {}
    for lab, g in zip(ORDRE_CORPUS, groupes):
        Rbar[lab] = ranks[idx:idx + len(g)].mean(); n[lab] = len(g); idx += len(g)
    lignes = [{"comparaison": "GLOBAL (Kruskal-Wallis)", "H": round(H, 2), "ddl": k - 1,
               "p_value": p, "eta2_H": round(eta2_H, 4), "z": np.nan, "p_dunn": np.nan, "p_dunn_holm": np.nan}]
    dunn = []
    for a, b in combinations(ORDRE_CORPUS, 2):
        se = np.sqrt(sigma2 * (1 / n[a] + 1 / n[b])); z = (Rbar[a] - Rbar[b]) / se
        pz = 2 * (1 - norm.cdf(abs(z)))
        dunn.append({"comparaison": f"{a} vs {b}", "H": np.nan, "ddl": np.nan, "p_value": np.nan,
                     "eta2_H": np.nan, "z": round(z, 2), "p_dunn": pz, "p_dunn_holm": np.nan})
    for x, pad in zip(dunn, p_adjust([d["p_dunn"] for d in dunn], "holm")):
        x["p_dunn_holm"] = pad
    return pd.DataFrame(lignes + dunn)


# ==========================================================================
# CSV COMBINES : corpus + unite (colonnes 'niveau', 'corpus')
# ==========================================================================
def _homog_niveau(sub, col, nom, grp, ordre, niveau):
    obs = pd.crosstab(sub[col], sub[grp])
    if ordre is not None:
        obs = obs.reindex(columns=ordre, fill_value=0)
    arr = obs.values
    chi2, p, dof, exp = chi2_contingency(arr, correction=False)
    V = _cramer(arr); lo, hi = cramer_ic(sub[col].values, sub[grp].values)
    _, p_ffh = _fisher_mc(sub[col], sub[grp])
    return {"croisement": nom, "niveau": niveau, "N": int(arr.sum()),
            "modalites": arr.shape[0], "n_groupes": arr.shape[1],
            "chi2": round(chi2, 1), "ddl": dof, "p_chi2_pearson": p, "p_fisher_mc": p_ffh,
            "cramer_v": round(V, 3), "cramer_v_corrige": round(_cramer_corrige(chi2, arr.sum(), *arr.shape), 3),
            "v_ic95_bas": round(lo, 3), "v_ic95_haut": round(hi, 3),
            "pct_cellules_attendu_lt5": round((exp < 5).mean() * 100, 1)}


def homogeneite_combine(df, bp, ecrire=True):
    rows = [
        _homog_niveau(df, "type_macro2", "type_macro", "document_set", ORDRE_CORPUS, "corpus"),
        _homog_niveau(df, "type_macro2", "type_macro", "unite", None, "unite"),
        _homog_niveau(bp, "second", "second_element", "document_set", ORDRE_CORPUS, "corpus"),
        _homog_niveau(bp, "second", "second_element", "unite", None, "unite"),
    ]
    return _ecrire(pd.DataFrame(rows), "tests_homogeneite.csv", ecrire)


def _residus_niveau(sub, col, nom, grp, ordre, niveau, parent):
    obs = pd.crosstab(sub[col], sub[grp])
    if ordre is not None:
        obs = obs.reindex(columns=ordre, fill_value=0)
    arr = obs.values.astype(float)
    _, _, _, exp = chi2_contingency(arr, correction=False)
    N = arr.sum(); row = arr.sum(1, keepdims=True); colm = arr.sum(0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = (arr - exp) / np.sqrt(exp * (1 - row / N) * (1 - colm / N))
    p = 2 * (1 - norm.cdf(np.abs(adj)))
    rows = []
    for i, lig in enumerate(obs.index):
        for j, gr in enumerate(obs.columns):
            par = gr if grp == "document_set" else parent[gr]
            rows.append({"croisement": nom, "niveau": niveau, "corpus": par, "groupe": gr,
                         "modalite": lig, "residu_ajuste": round(adj[i, j], 2), "p_brut": p[i, j]})
    t = pd.DataFrame(rows)
    t["p_bh"] = p_adjust(t["p_brut"], "bh"); t["p_holm"] = p_adjust(t["p_brut"], "holm")
    t["signif_bh"] = t["p_bh"] < ALPHA
    t["sens"] = np.where(t["residu_ajuste"] > 0, "sur-represente", "sous-represente")
    return t.sort_values("p_brut")


def residus_combine(df, bp, ecrire=True):
    par = _parent_map(df)
    parts = [
        _residus_niveau(df, "type_macro2", "type_macro", "document_set", ORDRE_CORPUS, "corpus", par),
        _residus_niveau(df, "type_macro2", "type_macro", "unite", None, "unite", par),
        _residus_niveau(bp, "second", "second_element", "document_set", ORDRE_CORPUS, "corpus", par),
        _residus_niveau(bp, "second", "second_element", "unite", None, "unite", par),
    ]
    return _ecrire(pd.concat(parts, ignore_index=True), "residus_significatifs.csv", ecrire)


def chute_combine(df, ecrire=True):
    om = df[df["type_fin"].isin(FORMES_OMISSION) & df["a_verbe"]].copy()
    om["ne_absent"] = om["type_fin"].str.endswith("_seul")
    par = _parent_map(df)
    rows = []
    for niveau, grp, ordre in [("corpus", "document_set", ORDRE_CORPUS), ("unite", "unite", None)]:
        gg = om.groupby(grp)["ne_absent"].agg(n="size", absent="sum")
        if ordre is not None:
            gg = gg.reindex(ordre)
        else:
            gg = gg.sort_values("absent", ascending=False)
        for nom_g, rr in gg.iterrows():
            n = int(rr["n"]); ab = int(rr["absent"])
            corpus = nom_g if niveau == "corpus" else par[nom_g]
            rows.append({"type_ligne": "taux", "niveau": niveau, "corpus": corpus, "groupe": nom_g,
                         "n": n, "ne_absent": ab,
                         "taux_absent_pct": round(ab / n * 100, 2) if n else np.nan})
    obs = pd.crosstab(om["ne_absent"], om["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    chi2, p, dof, _ = chi2_contingency(obs.values, correction=False)
    rows.append({"type_ligne": "omnibus", "niveau": "corpus", "groupe": "(global 3 corpus)",
                 "n": int(obs.values.sum()), "chi2": round(chi2, 1), "ddl": dof,
                 "p_value": p, "cramer_v": round(_cramer(obs.values), 3)})
    obs2 = pd.crosstab(om["ne_absent"], om["unite"]); _, pmc = _fisher_mc(om["ne_absent"], om["unite"])
    rows.append({"type_ligne": "omnibus", "niveau": "unite", "groupe": "(global unite)",
                 "n": int(obs2.values.sum()), "cramer_v": round(_cramer(obs2.values), 3), "p_fisher_mc": pmc})
    paires = _paires_or(om)
    for x, pad in zip(paires, p_adjust([x["p_fisher"] for x in paires], "holm")):
        rows.append({"type_ligne": "paire", "niveau": "corpus", "comparaison": x["comparaison"],
                     "n": x["n"], "odds_ratio": x["odds_ratio"], "or_ic95_bas": x["or_ic95_bas"],
                     "or_ic95_haut": x["or_ic95_haut"], "p_fisher": x["p_fisher"], "p_fisher_holm": pad})
    cols = ["type_ligne", "niveau", "corpus", "groupe", "comparaison", "n", "ne_absent",
            "taux_absent_pct", "chi2", "ddl", "p_value", "p_fisher_mc", "cramer_v",
            "odds_ratio", "or_ic95_bas", "or_ic95_haut", "p_fisher", "p_fisher_holm"]
    return _ecrire(pd.DataFrame(rows).reindex(columns=cols), "chute_ne_tests.csv", ecrire)


def kruskal_combine(df, ecrire=True, n_min=2):
    comp = df[df["a_une_portee"] & (df["longueur_portee_tokens"] > 0)]
    par = _parent_map(df)
    rows = []
    for niveau, grp, ordre in [("corpus", "document_set", ORDRE_CORPUS), ("unite", "unite", None)]:
        g = comp.groupby(grp)["longueur_portee_tokens"].agg(n="size", mediane="median", moyenne="mean")
        g = g.reindex(ordre) if ordre is not None else g.sort_values("n", ascending=False)
        for nom_g, rr in g.iterrows():
            if pd.isna(rr["n"]):
                continue
            corpus = nom_g if niveau == "corpus" else par[nom_g]
            rows.append({"type_ligne": "groupe", "niveau": niveau, "corpus": corpus, "groupe": nom_g,
                         "n": int(rr["n"]), "mediane": round(rr["mediane"], 2),
                         "moyenne": round(rr["moyenne"], 2)})
    for niveau, grp, ordre in [("corpus", "document_set", ORDRE_CORPUS), ("unite", "unite", None)]:
        labs = ordre if ordre is not None else sorted(comp[grp].unique())
        groupes = [comp.loc[comp[grp] == L, "longueur_portee_tokens"].values for L in labs]
        groupes = [g for g in groupes if len(g) >= n_min]
        H, p, eta2_H, *_ = _kw_stats(groupes)
        rows.append({"type_ligne": "omnibus", "niveau": niveau,
                     "groupe": f"(global {niveau}, {len(groupes)} groupes)",
                     "H": round(H, 2), "ddl": len(groupes) - 1, "p_value": p, "eta2_H": round(eta2_H, 4)})
    kd = kruskal_dunn(df)
    for _, r in kd[kd["comparaison"] != "GLOBAL (Kruskal-Wallis)"].iterrows():
        rows.append({"type_ligne": "dunn", "niveau": "corpus", "comparaison": r["comparaison"],
                     "z": r["z"], "p_dunn": r["p_dunn"], "p_dunn_holm": r["p_dunn_holm"]})
    # Hodges-Lehmann : decalage median (tokens) + IC, par paire de corpus
    for a, b in combinations(ORDRE_CORPUS, 2):
        xa = comp.loc[comp.document_set == a, "longueur_portee_tokens"].values
        xb = comp.loc[comp.document_set == b, "longueur_portee_tokens"].values
        hl, lo, hi, _ = hodges_lehmann(xa, xb)
        rows.append({"type_ligne": "hodges_lehmann", "niveau": "corpus", "comparaison": f"{a} vs {b}",
                     "hl_decalage_med": round(hl, 2), "hl_ic95_bas": round(lo, 2),
                     "hl_ic95_haut": round(hi, 2)})
    cols = ["type_ligne", "niveau", "corpus", "groupe", "comparaison", "n", "mediane", "moyenne",
            "H", "ddl", "p_value", "eta2_H", "z", "p_dunn", "p_dunn_holm",
            "hl_decalage_med", "hl_ic95_bas", "hl_ic95_haut"]
    return _ecrire(pd.DataFrame(rows).reindex(columns=cols), "kruskal_dunn.csv", ecrire)


# ==========================================================================
# DENSITE DE LA NEGATION (numerateur negations / denominateur tokens)
# ==========================================================================
def _poisson_ci_pour_1000(k, T, conf=0.95):
    """IC exact (gamma) du taux pour 1000 tokens : k negations sur T tokens."""
    from scipy.stats import chi2 as _chi2
    a = 1 - conf
    bas = _chi2.ppf(a / 2, 2 * k) / 2 if k > 0 else 0.0
    haut = _chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2
    return bas / T * 1000, haut / T * 1000


def charger_denominateurs(chemin=CHEMIN_DENOM):
    """Parts (tous, y compris 0 negation) -> tokens par groupe. None si absent."""
    p = Path(chemin)
    if not p.exists():
        print(f"  [!] {chemin} introuvable : densite ignoree (lancer denominateurs.py).")
        return None
    den = pd.read_parquet(p)
    den["unite"] = np.where(den["sous_corpus"] == den["document_set"], den["oeuvre"], den["sous_corpus"])
    return den


def densite_combine(df, den, ecrire=True):
    if den is None:
        return None
    par = den.drop_duplicates("unite").set_index("unite")["document_set"].to_dict()
    n_par_part = df.groupby("document_part_id").size().rename("n_neg")
    base = den.merge(n_par_part, on="document_part_id", how="left")
    base["n_neg"] = base["n_neg"].fillna(0)
    rows = []
    for niveau, grp, ordre in [("corpus", "document_set", ORDRE_CORPUS), ("unite", "unite", None)]:
        g = base.groupby(grp).agg(n_neg=("n_neg", "sum"), n_tokens=("n_tokens", "sum"),
                                  n_parts=("document_part_id", "size"))
        g = g.reindex(ordre) if ordre is not None else g.sort_values("n_neg", ascending=False)
        for nom_g, rr in g.iterrows():
            k = int(rr["n_neg"]); T = int(rr["n_tokens"])
            taux = k / T * 1000 if T else np.nan
            lo, hi = _poisson_ci_pour_1000(k, T) if T else (np.nan, np.nan)
            corpus = nom_g if niveau == "corpus" else par[nom_g]
            rows.append({"type_ligne": "taux", "niveau": niveau, "corpus": corpus, "groupe": nom_g,
                         "n_neg": k, "n_tokens": T, "n_parts": int(rr["n_parts"]),
                         "taux_pour_1000": round(taux, 2),
                         "ic95_bas": round(lo, 2), "ic95_haut": round(hi, 2)})
    # IRR entre corpus (rapport de taux + IC log-normal + test de Wald)
    gc = base.groupby("document_set").agg(k=("n_neg", "sum"), T=("n_tokens", "sum")).reindex(ORDRE_CORPUS)
    for a, b in combinations(ORDRE_CORPUS, 2):
        ka, Ta = gc.loc[a, "k"], gc.loc[a, "T"]; kb, Tb = gc.loc[b, "k"], gc.loc[b, "T"]
        irr = (ka / Ta) / (kb / Tb)
        se = np.sqrt(1 / ka + 1 / kb)
        lo, hi = np.exp(np.log(irr) - 1.96 * se), np.exp(np.log(irr) + 1.96 * se)
        z = np.log(irr) / se; p = 2 * (1 - norm.cdf(abs(z)))
        rows.append({"type_ligne": "irr", "niveau": "corpus", "comparaison": f"{a} vs {b}",
                     "irr": round(irr, 3), "irr_ic95_bas": round(lo, 3),
                     "irr_ic95_haut": round(hi, 3), "p_wald": p})
    cols = ["type_ligne", "niveau", "corpus", "groupe", "comparaison", "n_neg", "n_tokens",
            "n_parts", "taux_pour_1000", "ic95_bas", "ic95_haut", "irr", "irr_ic95_bas",
            "irr_ic95_haut", "p_wald"]
    return _ecrire(pd.DataFrame(rows).reindex(columns=cols), "densite_tests.csv", ecrire)


# ==========================================================================
# HODGES-LEHMANN : decalage median + IC distribution-free (longueur de portee)
# ==========================================================================
def hodges_lehmann(x, y, alpha=0.05):
    """Estimateur HL du decalage median (x - y) + IC base sur Mann-Whitney."""
    x = np.asarray(x, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    m, n = len(x), len(y); mn = m * n
    diffs = (x[:, None] - y[None, :]).ravel()
    hl = float(np.median(diffs))
    z = norm.ppf(1 - alpha / 2)
    C = mn / 2 - z * np.sqrt(mn * (m + n + 1) / 12.0)
    k = max(int(np.floor(C)), 1)
    # bornes = k-ieme plus petite et k-ieme plus grande des differences
    part = np.partition(diffs, [k - 1, mn - k])
    return hl, float(part[k - 1]), float(part[mn - k]), mn


# ==========================================================================
# VOLET OMISSION APPROFONDI (ex-phase4b) : discordance par negateur, chute par
# verbe (effet lexical), 'ne' litteraire, proprietes diverses. Stats sobres :
# proportions + IC de Wilson (robustes aux petits n) + chi2/Cramer V.
# ==========================================================================
def wilson(k, n, z=1.96):
    """IC de Wilson (en %) pour une proportion k/n ; robuste aux petits n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    demi = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round((centre - demi) * 100, 1), round((centre + demi) * 100, 1)


def _table_chute(df):
    """Sous-ensemble 'discordance' : negations verbales en un negateur a 2 formes."""
    formes = [x for p in PAIRES_OMISSION.values() for x in p]
    om = df[df["type_fin"].isin(formes) & df["a_verbe"]].copy()
    om["ne_absent"] = om["type_fin"].str.endswith("_seul")
    om["negateur"] = (om["type_fin"].str.replace("^ne_", "", regex=True)
                        .str.replace("_seul", "", regex=False))
    return om


# ---------- 1. discordance : chute par negateur ----------
def discordance_negateur(df, ecrire=True):
    om = _table_chute(df)
    rows = []
    for neg in PAIRES_OMISSION:
        s = om[om["negateur"] == neg]
        n = len(s); k = int(s["ne_absent"].sum())
        lo, hi = wilson(k, n)
        rows.append({"negateur": neg, "n": n, "ne_absent": k,
                     "taux_chute_pct": round(k / n * 100, 1) if n else np.nan,
                     "ic95_bas": lo, "ic95_haut": hi})
    t = pd.DataFrame(rows).sort_values("taux_chute_pct", ascending=False)
    # omnibus : la chute depend-elle du negateur ?
    tab = pd.crosstab(om["ne_absent"], om["negateur"])
    chi2, p, dof, _ = chi2_contingency(tab.values, correction=False)
    V = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    t.attrs["omnibus"] = (chi2, dof, p, V)
    print(f"  [discordance] chi2={chi2:.1f} ddl={dof} p={p:.2e} Cramer_V={V:.3f}")
    return _ecrire(t, "discordance_negateur.csv", ecrire)


# ---------- 2. chute generalisee par corpus ----------
def chute_corpus(df, ecrire=True):
    om = _table_chute(df)
    rows = []
    for c in ORDRE_CORPUS:
        s = om[om["document_set"] == c]; n = len(s); k = int(s["ne_absent"].sum())
        lo, hi = wilson(k, n)
        rows.append({"corpus": c, "n": n, "ne_absent": k,
                     "taux_chute_pct": round(k / n * 100, 1) if n else np.nan,
                     "ic95_bas": lo, "ic95_haut": hi, "champ": "tous_negateurs"})
    # rappel 'pas' seul, pour comparaison avec la Phase 4 d'origine
    omp = om[om["negateur"] == "pas"]
    for c in ORDRE_CORPUS:
        s = omp[omp["document_set"] == c]; n = len(s); k = int(s["ne_absent"].sum())
        lo, hi = wilson(k, n)
        rows.append({"corpus": c, "n": n, "ne_absent": k,
                     "taux_chute_pct": round(k / n * 100, 1) if n else np.nan,
                     "ic95_bas": lo, "ic95_haut": hi, "champ": "pas_seulement"})
    t = pd.DataFrame(rows)
    tab = pd.crosstab(om["ne_absent"], om["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    chi2, p, dof, _ = chi2_contingency(tab.values, correction=False)
    V = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    t.attrs["omnibus"] = (chi2, dof, p, V)
    print(f"  [chute corpus] chi2={chi2:.1f} ddl={dof} p={p:.2e} Cramer_V={V:.3f}")
    return _ecrire(t, "chute_corpus.csv", ecrire)


# ---------- 3. conditionnement lexical : chute par verbe ----------
def chute_verbe(df, ecrire=True, negateur="pas"):
    om = _table_chute(df)
    if negateur != "tous":
        om = om[om["negateur"] == negateur]
    top = om["verbe_lemme"].value_counts().head(TOP_VERBES).index
    rows = []
    for v in top:
        s = om[om["verbe_lemme"] == v]; n = len(s); k = int(s["ne_absent"].sum())
        lo, hi = wilson(k, n)
        rows.append({"verbe": v, "n": n, "ne_absent": k,
                     "taux_chute_pct": round(k / n * 100, 1), "ic95_bas": lo, "ic95_haut": hi})
    t = pd.DataFrame(rows).sort_values("taux_chute_pct", ascending=False)
    sub = om[om["verbe_lemme"].isin(top)]
    tab = pd.crosstab(sub["ne_absent"], sub["verbe_lemme"])
    chi2, p, dof, _ = chi2_contingency(tab.values, correction=False)
    t.attrs["omnibus"] = (chi2, dof, p)
    t.attrs["negateur"] = negateur
    print(f"  [chute verbe/{negateur}] chi2={chi2:.1f} ddl={dof} p={p:.2e}")
    return _ecrire(t, "chute_verbe.csv", ecrire)


# ---------- 3b. chute par verbe, PAYETON seul, CHAQUE verbe (demande encadrante) ----------
def chute_verbe_payeton(df, ecrire=True, min_occ=MIN_OCC_VERBE):
    """Omission du 'ne' par verbe dans PAYETON (la ou la chute existe), pour CHAQUE
    verbe + IC Wilson. Difference avec chute_verbe : payeton seul, tous les verbes
    (pas seulement le top 12), tous negateurs a deux formes confondus. Le drapeau
    'assez_frequent' (n >= min_occ) signale les taux interpretables."""
    om = _table_chute(df)
    om = om[om["document_set"] == "payetoncorpus"]
    rows = []
    for v, s in om.groupby("verbe_lemme"):
        n = len(s); k = int(s["ne_absent"].sum())
        lo, hi = wilson(k, n)
        rows.append({"verbe": v, "n": n, "ne_absent": k,
                     "taux_chute_pct": round(k / n * 100, 1) if n else np.nan,
                     "ic95_bas": lo, "ic95_haut": hi, "assez_frequent": n >= min_occ})
    t = (pd.DataFrame(rows)
         .sort_values(["assez_frequent", "taux_chute_pct", "n"], ascending=[False, False, False]))
    print(f"  [chute verbe payeton] {len(t)} verbes, {int((t['n'] >= min_occ).sum())} avec n>={min_occ}")
    return _ecrire(t, "chute_verbe_payeton.csv", ecrire)


# ---------- 3c. chute selon morpho du verbe (temps/mode/type), payeton ----------
def chute_morpho_payeton(df, ecrire=True):
    """Taux de chute du 'ne' selon TEMPS, MODE (subjonctif/indicatif...) et TYPE
    (modal/auxiliaire/lexical) du verbe, DANS payeton + IC Wilson. Repond a :
    le subjonctif / l'imparfait / les modaux favorisent-ils la chute du 'ne' ?
    Necessite les colonnes morpho (relancer build_negations_master avec md)."""
    if not set(MORPHO_COLS).issubset(df.columns):
        print("  [!] colonnes morpho absentes -> chute_morpho_payeton ignore (relancer build avec md).")
        return None
    om = _table_chute(df)
    om = om[om["document_set"] == "payetoncorpus"].copy()
    om["type_verbe"] = np.where(om["verbe_pos"] == "AUX", "auxiliaire",
                          np.where(om["verbe_lemme"].isin(MODAUX), "modal", "lexical"))
    rows = []
    for dim, col in {"temps": "verbe_temps", "mode": "verbe_mode", "type": "type_verbe"}.items():
        s = om.copy(); s[col] = s[col].fillna("(inconnu)").replace("", "(inconnu)")
        for mod, gg in s.groupby(col):
            n = len(gg); k = int(gg["ne_absent"].sum()); lo, hi = wilson(k, n)
            rows.append({"dimension": dim, "modalite": mod, "n": n, "ne_absent": k,
                         "taux_chute_pct": round(k / n * 100, 1) if n else np.nan,
                         "ic95_bas": lo, "ic95_haut": hi})
    t = pd.DataFrame(rows).sort_values(["dimension", "taux_chute_pct"], ascending=[True, False])
    return _ecrire(t, "chute_morpho_payeton.csv", ecrire)


# ---------- 3d. verbes des negations imbriquees (concord) ----------
def verbes_imbriquees(df, ecrire=True, min_occ=3):
    """Verbes portes par les negations imbriquees (concord/double) x corpus + TOTAL.
    Les verbes apparaissant < min_occ fois en contexte imbrique sont regroupes en
    '(autres)'. Complete proprietes_diverses (qui donne le taux d'imbrication)."""
    imb = df[df["imbriquee"] & df["a_verbe"]].copy()
    freq = imb["verbe_lemme"].value_counts()
    gardes = set(freq[freq >= min_occ].index)
    imb["verbe_aff"] = imb["verbe_lemme"].where(imb["verbe_lemme"].isin(gardes), "(autres)")
    t = pd.crosstab(imb["verbe_aff"], imb["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    t["TOTAL"] = t.sum(axis=1)
    t = t.sort_values("TOTAL", ascending=False)
    t.loc["TOTAL"] = t.sum(axis=0)
    print(f"  [verbes imbriquees] {len(gardes)} verbes >= {min_occ} occ en contexte imbrique")
    return _ecrire(t.reset_index(), "verbes_imbriquees.csv", ecrire)


# ---------- 3e. mode du verbe (TOUS les modes) + focus subjonctif ----------
def analyse_mode(df, ecrire=True):
    """Analyse detaillee du MODE du verbe nie pour TOUS les modes (indicatif,
    subjonctif, conditionnel, imperatif), avec le detail subjonctif demande par le
    superviseur. Quatre volets (colonne 'volet') dans mode_detail.csv :
      (a) distribution    : mode x corpus, incl. '(inconnu)' (infinitifs/participes) ;
      (b) mode_temps      : mode x temps pour CHAQUE mode (Ind-Pres, Sub-Past,
          Cnd-Pres...) -> situe chaque mode dans le temps ; le correctif
          verbe_fini_mode rapatrie le subjonctif/conditionnel PASSE via l'aux enfant
          du participe (sinon il tombe en '(inconnu)') ;
      (c) chute_payeton   : chute du 'ne' par mode dans payeton (IC Wilson) -> quel
          mode favorise la chute (subjonctif vs indicatif vs conditionnel...) ;
      (d) verbes_par_mode : verbes porteurs de chaque mode NON-indicatif (Sub/Cnd/Imp).
    Necessite verbe_mode (relancer build avec md + correctif verbe_fini_mode)."""
    if "verbe_mode" not in df.columns:
        print("  [!] verbe_mode absent -> analyse_mode ignore (relancer build avec md).")
        return None
    ORDRE_MODE = ["Ind", "Sub", "Cnd", "Imp", "(inconnu)"]
    ORDRE_TEMPS = ["Pres", "Past", "Imp", "Fut", "(inconnu)"]
    vb = df[df["a_verbe"]].copy()
    vb["verbe_mode"] = vb["verbe_mode"].fillna("(inconnu)").replace("", "(inconnu)")
    if "verbe_temps" in vb.columns:
        vb["verbe_temps"] = vb["verbe_temps"].fillna("(inconnu)").replace("", "(inconnu)")
    rows = []
    # (a) distribution mode x corpus (tous les modes)
    for mode in ORDRE_MODE:
        s = vb[vb["verbe_mode"] == mode]
        r = {"volet": "distribution", "cle": mode, "total": len(s),
             "pct_verbes": round(len(s) / len(vb) * 100, 2) if len(vb) else 0.0}
        for c in ORDRE_CORPUS:
            r[c] = int((s["document_set"] == c).sum())
        rows.append(r)
    # (b) mode x temps : situe chaque mode (incl. present/passe du subjonctif & conditionnel)
    if "verbe_temps" in vb.columns:
        for mode in ["Ind", "Sub", "Cnd", "Imp"]:
            sm = vb[vb["verbe_mode"] == mode]
            if len(sm) == 0:
                continue
            for tps in ORDRE_TEMPS:
                n = int((sm["verbe_temps"] == tps).sum())
                if n:
                    rows.append({"volet": "mode_temps", "cle": f"{mode}-{tps}", "total": n})
    # (c) chute par mode dans payeton (Wilson) -- tous les modes
    om = _table_chute(df)
    om = om[om["document_set"] == "payetoncorpus"].copy()
    om["verbe_mode"] = om["verbe_mode"].fillna("(inconnu)").replace("", "(inconnu)")
    for mode in ORDRE_MODE:
        s = om[om["verbe_mode"] == mode]
        n = len(s)
        if n == 0:
            continue
        k = int(s["ne_absent"].sum()); lo, hi = wilson(k, n)
        rows.append({"volet": "chute_payeton", "cle": mode, "total": n, "ne_absent": k,
                     "taux_chute_pct": round(k / n * 100, 1), "ic95_bas": lo, "ic95_haut": hi})
    # (d) verbes porteurs de chaque mode non-indicatif (top 8 par mode)
    for mode in ["Sub", "Cnd", "Imp"]:
        sm = vb[vb["verbe_mode"] == mode]
        for v, k in sm["verbe_lemme"].value_counts().head(8).items():
            rows.append({"volet": "verbes_par_mode", "cle": f"{mode}:{v}", "total": int(k)})
    t = pd.DataFrame(rows)
    compte = {m: int((vb["verbe_mode"] == m).sum()) for m in ORDRE_MODE}
    print(f"  [mode detail] {compte}")
    return _ecrire(t, "mode_detail.csv", ecrire)


# ---------- 4. 'ne' litteraire (ne_seul) ----------
def ne_litteraire(df, ecrire=True):
    ns = df[df["type_fin"] == "ne_seul"]
    rows = []
    for c in ORDRE_CORPUS:
        k = int((ns["document_set"] == c).sum())
        rows.append({"niveau": "corpus", "cle": c, "n_ne_seul": k,
                     "pct_du_total": round(k / len(ns) * 100, 1)})
    for v, k in ns["verbe_lemme"].value_counts().head(8).items():
        rows.append({"niveau": "verbe", "cle": v, "n_ne_seul": int(k),
                     "pct_du_total": round(k / len(ns) * 100, 1)})
    t = pd.DataFrame(rows); t.attrs["total"] = len(ns)
    print(f"  [ne litteraire] total ne_seul = {len(ns)}")
    return _ecrire(t, "ne_litteraire.csv", ecrire)


# ---------- 5-6. autres proprietes ----------
def proprietes_diverses(df, ecrire=True):
    rows = []
    # 'ne...que' : ne jamais absent (rappel)
    nq = int((df["type_fin"] == "ne_que").sum())
    rows.append({"propriete": "ne_que_total", "cle": "(corpus confondus)", "valeur": nq,
                 "detail": "negateur ou le 'ne' ne chute pas (aucune forme que_seul)"})
    # negation imbriquee (concord) par corpus
    for c in ORDRE_CORPUS:
        s = df[df["document_set"] == c]
        k = int(s["imbriquee"].sum()); n = len(s)
        rows.append({"propriete": "imbriquee_pct", "cle": c, "valeur": round(k / n * 100, 2),
                     "detail": f"{k}/{n} negations imbriquees (concord/double)"})
    # part des negations sans verbe (negation nominale/averbale) par corpus
    for c in ORDRE_CORPUS:
        s = df[df["document_set"] == c]
        k = int((~s["a_verbe"]).sum()); n = len(s)
        rows.append({"propriete": "sans_verbe_pct", "cle": c, "valeur": round(k / n * 100, 2),
                     "detail": f"{k}/{n} negations sans verbe rattache"})
    t = pd.DataFrame(rows)
    return _ecrire(t, "proprietes_diverses.csv", ecrire)



def lancer_tout(ecrire=True):
    df = charger()
    sec = df[df["second"].notna()].copy()
    res = {}
    res["homo"] = pd.DataFrame([homogeneite(df, "type_macro2", "type_macro"),
                                homogeneite(sec, "second", "second_element")])
    res["residus"] = pd.concat([residus_significatifs(df, "type_macro2", "type_macro"),
                                residus_significatifs(sec, "second", "second_element")], ignore_index=True)
    res["chute"] = chute_ne_tests(df)
    res["kw"] = kruskal_dunn(df)
    den = charger_denominateurs()
    res["densite"] = densite_combine(df, den, ecrire)
    homogeneite_combine(df, sec, ecrire)
    residus_combine(df, sec, ecrire)
    chute_combine(df, ecrire)
    kruskal_combine(df, ecrire)
    # --- volet omission approfondi (ex-phase4b) ---
    res["discordance"] = discordance_negateur(df, ecrire)
    res["chute_corpus"] = chute_corpus(df, ecrire)
    res["chute_verbe"] = chute_verbe(df, ecrire)
    res["chute_verbe_payeton"] = chute_verbe_payeton(df, ecrire)
    res["chute_morpho_payeton"] = chute_morpho_payeton(df, ecrire)
    res["verbes_imbriquees"] = verbes_imbriquees(df, ecrire)
    res["mode_detail"] = analyse_mode(df, ecrire)
    res["ne_litteraire"] = ne_litteraire(df, ecrire)
    res["proprietes"] = proprietes_diverses(df, ecrire)
    return res


if __name__ == "__main__":
    r = lancer_tout()
    print("=== Homogeneite (memoire, corpus) ==="); print(r["homo"].to_string(index=False))
    print("\n=== Chute (memoire, corpus) ==="); print(r["chute"].to_string(index=False))
    for f in OUT.glob("*.csv"):
        if f.name not in ATTENDUS:
            try: f.unlink()
            except OSError: pass
    print("\nFichiers :", ", ".join(ATTENDUS))