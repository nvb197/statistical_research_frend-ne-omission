# -*- coding: utf-8 -*-
"""
PHASE 3 - Analyse bivariee (FReND) : type de negation x CORPUS puis x UNITE
(niveau immediatement sous le corpus), reunis dans un meme fichier.

DEFINITION DE L'UNITE (niveau juste sous le corpus) :
  unite = sous_corpus, SAUF quand le corpus n'a pas de sous-decoupage (sous_corpus
  == document_set, cas de litbank) ou l'on prend alors l'OEUVRE. Ainsi chaque
  corpus a bien des sous-unites : payetoncorpus -> 14 sous_corpus, sequoia -> 5
  sous_corpus, litbank -> 18 oeuvres (37 unites au total). Une colonne 'corpus'
  (corpus parent) est ajoutee a chaque ligne pour suivre l'appartenance.

Sorties dans resultats_phase3/ (un fichier par phenomene, format LONG, colonnes
  analyse | niveau (corpus/unite) | corpus | groupe | modalite | n |
  pct_dans_groupe | residu_ajuste | notable_1_96) :
  - type_macro.csv, second_element.csv, omission_ne.csv
  - apercu_tests.csv     omnibus par (analyse x niveau)
  - longueur_portee.csv  longueur de portee par groupe (corpus + unite)

DEFINITIONS (alignees sur la Phase 4b) :
  - second_element = paradigme du negateur, mesure sur TOUTES les formes (ne_X et
    X_seul) et non les seules formes avec 'ne' : on mesure QUEL negateur, pas s'il
    garde le 'ne'. Sinon le paradigme de payeton (qui chute le 'ne') serait fausse.
  - omission_ne = chute du 'ne' generalisee aux 8 negateurs a deux formes (pas,
    plus, jamais, rien, personne, aucun, point, nul), pas seulement 'pas'.
  ATTENTION (typologie de SURFACE, pas de phenomene) : dans type_macro, 'ellipse'
  = pas_seul (chute du 'ne') + ne_seul ('ne' litteraire, phenomene INVERSE), et les
  autres formes sans 'ne' (rien_seul, jamais_seul, aucun_seul...) sont rangees dans
  'adverbiale' avec 'non'. Le residu 'ellipse' melange donc chute (payeton, via
  pas_seul) et 'ne' litteraire (litbank, via ne_seul) : NE PAS le lire comme "chute
  du 'ne'". Pour la chute, se fier a omission_ne (ne_X vs X_seul, propre) ; pour le
  'ne' litteraire, a ne_seul (phase 2 / phase 4b).

Methodo : residu ajuste (Haberman) ~ N(0,1), |r|>1.96 notable ; omnibus chi2 +
Fisher-Freeman-Halton Monte Carlo (lire p_fisher_mc quand cellules attendues < 5,
frequent au niveau unite). Aucune fusion de modalite.

NB : la valeur de retour en memoire (lancer_tout) reste au niveau CORPUS pour les
graphiques des notebooks ; ce sont les CSV qui portent corpus + unite.

Lancer :  python phase3_bivarie.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency, norm

# ====================== PARAMETRES ======================
ORDRE_CORPUS = ["payetoncorpus", "litbank", "sequoia"]
EXCLURE_EXPLETIFS_CONNUS = True
from expletifs import EXPLETIFS_CUE_ID  # source unique (voir expletifs.py)
ALPHA = 0.05
# negateurs du paradigme (second element), forme avec OU sans 'ne'. On les
# normalise pour mesurer QUEL negateur est employe (paradigme), independamment
# de la chute du 'ne' : sinon les formes sans ne (pas_seul...) seraient exclues
# et le paradigme de payeton (qui chute le 'ne') serait sous-estime.
NEG_PARADIGME = ["pas", "plus", "jamais", "rien", "personne", "aucun", "point", "nul", "guere", "ni"]
# negateurs a deux formes (avec/sans ne) -> analyse d'omission generalisee
PAIRES_OMISSION = {"pas": ("ne_pas", "pas_seul"), "plus": ("ne_plus", "plus_seul"),
                   "jamais": ("ne_jamais", "jamais_seul"), "rien": ("ne_rien", "rien_seul"),
                   "personne": ("ne_personne", "personne_seul"), "aucun": ("ne_aucun", "aucun_seul"),
                   "point": ("ne_point", "point_seul"), "nul": ("ne_nul", "nul_seul")}
# morpho du verbe nie (colonnes ajoutees par build_negations_master modifie : relancer
# le build avec fr_core_news_md). type_verbe = modal / auxiliaire / lexical.
MODAUX = {"pouvoir", "devoir", "vouloir", "savoir", "falloir"}
MORPHO_COLS = ["verbe_pos", "verbe_temps", "verbe_mode"]
# ========================================================

OUT = Path("resultats_phase3"); OUT.mkdir(exist_ok=True)
ATTENDUS = ["type_macro.csv", "second_element.csv", "omission_ne.csv",
            "apercu_tests.csv", "longueur_portee.csv",
            "temps_verbe.csv", "mode_verbe.csv", "type_verbe.csv"]


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
    # second = negateur du paradigme, normalise depuis ne_X ET X_seul (cf. NEG_PARADIGME)
    neg = df["type_fin"].astype(str).str.replace("^ne_", "", regex=True).str.replace("_seul", "", regex=False)
    df["second"] = neg.where(neg.isin(NEG_PARADIGME))
    df["a_verbe"] = df["verbe_lemme"].fillna("") != ""
    # type_verbe = modal / auxiliaire / lexical (depend de verbe_pos ; sinon 'lexical' par defaut)
    if "verbe_pos" in df.columns:
        df["type_verbe"] = np.where(df["verbe_pos"] == "AUX", "auxiliaire",
                              np.where(df["verbe_lemme"].isin(MODAUX), "modal", "lexical"))
        df.loc[~df["a_verbe"], "type_verbe"] = None
    # UNITE = niveau juste sous le corpus : sous_corpus, sauf litbank -> oeuvre
    df["unite"] = np.where(df["sous_corpus"] == df["document_set"], df["oeuvre"], df["sous_corpus"])
    return df


def _stats_residus(obs):
    arr = obs.values.astype(float)
    chi2, p, dof, exp = chi2_contingency(arr, correction=False)
    N = arr.sum()
    row = arr.sum(1, keepdims=True); col = arr.sum(0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = (arr - exp) / np.sqrt(exp)
        adj = (arr - exp) / np.sqrt(exp * (1 - row / N) * (1 - col / N))
    cramer = np.sqrt(chi2 / (N * (min(arr.shape) - 1))) if min(arr.shape) > 1 else np.nan
    idx, cols = obs.index, obs.columns
    return {"exp": pd.DataFrame(exp, index=idx, columns=cols),
            "pearson": pd.DataFrame(pearson, index=idx, columns=cols),
            "adj": pd.DataFrame(adj, index=idx, columns=cols),
            "chi2": chi2, "dof": dof, "p": p, "cramer_v": cramer, "N": int(N)}


def _cramer_corrige(chi2, N, r, c):
    """Cramer V corrige du biais (Bergsma 2013) : compare equitablement des
    tables de tailles differentes (corpus 3 col. vs unite 37 col.)."""
    phi2 = chi2 / N
    phi2t = max(0.0, phi2 - (r - 1) * (c - 1) / (N - 1))
    rt = r - (r - 1) ** 2 / (N - 1); ct = c - (c - 1) ** 2 / (N - 1)
    m = min(rt - 1, ct - 1)
    return np.sqrt(phi2t / m) if m > 0 else np.nan


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


# --------- niveau CORPUS, format large (pour les notebooks) ---------
def analyser(df, col, nom):
    obs = pd.crosstab(df[col], df["document_set"]).reindex(columns=ORDRE_CORPUS, fill_value=0)
    obs = obs.loc[obs.sum(axis=1).sort_values(ascending=False).index]
    r = _stats_residus(obs)
    prop = (obs / obs.sum(axis=0) * 100).round(1)
    prop["ensemble"] = (obs.sum(axis=1) / obs.values.sum() * 100).round(1)
    r["obs"], r["prop"], r["nom"], r["col"] = obs, prop, nom, col
    return r


# --------- format LONG combine : corpus + unite, colonne 'corpus' parent ---------
def phenomene_long(sub, col, nom, ecrire=True):
    parent = sub.drop_duplicates("unite").set_index("unite")["document_set"].to_dict()
    rows = []; apercu = []
    for niveau, grp in [("corpus", "document_set"), ("unite", "unite")]:
        obs = pd.crosstab(sub[col], sub[grp])
        if grp == "document_set":
            obs = obs.reindex(columns=ORDRE_CORPUS, fill_value=0)
        else:   # unites ordonnees par corpus parent puis par taille decroissante
            cols_tri = sorted(obs.columns, key=lambda g: (ORDRE_CORPUS.index(parent[g]), -int(obs[g].sum())))
            obs = obs[cols_tri]
        obs = obs.loc[obs.sum(axis=1).sort_values(ascending=False).index]
        arr = obs.values.astype(float)
        chi2, p, dof, exp = chi2_contingency(arr, correction=False)
        N = arr.sum(); row = arr.sum(1, keepdims=True); colm = arr.sum(0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            adj = (arr - exp) / np.sqrt(exp * (1 - row / N) * (1 - colm / N))
            colpct = arr / arr.sum(0) * 100
        cramer = np.sqrt(chi2 / (N * (min(arr.shape) - 1))) if min(arr.shape) > 1 else np.nan
        for i, mod in enumerate(obs.index):
            for j, gr in enumerate(obs.columns):
                par = gr if grp == "document_set" else parent[gr]
                rows.append({"analyse": nom, "niveau": niveau, "corpus": par, "groupe": gr,
                             "modalite": mod, "n": int(arr[i, j]),
                             "pct_dans_groupe": round(colpct[i, j], 1),
                             "residu_ajuste": round(adj[i, j], 2),
                             "notable_1_96": bool(abs(adj[i, j]) > 1.96)})
        _, p_ffh = _fisher_mc(sub[col], sub[grp])
        apercu.append({"analyse": nom, "niveau": niveau, "N": int(N),
                       "modalites": obs.shape[0], "n_groupes": obs.shape[1],
                       "chi2": round(chi2, 1), "ddl": dof, "p_chi2_pearson": p,
                       "p_fisher_mc": p_ffh, "cramer_v": round(cramer, 3),
                       "cramer_v_corrige": round(_cramer_corrige(chi2, N, *arr.shape), 3),
                       "pct_cellules_attendu_lt5": round((exp < 5).mean() * 100, 1)})
    long = pd.DataFrame(rows)
    _ecrire(long, f"{nom}.csv", ecrire)
    return long, apercu


def longueur_par_corpus(df, ecrire=True):
    """Tableau CORPUS uniquement (pour les notebooks)."""
    comp = df[df["a_une_portee"] & (df["longueur_portee_tokens"] > 0)]
    g = (comp.groupby("document_set")["longueur_portee_tokens"]
            .agg(n="size", mediane="median", moyenne="mean",
                 q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), maxi="max")
            .reindex(ORDRE_CORPUS).round(2).reset_index().rename(columns={"document_set": "groupe"}))
    return g


def longueur_combine(df, ecrire=True):
    comp = df[df["a_une_portee"] & (df["longueur_portee_tokens"] > 0)]
    parent = df.drop_duplicates("unite").set_index("unite")["document_set"].to_dict()
    def agg(grp, ordre=None, est_corpus=False):
        g = (comp.groupby(grp)["longueur_portee_tokens"]
                  .agg(n="size", mediane="median", moyenne="mean",
                       q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), maxi="max"))
        if ordre is not None:
            g = g.reindex(ordre)
        else:
            g = g.sort_values("n", ascending=False)
        g = g.round(2).reset_index().rename(columns={grp: "groupe"})
        g["corpus"] = g["groupe"] if est_corpus else g["groupe"].map(parent)
        return g
    c = agg("document_set", ORDRE_CORPUS, est_corpus=True); c.insert(0, "niveau", "corpus")
    u = agg("unite"); u.insert(0, "niveau", "unite")
    t = pd.concat([c, u], ignore_index=True)
    t = t[["niveau", "corpus", "groupe", "n", "mediane", "moyenne", "q1", "q3", "maxi"]]
    return _ecrire(t, "longueur_portee.csv", ecrire)


def lancer_tout(ecrire=True):
    df = charger()
    res = {}; apercu_all = []
    res["type_macro"] = analyser(df, "type_macro2", "type_macro")
    # second element = paradigme du negateur, sur TOUTES les formes (avec/sans ne)
    sec = df[df["second"].notna()].copy()
    res["second"] = analyser(sec, "second", "second_element")
    # omission generalisee : 8 negateurs a deux formes, negations verbales
    formes = [x for p in PAIRES_OMISSION.values() for x in p]
    om = df[df["type_fin"].isin(formes) & df["a_verbe"]].copy()
    om["ne"] = np.where(om["type_fin"].str.endswith("_seul"), "ne_absent", "ne_present")
    res["omission_ne"] = analyser(om, "ne", "omission_ne")

    _, ap = phenomene_long(df, "type_macro2", "type_macro", ecrire); apercu_all += ap
    _, ap = phenomene_long(sec, "second", "second_element", ecrire); apercu_all += ap
    _, ap = phenomene_long(om, "ne", "omission_ne", ecrire); apercu_all += ap
    longueur_combine(df, ecrire)

    # --- morpho du verbe nie x corpus/unite (temps, mode, type) : residus = quel
    #     corpus sur/sous-emploie subjonctif, imparfait, modaux... (necessite rebuild) ---
    if set(MORPHO_COLS).issubset(df.columns):
        vb = df[df["a_verbe"]].copy()
        for col, nom in [("verbe_temps", "temps_verbe"), ("verbe_mode", "mode_verbe"),
                         ("type_verbe", "type_verbe")]:
            sub = vb.copy()
            sub[col] = sub[col].fillna("(inconnu)").replace("", "(inconnu)")
            _, ap = phenomene_long(sub, col, nom, ecrire); apercu_all += ap
    else:
        print(f"  [!] colonnes morpho absentes {[c for c in MORPHO_COLS if c not in df.columns]} "
              "-> temps_verbe/mode_verbe/type_verbe ignores (relancer build avec md).")

    res["apercu"] = pd.DataFrame(apercu_all)
    _ecrire(res["apercu"], "apercu_tests.csv", ecrire)
    return res


if __name__ == "__main__":
    r = lancer_tout()
    print("APERCU DES TESTS (corpus + unite) :\n")
    print(r["apercu"].to_string(index=False))
    for f in OUT.glob("*.csv"):
        if f.name not in ATTENDUS:
            try:
                f.unlink(); print(f"  [nettoye] {f.name}")
            except OSError:
                pass
    print("\nFichiers dans", OUT, ":", ", ".join(ATTENDUS))