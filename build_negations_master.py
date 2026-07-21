# -*- coding: utf-8 -*-
"""
PHASE 0 — Construction de la table maitre des negations (negations.parquet).

A partir du corpus nettoye (my_xml_corpus_clean.xml), on produit UNE ligne par
negation, enrichie linguistiquement (spaCy), normalisee et typee. C'est la base
de toute l'analyse statistique (phases 1 a 6).

Pipeline NLP :
  - on essaie de charger 'fr_core_news_md' (tokenisation + phrases + POS +
    lemmes + dependances) ;
  - a defaut (pas de modele installe), on retombe sur spaCy blank('fr') +
    sentencizer + simplemma pour les lemmes. Les colonnes POS/dependance
    (cue_pos, verbe_lemme) restent vides dans ce mode et seront remplies en
    relancant avec le modele complet.

Sortie : negations.parquet  (superset des colonnes du CSV + colonnes d'analyse)
"""

import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
import spacy

XML_PATH = "my_xml_corpus_clean.xml"
OUT = "negations.parquet"

# --------------------------------------------------------------------------
# 0.1  Pipeline NLP (modele complet si dispo, sinon fallback)
# --------------------------------------------------------------------------
USING_FULL = True
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    USING_FULL = False
    nlp = spacy.blank("fr")
    nlp.add_pipe("sentencizer")
    try:
        import simplemma
        def lemma_of(word):
            return simplemma.lemmatize(word.lower(), lang="fr")
    except ImportError:
        def lemma_of(word):
            return word.lower()
nlp.max_length = 5_000_000

print(f"[pipeline] {'fr_core_news_md (complet)' if USING_FULL else 'blank fr + sentencizer (fallback)'}")


# --------------------------------------------------------------------------
# Lecture XML : texte lineaire + positions caractere de chaque Cue/Scope
# --------------------------------------------------------------------------
def inner_text(el):
    return "".join(t.text or "" for t in el.iter("Text"))


def build(dp):
    full = [""]
    elems = []

    def walk(node):
        for ch in node:
            if ch.tag == "Annotator":
                continue
            if ch.tag == "Text":
                full[0] += ch.text or ""
            elif ch.tag in ("Cue", "Scope"):
                s = len(full[0])
                walk(ch)
                elems.append((ch, s, len(full[0])))
            else:
                walk(ch)

    walk(dp)
    return full[0], elems


def iter_documents(node, chain):
    for child in node:
        if child.tag == "DocumentSet":
            yield from iter_documents(child, chain + [child.get("type")])
        elif child.tag == "Document":
            yield child, chain


# --------------------------------------------------------------------------
# 0.5  Normalisation de la forme de l'indice
# --------------------------------------------------------------------------
def normalise_indice(txt):
    t = txt.lower().strip()
    t = t.replace("’", "'")
    t = t.replace("...", " ")              # separateur de fragments discontinus
    t = re.sub(r"\bn'\b|\bn'", "ne ", t)   # n' -> ne
    t = t.replace("'", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\bn\b", "ne", t)          # n (n' elide separe) -> ne
    return t


# --------------------------------------------------------------------------
# 0.6  Typologie controlee de l'indice
# --------------------------------------------------------------------------
PREFIXES = {"in", "im", "ir", "il", "dé", "dés", "des", "dis", "mé", "més",
            "mal", "a", "an", "non", "anti"}
EXCEPTIVE = {"sauf", "hormis", "excepté", "excepter", "exception", "hors"}
LEXICAL_STEMS = ("refus", "empêch", "empech", "évit", "evit", "manqu",
                 "interdi", "exclu", "priv", "dépourv", "depourv", "incapab",
                 "impossib", "absen", "défau", "defau", "dén", "den", "rejet",
                 # ajouts (récupèrent ~29 cas auparavant 'autre', sans régression :
                 # ces branches ne s'exécutent qu'après ne/pas/sans/non/exceptive)
                 "arrêt", "arret", "préven", "preven", "prévien", "previen",
                 "annul", "échec", "echec", "renonc")
# negateurs lexicaux reperables par lemme (md), que les stems ne couvrent pas
NEG_LEX_LEMMES = {"nier", "rejeter"}


def _lexical(words):
    return any(any(x.startswith(st) for st in LEXICAL_STEMS) for x in words)


def _radical(words, rad):
    """True si un mot vaut rad ou commence par rad (couvre aucune/aucunes,
    nulle/nuls/nullement). Reserve aux negateurs sans faux-amis frequents."""
    return any(x == rad or x.startswith(rad) for x in words)


def classify(indice_norm, lemme, is_subtoken):
    w = set(indice_norm.split())
    lw = set(lemme.split())
    wl = w | lw                              # surface + lemmes (couvre pluriels : aucunes->aucun)
    has_ne = "ne" in wl

    if is_subtoken and indice_norm.replace(" ", "").replace("-", "") in PREFIXES:
        return "prefixe_morpho", "morphologique"
    if has_ne and "pas" in wl:
        return "ne_pas", "bipartite"
    if has_ne and "jamais" in wl:
        return "ne_jamais", "bipartite"
    if has_ne and "plus" in wl:
        return "ne_plus", "bipartite"
    if has_ne and "rien" in wl:
        return "ne_rien", "bipartite"
    if has_ne and "personne" in wl:
        return "ne_personne", "bipartite"
    if has_ne and _radical(wl, "aucun"):
        return "ne_aucun", "bipartite"
    if has_ne and ("guère" in wl or "guere" in wl):
        return "ne_guere", "bipartite"
    if has_ne and ("goutte" in wl or "mie" in wl):
        return "ne_litteraire_rare", "bipartite"
    if has_ne and "point" in wl:
        return "ne_point", "bipartite"
    if has_ne and _radical(wl, "nul"):
        return "ne_nul", "bipartite"
    # Un cue mene par 'sans' (ex. 'sans qu ... ne ... ni') est une negation
    # PREPOSITIONNELLE : le 'que'/'ni' y est subordonnant/coordonnant, PAS le
    # restrictif 'ne...que' ni une negation en 'ni'. (Les vrais bipartites
    # sans+pas/plus/rien... sont deja captes plus haut, avant cette garde.)
    if "sans" in wl and ("que" in wl or "ni" in wl):
        return "sans", "prepositionnelle"
    if has_ne and "que" in wl:
        return "ne_que", "restrictif"
    if has_ne and "ni" in wl:
        return "ne_ni", "bipartite"
    if has_ne:
        return "ne_seul", "ellipse"          # ne sans second element (omission de pas)
    if "pas" in wl:
        return "pas_seul", "ellipse"         # pas sans ne (chute du ne)
    if "sans" in wl:
        return "sans", "prepositionnelle"
    if indice_norm in ("non", "nan"):
        return "non", "adverbiale"
    if w & EXCEPTIVE or lw & EXCEPTIVE:
        return "exceptive", "exceptive"
    if _lexical(w) or _lexical(lw) or (lw & NEG_LEX_LEMMES):
        return "lexicale", "lexicale"
    if "ni" in wl:
        return "ni", "adverbiale"
    for adv in ("jamais", "plus", "rien", "personne", "guère", "point"):
        if adv in wl:
            return f"{adv}_seul", "adverbiale"
    if _radical(wl, "aucun"):
        return "aucun_seul", "adverbiale"
    if _radical(wl, "nul"):                   # couvre nulle, nuls, nullement
        return "nul_seul", "adverbiale"
    return "autre", "autre"


# --------------------------------------------------------------------------
# Outils d'alignement span-caractere -> tokens
# --------------------------------------------------------------------------
def span_tokens(doc, s, e):
    """Tokens couvrant l'intervalle caractere [s, e] ; 'expand' pour les sous-mots."""
    if e <= s:
        return None
    return doc.char_span(s, e, alignment_mode="expand")


def main():
    root = ET.parse(XML_PATH).getroot()
    rows = []

    for doc, chain in iter_documents(root, []):
        document_set = chain[0] if chain else ""
        sous_corpus = chain[-1] if chain else ""

        for part in doc.findall("DocumentPart"):
            pid = part.get("document_part_id")
            oeuvre = re.sub(r"_\d+$", "", pid)
            full, elems = build(part)
            if not full.strip():
                continue
            sdoc = nlp(full)

            # denominateurs au niveau du DocumentPart
            n_tokens = len(sdoc)
            n_mots = sum(1 for t in sdoc if t.is_alpha)
            sent_spans = [(s.start_char, s.end_char) for s in sdoc.sents]
            n_phrases = len(sent_spans)

            # grouper les elements par negation (ref ; no-ref separe par id)
            groups, order = {}, []
            for el, s, e in elems:
                r = el.get("ref")
                key = ("no-ref", el.get("id")) if r == "no-ref" else (r,)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append((el, s, e))

            # span global de chaque negation (pour detecter l'imbrication)
            gspan = {}
            for k in order:
                ss = min(s for _, s, _ in groups[k])
                ee = max(e for _, _, e in groups[k])
                gspan[k] = (ss, ee)

            for k in order:
                items = groups[k]
                ref = k[0]
                cues = [(el, s, e) for el, s, e in items if el.tag == "Cue"]
                scopes = [(el, s, e) for el, s, e in items if el.tag == "Scope"]

                cue_ids = list(dict.fromkeys(el.get("id") for el, _, _ in cues if el.get("id")))
                scope_ids = list(dict.fromkeys(el.get("id") for el, _, _ in scopes if el.get("id")))
                indice_brut = " ... ".join(inner_text(el) for el, _, _ in cues)
                texte_portee = " ".join(inner_text(el) for el, _, _ in scopes)

                # --- alignement cue : CHAQUE fragment separement ---
                # (un span global min..max aspirerait les tokens intermediaires :
                #  "ne [aime] pas" -> il faut [ne, pas], pas [ne, aime, pas])
                cue_tokens = []
                is_subtoken = False
                for el, s, e in cues:
                    sp = span_tokens(sdoc, s, e)
                    if not sp:
                        continue
                    cue_tokens.extend(list(sp))
                    # sous-mot : le fragment couvre moins qu'un token entier (prefixe in-)
                    if len(sp) == 1 and (e - s) < len(sp.text):
                        is_subtoken = True
                uniq = {t.i: t for t in cue_tokens}        # dedoublonner par index
                cue_tokens = [uniq[i] for i in sorted(uniq)]
                longueur_indice = len(cue_tokens)

                # --- alignement portee : par fragment, en EXCLUANT les tokens de
                #     l'indice (la portee de "ne…pas" est entrelacee avec l'indice :
                #     "Je [n'] attendis [pas] longtemps" -> portee = Je attendis
                #     longtemps, sans n' ni pas) ---
                a_portee = len(texte_portee.strip()) > 0
                longueur_portee = 0
                scope_traverse = False
                if a_portee and scopes:
                    # pour un indice morphologique (prefixe), indice et portee
                    # partagent le MEME token : on ne soustrait alors pas l'indice
                    cue_idx = set() if is_subtoken else {t.i for t in cue_tokens}
                    scope_tokens = []
                    for el, s, e in scopes:
                        sp = span_tokens(sdoc, s, e)
                        if sp:
                            scope_tokens.extend(list(sp))
                    scope_idx = {t.i for t in scope_tokens} - cue_idx
                    longueur_portee = len(scope_idx)
                    # la portee chevauche-t-elle plus d'une phrase ?
                    sc_s = min(s for _, s, _ in scopes)
                    sc_e = max(e for _, _, e in scopes)
                    touched = sum(1 for (a, b) in sent_spans if a < sc_e and b > sc_s)
                    scope_traverse = touched > 1

                # --- normalisation + lemme + typologie ---
                indice_norm = normalise_indice(indice_brut)
                # morpho du verbe (temps/mode/forme/pos) : valeurs par defaut
                verbe_pos = verbe_temps = verbe_mode = verbe_forme = None
                if USING_FULL and cue_tokens:
                    lemme = " ".join(t.lemma_ for t in cue_tokens)
                    cue_pos = cue_tokens[0].pos_                 # ADV pour ne/pas
                    cue_idx = {t.i for t in cue_tokens}
                    verbe_lemme = None
                    verbe_aux = None                            # repli : auxiliaire / verbe plein
                    verbe_token = None                          # token du verbe LEXICAL (lemme, forme, pos)
                    aux_token = None                            # token de l'AUXILIAIRE (porte temps/mode en temps compose)
                    for t in cue_tokens:                        # verbe-tete hors indice
                        h = t.head
                        if h.i in cue_idx:
                            continue
                        if h.pos_ == "VERB":                    # verbe lexical : on prend
                            verbe_lemme = h.lemma_; verbe_token = h
                            break
                        if h.pos_ == "AUX":                     # temps compose : viser le
                            if verbe_aux is None:               #   verbe lexical (tete de l'aux)
                                verbe_aux = h.lemma_; aux_token = h
                            gh = h.head
                            if gh.i not in cue_idx and gh.pos_ == "VERB":
                                # LIMITE CONNUE : avec une copule (etre/avoir + attribut),
                                # gh peut etre le verbe de la MATRICE au-dela d'une frontiere
                                # de proposition (ex. "...que la religion n'est rien" -> gh =
                                # 'comprendre'). Non corrige (impact non systemique ; un garde
                                # 'ne pas franchir un SCONJ/mark' romprait d'autres cas). A
                                # signaler comme limite plutot qu'a patcher a l'aveugle.
                                verbe_lemme = gh.lemma_; verbe_token = gh
                                break
                    if verbe_lemme is None:                     # avoir/etre verbe plein, ou aux isole
                        verbe_lemme = verbe_aux; verbe_token = aux_token

                    # Le TEMPS et le MODE sont portes par le verbe FINI : l'auxiliaire en
                    # temps compose ("n'a pas mange" -> avoir:Ind/Pres), sinon le verbe
                    # lexical lui-meme. La FORME (Inf/Part/Fin) est celle du verbe lexical.
                    def _m1(tok, feat):
                        if tok is None:
                            return None
                        v = tok.morph.get(feat)
                        return v[0] if v else None
                    verbe_fini = aux_token if (aux_token is not None and aux_token is not verbe_token) else verbe_token
                    verbe_temps = _m1(verbe_fini, "Tense")      # Pres / Past / Imp / Fut (INCHANGE)
                    # Correctif MODE : un verbe lexical NON-FINI (participe/infinitif d'un
                    # temps compose, ex. "n'a pas MANGE") ne porte pas de Mood ; celui-ci
                    # est sur son auxiliaire ENFANT (relation 'aux'), non atteignable par la
                    # remontee de tete depuis l'indice. Sans ce correctif, mode='(inconnu)'
                    # explose (participes comptes comme sans mode). Le TEMPS reste inchange.
                    verbe_fini_mode = verbe_fini
                    if verbe_token is not None:
                        vf = verbe_token.morph.get("VerbForm")
                        if vf and vf[0] != "Fin":
                            aux_child = next((c for c in verbe_token.children
                                              if c.pos_ == "AUX"
                                              and c.dep_ in ("aux", "aux:tense", "aux:pass")), None)
                            if aux_child is not None:
                                verbe_fini_mode = aux_child
                    verbe_mode = _m1(verbe_fini_mode, "Mood")   # Ind / Sub / Cnd / Imp (verbe fini)
                    verbe_forme = _m1(verbe_token, "VerbForm")  # Fin / Inf / Part (verbe lexical)
                    verbe_pos = verbe_token.pos_ if verbe_token is not None else None  # VERB / AUX
                else:
                    lemme = " ".join(lemma_of(x) for x in indice_norm.split()) if indice_norm else ""
                    cue_pos = None
                    verbe_lemme = None
                type_fin, type_macro = classify(indice_norm, lemme, is_subtoken)

                # --- discontinuite & imbrication ---
                indice_discontinu = len(cues) > 1
                ss, ee = gspan[k]
                imbriquee = any(kk != k and max(ss, gspan[kk][0]) < min(ee, gspan[kk][1])
                                for kk in order)

                rows.append({
                    "negation_id": f"{pid}::{ref}::{cue_ids[0] if cue_ids else 'NA'}",
                    "document_set": document_set,
                    "sous_corpus": sous_corpus,
                    "oeuvre": oeuvre,
                    "document_part_id": pid,
                    "ref": ref,
                    "indice_brut": indice_brut,
                    "indice_norm": indice_norm,
                    "lemme_indice": lemme,
                    "type_fin": type_fin,
                    "type_macro": type_macro,
                    "cue_pos": cue_pos,
                    "verbe_lemme": verbe_lemme,
                    "verbe_pos": verbe_pos,
                    "verbe_temps": verbe_temps,
                    "verbe_mode": verbe_mode,
                    "verbe_forme": verbe_forme,
                    "a_une_portee": a_portee,
                    "longueur_indice_tokens": longueur_indice,
                    "longueur_portee_tokens": longueur_portee,
                    "indice_discontinu": indice_discontinu,
                    "imbriquee": imbriquee,
                    "scope_traverse_phrase": scope_traverse,
                    "texte_portee": texte_portee,
                    "n_tokens_part": n_tokens,
                    "n_mots_part": n_mots,
                    "n_phrases_part": n_phrases,
                })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = main()
    print(f"\n[OK] {OUT} : {len(df)} negations, {df.shape[1]} colonnes\n")
    print("Negations / sous-corpus :")
    print(df["sous_corpus"].value_counts().to_string())
    print("\nType macro :")
    print(df["type_macro"].value_counts().to_string())
    print("\nType fin (top 12) :")
    print(df["type_fin"].value_counts().head(12).to_string())
    print(f"\nAvec portee : {df['a_une_portee'].sum()}  |  sans portee : {(~df['a_une_portee']).sum()}")
    print(f"Indice discontinu : {df['indice_discontinu'].sum()}  |  imbriquees : {df['imbriquee'].sum()}")
    print(f"Portee traversant >1 phrase : {df['scope_traverse_phrase'].sum()}")