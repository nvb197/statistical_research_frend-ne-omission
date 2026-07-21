# -*- coding: utf-8 -*-
"""
PHASE 0 (complement) - Table des DENOMINATEURS (FReND).

Probleme corrige : negations.parquet n'a qu'une ligne par negation, donc il ne
contient que les DocumentParts QUI CONTIENNENT une negation. Sommer n_tokens_part
sur ce fichier omet tous les parts SANS negation -> denominateur trop petit
(surtout sequoia). Ici on parcourt TOUS les DocumentParts du XML.

DEUX SORTIES, deux grains :
  1. denominateurs.parquet  (grain = DocumentPart)
       document_part_id | document_set | sous_corpus | oeuvre | n_tokens | n_phrases | n_mots
     -> sert au denominateur en TOKENS de la densite (jointure per-part). Le
        tokenizer est local : la somme per-part == le total "direct depuis le XML"
        a +/-0.05 %. NE PAS Y TOUCHER.
  2. phrases_par_document.parquet  (grain = Document)
       doc_id | document_set | sous_corpus | oeuvre | n_tokens | n_phrases
     -> nombre de phrases compte 'directement depuis le XML', unite <Document>.
        C'est CE fichier qu'il faut utiliser pour le denominateur en PHRASES
        (cf. phase1_stats.densite), pas la colonne n_phrases per-part.

POURQUOI deux grains pour les phrases : la segmentation de fr_core_news_md repose
sur le parseur (UD), SENSIBLE au contexte. Sommer les phrases per-part sur-compte
les phrases coupees par un bord de part (bords qui n'existent que pour brat,
textes tronques "pour tenir sur un ecran"). Verifie : 6336 documents / 6354 n'ont
qu'UN part (payeton, sequoia, presque tout) -> per-part == per-document pour eux ;
seuls les 18 documents MULTI-PARTS (oeuvres de litbank) doivent etre re-segmentes
sur leur texte concatene. On part donc de la somme per-part et on ne corrige que
ces 18.

NB : meme par <Document>, le total peut ne pas coincider exactement avec celui de
FReND : a tokens identiques, l'ecart restant tient a la VERSION du modele
spaCy (parseur UD), pas a l'unite de texte. Reproduire son chiffre exact suppose
la meme version de fr_core_news_md.

Phrases exactes seulement avec fr_core_news_md. Sans le modele : blank +
sentencizer (tokens EXACTS, phrases APPROXIMATIVES -> relancer avec md).
"""
import re
import xml.etree.ElementTree as ET
import pandas as pd

XML_DEFAUT = "my_xml_corpus_clean.xml"

# Reference : comptes publies de FReND, "directement depuis le XML" (spaCy, parseur UD).
REF_TOKENS = {"litbank": 224151, "payetoncorpus": 297445, "sequoia": 68921}
REF_PHRASES = {"litbank": 11744, "payetoncorpus": 17982, "sequoia": 3096}


def charger_nlp():
    import spacy
    try:
        nlp = spacy.load("fr_core_news_md")
        mode = "fr_core_news_md"
    except Exception:
        nlp = spacy.blank("fr"); nlp.add_pipe("sentencizer")
        mode = "blank+sentencizer"
    nlp.max_length = 3_000_000
    return nlp, mode


def _texte(el):
    """Texte lineaire d'un element (DocumentPart OU Document), EXACTEMENT comme
    build_negations_master : on ignore les <Annotator> et on ne prend que le
    contenu des <Text>. Appele sur un <Document>, concatene le texte de tous ses
    parts dans l'ordre."""
    buf = []
    def w(node):
        for ch in node:
            if ch.tag == "Annotator":
                continue
            if ch.tag == "Text":
                buf.append(ch.text or "")
            else:
                w(ch)
    w(el)
    return "".join(buf)


def parts_du_xml(chemin=XML_DEFAUT):
    """[(part_id, corpus, sous_corpus, texte), ...] pour TOUS les parts."""
    root = ET.parse(chemin).getroot()
    out = []
    def rec(el, corpus, sous):
        for ch in el:
            if ch.tag == "DocumentSet":
                t = ch.get("type")
                rec(ch, corpus or t, t)          # corpus = le plus externe ; sous = le plus interne
            elif ch.tag == "DocumentPart":
                pid = ch.get("document_part_id")
                out.append((pid, corpus, sous or corpus, _texte(ch)))
            else:
                rec(ch, corpus, sous)
    rec(root, None, None)
    return out


def documents_du_xml(chemin=XML_DEFAUT):
    """[(doc_id, corpus, sous_corpus, texte_concatene, [part_ids]), ...] : une
    entree par <Document>, texte = concatenation de tous ses parts (id = <DocID>)."""
    root = ET.parse(chemin).getroot()
    out = []
    def rec(el, corpus, sous):
        for ch in el:
            if ch.tag == "DocumentSet":
                t = ch.get("type")
                rec(ch, corpus or t, t)
            elif ch.tag == "Document":
                pids = [p.get("document_part_id") for p in ch.findall(".//DocumentPart")]
                out.append((ch.findtext("DocID"), corpus, sous or corpus, _texte(ch), pids))
            else:
                rec(ch, corpus, sous)
    rec(root, None, None)
    return out


def construire(chemin=XML_DEFAUT, ecrire=True):
    """Table des denominateurs PER-PART (schema inchange). Sert a la densite tokens."""
    nlp, mode = charger_nlp()
    parts = parts_du_xml(chemin)
    textes = [p[3] for p in parts]
    docs = nlp.pipe(textes, batch_size=64)
    lignes = []
    for (pid, corpus, sous, _), doc in zip(parts, docs):
        lignes.append({
            "document_part_id": pid,
            "document_set": corpus,
            "sous_corpus": sous,
            "oeuvre": re.sub(r"_\d+$", "", pid) if pid else None,
            "n_tokens": len(doc),
            "n_phrases": sum(1 for _ in doc.sents),   # phrases LOCALES (per-part)
            "n_mots": sum(1 for t in doc if t.is_alpha),
        })
    den = pd.DataFrame(lignes)
    if ecrire:
        den.to_parquet("denominateurs.parquet", index=False)
    print(f"[denominateurs] mode spaCy = {mode} | {len(den)} parts")
    if mode == "blank+sentencizer":
        print("  [!] PHRASES approximatives : relancer avec fr_core_news_md pour les phrases exactes.")
    return den, mode


def table_phrases_document(den, chemin=XML_DEFAUT, ecrire=True):
    """
    Construit la table des PHRASES au grain <Document> (le bon denominateur en
    phrases). Economique : pour un document mono-part, on reutilise n_phrases de
    den ; seuls les documents MULTI-PARTS (18 oeuvres de litbank) sont reparses
    sur leur texte concatene. oeuvre = meme regle que den (suffixe _\\d+ retire),
    de sorte que l'agregation hierarchique de phase1 reste coherente.
    """
    nlp, mode = charger_nlp()
    par_part = den.set_index("document_part_id")["n_phrases"].to_dict()
    docs = documents_du_xml(chemin)
    multipart = [d for d in docs if len(d[4]) > 1]
    n_doc_multi = {}
    if multipart:
        for (did, corpus, sous, _, pids), sd in zip(multipart, nlp.pipe([m[3] for m in multipart], batch_size=16)):
            n_doc_multi[did] = sum(1 for _ in sd.sents)
    toks_part = den.set_index("document_part_id")["n_tokens"].to_dict()
    lignes = []
    for did, corpus, sous, _, pids in docs:
        if len(pids) > 1:
            nphr = n_doc_multi[did]
        else:
            nphr = int(par_part.get(pids[0], 0))
        lignes.append({
            "doc_id": did,
            "document_set": corpus,
            "sous_corpus": sous,
            # oeuvre : MEME regle que den (depuis un part_id, pas le DocID qui peut
            # finir en '.txt' et fausser le strip) -> les cles hierarchiques de
            # phase1 (corpus/sous_corpus/oeuvre) restent alignees entre les 2 tables.
            "oeuvre": re.sub(r"_\d+$", "", pids[0]) if pids else None,
            "n_tokens": int(sum(toks_part.get(p, 0) for p in pids)),
            "n_phrases": nphr,
        })
    phr = pd.DataFrame(lignes)
    if ecrire and mode == "fr_core_news_md":
        phr.to_parquet("phrases_par_document.parquet", index=False)
    elif ecrire:
        print("  [!] phrases approximatives (sentencizer) : phrases_par_document.parquet NON ecrit (relancer avec md).")
    return phr, mode


if __name__ == "__main__":
    den, mode = construire()
    agg = den.groupby("document_set").agg(parts=("document_part_id", "nunique"),
                                          tokens=("n_tokens", "sum"),
                                          phrases=("n_phrases", "sum")).reset_index()
    print("\n=== Verification TOKENS (vs FReND) ===")
    for _, r in agg.iterrows():
        c = r.document_set
        if c in REF_TOKENS:
            print(f"  {c:14s}: {int(r.tokens):7d} tokens | ref. {REF_TOKENS[c]:7d} | écart {(r.tokens-REF_TOKENS[c])/REF_TOKENS[c]*100:+.2f}% "
                  f"| {int(r.parts)} parts")

    print("\n=== PHRASES per-document (vs FReND) ===")
    phr, _ = table_phrases_document(den)
    pp = den.groupby("document_set")["n_phrases"].sum().to_dict()       # per-part (pour reference)
    pd_ = phr.groupby("document_set")["n_phrases"].sum().to_dict()      # per-document (a utiliser)
    for c in ["litbank", "payetoncorpus", "sequoia"]:
        ed = (pd_[c] - REF_PHRASES[c]) / REF_PHRASES[c] * 100
        print(f"  {c:14s}: per-part {int(pp[c]):6d} | per-document {int(pd_[c]):6d} | ref. {REF_PHRASES[c]:6d} | écart doc {ed:+.1f}%")
    if mode != "fr_core_news_md":
        print("  [!] phrases APPROXIMATIVES (sentencizer). Relancer avec fr_core_news_md.")

    print("\n=== Verification mapping sous_corpus (XML vs parquet) ===")
    pq = pd.read_parquet("negations.parquet").drop_duplicates("document_part_id")
    j = pq.merge(den, on="document_part_id", suffixes=("_pq", "_xml"))
    print(f"  sous_corpus identique : {(j.sous_corpus_pq==j.sous_corpus_xml).mean()*100:.1f}%   |   "
          f"oeuvre identique : {(j.oeuvre_pq==j.oeuvre_xml).mean()*100:.1f}%")