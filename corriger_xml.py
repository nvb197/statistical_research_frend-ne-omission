# -*- coding: utf-8 -*-
"""
Corrige le XML : retire les 'ne' expletifs (listes dans expletifs.py) annotes a
tort comme indices de negation. Pour chaque expletif, on retire le tag <Cue> ET
les tags <Scope> de meme 'ref' dans le meme DocumentPart, MAIS on conserve le
texte (les <Text> internes restent). Le texte du corpus est donc intact ; seules
les annotations fautives disparaissent. Les denominateurs (tokens/phrases) ne
changent pas ; seul le nombre de negations diminue.

Sortie : my_xml_corpus_clean_corrige.xml  (l'entree n'est pas ecrasee).
Apres correction : reconstruire negations.parquet (build_negations_master.py)
sur ce fichier, puis relancer denominateurs.py + phases 1-4.

Necessite lxml.  Lancer : python corriger_xml.py
"""
import re
from lxml import etree
from expletifs import EXPLETIFS

SRC = "my_xml_corpus_clean.xml"
DST = "my_xml_corpus_clean_corrige.xml"


def part_de(cue_id):
    return re.sub(r"_T\d+$", "", cue_id)


def corriger(src=SRC, dst=DST):
    tree = etree.parse(src)
    root = tree.getroot()
    parts = {dp.get("document_part_id"): dp for dp in root.iter("DocumentPart")}
    rapport = []
    for e in EXPLETIFS:
        pid, ref, cid = part_de(e["cue_id"]), e["ref"], e["cue_id"]
        dp = parts.get(pid)
        if dp is None:
            rapport.append((cid, "PART INTROUVABLE", 0, 0)); continue
        cibles = [el for el in dp.iter() if el.tag in ("Cue", "Scope") and el.get("ref") == ref]
        nb_cue = sum(1 for el in cibles if el.tag == "Cue")
        nb_sco = sum(1 for el in cibles if el.tag == "Scope")
        cue_ok = any(el.tag == "Cue" and el.get("id") == cid for el in cibles)
        for el in cibles:
            el.tag = "_STRIP_"           # marque ; strip_tags conservera le <Text> interne
        rapport.append((cid, "OK" if cue_ok else "REF TROUVE MAIS PAS CE CUE_ID", nb_cue, nb_sco))
    etree.strip_tags(root, "_STRIP_")    # retire les tags marques, garde leur contenu
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return rapport


if __name__ == "__main__":
    rap = corriger()
    print(f"Ecrit : {DST}\n")
    print(f"{'cue_id':32s} {'etat':28s} {'#Cue':>5s} {'#Scope':>7s}")
    for cid, etat, nc, ns in rap:
        print(f"{cid:32s} {etat:28s} {nc:5d} {ns:7d}")