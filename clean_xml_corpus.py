# -*- coding: utf-8 -*-
"""
Nettoyage complet du corpus FReND (toutes les corrections validees par la tutrice).

CINQ operations, dans cet ordre, par DocumentPart :

  1. "Non"/"nan" etiquete <Scope> -> <Cue>            (faux type : c'est un indice)
  2. Paires indice/portee "no-ref" non reliees        -> nouveau ref commun
        (chevauchement ou adjacence <=2 caracteres, via Union-Find)
  3. Portee orpheline (no-ref OU ref reel sans indice) adjacente (<=2) a un indice
        reel -> la portee adopte le ref de cet indice
        (recupere les fragments de portee et les portees dont l'indice est sous
         un ref voisin, ex. "inaccessible" : in- en R3, accessible en R2)
  4. Portee orpheline sans indice adjacent mais avec un indice reel DANS LA MEME
        phrase -> elle adopte le ref de l'indice le plus proche de la phrase
  5. Portee parasite (aucun indice dans la phrase) -> on retire la balise <Scope>
        en conservant le texte (de-balisage)

Le parcours est RECURSIF : les negations imbriquees (chevauchantes) sont lues
correctement et ne sont jamais modifiees.

Sortie : my_xml_corpus_clean.xml + cleaning_log.csv (journal detaille par categorie)
"""

import re
import csv
import os
import sys
import itertools
import xml.etree.ElementTree as ET
from collections import defaultdict

SRC, OUT, LOG = "my_xml_corpus.xml", "my_xml_corpus_clean.xml", "cleaning_log.csv"
_noid = itertools.count(1)


def inner_text(el):
    return "".join(t.text or "" for t in el.iter("Text"))


def build(dp):
    """Parcours recursif : texte lineaire + (element, debut, fin) de chaque Cue/Scope."""
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


def sentence_spans(text):
    spans, start = [], 0
    for m in re.finditer(r'[.!?…]+["»\)]?(?=\s|$)|\n{2,}', text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def sentence_of(spans, pos):
    for a, b in spans:
        if a <= pos < b:
            return a, b
    return pos, pos


def gap(a0, a1, b0, b1):
    """Ecart entre deux intervalles (0 s'ils se chevauchent)."""
    return max(a0 - b1, b0 - a1)


def next_free_ref(elems):
    used = [int(m.group(1))
            for el, _, _ in elems
            for m in [re.fullmatch(r"R(\d+)", el.get("ref") or "")] if m]
    return [max(used) + 1 if used else 1]


def clean_part(dp, parent, log):
    pid = dp.get("document_part_id")

    # --- 1. "Non"/"nan" Scope -> Cue ---
    for el, _, _ in build(dp)[1]:
        if (el.tag == "Scope" and el.get("ref") == "no-ref"
                and inner_text(el).strip().lower() in ("non", "nan")):
            el.tag = "Cue"
            el.set("type", "negation")
            log.append([pid, "1_Non_vers_Cue", "", inner_text(el), ""])

    full, elems = build(dp)
    spans = sentence_spans(full)
    nxt = next_free_ref(elems)

    # --- 2. Appariement des paires "no-ref" (Union-Find, chevauchement/adjacence <=2) ---
    ncue, nsco = defaultdict(lambda: [10**9, -1, []]), defaultdict(lambda: [10**9, -1, []])
    for el, s, e in elems:
        if el.get("ref") != "no-ref":
            continue
        eid = el.get("id") or f"__noid_{next(_noid)}__"
        bucket = ncue if el.tag == "Cue" else nsco
        bucket[eid][0] = min(bucket[eid][0], s)
        bucket[eid][1] = max(bucket[eid][1], e)
        bucket[eid][2].append(el)

    parent_uf = {("c", k): ("c", k) for k in ncue}
    parent_uf.update({("s", k): ("s", k) for k in nsco})

    def find(x):
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    for c in ncue:
        for s in nsco:
            if gap(ncue[c][0], ncue[c][1], nsco[s][0], nsco[s][1]) <= 2:
                parent_uf[find(("c", c))] = find(("s", s))

    comp = defaultdict(list)
    for node in parent_uf:
        comp[find(node)].append(node)
    for members in comp.values():
        cm = [m for m in members if m[0] == "c"]
        sm = [m for m in members if m[0] == "s"]
        if cm and sm:
            new_ref = f"R{nxt[0]}"
            nxt[0] += 1
            for m in cm:
                for el in ncue[m[1]][2]:
                    el.set("ref", new_ref)
            for m in sm:
                for el in nsco[m[1]][2]:
                    el.set("ref", new_ref)
            log.append([pid, "2_paire_no_ref", new_ref,
                        " / ".join(inner_text(el) for m in cm for el in ncue[m[1]][2]),
                        " / ".join(inner_text(el) for m in sm for el in nsco[m[1]][2])])

    # --- 3/4/5 : traiter les portees encore orphelines ---
    full, elems = build(dp)
    real_cue_refs = {el.get("ref") for el, _, _ in elems
                     if el.tag == "Cue" and el.get("ref") not in (None, "no-ref")}
    real_cues = [(s, e, el.get("ref")) for el, s, e in elems
                 if el.tag == "Cue" and el.get("ref") not in (None, "no-ref")]

    # regrouper les portees : par ref reel, ou par id si "no-ref"
    sgroups = defaultdict(lambda: [10**9, -1, []])
    for el, s, e in elems:
        if el.tag != "Scope":
            continue
        r = el.get("ref")
        key = ("no-ref", el.get("id")) if r == "no-ref" else r
        sgroups[key][0] = min(sgroups[key][0], s)
        sgroups[key][1] = max(sgroups[key][1], e)
        sgroups[key][2].append(el)

    for key, (s0, e0, els) in sgroups.items():
        # la portee a-t-elle deja un indice ? (ref reel present dans real_cue_refs)
        if not isinstance(key, tuple) and key in real_cue_refs:
            continue
        sc_txt = " ".join(inner_text(e) for e in els)
        old_ref = key[1] if isinstance(key, tuple) else key

        # 3. indice reel adjacent (<=2) ?
        adj = [(gap(s0, e0, cs, ce), cr) for cs, ce, cr in real_cues if gap(s0, e0, cs, ce) <= 2]
        if adj:
            tref = min(adj)[1]
            for el in els:
                el.set("ref", tref)
            log.append([pid, "3_portee_vers_indice_adjacent", tref, sc_txt, f"(ancien {old_ref})"])
            continue

        # 4. indice reel dans la meme phrase ?
        sa, sb = sentence_of(spans, s0)
        in_sent = [(min(abs(s0 - ce), abs(cs - e0)), cr) for cs, ce, cr in real_cues if cs >= sa and ce <= sb]
        if in_sent:
            tref = min(in_sent)[1]
            for el in els:
                el.set("ref", tref)
            log.append([pid, "4_portee_vers_indice_phrase", tref, sc_txt, f"(ancien {old_ref})"])
            continue

        # 5. parasite : de-baliser. On conserve les <Text> enfants ET, par
        # robustesse (cf. pieges .text/.tail d'ElementTree), le .text et le .tail
        # de la balise retiree (None dans ce corpus, mais on ne perd rien ailleurs).
        for el in els:
            p = parent[el]
            idx = list(p).index(el)
            children = list(el)
            if el.text:                                   # texte avant les enfants
                head = ET.Element("Text"); head.text = el.text
                children = [head] + children
            if el.tail and children:                      # texte apres la balise
                children[-1].tail = (children[-1].tail or "") + el.tail
            p.remove(el)
            for j, child in enumerate(children):
                p.insert(idx + j, child)
        log.append([pid, "5_portee_parasite_retiree", "", sc_txt, ""])


def main():
    if not os.path.exists(SRC):
        sys.exit(f"Fichier introuvable : {SRC}")
    try:
        tree = ET.parse(SRC)
    except ET.ParseError as err:
        sys.exit(f"XML mal forme : {err}")
    root = tree.getroot()
    parent = {c: p for p in root.iter() for c in p}

    log = []
    for dp in root.iter("DocumentPart"):
        clean_part(dp, parent, log)

    tree.write(OUT, encoding="utf-8", xml_declaration=True)
    with open(LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["DocumentPart", "operation", "ref", "indice_ou_portee", "note"])
        w.writerows(log)

    from collections import Counter
    cats = Counter(r[1] for r in log)
    print(f"OK -> {OUT}")
    for k in sorted(cats):
        print(f"   {k}: {cats[k]}")


if __name__ == "__main__":
    main()