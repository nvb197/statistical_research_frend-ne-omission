# -*- coding: utf-8 -*-
"""
Source UNIQUE des 'ne' expletifs confirmes (annotes a tort comme indices).

Utilise par corriger_xml.py (pour retirer Cue+Scope du XML) ET par les phases
1-4 (exclusion de secours, au cas ou la table parquet n'aurait pas encore ete
reconstruite depuis le XML corrige). Une seule liste -> aucune divergence possible.

Chaque entree : cue_id (= id du Cue dans le XML) et ref (lien Cue<->Scope dans
le meme DocumentPart). Le part_id se deduit du cue_id (suffixe _Txx retire).
"""
EXPLETIFS = [
    {"cue_id": "La_morte_amoureuse_003_T84", "ref": "R42"},  # comparatif : qu'il n'aurait fallu
    {"cue_id": "Pauline_008_T2",             "ref": "R1"},   # comparatif : qu'elle ne l'est
    {"cue_id": "annodis.er_00385_T1",        "ref": "R1"},   # craindre que ... n'attente a ses jours
    # --- A CONFIRMER : etaient exclus dans une version precedente (sans contexte fourni).
    #     Si ce sont bien des expletifs, decommentez (ajoutez le bon 'ref' pour le XML).
    # {"cue_id": "payetontaf_037_T5", "ref": "?"},
    # {"cue_id": "export_010_T22",    "ref": "?"},
]
EXPLETIFS_CUE_ID = {e["cue_id"] for e in EXPLETIFS}