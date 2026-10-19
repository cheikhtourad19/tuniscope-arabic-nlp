"""Read-only native XLSX verification; optionally emit a corpus-free report."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sheet(path: Path, number: int) -> tuple[dict, ET.Element]:
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(item.itertext()) for item in
                       ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
        cells = {}
        for cell in sheet.findall(".//m:sheetData/m:row/m:c", NS):
            text = cell.find("m:v", NS)
            value = text.text if text is not None else ""
            if cell.get("t") == "s":
                value = strings[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(part.text or "" for part in cell.findall(".//m:t", NS))
            formula = cell.find("m:f", NS)
            cells[cell.get("r")] = (value or "", formula.text if formula is not None else None)
        return cells, sheet


def native_controls(sheet: ET.Element) -> dict:
    validations = []
    for item in sheet.findall(".//m:dataValidation", NS):
        formula = item.find("m:formula1", NS)
        validations.append((item.get("sqref"), item.get("type"),
                            formula.text if formula is not None else None))
    pane = sheet.find(".//m:pane", NS)
    # topLeftCell is saved scroll position, not the frozen boundary. Artifact
    # export resets that viewport; disclose it separately instead of asserting
    # that the entire pane/view was preserved byte for byte.
    return {"validations": sorted(validations),
            "pane": {k: v for k, v in pane.attrib.items() if k != "topLeftCell"} if pane is not None else None,
            "columns": [item.attrib for item in sheet.findall("m:cols/m:col", NS)]}


def verify() -> tuple[dict, dict]:
    ledger_path = ROOT / "data/interim/baseline_error_review_assisted_2026-10-17.json"
    ledger = json.loads(ledger_path.read_text())
    source = DIRECTORY / "tuniscope_revue_erreurs_baseline_2026-10-17.xlsx"
    output = DIRECTORY / "tuniscope_analyse_erreurs_baseline_assistee_2026-10-17.xlsx"
    assert digest(source) == ledger["provenance"]["human_workbook_sha256"]
    assert digest(output) == ledger["provenance"]["completed_workbook_sha256"]
    assert digest(ROOT / "scripts/complete_error_review.mjs") == ledger["provenance"]["script_sha256"]
    summaries = {}
    viewport_changes = {}
    for number, corpus in enumerate(("tsac", "madar"), 1):
        before, before_xml = read_sheet(source, number)
        after, after_xml = read_sheet(output, number)
        assert native_controls(before_xml) == native_controls(after_xml), "Native controls or widths changed"
        before_pane = before_xml.find(".//m:pane", NS)
        after_pane = after_xml.find(".//m:pane", NS)
        viewport_changes[corpus] = {"before": before_pane.get("topLeftCell"),
                                   "after": after_pane.get("topLeftCell")}
        allowed = {"A4", "G3", "G8", "H3"} | {f"{c}{r}" for c in "FG" for r in range(9, 39)}
        for address in set(before) | set(after):
            if address not in allowed:
                assert before.get(address, ("", None)) == after.get(address, ("", None)), address
        assert before["H3"][1] == after["H3"][1]
        assert float(after["H3"][0]) == 30
        counts = Counter()
        authors = Counter()
        for row, case in zip(range(9, 39), ledger["tasks"][corpus]["cases"], strict=True):
            assert after[f"A{row}"][0] == case["case_id"]
            for column in "ABCDEFGH":
                assert after[f"{column}{row}"][1] is None
            assert after[f"F{row}"][0] == case["category"]
            assert after[f"G{row}"][0] == case["hypothesis"]
            assert after[f"E{row}"][0] == case["label_opinion"]
            assert case["human_validation_of_ai_additions"] is False
            if before.get(f"F{row}", ("", None))[0]:
                assert before[f"F{row}"] == after[f"F{row}"]
                assert case["authors"]["category"] == "human"
            else:
                assert case["authors"]["category"] == "ai"
            assert after[f"G{row}"][0].startswith("IA — ")
            counts[case["category"]] += 1
            authors[case["authors"]["category"]] += 1
        expected = ledger["tasks"][corpus]["summary"]
        assert dict(counts) == expected["category_counts"]
        assert authors["human"] == expected["human_categories"]
        assert authors["ai"] == expected["ai_categories"]
        summaries[corpus] = expected
    return ledger, {"summary": summaries, "all_cases_documented": True,
                    "human_validation_of_ai_additions": False,
                    "native_xml_verification": "passed", "original_workbook_preserved": True,
                    "native_validation_panes_widths_preserved": True,
                    "saved_scroll_positions": viewport_changes,
                    "labels_predictions_and_human_fields_preserved": True,
                    "provenance": {**ledger["provenance"], "ledger_sha256": digest(ledger_path),
                                   "verifier_sha256": digest(Path(__file__))}}


def report(ledger: dict, summary: dict) -> None:
    directory = ROOT / "reports/error_analysis_assisted_2026-10-17"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["# TuniScope — Analyse qualitative assistée des erreurs classiques", "",
             "**Livrables :** contribution à L07, L10 et L12. Lot initial documenté, livrables complets non achevés.", "",
             "## Origine des champs", "",
             "Nous conservons 60 avis humains sur les labels et les 11 catégories déjà choisies. "
             "Sur délégation explicite, l'IA a proposé les 49 catégories manquantes et les 60 hypothèses courtes. "
             "Ces ajouts ne sont pas encore validés humainement. Le classeur original est conservé.", "",
             "| Corpus | Cas documentés | Avis humains | Catégories humaines | Catégories IA | Hypothèses IA |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for corpus, data in summary["summary"].items():
        lines.append(f"| {corpus.upper()} | {data['documented_cases']} | {data['human_label_opinions']} | "
                     f"{data['human_categories']} | {data['ai_categories']} | {data['ai_hypotheses']} |")
    lines += ["", "## Catégories principales du lot", "",
              "Une seule catégorie principale par cas. Les catégories humaines et IA sont agrégées ici, "
              "mais leur origine est distinguée dans le registre privé. Ce sont des descriptions proposées, pas des causes prouvées.", "",
              "| Corpus | Catégorie | Nombre |", "| --- | --- | ---: |"]
    for corpus, data in summary["summary"].items():
        for category, count in data["category_counts"].items():
            lines.append(f"| {corpus.upper()} | {category} | {count} |")
    lines += ["", "## Interprétation et réserves", "",
              "Pour TSAC, nous relevons notamment la polarité mixte ou ambiguë, les graphies Arabizi, "
              "les expressions figurées et les signes non verbaux. La catégorie Autre regroupe des phénomènes "
              "différents, détaillés seulement dans les notes privées. Leur regroupement n'est pas un mécanisme linguistique unique.", "",
              "Pour MADAR, nous examinons les indices morphologiques et le vocabulaire partagé. Une phrase "
              "courte peut manquer d'indices distinctifs. Un label issu de la provenance du corpus ne garantit "
              "pas que la variété soit identifiable dans chaque texte isolé.", "",
              "Une catégorie humaine de négation est conservée alors qu'aucune négation explicite n'est visible "
              "dans le fragment correspondant. Sa note signale la réserve. Le compte de deux cas de négation "
              "ne signifie donc pas deux négations linguistiquement confirmées.", "",
              "Certaines lectures restent incertaines, y compris lorsqu'une hypothèse est renseignée. "
              "Nous n'inventons ni contexte, ni intention ironique certaine, ni correction du label. "
              "L'absence de catégorie ironie/sarcasme comme catégorie principale n'établit pas une absence d'ironie.", "",
              "Le lot est qualitatif et non proportionnel, choisi parmi les erreurs de validation de la référence "
              "caractères brute v2. Ces comptes ne mesurent pas la prévalence des causes dans le corpus. "
              "Les labels et prédictions étaient visibles. Cette procédure ne fournit ni seconde annotation "
              "humaine indépendante, ni kappa, ni explication causale des poids du modèle.", "",
              "## Vérification et suite", "",
              "Le XLSX complété a été rendu, rouvert et contrôlé. Une lecture indépendante du XML vérifie "
              "les champs préservés, les nouvelles valeurs, les listes de choix, les largeurs, les volets et "
              "les deux compteurs à 30. Aucun essai dans l'application Excel native n'est revendiqué. "
              "La position de défilement enregistrée est réinitialisée par l'export, sans changer les limites des volets figés. "
              "Aucun label, score, source, manifeste ou modèle n'est modifié. Aucun texte du test officiel n'est chargé.", "",
              "La relecture des propositions IA reste à faire, en priorité pour les réserves et expressions "
              "incertaines. L07 reste partiel : notebooks, modèles ultérieurs, évaluation finale et bootstrap "
              "restent à réaliser. Le travail syntaxique sur 50 phrases est distinct. Aucun texte du lot ne "
              "doit être ajouté au train ni à un lexique induit depuis la validation."]
    for name, content in {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                          "README.md": "\n".join(lines) + "\n"}.items():
        target = directory / name
        if target.exists():
            assert target.read_text() == content, "Existing aggregate report differs"
        else:
            with target.open("x") as handle:
                handle.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    saved, result = verify()
    if args.report:
        report(saved, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
