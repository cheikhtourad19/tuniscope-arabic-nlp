"""Validate AI study spans against frozen train texts; render private/public notes."""
from collections import Counter
import html
import json

from export_linguistic_sample import CONFIG, OUTPUT, INDEX, ROOT, build_sample, digest

ANNOTATIONS = ROOT / "data/interim/linguistic_annotations_ai_train_v1.json"
PRIVATE_NOTES = ROOT / "explanation NLP/08_etude_linguistique_50_textes_train.md"
REPORT = ROOT / "reports/linguistic_study_train_v1"
ROLES = {"NOM", "VERBE", "ADJECTIF", "ADVERBE", "PARTICULE", "EXPRESSION"}


def validate_annotations(cases, annotations):
    if len(annotations) != len(cases) or len({a["case_id"] for a in annotations}) != len(annotations):
        raise ValueError("Missing or duplicate linguistic annotations")
    by_id = {a["case_id"]: a for a in annotations}
    if set(by_id) != {c["case_id"] for c in cases}:
        raise ValueError("Linguistic case IDs differ")
    result = []
    for case in cases:
        annotation = by_id[case["case_id"]]
        if case["split"] != "train":
            raise ValueError("Study must stay on train")
        if annotation["cue_role"] not in ROLES or annotation["negation_status"] not in {"présente", "absente", "incertaine"}:
            raise ValueError("Unknown linguistic choice")
        if not isinstance(annotation["negation_markers"], list):
            raise ValueError("Markers must be an explicit list")
        for span in [annotation["cue"], *annotation["negation_markers"]]:
            if not isinstance(span, str) or not span or span not in case["text_raw"]:
                raise ValueError(f"Unobserved evidence span in {case['case_id']}")
        if annotation["negation_status"] == "présente" and not annotation["negation_markers"]:
            raise ValueError("Proposed negation requires observed evidence")
        if annotation["negation_status"] == "absente" and annotation["negation_markers"]:
            raise ValueError("Absent negation cannot have negative markers")
        for field in ("negation_scope", "target", "reading", "irony"):
            if not isinstance(annotation[field], str) or not annotation[field].strip():
                raise ValueError("Every analysis axis must be documented")
        result.append((case, annotation))
    return result


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError("Existing linguistic output differs; preserve it and version the revision")
    else:
        with path.open("x") as stream:
            stream.write(content)


def main():
    sample = json.loads(OUTPUT.read_text())
    index = json.loads(INDEX.read_text())
    proposals = json.loads(ANNOTATIONS.read_text())
    if digest(OUTPUT) != index["payload_sha256"] or sample != build_sample():
        raise ValueError("Frozen sample differs from sources/configuration/selector")
    if proposals["human_verified"] is not False:
        raise ValueError("AI proposal file cannot assert human verification")
    pairs = validate_annotations(sample["cases"], proposals["cases"])
    per_corpus = {}
    for corpus in ("tsac", "madar"):
        subset = [(c,a) for c,a in pairs if c["corpus"] == corpus]
        per_corpus[corpus] = {"texts": len(subset),
            "proposed_negation_status": dict(Counter(a["negation_status"] for _,a in subset)),
            "cue_roles": dict(Counter(a["cue_role"] for _,a in subset)),
            "candidate_screen_with_ai_negation_absent": sum(bool(c["marker_candidates"]) and a["negation_status"] == "absente" for c,a in subset),
            "no_candidate_screen_with_ai_negation_present": sum(not c["marker_candidates"] and a["negation_status"] == "présente" for c,a in subset)}
    summary = {"selection": sample["summary"], "ai_documented_texts": len(pairs),
        "human_verified_annotations": 0, "annotations_are_gold_standard": False,
        "syntax_complete_token_tagging": False, "negation_rule_evaluated": False,
        "proposed_analysis": per_corpus,
        "provenance": {"sample_sha256": digest(OUTPUT), "index_sha256": digest(INDEX),
                       "annotations_sha256": digest(ANNOTATIONS), "config_sha256": digest(CONFIG),
                       "renderer_sha256": digest(__file_path()), **sample["provenance"]}}
    # Selection snapshot keeps its historical zero-annotation status. The
    # separate ai_documented_texts field records the current drafted analysis.
    private = ["# Étude linguistique de 50 textes du train", "",
        "Propositions IA à relire. Aucun label source n'est modifié. Les textes sont privés et ne doivent pas être publiés.", "",
        "Indice grammatical = extrait étudié, pas forcément un mot de sentiment. La portée désigne ce que la négation concerne. "
        "La cible est ce dont on parle ou ce qu'on apprécie. Une absence d'ironie établie n'est pas une preuve d'absence d'ironie.", "",
        "Les étiquettes sont légères : NOM, VERBE, ADJECTIF, ADVERBE (emploi adverbial), PARTICULE ou EXPRESSION. "
        "Ce ne sont ni des analyses exhaustives ni des sorties validées d'un analyseur automatique."]
    for case, a in pairs:
        private += ["", f"## {case['case_id']} — {case['corpus'].upper()} — {case['label']}", "",
            f"Source privée : {case['source_file']}:{case['source_line']} ; train v2 ; script {case['writing_system']}.", "",
            "> " + html.escape(case["text_raw"]).replace("\n", "\n> "), "",
            f"- Indice grammatical proposé : **{html.escape(a['cue'])}** — {a['cue_role']}.",
            f"- Négation proposée : **{a['negation_status']}**. {a['negation_scope']}",
            f"- Cible : {a['target']}", f"- Polarité et interprétation : {a['reading']}",
            f"- Ironie / pragmatique : {a['irony']}", "- Provenance : IA ; relecture humaine non effectuée."]
    write_once(PRIVATE_NOTES, "\n".join(private) + "\n")
    public = ["# TuniScope — Première étude linguistique assistée sur le train", "",
        "**Livrables :** L04, L10 et L12, contributions partielles. Cinquante textes documentés par IA, sans validation humaine indépendante.", "",
        "## Périmètre", "",
        "Nous étudions 25 commentaires TSAC et 25 phrases MSA MADAR issus exclusivement du train v2. "
        "Les quotas enrichissent les scripts et marqueurs candidats ; le lot n'est pas représentatif des corpus. "
        "Le sélecteur ne lit aucun score, prédiction ou texte de validation/test. "
        "Le snapshot de sélection conserve son état initial sans annotation ; un registre distinct contient les 50 propositions IA.", "",
        "## Observations proposées", "",
        "| Corpus | Textes | Négation proposée présente | Absente | Incertaine | Candidats avec négation proposée absente | Sans candidat avec négation proposée présente |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for corpus, s in per_corpus.items():
        n = s["proposed_negation_status"]
        public.append(f"| {corpus.upper()} | {s['texts']} | {n.get('présente',0)} | {n.get('absente',0)} | {n.get('incertaine',0)} | "
                      f"{s['candidate_screen_with_ai_negation_absent']} | {s['no_candidate_screen_with_ai_negation_present']} |")
    public += ["", "Ces comptes décrivent des propositions IA à relire. Ils ne constituent ni une mesure de précision/rappel "
        "du détecteur, ni une annotation de référence, ni une prévalence de négation dans le corpus.", "",
        "Nous relevons des négations grammaticales, des réponses négatives isolées, des constructions privatives, "
        "des exceptions et des locutions non négatives malgré un marqueur apparent. Une négation peut servir un éloge, "
        "porter sur une autre cible ou rester dans une proposition subordonnée. La présence de négation n'impose pas un label NEG.", "",
        "Plusieurs scripts MIXED sont dus à des restes techniques latins et non à un changement de langue. "
        "Des adjectifs peuvent décrire une quantité ou une taille, sans exprimer un avis. Les questions et impératifs "
        "du corpus dialectal ne reçoivent pas de nouveaux labels de sentiment.", "",
        "## Contrôles et suite", "",
        "Le rendu vérifie la sélection contre les sources/manifestes et les empreintes, les 50 identifiants, "
        "les extraits exacts, les catégories autorisées et les champs d'analyse. Ces contrôles de cohérence "
        "ne certifient pas la justesse linguistique des propositions. Les textes et notes détaillées restent privés.", "",
        "La relecture humaine, l'évaluation d'une règle de portée locale, la baseline à règles et les notebooks "
        "restent à faire. Le lexique éventuellement induit doit provenir du train uniquement. "
        "Aucun label, modèle, score, manifeste ou corpus d'origine n'a été modifié."]
    write_once(REPORT / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_once(REPORT / "README.md", "\n".join(public) + "\n")
    print(json.dumps({"ai_documented_texts": len(pairs), "human_verified_annotations": 0,
                      "proposed_analysis": per_corpus}, ensure_ascii=False, indent=2))


def __file_path():
    from pathlib import Path
    return Path(__file__)


if __name__ == "__main__":
    main()
