"""Export a source-bound, unannotated documentary review pack."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from search import CARDS, load_cards

SAMPLE = ROOT / 'rag/review_sample_v1.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    cards = load_cards()
    pairs = {}
    for card in cards:
        pairs.setdefault(card.id.rsplit('.', 1)[0], {})[card.lang] = card
    if len(cards) != 44 or any(set(pair) != {'fr', 'ar'} for pair in pairs.values()):
        raise ValueError('Le catalogue v1 doit contenir 22 paires FR/AR.')
    for topic, pair in pairs.items():
        if pair['fr'].version != pair['ar'].version or pair['fr'].references != pair['ar'].references:
            raise ValueError(f'Versions ou références différentes pour {topic}.')

    # Use the already disclosed blind sample, without reading proposed relevance labels.
    questions = []
    for line in SAMPLE.read_text().splitlines():
        if re.match(r'^\| Q\d+ \|', line):
            _, identifier, lang, question, judgment, remark, _ = [part.strip() for part in line.split('|')]
            if judgment or remark:
                raise ValueError('La grille source doit rester vierge pour un export indépendant.')
            questions.append((identifier, lang, question))
    if (len(questions) != 12 or len({row[0] for row in questions}) != 12
            or Counter(row[1] for row in questions) != {'fr': 6, 'ar': 6}):
        raise ValueError('La grille doit contenir 12 questions distinctes, dont 6 par langue.')

    # mkdir refuses an existing pack so previous human decisions cannot be overwritten.
    output.mkdir(parents=True, exist_ok=False)
    references = sorted({ref for card in cards for ref in card.references})
    for ref in references:
        target = output / 'sources' / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / ref, target)

    catalogue = [
        '# Relecture des fiches documentaires', '',
        'Nom ou pseudonyme du relecteur — À renseigner', '',
        'Date de lecture — À renseigner', '',
        f'Empreinte SHA-256 du catalogue — `{sha(CARDS)}`', '',
        'Lire les deux versions de chaque sujet et consulter les sources liées. '
        'Pour chaque langue, cocher une seule décision. « Conforme » signifie que le contenu '
        'est soutenu par les sources. Pour l’arabe, vérifier aussi la fidélité de la traduction. '
        'Préciser toute erreur ou ambiguïté dans la correction proposée. '
        'Une décision laissée vide reste à revoir.', '',
        'Les jugements humains sont à renseigner dans ce fichier. '
        'Les fichiers sous `sources/` sont des copies datées des références.', '',
    ]
    for number, (topic, pair) in enumerate(pairs.items(), 1):
        catalogue.extend([f'## {number}. {pair["fr"].title}', ''])
        for lang, label in [('fr', 'Français'), ('ar', 'Arabe standard')]:
            card = pair[lang]
            catalogue.extend([
                f'### {label} — `{card.id}`, version {card.version}', '',
                card.title, '', card.text, '',
                '- [ ] Conforme', '- [ ] À corriger', '- [ ] Incertain', '',
                'Correction proposée et référence précise — À renseigner si nécessaire', '',
            ])
        catalogue.append('Sources du sujet')
        catalogue.append('')
        catalogue.extend(f'- [{ref}](sources/{ref})' for ref in pair['fr'].references)
        catalogue.append('')
    (output / '01_fiches_a_relire.md').write_text('\n'.join(catalogue), encoding='utf-8')

    for reader in ('A', 'B'):
        review = [
            f'# Relecture indépendante — lecteur {reader}', '',
            'Nom ou pseudonyme du relecteur — À renseigner', '',
            'Date de lecture — À renseigner', '',
            f'Empreinte SHA-256 du catalogue — `{sha(CARDS)}`', '',
            'Utiliser le catalogue `01_fiches_a_relire.md` et ses sources. '
            'Chercher toutes les fiches qui répondent substantiellement à chaque question '
            'dans la langue indiquée. Inscrire leurs identifiants, ou « aucune » si aucune '
            'fiche ne permet de répondre. Une fiche liée au sujet peut être insuffisante.', '',
            'Remplir seul, avant de consulter le jugement de l’autre lecteur, '
            'les annotations proposées ou les résultats des moteurs. '
            'Conserver le fichier original rempli avant tout arbitrage.', '',
        ]
        for identifier, lang, question in questions:
            review.extend([
                f'## {identifier} — {lang}', '', question, '',
                'Fiches pertinentes ou aucune — À renseigner', '',
                'Remarque ou justification — À renseigner', '',
            ])
        (output / f'02_lecteur_{reader}.md').write_text('\n'.join(review), encoding='utf-8')

    guide = [
        '# Dossier de revue documentaire', '',
        '1. Remplir `01_fiches_a_relire.md` pour vérifier les faits et les traductions des 44 fiches.',
        '2. Fournir à chaque lecteur une copie vierge du catalogue, du dossier `sources/` '
        'et de sa grille `02_lecteur_A.md` ou `02_lecteur_B.md`.',
        '3. Recueillir deux avis humains indépendants. Une seconde passe de la même personne '
        'ou une réponse d’IA ne constitue pas un deuxième lecteur humain.',
        '4. Conserver les deux grilles remplies, comparer les identifiants retenus '
        'puis documenter les désaccords et leur arbitrage.',
        '5. Après les corrections, versionner les fiches ou annotations concernées, '
        'recalculer les empreintes et refaire la validation avant de figer les réglages.', '',
        'Dans VS Code, modifier les cases et les champs dans l’éditeur Markdown. '
        'Un champ vide garde le statut « à revoir ».', '',
        'La préparation du dossier vérifie la structure, les paires de langues et les fichiers '
        'de référence. Les décisions humaines restent vierges. '
        'L’évaluation des questions de test documentaire suivra la revue et le gel du protocole.', '',
    ]
    (output / '00_lire_en_premier.md').write_text('\n'.join(guide), encoding='utf-8')
    proof = {
        'cards': len(cards), 'bilingual_topics': len(pairs), 'source_files': len(references),
        'independent_review_questions': len(questions), 'human_review': 'pending',
        'cards_sha256': sha(CARDS), 'sample_sha256': sha(SAMPLE),
        'exporter_sha256': sha(Path(__file__)),
        'source_sha256': {ref: sha(ROOT / ref) for ref in references},
        'pack_sha256': {str(path.relative_to(output)): sha(path)
                        for path in sorted(output.rglob('*')) if path.is_file()},
    }
    (output / 'integrity.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Nouveau dossier de revue')
    args = parser.parse_args()
    proof = prepare(args.output)
    print(f"{proof['cards']} fiches, {proof['bilingual_topics']} sujets bilingues, "
          f"{proof['source_files']} sources et deux grilles vierges de "
          f"{proof['independent_review_questions']} questions.")
    print(f'Dossier de revue — {args.output.resolve()}')
