# Schémas générés pour le PowerPoint Silver

Les sept PNG de ce dossier sont des illustrations conceptuelles produites avec le générateur intégré `image_gen`, puis relues. Ils ne représentent ni une sortie de modèle, ni un résultat expérimental. Les graphiques de scores du PowerPoint utilisent séparément les résultats réels.

Les schémas ont été intégrés sans recadrage ni modification de leurs textes après génération. Le schéma CAMeLBERT a fait l’objet d’une correction pour représenter deux modèles indépendants. Les sources et variantes écartées restent dans les fichiers privés de préparation.

## Consignes des schémas retenus

## overview.png

```text
Use case: scientific-educational. One polished academic process-diagram image for a French engineering PowerPoint. Wide 3:1 landscape, high resolution. Warm ivory background #F4F1EB, flat cobalt #3255D9 and teal #117E78 blocks, dark ink #131923 connectors, large crisp French sans-serif lettering, ample padding. No slide title, no decorative objects, no invented metrics, no photographs, no 3D, no extra paragraphs. Draw a clear left-to-right chronological process with EXACTLY FIVE blocks labelled exactly « Caractères et BiGRU », « CAMeLBERT », « Qwen et LoRA », « RAG », « Application et bilan ». Straight visible arrows point only from each block to the next. No cycles or reverse arrows. These are research stages, not a data inference pipeline. Keep labels large and balanced.
```

## bigru.png

```text
Use case: scientific-educational. Create ONE clean academic process diagram image for a French engineering PowerPoint, wide landscape aspect ratio 3:1, high resolution. Background exactly warm ivory #F4F1EB. Flat polished scientific infographic, cobalt #3255D9 and teal #117E78, dark ink #131923, large very legible French sans-serif labels. No slide title, no photographs, no 3D, no decorative icons, no extra paragraphs, generous margins. Diagram for a BiGRU text classifier. LEFT one input block labelled exactly « Texte tokenisé ». It splits into TWO PARALLEL horizontal paths, upper block labelled « Lecture gauche vers droite », lower block labelled « Lecture droite vers gauche ». These TWO paths merge into ONE block labelled « Représentation de la phrase », then ONE final block labelled « Classification ». Every connector arrow goes from input to parallel paths, from both paths to merged representation, then to classification. The lower reading direction is a label about processing the text backwards, NOT a backwards pipeline arrow. Exactly five blocks, two directed parallel paths. Keep all text crisp, no invented scores or numbers. This is a conceptual mechanism, not a statistical chart.
```

## camelbert_v2.png

```text
Correct this CAMeLBERT educational diagram while preserving the ivory background, cobalt and teal palette, polished flat style and legible French typography. IMPORTANT scientific correction: the project has TWO INDEPENDENT adapted encoders, NOT one shared encoder with two heads. Replace the shared branching pipeline with TWO COMPLETE PARALLEL ROWS, with NO connections between rows. UPPER ROW exactly « Texte TSAC » → « Tokenizer » → « CAMeLBERT adapté TSAC » → « Tête TSAC » → « NEG / POS ». LOWER ROW exactly « Texte MADAR » → « Tokenizer » → « CAMeLBERT adapté MADAR » → « Tête MADAR » → « MSA / TN / EG ». Each row has its own CAMeLBERT block and own head. Large left-to-right arrows within each row only. A small caption « Deux modèles distincts » may sit between the rows. Remove the previous shared encoder and branching group. No other text, no title, no metrics. High resolution wide landscape 3:1, text exact and comfortably padded.
```

## lora.png

```text
Use case: scientific-educational. One polished academic process-diagram image for a French engineering PowerPoint. Wide 3:1 landscape, high resolution. Warm ivory background #F4F1EB, flat cobalt #3255D9 and teal #117E78 blocks, dark ink #131923 connectors, large crisp French sans-serif lettering, ample padding. No slide title, no decorative objects, no invented metrics, no photographs, no 3D, no extra paragraphs. Explain LoRA as a frozen base plus a small learned update, NOT as replacing or compressing the base. LEFT an input labelled « Texte et consigne ». CENTER two parallel routes: a larger upper block « Qwen : poids gelés » with a subtle lock symbol and a small lower block « Matrices LoRA entraînées ». Both routes receive the input and merge at a plus symbol on the right. From plus, an arrow goes to the output labelled « Label prédit ». The larger base remains present and fixed while the small route learns. No numbers, no QLoRA, no GPU images. Diagram should be elegant and technically accurate, five conceptual elements with a clear additive merge.
```

## index.png

```text
Use case: scientific-educational. Create ONE clean academic diagram for a French engineering PowerPoint. Wide landscape 3:1 or 16:7, high resolution, warm ivory #F4F1EB background, flat cobalt #3255D9 and teal #117E78 blocks, dark ink #131923 clear directed arrows, very legible French sans-serif labels with generous padding. No slide title, no photographs, no 3D, no decorative objects, no numeric metrics, no extra explanation paragraphs. Diagram of preparation at service startup. LEFT block « Fiches documentaires ». It splits into TWO PARALLEL branches. Upper path « Indexation BM25 » leads to « Mots et fréquences ». Lower path « Encodage E5 » leads to « Vecteurs des fiches ». The two right-hand outputs stay separate but are grouped by a simple bracket labelled « Conservés en mémoire ». One small label above the whole pipeline « Au chargement ». All arrows go from catalogue to processing to stored outputs. No question in this diagram, no database cylinder suggesting an external database, no generation. Exactly five blocks.
```

## embedding.png

```text
Use case: scientific-educational. Create ONE clean academic diagram for a French engineering PowerPoint. Wide landscape 3:1 or 16:7, high resolution, warm ivory #F4F1EB background, flat cobalt #3255D9 and teal #117E78 blocks, dark ink #131923 clear directed arrows, very legible French sans-serif labels with generous padding. No slide title, no photographs, no 3D, no decorative objects, no numeric metrics, no extra explanation paragraphs. Diagram of dense retrieval with E5. TOP ROW « Question » then « Encodeur E5 » then « Vecteur de la question ». LOWER LEFT « Vecteurs des fiches déjà calculés ». Both the question vector and the precomputed card vectors feed a block « Similarité cosinus », then a final output block « Classement des fiches ». Use directional arrows showing both inputs merging into comparison. The catalogue is NOT re-embedded at every query. No translation stage. No invented coordinate values or results. All labels exact French, clearly differentiate question vector and stored card vectors.
```

## rag.png

```text
Use case: scientific-educational. Create ONE clean academic diagram for a French engineering PowerPoint. Wide landscape 3:1 or 16:7, high resolution, warm ivory #F4F1EB background, flat cobalt #3255D9 and teal #117E78 blocks, dark ink #131923 clear directed arrows, very legible French sans-serif labels with generous padding. No slide title, no photographs, no 3D, no decorative objects, no numeric metrics, no extra explanation paragraphs. Create a complete readable request-to-answer process flow with a decision branch. TOP ROW from left to right exactly « Question, langue et moteur » then « Fiches de la langue choisie » then « Recherche BM25 ou E5 ». Arrow goes DOWN from search to a middle box « Trois passages classés », then to a decision diamond « Appui suffisant ? ». The decision has TWO OUTGOING branches. Green branch labelled « Oui » leads to « Extrait de la première fiche + source ». Red branch labelled « Non » leads to « Abstention ». Exactly this ordering. No generation model on the served path, no automatic language detection, no reranker, no fictitious answer. Clear large labels, arrows must all point in the correct direction. Make this the most legible diagram with ample space around the branching diamond.
```

