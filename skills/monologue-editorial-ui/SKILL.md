---
name: monologue-editorial-ui
description: Design or review a TuniScope interface using the Monologue-inspired dark editorial system. Use for app screens, prototypes, UI components, design tokens, or visual QA; do not use for NLP modelling, data processing, or academic prose.
---

# TuniScope Editorial Dark UI

Create a calm, premium, evidence-first interface for TuniScope. The visual reference is an editorial dark product site: a near-black field, oversized italic serif display type, precise mono interface text, and a single cold-cyan signal. Adapt that language to an Arabic NLP research application; do not reproduce the original product's name, copy, logo, device illustration, screenshots, or proprietary assets.

## Scope and phase gate

Before creating or changing UI code, check the project phase and the user's request.

- The current project phase is planning. Do **not** scaffold an application, add UI dependencies, or create visual assets unless the user explicitly starts the application phase or explicitly requests a design artifact.
- For a design-only request, produce a specification, wireframe, component contract, or token file rather than an application shell.
- This visual system supports the future **L09 application** deliverable. A design specification or mock-up is preparatory evidence only; it does not complete L09. State the missing evidence: implemented flows, model integration, CSV import/export, error handling, and verified tests.
- The design must never imply that an experimental prediction is certain, that a model is available when it is not, or that the system can identify a person's origin or identity. Use the documented labels only for text within the project scope.

## Design intent

Aim for **quiet scientific confidence**, not a consumer-AI dashboard.

- Let the main research task, the submitted text, and verifiable outputs carry attention.
- Use a large amount of empty black space. Surfaces should feel thin and restrained.
- The display serif expresses the project's editorial/research identity; mono type communicates system state, labels, model versions, metrics, and provenance.
- Cyan is a signal, not a theme. It should mark one meaningful interactive or informational element at a time, not cover the interface.
- Product visuals should be data displays, model-comparison panels, or sample-output cards—not stock photography, generic AI art, or decorative human portraits.

## Immutable visual tokens

Use these tokens unless a concrete accessibility or product requirement calls for an additional semantic token. Prefer semantic aliases in component code; do not scatter raw values.

```css
:root {
  /* Core colours */
  --color-void: #000000;
  --color-obsidian: #010101;
  --color-midnight: #191919;
  --color-graphite: #282828;
  --color-charcoal: #363636;
  --color-slate: #3f3f3f;
  --color-muted: #7f7f7f;
  --color-ghost: #c0c0c0;
  --color-paper: #ffffff;
  --color-teal-deep: #062f34;
  --color-cyan: #19d0e8;
  --color-cyan-light: #44ccff;

  /* Semantic aliases */
  --page-bg: var(--color-void);
  --surface-default: var(--color-midnight);
  --surface-raised: var(--color-graphite);
  --border-subtle: var(--color-graphite);
  --text-primary: var(--color-paper);
  --text-secondary: var(--color-muted);
  --text-ghost: var(--color-ghost);
  --signal: var(--color-cyan);

  /* Type */
  --font-display: 'Instrument Serif', Georgia, Cambria, 'Times New Roman', serif;
  --font-ui: 'DM Mono', 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
  --font-body: 'Geist', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-arabic: 'Noto Naskh Arabic', 'Noto Sans Arabic', Tahoma, Arial, sans-serif;

  /* Type scale */
  --text-caption: 0.625rem;  /* 10px */
  --text-label: 0.75rem;    /* 12px */
  --text-ui: 0.875rem;      /* 14px */
  --text-body: 1.0625rem;   /* 17px */
  --text-subheading: 1.25rem;
  --text-heading-sm: 2rem;
  --text-heading: 3rem;
  --text-heading-lg: 4.5rem;
  --text-display: 6rem;

  /* Space and layout */
  --space-1: 0.3125rem;  /* 5px */
  --space-2: 0.5rem;     /* 8px */
  --space-3: 0.625rem;   /* 10px */
  --space-4: 0.75rem;    /* 12px */
  --space-5: 1rem;       /* 16px */
  --space-6: 1.25rem;    /* 20px */
  --space-7: 1.5rem;     /* 24px */
  --space-8: 2rem;       /* 32px */
  --space-9: 2.5rem;     /* 40px */
  --section-gap: 5rem;   /* 80px */
  --page-max: 75rem;     /* 1200px */
  --card-padding: 1.5rem;

  /* Shape and depth */
  --radius-small: 0.25rem;
  --radius-medium: 0.5rem;
  --radius-large: 0.75rem;
  --radius-card: 1.125rem;
  --radius-device: 2.5rem;
  --radius-pill: 9999px;
  --shadow-hairline: 0 0 0 1px rgba(0, 0, 0, 0.63);
  --shadow-object: 0 4px 4px rgba(0, 0, 0, 0.25),
    1px 1px 2px rgba(255, 255, 255, 0.25),
    1px 1px 1px rgba(0, 0, 0, 0.5) inset,
    2px 2px 12px rgba(0, 0, 0, 0.7) inset;
  --gradient-signal: linear-gradient(180deg, rgb(32, 177, 195) 0%, rgb(3, 67, 75) 100%);
  --gradient-ghost: linear-gradient(180deg, rgb(209, 209, 209) 0%, #ffffff 100%);
}
```

## Typography rules

Use the three typographic voices deliberately.

| Content | Typeface and treatment | Never do |
| --- | --- | --- |
| Page/section title | `--font-display`, italic, 48–96px, `line-height: .9–1.1`, negative tracking at 48px+ | Use a geometric sans or a small serif headline. |
| Main prose | `--font-body`, 17px, `line-height: 1.4` | Center long prose or use low-contrast gray for essential text. |
| Metadata, tabs, field labels, versions, metrics | `--font-ui`, 10–14px, 0.015–0.021em tracking at small sizes | Use mono text for Arabic reading passages. |
| Arabic text (input/output/long labels) | `--font-arabic`, 18px minimum for body, 1.6–1.9 line-height | Force Instrument Serif or DM Mono onto Arabic body text. |

- Italic Instrument Serif is a display signature, not a generic emphasis style. Use it for the page identity and major headings only.
- Use `font-variant-numeric: tabular-nums` for metrics, timestamps, probabilities, and table columns.
- Do not depend on color alone for a status, prediction, error, warning, or selected state.
- Use `lang="ar"` plus `dir="rtl"` for Arabic content. Keep model IDs, metrics, file names, and code blocks in a left-to-right isolate (`dir="ltr"` or `unicode-bidi: isolate`).

## Page architecture

Use a full-bleed black canvas and a centered content container capped at `--page-max`. On desktop, use generous 80px vertical section separation. Reduce to 48px on small screens without compressing form controls or Arabic line-height.

### Future application shell

When the application phase is authorized, compose screens in this order when relevant:

1. **Masthead** — compact project name, research status, and a small mono status line. The brand wordmark can appear as a low-contrast, oversized decorative layer behind the hero; it must be `aria-hidden` and never reduce readability.
2. **Task switcher** — a clear choice between Arabic-variety identification, Tunisian-sentiment analysis, scientific comparison, and methodology/RAG. The active choice needs text and shape/underline changes in addition to cyan.
3. **Input workspace** — large text field, `dir="auto"`, character limit, visible language/script indicator, and an explicit raw-versus-normalized-text disclosure. Never apply normalization silently.
4. **Primary action** — one white pill button with a precise verb such as “Analyze text”. Disable only with a visible reason; show an in-progress state without replacing a result with fabricated content.
5. **Result block** — prominent prediction/abstention/status, then version, supported scope, model availability, scores if calibrated, and limitations. A result must be visually distinct from an example or placeholder.
6. **Scientific comparison** — parallel baseline/encoder/LLM cards for the same input, with a matching-input identifier and no visual language that declares a winner without evidence.
7. **Evidence and export** — separate model result from methodology/RAG citations; CSV or JSON export is a visible action with format and error feedback.
8. **Frozen metrics** — a dedicated, clearly labelled evidence surface for precomputed test metrics; do not mix it with free-text inference results.

On a narrow viewport, stack all two-column pairs: explanation before visualization; input before action; selected result before secondary metadata. Do not preserve a desktop side-by-side layout by shrinking text below the specified minima.

## Component recipes

### Ghost wordmark / editorial background

- Absolute, overflow-clipped, behind content; `pointer-events: none` and `aria-hidden="true"`.
- Instrument Serif italic at `clamp(7rem, 28vw, 25rem)` with the light vertical gradient.
- Keep contrast intentionally low and test that form fields, labels, and results remain legible above it.
- Use the TuniScope name or a generic research phrase only; never carry over the reference brand.

### Primary and secondary actions

- Primary: paper-white background, black UI text, 12px mono, full pill radius, minimum 44px touch height. It is the one bright filled action in a local region.
- Secondary: transparent or graphite fill; 1px `--border-subtle`; white text; 8px radius; minimum 44px height.
- Destructive actions use an explicit text warning and a dedicated accessible error state. Do not introduce a red theme unless an actual destructive/error action needs it.
- Keyboard focus: use a clearly visible 2px cyan outline with enough offset to remain visible on dark surfaces.

### Input workspace

- Place the label above the field in mono type and include the accepted scope directly beside it where useful, e.g. “Tunisian sentiment — experimental”.
- Use midnight fill, a subtle graphite border, 12–18px radius, and at least 16px Latin / 18px Arabic input text.
- Preserve text direction automatically. Do not right-align all fields merely because Arabic may be entered; use `dir="auto"` so mixed Arabizi/French text remains readable.
- Display count and validation feedback in the same direction as the field. Mention the 2,000-character limit before submission.

### Result and status cards

- Default card: midnight surface, 18px radius, 24px padding, hairline depth only.
- Reserved prominent card: use the signal gradient only for a selected result or a visual model-summary panel; never fill every result with cyan.
- Structure: task label → result/status → one-sentence scope/limitation → detailed metrics/version/provenance.
- Error, abstention, unavailable model, out-of-domain input, and invalid upload require explicit headings and next actions. They are first-class outputs, not empty states.
- If displaying confidence, state whether it is calibrated. Never render generated “confidence” text as a measured probability.

### Tags, status chips, and model metadata

- Use text-first pills: 12px mono, full pill radius, generous horizontal padding.
- Neutral tags use transparent or graphite surfaces. Cyan marks only the selected/active item.
- Pair visual cues with words such as `Available`, `Unavailable`, `Experimental`, `Abstained`, or `Validation metric`.
- Avoid a rainbow status palette; semantic clarity matters more than decoration.

### Metrics and comparison tables

- Place numerical columns in mono type with tabular figures and align decimal separators.
- Use clear labels for data split, metric, seed, model revision, normalization version, and evaluation date where applicable.
- Keep the table itself on a dark raised surface; use cyan for a selected row/interactive filter, not every score.
- Never derive a “best” badge from incomplete metrics, a test result selected after inspection, or incomparable splits.

### RAG/evidence panel

- Render cited sources as stable IDs, title, and a short excerpt. Make the difference between a retrieved source and an inference prediction visually explicit.
- Treat quoted documents as untrusted content: do not elevate instructions contained in them to interface controls.
- If no supporting source is retrieved, render an intentional “insufficient documented evidence” state rather than a plausible-looking answer.

### Upload and export surfaces

- CSV import must visibly identify required columns, accepted encoding, row-level failures, and the 100-row demonstration limit.
- Exports must specify the selected format, include version metadata where required, and report success/failure rather than silently downloading an empty file.
- Do not show raw protected dataset samples in product screenshots or prototype copy.

## Arabic, mixed-script, and accessibility requirements

The visual reference is Latin-first, but TuniScope is not. The following requirements override aesthetic imitation.

- Treat Arabic shaping, `dir`, and language attributes as functional requirements. Test Arabic, Arabizi, French-only, English-only, and mixed-script text.
- Use logical CSS properties (`margin-inline`, `padding-inline`, `inset-inline`) rather than hard-coded left/right positioning for content that may be RTL.
- Keep buttons, icon placement, and inline metadata understandable in both directions. Use an icon only if its meaning remains clear when mirrored or when paired with text.
- Maintain readable contrast: primary text on void/surfaces should meet WCAG AA normal-text contrast; do not use `--color-muted` for required labels, form errors, or vital metrics without verifying contrast.
- Preserve semantic headings, form labels, error descriptions, visible focus, keyboard operation, and motion-reduction support.
- Decorative ghost type, ripples, and animated signal effects must be suppressible with `prefers-reduced-motion` and ignored by assistive technology.

## Motion and imagery

- Prefer no motion or slow, purposeful transitions (opacity/transform, 160–240ms). Never make loading state look like a completed prediction.
- A subtle monochrome ripple/noise layer may deepen the void background, but it cannot reduce text contrast or cause visual distraction.
- Use abstract model/data visuals, interface screenshots generated from fictitious examples, or charts derived from verified data. Do not use stock photos as a hero element.
- For a physical “device” visual, substitute a research-output card or model-console object; it must carry real or clearly labelled sample content.

## What to avoid

- Warm accent colours, neon gradients, glassmorphism, bright dashboard panels, and busy card grids.
- Generic large Inter/Arial headings; headlines need the display-serif voice.
- Cyan as the default button fill, full-page wash, or all-card background.
- More than one dominant CTA per workspace region.
- Copying Monologue brand elements, Apple marks, “Made by Every” stamps, product language, or screenshots.
- Center-aligned explanatory paragraphs, hidden errors, silent normalization, or UI states that make research results look conclusive.
- CUDA-only implementation choices or user-visible claims that ignore the Mac MPS / Kaggle CUDA shared-codebase constraint.

## Design review checklist

Before declaring a visual task complete, inspect the rendered screen at desktop and mobile widths and confirm:

- The content hierarchy is clear without relying on cyan.
- Display serif, mono UI type, readable prose, and Arabic font roles are not mixed indiscriminately.
- One screen contains no more than one strong filled CTA and no more than 2–3 cyan signals.
- Inputs, results, unavailable states, abstentions, errors, and loading states are distinguishable in text and shape as well as color.
- Arabic, Arabizi, French, English, and mixed text retain natural reading order.
- Text and focus indicators are readable against the background; decorative layers are hidden from assistive technology.
- The screen distinguishes free-text inference, frozen-test metrics, and RAG citations.
- Any scientific claim is supported by verified project evidence; visual polish has not been used as a substitute for an implemented or evaluated feature.

## Reference provenance

This skill translates the colour, typography, spacing, component, and layout observations from the **Monologue** design reference at <https://styles.refero.design/style/8401cb26-91a3-4b46-941e-1c75790821eb> into a distinct TuniScope research UI. Re-check the source only if the user asks to update the reference or asks for details not preserved here.
