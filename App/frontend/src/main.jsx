import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUpRight,
  ArrowRight,
  Check,
  ChevronDown,
  Download,
  FileText,
  Layers3,
  LoaderCircle,
  Menu,
  Plus,
  RefreshCw,
  ScanText,
  Upload,
  X,
  CircleHelp,
} from "lucide-react";
import "@fontsource/instrument-serif/latin-400-italic.css";
import "@fontsource/dm-mono/latin-400.css";
import "@fontsource/geist/latin-400.css";
import "@fontsource/geist/latin-500.css";
import "@fontsource/noto-naskh-arabic/arabic-400.css";
import "./styles.css";

const taskNames = {
  sentiment: "Sentiment tunisien",
  dialect: "Variété d’arabe",
};
const labelNames = {
  POS: "Positif",
  NEG: "Négatif",
  MSA: "Arabe standard",
  TN: "Tunisien",
  EG: "Égyptien",
};
const examples = {
  sentiment: [
    "el service behi barcha",
    "el service moch behi",
    "الخدمة باهية برشا",
  ],
  dialect: ["وين المحطة؟", "أين المحطة؟", "فين المحطة؟"],
};
const percent = (value) => `${(value * 100).toFixed(2).replace(".", ",")} %`;
const number = (value) => Number(value).toLocaleString("fr-FR");
const modelName = (id) =>
  `${id.split("__")[0].replace("tfidf_", "TF-IDF ").replace("char", "caractères").replace("word", "mots")} · ${id.endsWith("raw-v0") ? "brut" : "normalisé"}`;

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(
      typeof payload.detail === "string"
        ? payload.detail
        : "Requête invalide ou serveur indisponible.",
    );
  return payload;
}

function csvCell(value) {
  const text = String(value ?? "");
  // Spreadsheet formula protection for user-controlled text and identifiers.
  return `"${(/^[\s]*[=+\-@\t\r]/.test(text) ? "'" : "") + text.replaceAll('"', '""')}"`;
}
function download(data, format, name) {
  let content;
  if (format === "json") content = JSON.stringify(data, null, 2);
  else {
    const rows = Array.isArray(data) ? data : [data];
    const columns = [
      "id",
      "task",
      "model",
      "status",
      "label",
      "text_raw",
      "text_norm",
      "script",
      "scores",
      "scores_calibrated",
      "versions",
      "warnings",
      "error",
    ];
    content =
      "\ufeff" +
      [
        columns.join(","),
        ...rows.map((row) =>
          columns
            .map((key) =>
              csvCell(
                typeof row[key] === "object"
                  ? JSON.stringify(row[key])
                  : row[key],
              ),
            )
            .join(","),
        ),
      ].join("\r\n");
  }
  const url = URL.createObjectURL(
    new Blob([content], {
      type: format === "json" ? "application/json" : "text/csv;charset=utf-8",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `${name}.${format}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function App() {
  const [page, setPage] = useState("workspace");
  const [task, setTask] = useState("sentiment");
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [text, setText] = useState("");
  const [model, setModel] = useState("tfidf_char__raw-v0");
  const [comparison, setComparison] = useState(false);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [exportNotice, setExportNotice] = useState("");
  const [file, setFile] = useState(null);
  const [batch, setBatch] = useState(null);
  const [menu, setMenu] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const choices =
    health?.models?.filter((m) => m.task === task && m.available) || [];
  const available = choices.some((m) => m.id === model);
  const length = [...text].length;
  async function refresh() {
    setHealthError("");
    try {
      const [h, m] = await Promise.all([api("/health"), api("/metrics")]);
      setHealth(h);
      setMetrics(m);
    } catch {
      setHealth(null);
      setHealthError(
        "Connexion au backend impossible. Vérifier son démarrage, puis réessayer.",
      );
    }
  }
  useEffect(() => {
    refresh();
  }, []);
  useEffect(() => {
    if (choices.length && !available) setModel(choices[0].id);
  }, [task, health]);
  function navigate(next) {
    setPage(next);
    setMenu(false);
    setError("");
    setExportNotice("");
  }
  function changeTask(next) {
    setTask(next);
    setResults([]);
    setBatch(null);
    setError("");
    setExportNotice("");
  }
  function exportData(data, format, name) {
    try {
      download(data, format, name);
      setExportNotice(`Export ${format.toUpperCase()} préparé.`);
    } catch {
      setError("L’export a échoué. Réessayer.");
    }
  }
  async function analyze(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResults([]);
    setExportNotice("");
    try {
      const response = await api(comparison ? "/compare" : "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, task, model }),
      });
      setResults(comparison ? response.results : [response]);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function analyzeBatch(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setBatch(null);
    setExportNotice("");
    const data = new FormData();
    data.append("file", file);
    data.append("task", task);
    data.append("model", model);
    try {
      setBatch(await api("/batch", { method: "POST", body: data }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  const readyText = healthError
    ? "Hors connexion"
    : !health
      ? "Connexion…"
      : health.ready
        ? "Modèles chargés"
        : "Modèles indisponibles";
  return (
    <>
      <header className="masthead">
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("workspace");
          }}
          className="brand"
        >
          tuni<span>scope</span>
          <i />
        </a>
        <nav aria-label="Navigation principale" className={menu ? "open" : ""}>
          {[
            ["workspace", "Atelier"],
            ["batch", "Traitement CSV"],
            ["metrics", "Résultats"],
            ["about", "Le projet"],
          ].map(([key, name]) => (
            <button
              key={key}
              onClick={() => navigate(key)}
              aria-current={page === key ? "page" : undefined}
            >
              {name}
            </button>
          ))}
        </nav>
        <div className="header-right">
          <span className="research-badge">RECHERCHE · V0.1</span>
          <button
            className="icon mobile-toggle"
            onClick={() => setMenu(!menu)}
            aria-label={menu ? "Fermer le menu" : "Ouvrir le menu"}
          >
            {menu ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </header>
      <main>
        <section className="hero">
          <div>
            <p className="eyebrow">
              <span className="tiny-line" /> LANGUE, VARIÉTÉS & SENTIMENT
            </p>
            <h1>
              {page === "workspace" ? (
                <>
                  Les mots ont
                  <br />
                  plusieurs <em>nuances.</em>
                </>
              ) : page === "batch" ? (
                <>
                  Un fichier.
                  <br />
                  <em>Plusieurs lectures.</em>
                </>
              ) : page === "metrics" ? (
                <>
                  Mesurer, puis
                  <br />
                  <em>comprendre.</em>
                </>
              ) : (
                <>
                  Un terrain
                  <br />
                  <em>d’expérimentation.</em>
                </>
              )}
            </h1>
            <p className="hero-copy">
              {page === "workspace"
                ? "Explorez le sentiment tunisien et les variétés de l’arabe. Un texte, des modèles, une lecture à comparer."
                : page === "batch"
                  ? "Analysez jusqu’à 100 textes en une fois. Conservez les identifiants et exportez chaque résultat, y compris les erreurs."
                  : page === "metrics"
                    ? "Les performances enregistrées de nos modèles, sur des exemples séparés de leur entraînement."
                    : "TuniScope étudie ce que les modèles reconnaissent dans les textes arabes — et ce qui leur échappe."}
            </p>
          </div>
          <aside className="hero-note">
            <span className="orbit" aria-hidden="true">
              <i />
              <b />
            </span>
            <p className="mono">
              DEUX TÂCHES.
              <br />
              DES LIMITES EXPLICITES.
            </p>
            <span>
              Arabe standard · Tunisien · Égyptien
              <br />
              Sentiment positif · négatif
            </span>
          </aside>
        </section>

        <div className="workspace-top">
          <span className="eyebrow">
            {page === "workspace"
              ? "01 / VOTRE TEXTE"
              : page === "batch"
                ? "02 / VOTRE FICHIER"
                : page === "metrics"
                  ? "03 / PREUVES EXPÉRIMENTALES"
                  : "04 / MÉTHODE & PÉRIMÈTRE"}
          </span>
          <div className="connection">
            <span
              className={health?.ready ? "status-dot active" : "status-dot"}
            />
            {readyText}
            <button
              className="icon"
              onClick={refresh}
              aria-label="Actualiser la connexion"
            >
              <RefreshCw size={14} />
            </button>
          </div>
        </div>
        {healthError && (
          <div className="notice error" role="alert">
            {healthError}
          </div>
        )}
        {health && !health.ready && (
          <div className="notice" role="status">
            <strong>Modèles indisponibles.</strong>{" "}
            {health.error ||
              "Le bundle de modèles n’est pas complet. Vérifier les fichiers puis reconstruire le backend."}
          </div>
        )}
        {error && (
          <div className="notice error" role="alert">
            {error}
          </div>
        )}
        <p className="sr-only" role="status" aria-live="polite">
          {busy ? "Traitement en cours." : exportNotice}
        </p>

        {(page === "workspace" || page === "batch") && (
          <>
            <div
              className="task-switch"
              role="group"
              aria-label="Choisir la tâche"
            >
              {Object.entries(taskNames).map(([key, name]) => (
                <button
                  disabled={busy}
                  className={task === key ? "selected" : ""}
                  key={key}
                  onClick={() => changeTask(key)}
                >
                  {key === "sentiment" ? (
                    <ScanText size={16} />
                  ) : (
                    <Layers3 size={16} />
                  )}{" "}
                  {name}
                  <span>
                    {key === "sentiment" ? "POS / NEG" : "MSA / TN / EG"}
                  </span>
                </button>
              ))}
            </div>
            <div className="work-grid">
              <section className="input-panel">
                <div className="panel-heading">
                  <h2>
                    {page === "workspace"
                      ? "Une phrase pour commencer."
                      : "Votre collection de textes."}
                  </h2>
                  <span className="mini-tag">
                    {page === "workspace" ? "TEXTE LIBRE" : "CSV UTF-8"}
                  </span>
                </div>
                <form onSubmit={page === "workspace" ? analyze : analyzeBatch}>
                  {page === "workspace" ? (
                    <>
                      <label htmlFor="input-text" className="field-label">
                        TEXTE À ANALYSER <span>2 000 caractères maximum</span>
                      </label>
                      <textarea
                        id="input-text"
                        dir="auto"
                        lang={/\p{Script=Arabic}/u.test(text) && !/\p{Script=Latin}/u.test(text) ? "ar" : undefined}
                        value={text}
                        disabled={busy}
                        onChange={(e) => {
                          setText(e.target.value);
                          setResults([]);
                          setExportNotice("");
                        }}
                        placeholder={
                          task === "sentiment"
                            ? "Écrivez un avis tunisien…\nel service behi barcha"
                            : "Écrivez une phrase en arabe…\nوين المحطة؟"
                        }
                        aria-describedby="text-count scope-note"
                      />
                      <div className="input-bottom">
                        <span>Arabe, Arabizi ou écriture mixte</span>
                        <span
                          id="text-count"
                          className={length > 2000 ? "invalid" : ""}
                        >
                          {number(length)} / 2 000
                        </span>
                      </div>
                      <div className="examples">
                        <span>EXEMPLES FICTIFS</span>
                        {examples[task].map((example) => (
                          <button
                            type="button"
                            dir="auto"
                            lang={/\p{Script=Arabic}/u.test(example) ? "ar" : undefined}
                            disabled={busy}
                            key={example}
                            onClick={() => {
                              setText(example);
                              setResults([]);
                            }}
                          >
                            <Plus size={12} />
                            {example}
                          </button>
                        ))}
                      </div>
                    </>
                  ) : (
                    <>
                      <label htmlFor="csv-file" className="upload-zone">
                        <Upload size={28} />
                        <strong>
                          {file ? file.name : "Choisir un fichier CSV"}
                        </strong>
                        <span>
                          Colonnes id,text · UTF-8 · 100 lignes · 1 Mo max.
                        </span>
                        <input
                          id="csv-file"
                          type="file"
                          accept=".csv,text/csv"
                          disabled={busy}
                          onChange={(e) => {
                            setFile(e.target.files[0] || null);
                            setBatch(null);
                          }}
                        />
                      </label>
                      <a href="/exemple.csv" download className="text-link">
                        <Download size={14} /> Télécharger un exemple fictif
                      </a>
                    </>
                  )}
                  <div className="model-row">
                    <label htmlFor="model">
                      MODÈLE
                      <select
                        disabled={busy || !choices.length}
                        value={model}
                        id="model"
                        onChange={(e) => {
                          setModel(e.target.value);
                          setResults([]);
                          setBatch(null);
                        }}
                      >
                        {!choices.length ? (
                          <option value={model}>Aucun modèle chargé</option>
                        ) : (
                          choices.map((m) => (
                            <option value={m.id} key={m.id}>
                              {modelName(m.id)}
                            </option>
                          ))
                        )}
                      </select>
                    </label>
                    {page === "workspace" && (
                      <label className="check-label">
                        <input
                          type="checkbox"
                          checked={comparison}
                          disabled={busy}
                          onChange={(e) => {
                            setComparison(e.target.checked);
                            setResults([]);
                          }}
                        />
                        <span>Comparer les variantes</span>
                      </label>
                    )}
                  </div>
                  <div className="submit-row">
                    <button
                      className="primary"
                      type="submit"
                      disabled={
                        busy ||
                        !available ||
                        (page === "workspace"
                          ? !text.trim() || length > 2000
                          : !file)
                      }
                    >
                      {busy ? (
                        <LoaderCircle size={16} className="spin" />
                      ) : page === "workspace" ? (
                        <ArrowUpRight size={17} />
                      ) : (
                        <FileText size={17} />
                      )}{" "}
                      {busy
                        ? "Analyse en cours…"
                        : page === "workspace"
                          ? "Analyser le texte"
                          : "Analyser le fichier"}
                    </button>
                    <span>
                      {!available
                        ? "Un modèle chargé est nécessaire."
                        : "Inférence locale · aucun réentraînement"}
                    </span>
                  </div>
                </form>
              </section>
              <aside className="context-panel">
                <span className="eyebrow">LE CADRE DE CETTE LECTURE</span>
                <h2>
                  {task === "sentiment" ? (
                    <>
                      Une opinion.{" "}
                      <br />
                      Deux polarités.
                    </>
                  ) : (
                    <>
                      Trois variétés.{" "}
                      <br />
                      Un périmètre précis.
                    </>
                  )}
                </h2>
                <p id="scope-note">
                  {task === "sentiment"
                    ? "Le modèle a appris à distinguer les avis positifs et négatifs dans des commentaires tunisiens. Il ne possède pas de classe « neutre »."
                    : "MSA désigne l’arabe standard. TN regroupe Tunis et Sfax ; EG représente l’arabe du Caire. Les phrases d’apprentissage viennent du domaine du voyage."}
                </p>
                <div className="scope-labels">
                  {(task === "sentiment"
                    ? ["POS", "NEG"]
                    : ["MSA", "TN", "EG"]
                  ).map((label) => (
                    <span key={label}>
                      {label}
                      <small>{labelNames[label]}</small>
                    </span>
                  ))}
                </div>
                <div className="context-foot">
                  <CircleHelp size={17} />
                  <p>
                    Une prédiction reste expérimentale. Le type d’écriture
                    n’identifie ni la langue ni l’origine d’une personne.
                  </p>
                </div>
              </aside>
            </div>
          </>
        )}

        {page === "workspace" && (
          <section className="result-section" aria-live="polite">
            <div className="section-label">
              <span className="eyebrow">LECTURE DU MODÈLE</span>
              {results.length > 0 && (
                <div className="export-actions">
                  <button
                    onClick={() =>
                      exportData(results, "json", "tuniscope-predictions")
                    }
                  >
                    <Download size={14} /> JSON
                  </button>
                  <button
                    onClick={() =>
                      exportData(results, "csv", "tuniscope-predictions")
                    }
                  >
                    <Download size={14} /> CSV
                  </button>
                </div>
              )}
            </div>
            {exportNotice && <p className="mono">{exportNotice}</p>}
            {!results.length ? (
              <div className="empty-result">
                <span className="empty-symbol" aria-hidden="true">
                  ↗
                </span>
                <div>
                  <h3>
                    {busy
                      ? "Les modèles lisent votre texte…"
                      : "Votre prochaine lecture apparaît ici."}
                  </h3>
                  <p>
                    {busy
                      ? "Le résultat sera affiché dès la fin de l’inférence."
                      : "Saisissez un texte, choisissez une tâche et lancez l’analyse."}
                  </p>
                </div>
                <span className="mono">{busy ? "EN COURS" : "EN ATTENTE"}</span>
              </div>
            ) : (
              <>
                <div className="result-grid">
                  {results.map((result) => (
                    <Result key={result.model} result={result} />
                  ))}
                </div>
                <details className="text-details">
                  <summary>
                    Voir le texte brut et sa normalisation{" "}
                    <ChevronDown size={15} />
                  </summary>
                  <div className="text-pair">
                    <div>
                      <span className="field-label">BRUT</span>
                      <p dir="auto" lang={results[0].script === "AR" ? "ar" : undefined}>{results[0].text_raw}</p>
                    </div>
                    <div>
                      <span className="field-label">NFC + ESPACES</span>
                      <p dir="auto" lang={results[0].script === "AR" ? "ar" : undefined}>{results[0].text_norm}</p>
                    </div>
                  </div>
                  <p className="subtle">
                    {results[0].text_raw === results[0].text_norm
                      ? "Le normaliseur n’a pas modifié ce texte."
                      : "Unicode et espaces ont été normalisés. Le texte brut reste conservé dans la réponse."}{" "}
                    Le modèle « brut » reçoit le texte brut.
                  </p>
                </details>
              </>
            )}
          </section>
        )}

        {page === "batch" && batch && (
          <section className="result-section">
            <div className="section-label">
              <span className="eyebrow">
                {batch.rows} ENTRÉES · {batch.errors} ERREURS
              </span>
              <div className="export-actions">
                <button
                  onClick={() =>
                    exportData(batch.results, "json", "tuniscope-batch")
                  }
                >
                  <Download size={14} /> JSON
                </button>
                <button
                  onClick={() =>
                    exportData(batch.results, "csv", "tuniscope-batch")
                  }
                >
                  <Download size={14} /> CSV
                </button>
              </div>
            </div>
            {exportNotice && <p className="mono">{exportNotice}</p>}
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Texte</th>
                    <th>État</th>
                    <th>Résultat</th>
                  </tr>
                </thead>
                <tbody>
                  {batch.results.map((r, index) => (
                    <tr key={index}>
                      <td>{r.id || "—"}</td>
                      <td dir="auto">{r.text_raw || r.error}</td>
                      <td>
                        {r.status === "error"
                          ? "Erreur"
                          : r.status === "abstained"
                            ? "Abstention"
                            : "Prédit"}
                      </td>
                      <td>{labelNames[r.label] || r.reason || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {page === "metrics" && (
          <section className="metrics-section">
            <div className="notice">
              <strong>Validation uniquement.</strong> Le test officiel n’a pas
              été évalué. Ces scores proviennent des entraînements sauvegardés,
              pas des textes saisis dans l’atelier.
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Tâche / modèle</th>
                    <th>Macro-F1</th>
                    <th>Accuracy</th>
                    <th>Train</th>
                    <th>Validation</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics?.results?.map((r, i) => (
                    <tr key={i}>
                      <td>
                        <strong>{taskNames[r.task]}</strong>
                        <span>{modelName(r.id)}</span>
                      </td>
                      <td>{r.metrics.macro_f1.toFixed(6)}</td>
                      <td>{percent(r.metrics.accuracy)}</td>
                      <td>{number(r.train_rows)}</td>
                      <td>{number(r.validation_rows)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!metrics?.results?.length && (
              <p className="notice">
                Aucun résultat chargé. Préparer le bundle pour retrouver ses
                métriques.
              </p>
            )}
            <div className="metric-cards">
              <article>
                <span className="eyebrow">MACRO-F1</span>
                <h3>Chaque classe compte.</h3>
                <p>
                  La moyenne des F1 de toutes les classes. Elle accorde le même
                  poids aux classes, même si leurs effectifs diffèrent.
                </p>
              </article>
              <article>
                <span className="eyebrow">ACCURACY</span>
                <h3>La part de réponses correctes.</h3>
                <p>
                  Le nombre de prédictions correctes divisé par le nombre
                  d’exemples. Elle peut favoriser les classes fréquentes.
                </p>
              </article>
            </div>
            <p className="subtle mono">
              SOURCE : {health?.source_run || "AUCUN RUN"} · MANIFESTES V2 · UNE
              GRAINE EXPLORATOIRE
            </p>
          </section>
        )}

        {page === "about" && (
          <section className="about-section">
            <div className="about-grid">
              <article>
                <span className="eyebrow">LE PROJET</span>
                <h2>
                  Comparer avant
                  <br />
                  de conclure.
                </h2>
                <p>
                  TuniScope compare des approches de classification sur deux
                  corpus : TSAC pour le sentiment tunisien, MADAR pour les
                  variétés de l’arabe. Les originaux sont conservés ; les
                  groupes de variantes restent dans une seule partition
                  d’apprentissage ou de validation.
                </p>
                <p>
                  Cette version de l’application utilise des pipelines TF-IDF et
                  une régression logistique déjà entraînés. Aucun corpus n’est
                  nécessaire pour analyser un nouveau texte.
                </p>
                <div className="reference-links">
                  <a
                    href="https://github.com/fbougares/TSAC"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Source TSAC <ArrowUpRight size={15} />
                  </a>
                  <a
                    href="https://camel.abudhabi.nyu.edu/madar-parallel-corpus/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Source MADAR <ArrowUpRight size={15} />
                  </a>
                </div>
              </article>
              <article className="component-list">
                <span className="eyebrow">COMPOSANTS</span>
                {[
                  [
                    "TF-IDF + régression logistique",
                    health?.ready ? "Disponible" : "Indisponible",
                  ],
                  ["Encodeur Transformer", "À implémenter"],
                  ["LLM et adaptateur LoRA", "À implémenter"],
                  ["Recherche documentaire RAG", "À implémenter"],
                ].map(([name, state]) => (
                  <div key={name}>
                    <span>{name}</span>
                    <span className="mini-tag">{state}</span>
                  </div>
                ))}
                <p>
                  Les scores des classifieurs ne sont pas calibrés. L’abstention
                  actuelle couvre uniquement l’absence totale d’indices dans le
                  vocabulaire ; elle ne détecte pas tous les textes hors
                  domaine.
                </p>
              </article>
            </div>
            <section className="rag-panel">
              <span className="eyebrow">ASSISTANCE DOCUMENTAIRE</span>
              <h2>Une question sur la méthode ?</h2>
              <p>
                Le futur RAG répondra depuis des fiches sourcées. Il
                n’expliquera pas automatiquement pourquoi une prédiction est
                correcte.
              </p>
              <form
                onSubmit={async (e) => {
                  e.preventDefault();
                  setBusy(true);
                  setError("");
                  try {
                    setAnswer(
                      await api("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question }),
                      }),
                    );
                  } catch (err) {
                    setError(err.message);
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                <label htmlFor="question" className="field-label">
                  QUESTION DOCUMENTAIRE
                </label>
                <div className="question-row">
                  <input
                    id="question"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Comment les données ont-elles été séparées ?"
                    maxLength={2000}
                  />
                  <button
                    className="secondary"
                    disabled={busy || !question.trim()}
                  >
                    {busy ? "Vérification…" : "Vérifier la disponibilité"}
                    <ArrowRight size={16} />
                  </button>
                </div>
              </form>
              {answer && (
                <div className="notice" role="status">
                  {answer.reason} Aucune réponse ni citation n’a été générée.
                </div>
              )}
            </section>
          </section>
        )}

        <footer>
          <span className="brand small">
            tuni<span>scope</span>
            <i />
          </span>
          <p>
            Une application expérimentale de NLP arabe.
            <br />
            Les textes saisis ne sont pas enregistrés par l’application.
          </p>
          <a href="/api/docs" target="_blank" rel="noreferrer">
            Documentation API <ArrowUpRight size={14} />
          </a>
          <span className="mono">CONSTRUIRE · MESURER · COMPRENDRE</span>
        </footer>
      </main>
    </>
  );
}

function Result({ result }) {
  return (
    <article className="result-card">
      <div className="section-label">
        <span className="eyebrow">{modelName(result.model)}</span>
        <span className="mini-tag">{result.script}</span>
      </div>
      <div className="prediction">
        <span className="prediction-mark">
          {result.status === "predicted" ? (
            <Check size={20} />
          ) : (
            <CircleHelp size={20} />
          )}
        </span>
        <div>
          <span className="mono">
            {result.status === "predicted" ? "PRÉDICTION" : "ABSTENTION"}
          </span>
          <h3>
            {labelNames[result.label] || "Indices insuffisants"}
            {result.label && <small>{result.label}</small>}
          </h3>
        </div>
      </div>
      {result.reason && <p>{result.reason}</p>}
      <div className="score-bars">
        {Object.entries(result.scores)
          .sort((a, b) => b[1] - a[1])
          .map(([label, score]) => (
            <div key={label}>
              <div>
                <span>{labelNames[label]}</span>
                <span className="mono">{percent(score)}</span>
              </div>
              <div className="bar-track">
                <div
                  style={{ width: `${score * 100}%` }}
                  className={label === result.label ? "winning" : ""}
                />
              </div>
            </div>
          ))}
      </div>
      <p className="score-note">Scores du modèle · non calibrés</p>
      <details>
        <summary>
          Limites et provenance <ChevronDown size={14} />
        </summary>
        <ul>
          {result.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
        <p className="mono provenance">
          RUN {result.versions.source_run}
          <br />
          SHA {result.versions.checkpoint_sha256.slice(0, 16)}…<br />
          ID {result.id}
        </p>
      </details>
    </article>
  );
}

createRoot(document.getElementById("root")).render(<App />);
