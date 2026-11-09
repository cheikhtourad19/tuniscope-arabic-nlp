import { test, expect } from "@playwright/test";
import fs from "node:fs/promises";

test("text inference, paired comparison, JSON export and raw/normalized disclosure", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByText("Modèles chargés", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("TEXTE À ANALYSER").fill("el service behi barcha");
  await page.getByLabel("Comparer les modèles").check();
  await page.getByRole("button", { name: "Analyser le texte" }).click();
  await expect(page.locator(".result-card")).toHaveCount(5, { timeout: 30_000 });
  await expect(page.locator(".result-card").filter({ hasText: "CAMeLBERT adapté" })).toHaveCount(1);
  await expect(page.locator(".result-card").filter({ hasText: "Qwen3-0.6B · LoRA" })).toHaveCount(1);
  await expect(page.locator(".prediction h3").first()).toContainText("Positif");
  await page.getByText("Voir le texte brut et sa normalisation").click();
  await expect(page.locator(".text-pair")).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON", exact: true }).click();
  const downloaded = await downloadPromise;
  const rows = JSON.parse(await fs.readFile(await downloaded.path(), "utf8"));
  expect(rows).toHaveLength(5);
  expect(new Set(rows.map((row) => row.id)).size).toBe(1);
  expect(rows[0].versions.checkpoint_sha256).toHaveLength(64);
  expect(errors).toEqual([]);
});

test("CSV retains an invalid row, exports errors and protects spreadsheet formulas", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Traitement CSV", exact: true })
    .click();
  await page
    .locator("#csv-file")
    .setInputFiles({
      name: "fictif.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "id,text\na,el service behi barcha\nb,\n=1+1,el service moch behi\n",
      ),
    });
  await page.getByRole("button", { name: "Analyser le fichier" }).click();
  await expect(page.getByText("3 ENTRÉES · 1 ERREURS")).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(3);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "CSV", exact: true }).click();
  const downloaded = await downloadPromise;
  const csv = await fs.readFile(await downloaded.path(), "utf8");
  expect(csv).toContain("'=1+1");
  expect(csv).toContain('"error"');
});

test("validation metrics and cited documentary answers are available", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Résultats", exact: true }).click();
  await expect(
    page.getByText("Validation uniquement.", { exact: true }),
  ).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(10);
  await expect(page.getByText("Moyenne ± écart-type")).toBeVisible();
  await page.getByRole("button", { name: "Le projet", exact: true }).click();
  await expect(page.getByText("Validation documentaire · 20 questions")).toBeVisible();
  const question = page.getByLabel("QUESTION DOCUMENTAIRE");
  const ask = page.getByRole("button", { name: "Chercher dans les fiches" });
  await question.fill("Que signifie macro-F1 et comment calcule-t-on la moyenne des F1 des classes ?");
  await ask.click();
  await expect(page.getByText("macro-f1.fr", { exact: true })).toBeVisible();
  const source = await page.getByRole("link", { name: /19_protocole_ablation/ }).getAttribute("href");
  expect((await page.request.get(source)).ok()).toBeTruthy();
  await question.fill("Quel est le macro-F1 de TuniScope sur un corpus marocain indépendant ?");
  await ask.click();
  await expect(page.getByText("Sources insuffisantes", { exact: true })).toBeVisible();
  await expect(page.locator(".documentary-source")).toHaveCount(0);
});

test("CSV can use the shared Qwen LoRA adapter and export the selected seed", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Traitement CSV", exact: true }).click();
  await page.locator("#model").selectOption("qwen_lora__raw-v0");
  await page.locator("#csv-file").setInputFiles({
    name: "fictif-lora.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("id,text\na,الخدمة باهية برشا\nb,el service moch behi\n"),
  });
  await page.getByRole("button", { name: "Analyser le fichier" }).click();
  await expect(page.locator("tbody tr")).toHaveCount(2, { timeout: 30_000 });
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON", exact: true }).click();
  const downloaded = await downloadPromise;
  const rows = JSON.parse(await fs.readFile(await downloaded.path(), "utf8"));
  expect(rows.map((row) => row.id)).toEqual(["a", "b"]);
  for (const row of rows) {
    expect(row.model).toBe("qwen_lora__raw-v0");
    expect(row.versions.seed).toBe(42);
    expect(row.versions.prompt_mode).toBe("zero_shot");
    expect(row.scores).toEqual({});
  }
});

test("Arabic and mobile layout keep natural text direction and fit the viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: /Variété d’arabe/ }).click();
  await page.getByLabel("TEXTE À ANALYSER").fill("وين المحطة؟");
  await page.getByRole("button", { name: "Analyser le texte" }).click();
  await expect(page.locator(".result-card")).toHaveCount(1);
  await expect(page.locator(".result-card .mini-tag")).toHaveText("AR");
  await expect(page.locator("textarea")).toHaveAttribute("dir", "auto");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Ouvrir le menu" }).click();
  await page.getByRole("button", { name: "Résultats", exact: true }).click();
  await expect(
    page.getByText("Validation uniquement.", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Ouvrir le menu" }).click();
  await page.getByRole("button", { name: "Le projet", exact: true }).click();
  await page.locator("#documentary-lang").selectOption("ar");
  await page.locator("#documentary-method").selectOption("e5");
  await page.getByLabel("QUESTION DOCUMENTAIRE").fill("ما المدن التي يمثلها الرمزان TN وEG في MADAR، وما المقصود بالرمز MSA؟");
  await page.getByRole("button", { name: "Chercher dans les fiches" }).click();
  await expect(page.getByText("madar.ar", { exact: true })).toBeVisible();
  await expect(page.locator(".documentary-answer p")).toHaveAttribute("dir", "rtl");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("missing models visibly disable analysis, without fabricated predictions", async ({
  page,
}) => {
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: {
        ready: false,
        status: "unavailable",
        models: [],
        error: "Bundle absent.",
      },
    }),
  );
  await page.route("**/api/metrics", (route) =>
    route.fulfill({ json: { results: [] } }),
  );
  await page.goto("/");
  await expect(
    page.getByText("Modèles indisponibles.", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("TEXTE À ANALYSER").fill("avis fictif");
  await expect(
    page.getByRole("button", { name: "Analyser le texte" }),
  ).toBeDisabled();
  await expect(page.locator(".result-card")).toHaveCount(0);
});
