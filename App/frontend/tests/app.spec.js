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
  await page.getByLabel("Comparer les variantes").check();
  await page.getByRole("button", { name: "Analyser le texte" }).click();
  await expect(page.locator(".result-card")).toHaveCount(2);
  await expect(page.locator(".prediction h3").first()).toContainText("Positif");
  await page.getByText("Voir le texte brut et sa normalisation").click();
  await expect(page.locator(".text-pair")).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON", exact: true }).click();
  const downloaded = await downloadPromise;
  const rows = JSON.parse(await fs.readFile(await downloaded.path(), "utf8"));
  expect(rows).toHaveLength(2);
  expect(rows[0].id).toBe(rows[1].id);
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

test("metrics are validation only; unimplemented RAG never invents an answer", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Résultats", exact: true }).click();
  await expect(
    page.getByText("Validation uniquement.", { exact: true }),
  ).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(4);
  await page.getByRole("button", { name: "Le projet", exact: true }).click();
  await page
    .getByLabel("QUESTION DOCUMENTAIRE")
    .fill("Comment fonctionne le modèle ?");
  await page.getByRole("button", { name: "Vérifier la disponibilité" }).click();
  await expect(
    page.getByText(/Aucune réponse ni citation n’a été générée/),
  ).toBeVisible();
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
