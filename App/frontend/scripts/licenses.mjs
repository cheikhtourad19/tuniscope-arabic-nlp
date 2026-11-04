// Keep redistribution notices beside the locally hosted fonts and JS assets.
import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const packages = ["react", "react-dom", "scheduler", "lucide-react",
  "@fontsource/instrument-serif", "@fontsource/dm-mono", "@fontsource/geist", "@fontsource/noto-naskh-arabic"];
const notices = [];
for (const name of packages) {
  const folder = path.join("node_modules", name);
  const info = JSON.parse(await readFile(path.join(folder, "package.json"), "utf8"));
  let license;
  for (const file of ["LICENSE", "LICENSE.txt", "LICENSE.md"]) {
    try { license = await readFile(path.join(folder, file), "utf8"); break; }
    catch (error) { if (error.code !== "ENOENT") throw error; }
  }
  if (!license) throw new Error(`Missing license notice: ${name}`);
  notices.push(`${name} ${info.version}\n${"=".repeat(60)}\n${license}`);
}
await mkdir("dist", { recursive: true });
await writeFile("dist/licenses.txt", notices.join("\n\n"));
