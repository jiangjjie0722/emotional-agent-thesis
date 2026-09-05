import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const sourcePath = process.argv[2];
if (!sourcePath) {
  throw new Error("Usage: node build_english_csv.mjs SOURCE.csv [OUTPUT_DIRECTORY]");
}
const mappingPath = path.join(path.dirname(new URL(import.meta.url).pathname), "translations.json");
const outputDir = process.argv[3] ?? path.resolve("outputs/emotion_english");
const outputPath = path.join(outputDir, "Emotion_English.csv");

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field !== "" || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function encodeCsv(rows) {
  const quote = (value) => {
    const text = String(value ?? "");
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return `\uFEFF${rows.map((row) => row.map(quote).join(",")).join("\r\n")}\r\n`;
}

const sourceText = await fs.readFile(sourcePath, "utf8");
const sourceRows = parseCsv(sourceText);
const mapping = JSON.parse(await fs.readFile(mappingPath, "utf8"));
let translatedCellCount = 0;
const translatedRows = sourceRows.map((row) => row.map((value) => {
  if (Object.hasOwn(mapping, value)) {
    translatedCellCount += 1;
    return mapping[value];
  }
  return value;
}));

const translatedCsv = encodeCsv(translatedRows);
if (/[\u3400-\u9fff]/u.test(translatedCsv)) {
  throw new Error("Chinese characters remain in the translated CSV");
}
if (sourceRows.length !== translatedRows.length) {
  throw new Error("Row count changed during translation");
}
for (let r = 0; r < sourceRows.length; r += 1) {
  if (sourceRows[r].length !== translatedRows[r].length) {
    throw new Error(`Column count changed on row ${r + 1}`);
  }
  for (let c = 0; c < sourceRows[r].length; c += 1) {
    const original = sourceRows[r][c];
    if (!Object.hasOwn(mapping, original) && translatedRows[r][c] !== original) {
      throw new Error(`Non-Chinese value changed at row ${r + 1}, column ${c + 1}`);
    }
  }
}

const workbook = await Workbook.fromCSV(translatedCsv, { sheetName: "Emotion" });
const sheet = workbook.worksheets.getItem("Emotion");
const used = sheet.getUsedRange();
used.format = {
  font: { name: "Arial", size: 9, color: "#1F2937" },
  verticalAlignment: "center",
  wrapText: true,
};
sheet.getRange("A1:AG1").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#163A5B" },
};
sheet.getRange("A1:D44").format.columnWidth = 14;
sheet.getRange("E1:W44").format.columnWidth = 28;
sheet.getRange("X1:AD44").format.columnWidth = 20;
sheet.getRange("AE1:AG44").format.columnWidth = 34;
sheet.getRange("AF1:AF44").format.columnWidth = 52;
sheet.getRange("A1:AG1").format.rowHeight = 52;
sheet.getRange("A2:AG44").format.rowHeight = 68;
sheet.freezePanes.freezeRows(1);
sheet.showGridLines = false;

const overview = await workbook.inspect({
  kind: "workbook,sheet,region",
  sheetId: "Emotion",
  range: "A1:AG6",
  maxChars: 7000,
  tableMaxRows: 6,
  tableMaxCols: 33,
  tableMaxCellChars: 110,
});
console.log(overview.ndjson);

await fs.mkdir(outputDir, { recursive: true });
for (const [name, range] of [["left", "A1:H8"], ["middle", "I1:Q8"], ["right", "R1:AG8"]]) {
  const preview = await workbook.render({ sheetName: "Emotion", range, scale: 1.2, format: "png" });
  await fs.writeFile(path.join(outputDir, `preview_${name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(outputPath, translatedCsv, "utf8");

const roundTripRows = parseCsv(await fs.readFile(outputPath, "utf8"));
const sourceWidths = sourceRows.map((row) => row.length);
const outputWidths = roundTripRows.map((row) => row.length);
if (JSON.stringify(sourceWidths) !== JSON.stringify(outputWidths)) {
  throw new Error("CSV round-trip changed row widths");
}
console.log(JSON.stringify({
  outputPath,
  rows: translatedRows.length,
  columns: Math.max(...translatedRows.map((row) => row.length)),
  translatedUniqueValues: Object.keys(mapping).length,
  translatedCellCount,
  chineseCharactersRemaining: 0,
}));
