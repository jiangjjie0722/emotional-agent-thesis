import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const sourcePath = process.argv[2];
if (!sourcePath) {
  throw new Error("Usage: node inspect_csv.mjs SOURCE.csv");
}
const csvText = await fs.readFile(sourcePath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Emotion" });
const summary = await workbook.inspect({
  kind: "workbook,sheet,region",
  sheetId: "Emotion",
  range: "A1:AH6",
  maxChars: 6000,
  tableMaxRows: 6,
  tableMaxCols: 34,
  tableMaxCellChars: 100,
});
console.log(summary.ndjson);
console.log(workbook.help("export csv", {
  search: "csv|CSV|export",
  include: "index,examples,notes",
  maxChars: 5000,
}).ndjson);
