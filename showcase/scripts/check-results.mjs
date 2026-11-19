// Fails when the dashboard shows a result that does not come from the results registry.
//
// 1. No hand-typed results: a decimal with two or more places inside a string, template text or JSX text, in any
//    file the pages import. Code constants (axis padding, colour alpha) are not text and are allowed; so are the
//    `code` listings, links and DOIs.
// 2. Every registry id passed to r(), fmt(), ci() or pct() exists in content/generated/results-registry.json.
//
// Run: node scripts/check-results.mjs   (also runs before `npm run build`)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const registry = JSON.parse(fs.readFileSync(path.join(root, "content/generated/results-registry.json"), "utf8")).entries;
const DECIMAL = /(?<![\w.\/])\d+\.\d{2,}(?!\w|\.\d)/g;
const SKIP_PROPERTIES = new Set(["code", "href", "url", "doi", "source"]);
const REF_CALLS = new Set(["r", "fmt", "pct"]);
// Numeric literals that are results when they initialise a property with one of these names.
const RESULT_PROPERTIES = /^(score|macroF1|macro_f1|kappa|accuracy|f1|stageF1|value|learned)$/;

function resolveImport(from, spec) {
  if (!spec.startsWith(".")) return undefined;
  const base = path.resolve(path.dirname(from), spec);
  return ["", ".ts", ".tsx", "/index.ts", "/index.tsx"].map((ext) => base + ext).find((p) => fs.existsSync(p) && fs.statSync(p).isFile());
}

// Files reachable from the app's pages and layout.
const entries = fs.readdirSync(path.join(root, "app"), { recursive: true })
  .filter((f) => /(^|[\\/])(page|layout)\.tsx$/.test(f)).map((f) => path.join(root, "app", f));
const files = new Set();
const queue = [...entries];
while (queue.length) {
  const file = queue.pop();
  if (files.has(file) || !/\.tsx?$/.test(file)) continue;
  files.add(file);
  for (const imp of ts.preProcessFile(fs.readFileSync(file, "utf8")).importedFiles) {
    const next = resolveImport(file, imp.fileName);
    if (next) queue.push(next);
  }
}

const problems = [];
for (const file of files) {
  const text = fs.readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const where = (node) => `${path.relative(root, file)}:${sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1}`;
  const skipped = (node) => {
    for (let n = node.parent; n; n = n.parent) {
      if (ts.isPropertyAssignment(n) && SKIP_PROPERTIES.has(n.name.getText(sf))) return true;
      if (ts.isJsxAttribute(n) && SKIP_PROPERTIES.has(n.name.getText(sf))) return true;
    }
    return false;
  };
  const visit = (node) => {
    let literal;
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) literal = node.text;
    else if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) literal = node.text;
    else if (ts.isJsxText(node)) literal = node.text;
    if (literal !== undefined && !skipped(node) && !/doi\.org|arxiv|https?:|rgba?\(|hsla?\(/i.test(literal)) {
      for (const m of literal.matchAll(DECIMAL)) problems.push(`${where(node)}  hand-typed ${m[0]}`);
    }
    if (ts.isNumericLiteral(node) && node.text.includes(".")) {
      let owner = node.parent;
      while (owner && (ts.isArrayLiteralExpression(owner) || ts.isPrefixUnaryExpression(owner))) owner = owner.parent;
      if (owner && ts.isPropertyAssignment(owner) && RESULT_PROPERTIES.test(owner.name.getText(sf))) {
        problems.push(`${where(node)}  hand-typed ${node.text} (${owner.name.getText(sf)})`);
      }
    }
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && REF_CALLS.has(node.expression.text)) {
      const arg = node.arguments[0];
      if (arg && ts.isStringLiteral(arg) && !(arg.text in registry)) problems.push(`${where(node)}  unknown registry id "${arg.text}"`);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
}

if (problems.length) {
  console.error(`check-results: ${problems.length} problem(s). Register results in scripts/build_results_registry.py and reference them with r()/fmt().`);
  for (const p of problems) console.error("  " + p);
  process.exit(1);
}
console.log(`check-results: ${files.size} files, no hand-typed results, all registry ids resolve.`);
