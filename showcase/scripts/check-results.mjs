// Fails when the dashboard drifts from generated artifacts, registry entries, or cited metadata.
//
// 1. No hand-typed results: a decimal with two or more places inside a string, template text or JSX text, in any
//    file the pages import. Code constants (axis padding, colour alpha) are not text and are allowed; so are the
//    `code` listings, links and DOIs.
// 2. Every registry id passed to r(), fmt() or pct() exists in content/generated/results-registry.json.
// 3. Citation integrity: every [N] cited in TSX/TS exists in `references`, every entry in `references` is cited,
//    and every literature entry's `cite` in results-registry.json exists in `references`.
// 4. Feature explorer integrity: every id in `stageFeatureTargets` exists in `featureFamilies` and
//    staging-dashboard.json distributions, has unique display names per stage, and never assigns a directional
//    expectation ("high"/"low") to a zero-mean signal whose max |Cohen's d| < 0.05.
// 5. History log hygiene: no unfinished stub findings ("Declared", "TODO", "TBD") in `historyPhases`.
//
// Run: node scripts/check-results.mjs   (also runs before `npm run build` and via pytest)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const registry = JSON.parse(fs.readFileSync(path.join(root, "content/generated/results-registry.json"), "utf8")).entries;
const dashboardData = JSON.parse(fs.readFileSync(path.join(root, "content/generated/staging-dashboard.json"), "utf8"));
const algorithmMetadata = JSON.parse(fs.readFileSync(path.join(root, "content/generated/algorithm-metadata.json"), "utf8"));

const DECIMAL = /(?<![\w.\/])\d+\.\d{2,}(?!\w|\.\d)/g;
const SKIP_PROPERTIES = new Set(["code", "href", "url", "doi", "source"]);
const REF_CALLS = new Set(["r", "fmt", "pct"]);
const RESULT_PROPERTIES = /^(score|macroF1|macro_f1|kappa|accuracy|f1|stageF1|value|learned)$/;
const STUB_FINDINGS = /^(declared|todo|tbd)$/i;

function resolveImport(from, spec) {
  if (!spec.startsWith(".")) return undefined;
  const base = path.resolve(path.dirname(from), spec);
  return ["", ".ts", ".tsx", "/index.ts", "/index.tsx"].map((ext) => base + ext).find((p) => fs.existsSync(p) && fs.statSync(p).isFile());
}

function propName(node, sf) {
  if (!node) return undefined;
  if (ts.isIdentifier(node) || ts.isStringLiteral(node)) return node.text;
  return node.getText(sf).replace(/^['"]|['"]$/g, "");
}

function intLiteral(node) {
  return node && ts.isNumericLiteral(node) ? Number(node.text) : undefined;
}

function findVariableInit(sf, name) {
  let found;
  const walk = (node) => {
    if (found) return;
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text === name && node.initializer) {
      found = node.initializer;
      return;
    }
    ts.forEachChild(node, walk);
  };
  walk(sf);
  return found;
}

// Files reachable from the app's pages and layout.
const entries = fs
  .readdirSync(path.join(root, "app"), { recursive: true })
  .filter((f) => /(^|[\\/])(page|layout)\.tsx$/.test(f))
  .map((f) => path.join(root, "app", f));
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
const citedNumbers = new Set();
let contentSf;

for (const file of files) {
  const text = fs.readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  if (path.basename(file) === "staging-content.ts") contentSf = sf;
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
    // Collect <Citation number={N} />
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(sf) === "Citation") {
      for (const attr of node.attributes.properties) {
        if (ts.isJsxAttribute(attr) && attr.name.text === "number" && attr.initializer && ts.isJsxExpression(attr.initializer)) {
          const n = intLiteral(attr.initializer.expression);
          if (n !== undefined) citedNumbers.add(n);
        }
      }
    }
    // Collect cite: N, cites: [...], sources: [...]
    if (ts.isPropertyAssignment(node)) {
      const key = propName(node.name, sf);
      if (key === "cite") {
        const n = intLiteral(node.initializer);
        if (n !== undefined) citedNumbers.add(n);
      } else if ((key === "cites" || key === "sources") && ts.isArrayLiteralExpression(node.initializer)) {
        for (const el of node.initializer.elements) {
          const n = intLiteral(el);
          if (n !== undefined) citedNumbers.add(n);
        }
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
}

// 3. Citation & 4. Feature-catalog & 5. History checks on staging-content.ts
if (contentSf) {
  const refNumbers = new Set();
  const refsInit = findVariableInit(contentSf, "references");
  if (refsInit && ts.isArrayLiteralExpression(refsInit)) {
    for (const obj of refsInit.elements) {
      if (!ts.isObjectLiteralExpression(obj)) continue;
      for (const prop of obj.properties) {
        if (ts.isPropertyAssignment(prop) && propName(prop.name, contentSf) === "number") {
          const n = intLiteral(prop.initializer);
          if (n !== undefined) {
            if (refNumbers.has(n)) problems.push(`staging-content.ts: duplicate reference number ${n}`);
            refNumbers.add(n);
          }
        }
      }
    }
  }
  for (const n of citedNumbers) {
    if (!refNumbers.has(n)) problems.push(`Citation [${n}] used in UI has no entry in references`);
  }
  for (const n of refNumbers) {
    if (!citedNumbers.has(n)) problems.push(`Reference [${n}] in references is never cited in the dashboard`);
  }
  for (const [id, entry] of Object.entries(registry)) {
    if (entry.kind === "literature" && typeof entry.cite === "number" && !refNumbers.has(entry.cite)) {
      problems.push(`results-registry.json "${id}" cites [${entry.cite}], missing from references`);
    }
  }

  // Extract feature id -> display name from eegSpectrumFeatures + featureFamilies
  const featureNames = new Map();
  const extractFeatureNames = (init) => {
    if (!init) return;
    const walk = (node) => {
      if (ts.isObjectLiteralExpression(node)) {
        let id, name;
        for (const prop of node.properties) {
          if (!ts.isPropertyAssignment(prop)) continue;
          const k = propName(prop.name, contentSf);
          if (k === "id" && ts.isStringLiteral(prop.initializer)) id = prop.initializer.text;
          if (k === "name" && ts.isStringLiteral(prop.initializer)) name = prop.initializer.text;
        }
        if (id && name && (id.includes(".") || id === "spindle_bursts")) featureNames.set(id, name);
      }
      ts.forEachChild(node, walk);
    };
    walk(init);
  };
  extractFeatureNames(findVariableInit(contentSf, "eegSpectrumFeatures"));
  extractFeatureNames(findVariableInit(contentSf, "featureFamilies"));

  const distById = new Map((dashboardData.feature_evidence?.distributions ?? []).map((d) => [d.id, d]));
  const targetsInit = findVariableInit(contentSf, "stageFeatureTargets");
  if (targetsInit && ts.isArrayLiteralExpression(targetsInit)) {
    for (const stageObj of targetsInit.elements) {
      if (!ts.isObjectLiteralExpression(stageObj)) continue;
      let stageName = "";
      let featureIds = [];
      const expectMap = new Map();
      for (const prop of stageObj.properties) {
        if (!ts.isPropertyAssignment(prop)) continue;
        const k = propName(prop.name, contentSf);
        if (k === "stage" && ts.isStringLiteral(prop.initializer)) stageName = prop.initializer.text;
        if (k === "featureIds" && ts.isArrayLiteralExpression(prop.initializer)) {
          featureIds = prop.initializer.elements.filter(ts.isStringLiteral).map((el) => el.text);
        }
        if (k === "expect" && ts.isObjectLiteralExpression(prop.initializer)) {
          for (const ep of prop.initializer.properties) {
            if (ts.isPropertyAssignment(ep) && ts.isStringLiteral(ep.initializer)) {
              expectMap.set(propName(ep.name, contentSf), ep.initializer.text);
            }
          }
        }
      }
      const seenNames = new Set();
      for (const fid of featureIds) {
        if (!distById.has(fid)) problems.push(`stageFeatureTargets.${stageName}: "${fid}" missing from staging-dashboard.json distributions`);
        const displayName = featureNames.get(fid);
        if (displayName) {
          if (seenNames.has(displayName)) problems.push(`stageFeatureTargets.${stageName}: duplicate display name "${displayName}" (${fid})`);
          seenNames.add(displayName);
        }
        if (expectMap.has(fid) && distById.has(fid)) {
          const maxD = Math.max(...distById.get(fid).stages.map((s) => Math.abs(s.one_vs_rest_cohens_d ?? 0)));
          if (maxD < 0.05) {
            problems.push(`stageFeatureTargets.${stageName}: "${fid}" has directional expect="${expectMap.get(fid)}" but max |d|=${maxD.toFixed(3)}`);
          }
        }
      }
    }
  }

  // History stubs check
  const historyInit = findVariableInit(contentSf, "historyPhases");
  if (historyInit) {
    const walkHistory = (node) => {
      if (ts.isPropertyAssignment(node) && propName(node.name, contentSf) === "finding" && ts.isStringLiteral(node.initializer)) {
        if (STUB_FINDINGS.test(node.initializer.text.trim())) {
          problems.push(`historyPhases contains unfinished stub finding "${node.initializer.text}"`);
        }
      }
      ts.forEachChild(node, walkHistory);
    };
    walkHistory(historyInit);
  }
}

if (!algorithmMetadata?.epoch_ratio_bands?.alpha || !algorithmMetadata?.track_bands?.alpha) {
  problems.push("algorithm-metadata.json missing canonical band definitions");
}

if (problems.length) {
  console.error(`check-results: ${problems.length} problem(s).`);
  for (const p of problems) console.error("  " + p);
  process.exit(1);
}
console.log(
  `check-results: ${files.size} files, no hand-typed results, all registry ids, ${citedNumbers.size} citations, feature expectations, and history entries verified.`
);
