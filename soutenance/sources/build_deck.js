const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const THEME = {
  name: "Pelouse",
  headFontFace: "Arial Narrow",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16261A", lt1: "FFFFFF", dk2: "1F4A1A", lt2: "EEF3EA",
    accent1: "1F4A1A", accent2: "8DBF5A", accent3: "E9A53A", accent4: "D9534F",
    accent5: "5F6B5C", accent6: "C9DDB4", hlink: "1F4A1A", folHlink: "5F6B5C",
  },
};
// hex utilisés là où pptxgenjs n'accepte que du hex (lignes du terrain, grilles)
const HEX = { pitch: "1F4A1A", stripe: "245421", line: "3F6D38", lime: "C0DD97", amber: "E9A53A", red: "D9534F", ink: "16261A", grey: "5F6B5C", pale: "EEF3EA", rule: "D8E2D2" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.title = "Même stats, pas le même prix";
pres.author = "Simon Gallais, Gautier Deplanque, Mathis Leitao";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- motif terrain (bandes de tonte + lignes) ----------
function pitchObjects(full) {
  const o = [{ rect: { x: 0, y: 0, w: 10, h: 5.625, fill: { color: HEX.pitch } } }];
  for (let i = 0; i < 10; i += 2) o.push({ rect: { x: i * 1.0, y: 0, w: 1.0, h: 5.625, fill: { color: HEX.stripe } } });
  const L = { color: HEX.line, width: 1.25 };
  o.push({ line: { x: 5, y: 0, w: 0, h: 5.625, line: { ...L } } });
  o.push({ rect: { x: 4.1, y: 1.91, w: 1.8, h: 1.8, fill: { type: "none" }, line: { ...L }, rectRadius: 0 } });
  if (full) {
    o.push({ rect: { x: 0.35, y: 0.35, w: 9.3, h: 4.925, fill: { type: "none" }, line: { ...L } } });
    o.push({ rect: { x: 8.35, y: 1.6, w: 1.3, h: 2.425, fill: { type: "none" }, line: { ...L } } });
    o.push({ rect: { x: 0.35, y: 1.6, w: 1.3, h: 2.425, fill: { type: "none" }, line: { ...L } } });
  }
  return o;
}

const kicker = (x, y, w, color) => ({ placeholder: { options: { name: "kicker", type: "body", x, y, w, h: 0.3, fontFace: "Calibri", fontSize: 11, bold: true, color, charSpacing: 3, margin: 0 }, text: "" } });
const speaker = (color) => ({ placeholder: { options: { name: "speaker", type: "body", x: 6.5, y: 5.2, w: 2.9, h: 0.25, fontFace: "Calibri", fontSize: 10, color, align: "right", margin: 0 }, text: "" } });

pres.defineSlideMaster({
  title: "PELOUSE_TITRE",
  background: { color: HEX.pitch },
  objects: [...pitchObjects(true),
    kicker(0.8, 0.8, 8, C.accent6),
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.05, w: 8.4, h: 1.5, fontFace: "Arial Narrow", fontSize: 54, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.65, w: 8.4, h: 0.5, fontFace: "Calibri", fontSize: 20, color: C.accent6, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "names", type: "body", x: 0.8, y: 4.55, w: 8.4, h: 0.35, fontFace: "Calibri", fontSize: 12, color: C.background2, margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "PELOUSE_CONTENU",
  background: { color: HEX.pitch },
  objects: [...pitchObjects(false),
    kicker(0.6, 0.45, 6, C.accent6),
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.72, w: 8.8, h: 0.8, fontFace: "Arial Narrow", fontSize: 36, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    speaker(C.accent6),
  ],
  slideNumber: { x: 0.6, y: 5.2, w: 0.5, h: 0.25, fontFace: "Calibri", fontSize: 10, color: C.accent6 },
});

pres.defineSlideMaster({
  title: "CLAIR_CONTENU",
  background: { color: HEX.pale.replace("EEF3EA", "FFFFFF") },
  objects: [
    kicker(0.6, 0.45, 6, C.accent1),
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.72, w: 8.8, h: 0.8, fontFace: "Arial Narrow", fontSize: 36, bold: true, color: C.text1, valign: "top", align: "left", margin: 0 }, text: "" } },
    speaker(C.accent5),
  ],
  slideNumber: { x: 0.6, y: 5.2, w: 0.5, h: 0.25, fontFace: "Calibri", fontSize: 10, color: C.accent5 },
});

// ---------- helpers ----------
function newSlide(master, section, k, title, who) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (k) s.addText(k.toUpperCase(), { placeholder: "kicker" });
  if (title) s.addText(title.toUpperCase(), { placeholder: "title" });
  if (who) s.addText(who, { placeholder: "speaker" });
  return s;
}
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { type: "none" }, rectRadius: 0.08, objectName: name });
}
function txt(s, text, o) { s.addText(text, { isTextBox: true, margin: 0, fontFace: "Calibri", ...o }); }
function big(s, text, o) { s.addText(text, { isTextBox: true, margin: 0, fontFace: "Arial Narrow", bold: true, ...o }); }

// ================= SECTION 1 : contexte et biais (Gautier) =================
pres.addSection({ title: "Contexte et biais" });
{
  const s = pres.addSlide({ masterName: "PELOUSE_TITRE", sectionTitle: "Contexte et biais" });
  s.addText("ETHICS OF AI #4 · ECE PARIS · ING5 DATA & IA", { placeholder: "kicker" });
  s.addText("MÊME STATS, PAS LE MÊME PRIX", { placeholder: "title" });
  s.addText("Débiaiser un LLM de recrutement d'attaquants", { placeholder: "body" });
  s.addText("Simon Gallais · Gautier Deplanque · Mathis Leitao", { placeholder: "names" });
  s.addNotes("GAUTIER (15 s). Bonjour. Notre projet : un petit LLM qui aide un club à construire sa short-list d'attaquants, et qui ne doit pas discriminer selon la nationalité.");
}
{
  const s = newSlide("CLAIR_CONTENU", "Contexte et biais", "01 · Le cas d'usage", "Un assistant de short-list", "Gautier");
  txt(s, "Le club demande : cet attaquant doit-il entrer dans la short-list des cibles prioritaires (valeur attendue ≥ 10 M€) ?", { x: 0.6, y: 1.55, w: 4.1, h: 1.0, fontSize: 16, color: C.text1, valign: "top" });
  txt(s, "Une décision à fort impact sur des personnes : transfert, salaire, carrière.", { x: 0.6, y: 2.7, w: 4.1, h: 0.7, fontSize: 14, italic: true, color: C.accent1, valign: "top" });
  const sh = [["Joueurs évalués", "Risque de discrimination"], ["Club", "Talents écartés à tort"], ["Agents", "Pouvoir de négociation"], ["Régulateurs", "AI Act : transparence"]];
  txt(s, "PARTIES PRENANTES", { x: 5.2, y: 1.55, w: 4.2, h: 0.3, fontSize: 11, bold: true, charSpacing: 3, color: C.accent5 });
  sh.forEach(([a, b], i) => {
    const y = 1.95 + i * 0.75;
    card(s, 5.2, y, 4.2, 0.62, "EEF3EA", `partie_${i}`);
    txt(s, a, { x: 5.4, y: y + 0.08, w: 1.8, h: 0.46, fontSize: 14, bold: true, color: C.accent1, valign: "middle" });
    txt(s, b, { x: 7.2, y: y + 0.08, w: 2.1, h: 0.46, fontSize: 13, color: C.text1, valign: "middle" });
  });
  s.addNotes("GAUTIER (40 s). Le cas d'usage : un club veut un assistant qui décide si un attaquant entre dans sa short-list. C'est une décision qui touche des personnes, donc on a commencé par cartographier les parties prenantes : les joueurs, le club, les agents, et les régulateurs, puisque le recrutement serait un usage à fort impact au sens de l'AI Act.");
}
{
  const s = newSlide("CLAIR_CONTENU", "Contexte et biais", "02 · Les données", "Transfermarkt, 14 saisons", "Gautier");
  const stats = [["6 872", "saisons joueur"], ["Big 5", "championnats"], ["2012-26", "14 saisons"], ["≥ 450", "minutes jouées"]];
  stats.forEach(([n, l], i) => {
    const x = 0.6 + i * 2.25;
    card(s, x, 1.7, 2.05, 1.6, "EEF3EA", `stat_${i}`);
    big(s, n, { x: x + 0.15, y: 1.85, w: 1.75, h: 0.85, fontSize: 40, color: C.accent1, valign: "bottom" });
    txt(s, l, { x: x + 0.15, y: 2.75, w: 1.75, h: 0.4, fontSize: 13, color: C.accent5 });
  });
  txt(s, "Une ligne par attaquant et par saison : matchs, minutes, buts, passes, âge, nationalité, club, championnat et valeur marchande.", { x: 0.6, y: 3.65, w: 8.8, h: 0.7, fontSize: 15, color: C.text1 });
  txt(s, "Source : Football Data from Transfermarkt (Kaggle)", { x: 0.6, y: 4.5, w: 8.8, h: 0.3, fontSize: 10, color: C.accent5 });
  s.addNotes("GAUTIER (30 s). Les données viennent de Transfermarkt via Kaggle : près de 6 900 saisons d'attaquants dans les cinq grands championnats, sur 14 saisons. Pour chaque saison on a les stats, la nationalité, le club et la valeur marchande.");
}
{
  const s = newSlide("PELOUSE_CONTENU", "Contexte et biais", "03 · Le biais", "Même stats, pas le même prix", "Gautier");
  // tableau des scores
  card(s, 0.6, 1.75, 8.8, 1.55, "0F2A0C", "scoreboard");
  txt(s, "GERVINHO · CÔTE D'IVOIRE · PARMA", { x: 0.85, y: 1.9, w: 3.3, h: 0.3, fontSize: 11, bold: true, charSpacing: 2, color: C.accent6 });
  big(s, "5,5 M€", { x: 0.85, y: 2.2, w: 3.0, h: 0.95, fontSize: 54, color: C.background1, valign: "middle" });
  txt(s, [{ text: "SERIE A 2019/20", options: { fontSize: 10, color: C.accent6, breakLine: true } }, { text: "6 buts · 4 passes", options: { fontSize: 15, bold: true, color: C.background1 } }],
    { x: 3.9, y: 2.15, w: 2.2, h: 0.9, align: "center", valign: "middle" });
  txt(s, "HIGUAÍN · ARGENTINE · JUVENTUS", { x: 5.85, y: 1.9, w: 3.3, h: 0.3, fontSize: 11, bold: true, charSpacing: 2, color: C.accent6, align: "right" });
  big(s, "25,5 M€", { x: 6.15, y: 2.2, w: 3.0, h: 0.95, fontSize: 54, color: C.accent3, align: "right", valign: "middle" });
  // primes
  const pr = [["+25 %", "Amérique du Sud", C.accent3], ["-8 %", "Afrique", C.accent4], ["0,65", "disparate impact du label historique (seuil : 0,8)", C.background1]];
  pr.forEach(([n, l, col], i) => {
    const x = 0.6 + i * 2.95;
    big(s, n, { x, y: 3.55, w: 2.7, h: 0.75, fontSize: 40, color: col });
    txt(s, l, { x, y: 4.3, w: 2.7, h: 0.6, fontSize: 13, color: C.background2, valign: "top" });
  });
  s.addNotes("GAUTIER (45 s). Exemple : Gervinho et Higuaín, même saison, même championnat, exactement 6 buts et 4 passes. L'un vaut 5,5 millions, l'autre 25,5. Sur tout le dataset, à performance égale, un attaquant sud-américain vaut 25 % de plus, un africain 8 % de moins. Si on apprend la short-list sur ces valeurs historiques, le disparate impact tombe à 0,65, sous le seuil légal de 80 %. Je laisse Simon expliquer comment on corrige ça.");
}

// ================= SECTION 2 : méthode (Simon) =================
pres.addSection({ title: "Méthode" });
{
  const s = newSlide("CLAIR_CONTENU", "Méthode", "04 · Réduction du biais", "Trois leviers, à trois endroits", "Simon");
  const lv = [["Données", "Relabeling et reweighing", "Label corrigé : valeur attendue d'après la seule performance. Poids qui rendent le label indépendant du groupe."],
              ["Entrée", "Masquage de la nationalité", "La nationalité disparaît du prompt. Insuffisant seul : le club peut servir de proxy."],
              ["Sortie", "Seuils par groupe", "Post-traitement sur P(YES) pour viser la parité démographique."]];
  lv.forEach(([a, b, c], i) => {
    const x = 0.6 + i * 3.0;
    card(s, x, 1.65, 2.8, 3.2, "EEF3EA", `levier_${i}`);
    big(s, `0${i + 1}`, { x: x + 0.2, y: 1.8, w: 1.0, h: 0.6, fontSize: 32, color: C.accent2 });
    txt(s, a.toUpperCase(), { x: x + 0.2, y: 2.45, w: 2.4, h: 0.3, fontSize: 11, bold: true, charSpacing: 3, color: C.accent5 });
    txt(s, b, { x: x + 0.2, y: 2.75, w: 2.4, h: 0.65, fontSize: 16, bold: true, color: C.accent1, valign: "top" });
    txt(s, c, { x: x + 0.2, y: 3.45, w: 2.4, h: 1.3, fontSize: 13, color: C.text1, valign: "top" });
  });
  s.addNotes("SIMON (50 s). On agit à trois endroits. Sur les données : on remplace le label historique par un label corrigé, la valeur que justifie la seule performance, et on repondère les exemples (reweighing de Kamiran et Calders). Sur l'entrée : on masque la nationalité du prompt, mais l'ablation montre que ça ne suffit pas, le club sert de proxy. En sortie : des seuils par groupe pour la parité démographique.");
}
{
  const s = newSlide("CLAIR_CONTENU", "Méthode", "05 · Adapter le LLM", "Qwen2.5-0.5B, trois méthodes", "Simon");
  const m = [["Fine-tuning complet", "494 M", "paramètres entraînés", "Tous les poids sont ajustés. Le plus performant attendu, le plus coûteux."],
             ["LoRA", "2,2 M", "paramètres entraînés", "Matrices de rang 16 sur l'attention, environ 0,4 % du modèle."],
             ["Distillation", "8 / 24", "couches pour l'élève", "Un élève plus petit imite le modèle fine-tuné : KL + vrai label."]];
  m.forEach(([a, n, l, d], i) => {
    const x = 0.6 + i * 3.0;
    card(s, x, 1.65, 2.8, 2.75, i === 2 ? "1F4A1A" : "EEF3EA", `methode_${i}`);
    const dark = i === 2;
    txt(s, a, { x: x + 0.2, y: 1.8, w: 2.4, h: 0.4, fontSize: 16, bold: true, color: dark ? C.background1 : C.accent1 });
    big(s, n, { x: x + 0.2, y: 2.25, w: 2.4, h: 0.75, fontSize: 40, color: dark ? C.accent6 : C.accent1 });
    txt(s, l, { x: x + 0.2, y: 3.0, w: 2.4, h: 0.3, fontSize: 11, color: dark ? C.accent6 : C.accent5 });
    txt(s, d, { x: x + 0.2, y: 3.35, w: 2.4, h: 0.95, fontSize: 13, color: dark ? C.background2 : C.text1, valign: "top" });
  });
  txt(s, "Chaque méthode en version biaisée et corrigée. Références sans entraînement : zero-shot, few-shot, RAG.", { x: 0.6, y: 4.6, w: 8.8, h: 0.4, fontSize: 13, italic: true, color: C.accent5 });
  s.addNotes("SIMON (50 s). Le LLM est Qwen2.5 0,5 milliard, il tourne sur un GPU Colab. On l'adapte avec les trois méthodes vues en cours : fine-tuning complet, LoRA, et distillation, où un élève de 8 couches imite le modèle fine-tuné. Chacune existe en version biaisée, sur les labels historiques, et corrigée. Few-shot et RAG servent de références sans entraînement.");
}
{
  const s = newSlide("CLAIR_CONTENU", "Méthode", "06 · Benchmark", "Performance, équité, efficacité", "Simon");
  const hdr = ["Méthode (version corrigée)", "F1 vs y_fair", "Disparate impact", "Paramètres entraînés", "Entraînement"].map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.pitch }, fontSize: 12 } }));
  const rows = [["Fine-tuning complet", "à venir", "à venir", "494 M", "à venir"], ["LoRA", "à venir", "à venir", "2,2 M", "à venir"], ["Distillation", "à venir", "à venir", "≈ 255 M", "à venir"], ["Few-shot (référence)", "à venir", "à venir", "0", "0"]]
    .map((r, i) => r.map(t => ({ text: t, options: { fontSize: 12, color: HEX.ink, fill: { color: i % 2 ? "FFFFFF" : HEX.pale } } })));
  s.addTable([hdr, ...rows], { x: 0.6, y: 1.6, w: 8.8, colW: [2.6, 1.4, 1.6, 1.7, 1.5], rowH: 0.42, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: HEX.rule }, valign: "middle", objectName: "tableau_benchmark" });
  card(s, 0.6, 3.95, 8.8, 0.85, "EEF3EA", "critere");
  txt(s, [{ text: "Critère de choix  ", options: { bold: true, color: C.accent1 } }, { text: "score = F1(y_fair) × min(1, DI / 0,8). À score proche, le modèle le plus léger.", options: { color: C.text1 } }], { x: 0.8, y: 4.05, w: 8.4, h: 0.65, fontSize: 14, valign: "middle" });
  s.addNotes("SIMON (60 s). [Chiffres à compléter après le run Colab.] On compare la F1 par rapport au label corrigé, le disparate impact, et l'efficacité : paramètres entraînés et temps d'entraînement. Le critère est explicite : la F1, pénalisée si le disparate impact passe sous 0,8. Point d'honnêteté : les versions corrigées sont notées sur le label qu'elles apprennent, donc on regarde aussi la F1 sur le label historique.");
}
{
  const s = newSlide("CLAIR_CONTENU", "Méthode", "07 · Explicabilité", "LIME et SHAP sur le LLM", "Simon");
  [["LIME", "Quelles variables portent cette décision ?", "Perturbe le profil, refait le prompt, ajuste un modèle linéaire local."],
   ["SHAP", "Combien chaque variable pèse-t-elle ?", "Valeurs de Shapley : contribution de chaque variable par rapport à la moyenne."]].forEach(([a, q, d], i) => {
    const x = 0.6 + i * 4.5;
    card(s, x, 1.6, 4.3, 1.45, "EEF3EA", `xai_${i}`);
    big(s, a, { x: x + 0.2, y: 1.7, w: 1.4, h: 0.6, fontSize: 30, color: C.accent1 });
    txt(s, q, { x: x + 1.6, y: 1.75, w: 2.55, h: 0.55, fontSize: 14, bold: true, color: C.text1, valign: "middle" });
    txt(s, d, { x: x + 0.2, y: 2.35, w: 3.9, h: 0.6, fontSize: 13, color: C.text1, valign: "top" });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.25, w: 8.8, h: 1.65, fill: { color: "FFFFFF" }, line: { color: HEX.rule, dashType: "dash", width: 1 }, objectName: "emplacement_figure_shap" });
  txt(s, "Emplacement figure : SHAP waterfall avant / après correction (notebook, section 5.2)", { x: 0.8, y: 3.3, w: 8.4, h: 1.55, fontSize: 13, color: C.accent5, align: "center", valign: "middle" });
  s.addNotes("SIMON (60 s). Le LLM lit du texte, donc on encode le profil en variables, on les perturbe, on reconstruit un prompt à chaque fois et on lit P(YES). LIME donne les variables qui portent la décision localement, SHAP la contribution de chacune. [Commenter la figure : poids de la nationalité avant, puis après correction.] Limite assumée : une explication n'est pas une justification.");
}

// ================= SECTION 3 : démo et limites (Mathis) =================
pres.addSection({ title: "Démo et limites" });
{
  const s = newSlide("PELOUSE_CONTENU", "Démo et limites", "08 · Démo finale", "Gervinho contre Higuaín", "Mathis");
  txt(s, "Serie A 2019/20 · 6 buts · 4 passes chacun · probabilité d'entrer dans la short-list", { x: 0.6, y: 1.55, w: 8.8, h: 0.4, fontSize: 16, color: C.accent6 });
  [["GERVINHO", "Côte d'Ivoire · Parme", "Avec la nationalité argentine", 0.6], ["HIGUAÍN", "Argentine · Juventus", "Avec la nationalité ivoirienne", 5.1]].forEach(([n, sub, swap, x], i) => {
    card(s, x, 2.1, 4.3, 2.75, "0F2A0C", `demo_${i}`);
    big(s, n, { x: x + 0.25, y: 2.22, w: 3.8, h: 0.5, fontSize: 28, color: C.background1 });
    txt(s, sub, { x: x + 0.25, y: 2.72, w: 3.8, h: 0.3, fontSize: 12, color: C.accent6 });
    [["Modèle biaisé", C.accent3], ["Modèle corrigé", C.background1]].forEach(([lab, col], j) => {
      const y = 3.15 + j * 0.45;
      txt(s, lab, { x: x + 0.25, y, w: 2.6, h: 0.35, fontSize: 13, color: C.background2, valign: "middle" });
      big(s, "à venir", { x: x + 2.85, y, w: 1.2, h: 0.35, fontSize: 20, color: col, align: "right", valign: "middle" });
    });
    txt(s, [{ text: swap + " (biaisé) : ", options: { color: C.accent6 } }, { text: "à venir", options: { bold: true, color: C.accent3 } }], { x: x + 0.25, y: 4.18, w: 3.8, h: 0.5, fontSize: 12, valign: "middle" });
  });
  s.addNotes("MATHIS (50 s). Plutôt qu'un profil inventé, on reprend notre vraie paire : Gervinho et Higuaín, mêmes stats. Avec le modèle biaisé, [chiffres] : Higuaín est favorisé. Avec le modèle corrigé, [chiffres] : les deux sont traités de la même façon. Et si on donne à Gervinho la nationalité argentine, sans rien changer d'autre, le modèle biaisé [chiffre] : c'est la preuve que la nationalité seule faisait la différence. SHAP, qu'a montré Simon, explique pourquoi.");
}
{
  const s = newSlide("PELOUSE_CONTENU", "Démo et limites", "09 · Bonus interactif", "Surcoté ou sous-coté ?", "Mathis");
  txt(s, "N'importe quel attaquant, n'importe quelle saison : sa valeur Transfermarkt face à la valeur que justifient ses stats.", { x: 0.6, y: 1.55, w: 8.8, h: 0.7, fontSize: 16, color: C.background2 });
  [["SURCOTÉ", "> +20 %", C.accent3], ["JUSTE PRIX", "± 20 %", C.background1], ["SOUS-COTÉ", "< -20 %", C.accent6]].forEach(([a, b, col], i) => {
    const x = 0.6 + i * 3.0;
    card(s, x, 2.5, 2.8, 1.5, "0F2A0C", `verdict_${i}`);
    big(s, a, { x: x + 0.2, y: 2.62, w: 2.4, h: 0.55, fontSize: 26, color: col });
    txt(s, b, { x: x + 0.2, y: 3.25, w: 2.4, h: 0.4, fontSize: 15, color: C.background2 });
  });
  txt(s, "Le LLM commente le verdict en langage naturel, avec la part de l'écart liée à la prime de la confédération.", { x: 0.6, y: 4.25, w: 8.8, h: 0.6, fontSize: 13, italic: true, color: C.accent6 });
  s.addNotes("MATHIS (40 s). En bonus, un widget dans le notebook : on choisit un joueur et une saison, et on compare sa valeur Transfermarkt à la valeur que justifient ses stats. Au-delà de +20 % il est surcoté, en dessous de -20 % sous-coté. [Démo live possible : un joueur connu.]");
}
{
  const s = newSlide("CLAIR_CONTENU", "Démo et limites", "10 · Red teaming et limites", "Penser comme l'adversaire", "Mathis");
  txt(s, "ATTAQUES TESTÉES", { x: 0.6, y: 1.55, w: 4.2, h: 0.3, fontSize: 11, bold: true, charSpacing: 3, color: C.accent5 });
  [["Injection", "« les Sud-Américains se vendent toujours cher » glissé dans le profil"], ["Proxy", "même joueur placé dans un club brésilien"]].forEach(([a, b], i) => {
    const y = 1.95 + i * 1.05;
    card(s, 0.6, y, 4.2, 0.9, "EEF3EA", `attaque_${i}`);
    txt(s, a, { x: 0.8, y: y + 0.1, w: 3.8, h: 0.3, fontSize: 14, bold: true, color: C.accent1 });
    txt(s, b, { x: 0.8, y: y + 0.42, w: 3.8, h: 0.4, fontSize: 12, color: C.text1 });
  });
  txt(s, "LIMITES", { x: 5.2, y: 1.55, w: 4.2, h: 0.3, fontSize: 11, bold: true, charSpacing: 3, color: C.accent5 });
  txt(s, [
    { text: "Transfermarkt est une estimation communautaire, pas un prix réel", options: { bullet: true, breakLine: true } },
    { text: "Le label corrigé ignore xG, dribbles, profil tactique", options: { bullet: true, breakLine: true } },
    { text: "Contrats absents : durée restante et salaire pèsent sur la valeur", options: { bullet: true, breakLine: true } },
    { text: "Une explication n'est pas une justification", options: { bullet: true, breakLine: true } },
    { text: "Équité contre performance : un arbitrage assumé", options: { bullet: true } },
  ], { x: 5.2, y: 1.95, w: 4.2, h: 2.6, fontSize: 13, color: C.text1, paraSpaceAfter: 8, valign: "top" });
  s.addNotes("MATHIS (40 s). On a joué l'adversaire : injection de texte dans le profil, et proxy en plaçant le joueur dans un club brésilien. Côté limites : Transfermarkt reste une estimation communautaire, les contrats ne sont pas dans les données alors qu'un joueur en fin de contrat vaut moins, notre label corrigé repose sur un modèle simple de la performance, et SHAP montre où le modèle regarde sans prouver l'absence de biais.");
}
{
  const s = pres.addSlide({ masterName: "PELOUSE_TITRE", sectionTitle: "Démo et limites" });
  s.addText("CONCLUSION", { placeholder: "kicker" });
  s.addText("ETHICS BY DESIGN", { placeholder: "title" });
  s.addText("Mesurer le biais, le réduire à trois niveaux, choisir le modèle sur un critère explicite, expliquer chaque décision.", { placeholder: "body" });
  s.addText("Merci. Questions ?", { placeholder: "names" });
  s.addNotes("MATHIS (20 s). Pour conclure : on a mesuré le biais, on l'a réduit sur les données, l'entrée et la sortie, on a choisi le modèle sur un critère qui combine performance et équité, et chaque décision est expliquée. Merci, place aux questions.");
}

(async () => {
  await pres.writeFile({ fileName: "ethics_soutenance.pptx" });
  await applyTheme("ethics_soutenance.pptx", THEME);
  console.log("ok");
})();
