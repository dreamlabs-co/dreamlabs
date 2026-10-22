// The setup wizard page: one step at a time, each a card, talking to the routes in robots/setup_wizard.py.
import { ArmView, MARKER_ROOM, UrdfView, cadNavigation, folderFiles, unzip } from "/setup_viewer.js";

const $ = (id) => document.getElementById(id);
const el = (tag, attrs, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) node.append(kid instanceof Node ? kid : String(kid));
  return node;
};
// The poll redraws every 150 ms, and rewriting a node drops any text selected in it: write only what changed.
const setText = (node, text) => { if (node.textContent !== text) node.textContent = text; };
const setKids = (node, key, kids) => {
  if (node.dataset.drawn === key) return;
  node.dataset.drawn = key;
  node.replaceChildren(...kids());
};
const SIDES = ["left", "right"];

const S = {
  state: null, step: 0, pick: null, robot: null, labels: null, prefill: null, scan: null, ticked: new Set(), opened: false,
  cams: {}, shots: [], dataset: null, status: null,
  so101Files: [], so101Side: { left: null, right: null }, urdf: null, homeBusy: false, embodiments: null, finished: false, missedPolls: 0, gone: false,
  reach: false, reachSpan: {}, way: "wizard",
};
// The 3D views: how the arm is drawn (GET /api/view), and both arms in the home step and in the powered checks.
const V = { drawing: null, home: null, power: null, busy: false };

const GONE = "Setup stopped running on the station, and the arms may still be powered on. Support the arms, then run dreamcli setup again.";
class Gone extends Error { constructor() { super(GONE); } }

async function api(method, path, body) {
  let reply;
  try {
    reply = await fetch(path, {
      method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined,
    });
  } catch { throw new Gone(); }
  const data = await reply.json().catch(() => ({}));
  if (!reply.ok) throw new Error(data.error || `${method} ${path} answered ${reply.status}`);
  return data;
}

async function guard(errId, fn) {
  $(errId).textContent = "";
  try { await fn(); } catch (exc) {
    $(errId).textContent = exc.message;
    $(errId).scrollIntoView({ block: "center" });  // on a phone the words land below the screen otherwise (B32)
  }
}

// A question in the page's own dialog, not the browser's: resolves true on its yes button, false on Go back or Esc.
function ask(title, text, yes) {
  $("askTitle").textContent = title;
  $("askText").textContent = text;
  $("askYes").textContent = yes;
  $("askDialog").showModal();
  return new Promise((resolve) => {
    const done = (answer) => () => { $("askDialog").close(); resolve(answer); };
    $("askYes").onclick = done(true);
    $("askNo").onclick = done(false);
    $("askDialog").oncancel = done(false);
  });
}

async function busy(buttons, fn) {
  for (const button of buttons) button.disabled = true;
  try { await fn(); } finally { for (const button of buttons) button.disabled = false; }
}

// The cards: the one being done is open, those before are done with a summary, those after wait.
// The step list (setup_page.js) goes back to a done step the way a summary's Change once did.
globalThis.setupGo = (step) => go(step);

function go(step) {
  S.step = step;
  document.querySelectorAll("section.card").forEach((card) => {
    const i = Number(card.dataset.step);
    card.classList.toggle("open", i === step);
    card.classList.toggle("done", i < step);
    card.classList.toggle("later", i > step);
    const pill = card.querySelector("[data-pill]");
    pill.textContent = i < step ? "Done" : i === step ? "Current" : "";
    pill.className = "pill" + (i < step ? " good" : i > step ? " hidden" : "");  // none on a step not reached (B13)
  });
  document.querySelector(`section.card[data-step="${step}"]`).scrollIntoView({ behavior: "smooth", block: "start" });
  $("title").focus({ preventScroll: true });  // focus leaves the hidden card's button for the new step's heading (2.4.3)
  if (S.reach) api("POST", "/api/reach/paint", { on: step === 4 }).catch((exc) => { $("reachErr").textContent = exc.message; });
}

// The badges count the cards shown: a robot without the Reach area step (the YAM) goes 0 to 5.
function numberSteps() {
  $("reachCard").classList.toggle("hidden", !S.reach);
  let shown = 0;
  document.querySelectorAll("section.card").forEach((card) => {
    if (!card.classList.contains("hidden")) card.querySelector(".step").textContent = String(shown++);
  });
}

function summary(step, text) {
  document.querySelector(`section.card[data-step="${step}"] [data-summary]`).replaceChildren(el("span", {}, text));
}

// Step 0: the robot.
// A robot card's name where the page says more than the body's label: OpenYAM arms are set up as a YAM.
const CARD_LABEL = { yam: "YAM / OpenYAM" };

function renderRobots() {
  const choices = S.state.robots.map((robot) => ({
    key: robot.body, title: CARD_LABEL[robot.body] || robot.label, sub: robot.unavailable || "", off: !!robot.unavailable, note: robot.note,
  }));
  choices.push({ key: "custom", title: "My own robot", sub: "", off: false, note: null });
  $("robots").replaceChildren(...choices.map((c) => el("button", {
    class: "robot" + (S.pick === c.key ? " on" : ""), disabled: c.off,
    onclick: () => { S.pick = c.key; showPick(); if (c.key === "so101") checkSo101(); },
  }, el("b", {}, c.title), el("span", {}, c.sub))));
  const picked = choices.find((c) => c.key === S.pick);
  const self = S.way === "self";
  $("custom").classList.toggle("hidden", self || S.pick !== "custom");
  $("so101").classList.toggle("hidden", self || S.pick !== "so101");
  $("embodimentBox").classList.toggle("hidden", self || (S.pick !== "so101" && S.pick !== "custom"));
  $("newBox").classList.toggle("hidden", self || (S.pick !== "so101" && S.pick !== "custom") || $("embodiment").value !== NEW);
  $("pickActions").classList.toggle("hidden", self);
  $("calibHelp").textContent = S.pick === "so101" ? "Setup reads each arm's LeRobot calibration files so joint angles match the trained arm."
    : "Zero each motor the way your robot's maker describes. Uncalibrated motors can jump during the power test.";
  $("selfBox").classList.toggle("hidden", !self || !picked || picked.off);
  renderSo101Files();
  if (S.pick === "custom") renderOwn();
  const needs = !picked || picked.off ? [["a robot", "robots"]] : stillNeeded(picked);
  gate("pickRobot", needs);
  $("pickNeed").replaceChildren(...(!needs.length ? [] : ["Still needed: ",
    ...needs.flatMap(([text, id], i) => [i ? "; " : "", el("button", { class: "linkish", onclick: () => showNeed(id) }, text)]), "."]));
}

// What the picked robot still lacks before Continue, each as [what, the id of the block that gives it], so a gray
// Continue says why rather than sitting grayed out with no explanation.
function stillNeeded(picked) {
  if (!picked || picked.off) return [];
  const own = S.pick === "custom", needs = [];
  if (own && !ownArm()) {
    needs.push(S.armWay === "urdf" ? ["your arm's tool point (set it, or skip it)", "urdfBox"]
      : S.armWay === "ours" ? ["an arm model", "armList"] : ["an arm model", "armUrdf"]);
  }
  if (own && !S.driver) needs.push(["your driver", "driverBox"]);
  if (S.pick === "so101" && !(S.so101Side.left && S.so101Side.right)) needs.push(["a calibration file for each arm", "so101"]);
  if ((own || S.pick === "so101") && !$("embodiment").value) needs.push(["an embodiment", "embodimentBox"]);
  return needs;
}

// A step's Continue stays clickable, and a refused click outlines what is incomplete. gate(): the button looks blocked while `needs` ([what,
// the id of the block that gives it] pairs) is not empty, and once a click was refused each need's block has a thin
// red line, gone as soon as that need is met. refused(): true for a click on a blocked button, which lines the
// blocks and scrolls to the first.
const GATES = {};
function gate(id, needs) {
  const g = (GATES[id] ||= { needs: [], tried: false });
  for (const [, box] of g.needs) $(box).classList.remove("missing");
  g.needs = needs;
  g.tried = g.tried && needs.length > 0;
  if (g.tried) for (const [, box] of needs) $(box).classList.add("missing");
  $(id).classList.toggle("blocked", needs.length > 0);
  $(id).setAttribute("aria-disabled", String(needs.length > 0));
  // What a blocked button waits for, in words beside it, not only as an outline (1.4.1).
  const note = $(`${id}Need`);
  if (note) {
    note.textContent = needs.length ? `Still needed: ${needs.map(([what]) => what).join(", ")}.` : "";
    $(id).setAttribute("aria-describedby", note.id);
  }
}
function refused(id) {
  const g = GATES[id];
  if (!g) return $(id).classList.contains("blocked");  // blocked in the markup, before its step's first render
  if (!g.needs.length) return false;
  g.tried = true;
  gate(id, g.needs);
  showNeed(g.needs[0][1]);
  return true;
}

// A still-needed block scrolled to the middle of the window and flashed.
function showNeed(id) {
  const box = $(id);
  box.scrollIntoView({ behavior: "smooth", block: "center" });
  box.classList.remove("flash");
  requestAnimationFrame(() => box.classList.add("flash"));
}

// The picked robot, and the account's embodiments once an SO-101 or a robot of your own is picked: its fine-tune's,
// which names its cameras.
async function showPick() {
  renderRobots();
  if (S.way === "self") {
    if (!$("selfBox").classList.contains("hidden")) await guard("pickErr", showTemplate);
    return;
  }
  if (S.pick !== "so101" && S.pick !== "custom") return;
  if (S.pick === "custom" && !D.listing) await guard("pickErr", openDriver);
  await guard("pickErr", loadEmbodiments);
  renderRobots();
}

// Step 0's two ways: the wizard, or Set it up myself, the station file and the commands that run it with no step and
// no check (the later steps' cards are hidden).
function pickWay(way) {
  S.way = way;
  $("wayWizard").classList.toggle("on", way === "wizard");
  $("waySelf").classList.toggle("on", way === "self");
  document.querySelector("main").classList.toggle("self-way", way === "self");
  return showPick();
}
// The station file with what the operator must fill in colored: each REPLACE_ value, and the key it is the value of.
// A home's joint angles show as 0.0, still marked, since the person starts from zeros and measures them later.
function markReplace(text) {
  const parts = [];
  let at = 0;
  for (const m of text.matchAll(/\b\w+(?=:\s*\[?\s*REPLACE_)|REPLACE_\w+/g)) {
    const shown = m[0] === "REPLACE_RAD" ? "0.0" : m[0];
    parts.push(text.slice(at, m.index), el("span", { class: m[0].startsWith("REPLACE_") ? "yr" : "yk" }, shown));
    at = m.index + m[0].length;
  }
  return [...parts, text.slice(at)];
}

// Set it up myself: the file Save writes for the picked robot, every value the steps measure left as REPLACE_
// (GET /api/template), and the run the wizard's file gets: a dry run first, then the run.
async function showTemplate() {
  const path = S.state.config_path, model = S.pick === "yam" ? "" : ' --checkpoint "<your model>"';
  const run = `dreamcli robot run --config ${path}${model} --task "<what to do>"`;
  $("selfPath").textContent = path;
  const yaml = (await api("GET", `/api/template?body=${encodeURIComponent(S.pick)}`)).yaml;
  $("selfYaml").replaceChildren(...markReplace(yaml.replace(/^(#.*\n)+\n?/, "")));  // the table says what the comments did
  $("selfCanRow").classList.toggle("hidden", !yaml.includes("can_channel"));
  showCommand("selfDry", `${run} --dry-run`);
  $("selfTcp").classList.toggle("hidden", S.pick !== "custom");
  // Set each arm's home is the third card, the fourth when the URDF card comes before it.
  for (const id of ["homeStep", "homeStepRow"]) $(id).textContent = S.pick === "custom" ? "4" : "3";
  showCommand("selfRun", run);
  $("selfDocs").href = `${S.state.connection.platform_url}/docs/set-it-up-yourself/`;
}

// A robot of your own's driver, picked on this computer's own files: the page runs on the operator's laptop and the
// wizard on the station, so the station lists its folders (GET /api/files) and a picked Python file's classes (GET
// /api/classes, read, not run). It opens on the driver the file names, else in the station file's folder. A driver
// installed as a package is typed as package.module:Class instead.
const D = { listing: null, file: null, classes: null };
const inFolder = (dir, name) => (dir.endsWith("/") ? dir + name : `${dir}/${name}`);

async function openDriver() {
  await openFolder(S.state.driver_dir);
  const [, file] = /^(\/.*\.py):\w+$/.exec(S.driver) || [];
  if (!file) return;
  await openFolder(file.slice(0, file.lastIndexOf("/")) || "/");
  await openFile(file);
}

async function openFolder(dir) {
  D.listing = await api("GET", `/api/files?dir=${encodeURIComponent(dir)}`);
  Object.assign(D, { file: null, classes: null });
  renderDriver();
}

// A Python file picked in the driver's dialog: its one class is the driver at once and the dialog closes; several ask
// which beside the button; none keeps the dialog open and says so.
async function openFile(file) {
  D.classes = (await api("GET", `/api/classes?file=${encodeURIComponent(file)}`)).classes;
  D.file = file;
  if (D.classes.length) $("driverDialog").close();
  if (D.classes.length === 1) pickDriver(`${file}:${D.classes[0]}`);
  renderDriver();
}

function pickDriver(value) {
  S.driver = value;
  renderRobots();
}

// The driver's block, kept to one obvious control: one button opens the station's folders in a dialog; once picked, the
// block says which file and class with a check, and a file of several classes asks which.
function renderDriver() {
  const l = D.listing, name = D.file ? D.file.split("/").pop() : "";
  $("driverDir").textContent = l ? l.dir : "";
  $("driverUp").disabled = !l || !l.parent;
  const open = (fn) => () => guard("pickErr", fn);
  const rows = l ? [
    ...l.dirs.map((dir) => el("button", { onclick: open(() => openFolder(inFolder(l.dir, dir))) }, `${dir}/`)),
    ...l.files.map((file) => el("button", { class: D.file === inFolder(l.dir, file) ? "on" : "", onclick: open(() => openFile(inFolder(l.dir, file))) }, file)),
  ] : [];
  $("driverList").replaceChildren(...(!l ? [] : rows.length ? rows : [el("p", { class: "note empty" }, "No folders or Python files here.")]));
  $("driverDialogNote").textContent = D.file && !D.classes.length ? `${name} defines no class: pick another file.` : "";
  const several = D.file && D.classes.length > 1;
  $("driverClasses").classList.toggle("hidden", !several);
  $("driverClasses").replaceChildren(...(!several ? [] : [el("span", { class: "note" }, `${name} has ${D.classes.length} classes. Which one runs one arm?`),
    ...D.classes.map((cls) => el("label", {}, el("input", { type: "radio", name: "driverClass", checked: S.driver === `${D.file}:${cls}`, onchange: () => pickDriver(`${D.file}:${cls}`) }), cls))]));
  const [, file, cls] = /^(\/.*\.py):(\w+)$/.exec(S.driver) || [];
  $("driverPicked").classList.toggle("hidden", !S.driver);
  $("driverPicked").textContent = !S.driver ? "" : `\u2713 Using ${file ? `${file.split("/").pop()}: ${cls}` : S.driver}`;
  $("driverBox").classList.toggle("done", !!S.driver);
  $("driverReq").textContent = S.driver ? "Done" : "Required";
  $("driverChoose").replaceChildren(el("span", { class: "fileicon", "aria-hidden": "true" }), S.driver ? "Change driver file" : "Choose driver file (.py)");
}

// dream's model of a robot of your own's arm, one of two ways: its URDF (the step below), or an arm dream has a model
// of. The arms the robot tiles above already are, and second names of one of these, are not offered (GET /api/state
// ik_arms). None is picked until the operator picks (or the file's, on a re-run): a wrong model of the arm draws and
// solves another arm.
const ownArm = () => (S.armWay === "urdf" ? S.urdfIk : S.armWay === "ours" ? S.ourIk : "");

function renderOwn() {
  renderDriver();
  $("armUrdf").classList.toggle("on", S.armWay === "urdf");
  $("armOurs").classList.toggle("on", S.armWay === "ours");
  $("armList").classList.toggle("hidden", S.armWay !== "ours");
  $("urdfBox").classList.toggle("hidden", S.armWay !== "urdf");
  $("armSelect").replaceChildren(el("option", { value: "", disabled: true }, "Choose your arm"),
    ...S.state.ik_arms.map((arm) => el("option", { value: arm.name }, arm.label)));
  $("armSelect").value = S.ourIk || "";
  const count = S.state.ik_arms.length;
  $("armOursSub").replaceChildren(`${count} model${count === 1 ? "" : "s"}. Yours not here? Switch to `,
    el("button", { class: "linkish strong", onclick: () => { S.armWay = "urdf"; renderRobots(); } }, "From my URDF"), ".");
}

// The embodiment list's last option: a new one, which Continue registers from the arm.
const NEW = "new";
async function loadEmbodiments() {
  if (S.embodiments) return;
  S.embodiments = (await api("GET", "/api/embodiments")).embodiments;
  $("embodiment").replaceChildren(...S.embodiments.map((e) => el("option", { value: e.name }, e.name)),
    el("option", { value: NEW }, "New embodiment for Dream Labs base models"));
  if (S.embodiments.some((e) => e.name === S.state.existing_embodiment)) $("embodiment").value = S.state.existing_embodiment;
  $("embodimentNote").textContent = S.embodimentNote
    || (S.embodiments.length ? "" : "Your account doesn't have an embodiment yet, so Continue registers a new one.");
  renderEmbodimentCameras();
}

// The cameras the picked embodiment expects, as chips under it; the cameras step matches them to real ones.
function renderEmbodimentCameras() {
  const picked = (S.embodiments || []).find((e) => e.name === $("embodiment").value);
  $("embodimentCams").replaceChildren(...(picked && (picked.cameras || []).length ? ["Expects cameras", ...picked.cameras.map((name) => el("code", {}, name)),
    el("span", {}, `· You'll match them to real cameras in step ${$("step-2").querySelector(".step").textContent}.`)] : []));
}

// A robot of your own from its URDF, with nothing typed: the operator gives the file (POST /api/urdf reads its links),
// answers "Is this the end of your arm?" on the drawing, clicking the right link if not, then puts the tool marker
// where the fingertips close, beside a YAM whose marker shows where it goes. The marker is six numbers in the end
// link's frame, each with its minus and plus buttons and its field. POST /api/arm_file gets the links and the marker.
const U = { tree: null, base: 0, end: 0, picking: "end", ok: false, tcp: null, rpy: null, move: 10, turn: 15, view: null, yam: null, held: {} };
// The steps of the plus and minus buttons: millimetres, and degrees.
const MOVE_STEPS = [[1, "1 mm"], [5, "5 mm"], [10, "1 cm"], [50, "5 cm"]];
const TURN_STEPS = [1, 5, 15, 90].map((d) => [d, `${d}°`]);
// The marker's numbers, in the end link's frame (the small x, y and z arrows on the drawing): where it sits, mm, and
// how it is turned, degrees, as a URDF's rpy (turns about the fixed x, then y, then z).
const DIMS = [["x", "mm"], ["y", "mm"], ["z", "mm"], ["roll", "°"], ["pitch", "°"], ["yaw", "°"]];

// (3, 3) turns as rows; vectors as [x, y, z].
const mul = (a, b) => a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
const flip = (a) => [0, 1, 2].map((j) => a.map((row) => row[j]));
const apply = (a, v) => a.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function about(axis, degrees) {
  const r = (degrees * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r);
  return [[[1, 0, 0], [0, c, -s], [0, s, c]], [[c, 0, s], [0, 1, 0], [-s, 0, c]], [[c, -s, 0], [s, c, 0], [0, 0, 1]]][axis];
}
// A pose's turn: [qw, qx, qy, qz, ...] (rounded by the server, so made unit first).
function turnOf(pose) {
  const n = Math.hypot(...pose.slice(0, 4)), [w, x, y, z] = pose.slice(0, 4).map((v) => v / n);
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
}
// A URDF rpy, degrees, as a turn, Rz(yaw) Ry(pitch) Rx(roll); and a turn as one (pitch within 90 degrees, roll 0 when
// pitch is at 90).
const rpyTurn = ([r, p, y]) => mul(about(2, y), mul(about(1, p), about(0, r)));
function rpyOf(m) {
  const deg = (a) => (a * 180) / Math.PI, side = Math.hypot(m[0][0], m[1][0]);
  const yaw = side < 1e-9 ? Math.atan2(-m[0][1], m[1][1]) : Math.atan2(m[1][0], m[0][0]);
  return [side < 1e-9 ? 0 : Math.atan2(m[2][1], m[2][2]), Math.atan2(-m[2][0], side), yaw].map(deg);
}
// Degrees into (-180, 180]; any number to a thousandth, without -0.
const wrap = (d) => { const w = ((((d + 180) % 360) + 360) % 360) - 180; return w === -180 ? 180 : w; };
const fix = (v) => Math.round(v * 1000) / 1000 + 0;

// The marker's first place on an end link: at its origin, out of the gripper along the base axis nearest the arm's
// last bone (from the nearest link back whose origin is not the end's, or the end's own z when there is none), the
// fingers closing along the base's y (x when that is taken).
function firstMarker(tree, end) {
  const at = (link) => tree.poses[link].slice(4);
  const gap = (link) => Math.hypot(...at(end).map((v, k) => v - at(link)[k]));
  let from = tree.parents[end];
  while (from >= 0 && gap(from) < 0.005) from = tree.parents[from];
  const way = from < 0 ? flip(turnOf(tree.poses[end]))[2] : at(end).map((v, k) => v - at(from)[k]);
  const k = [0, 1, 2].reduce((a, b) => (Math.abs(way[b]) > Math.abs(way[a]) ? b : a));
  const z = [0, 1, 2].map((i) => (i === k ? Math.sign(way[k]) || 1 : 0));
  const x = k === 1 ? [1, 0, 0] : [0, 1, 0];
  const local = mul(flip(turnOf(tree.poses[end])), flip([x, cross(z, x), z]));
  return { tcp: [0, 0, 0], rpy: rpyOf(local).map((d) => fix(wrap(d))) };
}

// The marker as the arm file takes it (`dream.arm_file`), in the end link's frame: the tool point, metres, the way
// out of the gripper (the marker's z) and the way the fingers close (its x), to a micrometre.
function markerTool() {
  const local = rpyTurn(U.rpy), round = (v) => v.map((x) => Math.round(x * 1e6) / 1e6 + 0);
  return {
    tcp: round(U.tcp.map((v) => v / 1000)),
    approach: round(local.map((row) => row[2])),
    jaw: round(local.map((row) => row[0])),
  };
}

// The URDF the operator gives: a .zip of its URDF and meshes, or a .urdf alone (urdfFile), or its folder (urdfFolder,
// or dropped), which the drawing then shows.
async function readUrdf() {
  const file = $("urdfFile").files[0];
  $("urdfPick").classList.add("hidden");
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".zip")) {
    await loadUrdf(file.name, await file.text(), null, file.name);
    return;
  }
  await guard("pickErr", async () => pickUrdf(file.name, unzip(await file.arrayBuffer())));
}

// The URDF's folder, picked (its FileList) or dropped (its entry): the files in it, read as a .zip's are
// (folderFiles()).
async function readUrdfFolder(given) {
  $("urdfPick").classList.add("hidden");
  const name = given.isDirectory ? given.name : given.length ? given[0].webkitRelativePath.split("/")[0] : "";
  if (name) await guard("pickErr", async () => pickUrdf(name, await folderFiles(given)));
}

// The URDFs among the files of a .zip or folder (`name`): one is drawn; several are listed to choose one,
// and nothing is drawn until then.
async function pickUrdf(name, files) {
  const urdfs = [...files.keys()].filter((path) => path.toLowerCase().endsWith(".urdf"));
  if (!urdfs.length) throw new Error(`${name} doesn't contain a .urdf file. Give your arm's URDF as a .zip with its meshes, or its folder.`);
  const texts = await Promise.all(urdfs.map(async (path) => new TextDecoder().decode(await files.get(path)())));
  if (urdfs.length === 1) {
    await loadUrdf(urdfs[0].split("/").pop(), texts[0], files, urdfs[0]);
    return;
  }
  forgetArm();
  S.urdf = null;
  U.tree = null;
  renderUrdf();
  $("urdfDrawn").textContent = "";
  $("urdfPickAsk").textContent = `${name} contains ${urdfs.length} URDFs. Choose the one for your arm:`;
  $("urdfPickList").replaceChildren(...urdfs.map((path, i) => el("label", { class: "device" },
    el("input", { type: "radio", name: "urdfPick", onchange: () => loadUrdf(path.split("/").pop(), texts[i], files, path) }),
    el("div", {}, el("div", { class: "name" }, path), el("div", { class: "sub2" }, urdfCounts(texts[i]))), el("span"))));
  $("urdfPick").classList.remove("hidden");
}

// A URDF's links and moving joints, counted from its text, for the pick of one of several: a hint, not a parse.
function urdfCounts(text) {
  const links = (text.match(/<link\s[^>]*name=/g) || []).length;
  const joints = (text.match(/<joint\s[^>]*type="(revolute|prismatic|continuous)"/g) || []).length;
  return `${links} links, ${joints} moving joints`;
}

// A new URDF, end or base has no tool point yet: Continue waits for Set tool center point or Skip calibration again.
function forgetArm() {
  S.urdfIk = "";
  $("urdfNote").textContent = "";
  $("tcpSaved").classList.add("hidden");
  if (S.state.tool !== "tcp") renderRobots();
}

// files: unzip() or folderFiles() of the .zip or folder the URDF is `path` in, or null for a bare URDF.
async function loadUrdf(name, text, files, path) {
  forgetArm();
  S.urdf = null;
  U.tree = null;
  $("urdfDrawn").textContent = "";
  renderUrdf();
  await guard("pickErr", async () => {
    const tree = await api("POST", "/api/urdf", { urdf: text });
    S.urdf = { name, text };
    Object.assign(U, { tree, base: tree.base, end: tree.flange, picking: "end", ok: false, held: {}, ...firstMarker(tree, tree.flange) });
    if (!U.view) {
      U.view = new UrdfView($("urdfView"), pickLink, dragMarker);
      cadNavigation(U.view, MARKER_ROOM);
    }
    $("urdfDrawn").textContent = drawnLine(tree, await U.view.tree(tree, files, path), files);
    renderUrdf();
  });
}

// What the drawing shows of how the arm looks (UrdfView.tree()'s answer): the links its meshes draw, and the ones
// drawn plain and why.
function drawnLine(tree, drawn, files) {
  if (!files) return drawn.plain.length ? "Your URDF's meshes are separate files, so its links are drawn as simple shapes. To see your arm as it looks, give the URDF as a .zip with its meshes, or its folder." : "";
  const plain = drawn.plain.map((link) => tree.links[link]).join(", ");
  return `${drawn.shown.length} links drawn from your meshes.` + (plain ? ` Drawn as simple shapes: ${plain} (missing from the zip, not STL, OBJ, COLLADA, or glTF, or a compressed glTF: ${drawn.missing.join(", ")}).` : "");
}

// A click on the drawing: the end of the arm, the marker starting again on it, or the link it stands on after "It
// stands on another link".
function pickLink(link) {
  forgetArm();
  if (U.picking === "base") U.base = link;
  else Object.assign(U, { end: link, ...firstMarker(U.tree, link) });
  U.picking = "end";
  U.ok = false;
  renderUrdf();
}

// The links from the base out to the end, or null when the end does not hang below the base.
function chain() {
  const links = [];
  for (let link = U.end; link !== U.base; link = U.tree.parents[link]) {
    if (link < 0) return null;
    links.unshift(link);
  }
  return links;
}

// The chain's moving joints a person decides on (U.held): in `dreamcli tcp`, those the station's embodiment does not
// drive (the server's `joint_map`, TcpServer.joint_map()), or every one when it has no joints to read; none in the
// wizard (`joint_map` null), whose arm drives its whole chain. Never the page: a passive mount may be fixed, or may
// turn (review, 2026-10-18: a Vega-1's hand mount roll, continuous in its URDF).
function spare(links) {
  const map = U.tree.joint_map;
  if (!map) return [];
  return links.filter((link) => U.tree.moving[link] && (map.joints === null || !map.joints.includes(U.tree.joints[link]))).map((link) => U.tree.joints[link]);
}

// One line per spare() joint with its Moves and Held still buttons, neither pressed until the operator presses one,
// under why the account could not be asked when it could not (`joint_map.error`).
function renderSpare(extra) {
  const error = U.tree.joint_map && U.tree.joint_map.error;
  $("urdfSpare").classList.toggle("hidden", U.ok || (!extra.length && !error));
  $("urdfSpare").replaceChildren(...(error ? [el("div", { class: "err" }, error)] : []), ...extra.map((joint) => el("div", { class: "pad" },
    el("code", {}, joint), ": does your robot move this joint? ",
    [[false, "Moves"], [true, "Held still"]].map(([held, text]) => el("button", {
      [held ? "data-held" : "data-moves"]: joint, class: U.held[joint] === held ? "on" : "", "aria-pressed": String(U.held[joint] === held),
      onclick: () => { U.held[joint] = held; renderUrdf(); },
    }, text)))));
}

async function confirmEnd() {
  U.ok = true;
  renderUrdf();
  if (U.yam) return;
  U.yam = new ArmView($("yamView"));
  const yam = await api("GET", "/api/reference_arm");
  await U.yam.load(yam, yam.frame);
  U.yam.aim(yam.frame, MARKER_ROOM);
  U.yam.mark(yam.tool_link);
  U.yam.label("Reference: YAM (for comparison only)");
  cadNavigation(U.yam, MARKER_ROOM);
}

// The marker's number `i` (DIMS) set to `value`, an angle wrapped into (-180, 180]; a field left blank or not a
// number shows the number again.
function setDim(i, value) {
  if (Number.isFinite(value)) {
    if (i < 3) U.tcp[i] = fix(value);
    else U.rpy[i - 3] = fix(wrap(value));
  }
  renderUrdf();
}

const nudge = (i, sign) => setDim(i, i < 3 ? U.tcp[i] + sign * U.move : U.rpy[i - 3] + sign * U.turn);

// The marker `amount` mm along its own axis k (0, 1, 2: x, y, z), or turned `amount` degrees about axis k - 3, from
// where a drag on its triad or the number box at a handle (UrdfView) found it: `from`, the six numbers then.
function dragMarker(k, amount, [x, y, z, ...rpy]) {
  const local = rpyTurn(rpy);
  if (k < 3) U.tcp = [x, y, z].map((v, i) => fix(v + amount * local[i][k]));
  else U.rpy = rpyOf(mul(local, about(k - 3, amount))).map((d) => fix(wrap(d)));
  renderUrdf();
}

// The marker back where it started on this end link (firstMarker()).
function resetMarker() {
  Object.assign(U, firstMarker(U.tree, U.end));
  renderUrdf();
}

// The marker turned half a turn about its own "out of the gripper" arrow: its "up" and "fingers close" arrows reversed.
function flipUp() {
  U.rpy = rpyOf(mul(rpyTurn(U.rpy), about(2, 180))).map((d) => fix(wrap(d)));
  renderUrdf();
}

// The marker's buttons, drawn once: per kind its steps, then per number its minus, its field and its plus.
function buildPad() {
  const steps = (kind, title, list) => el("div", { class: "pad" }, el("span", { class: "padlabel" }, title),
    list.map(([step, text]) => el("button", { [`data-${kind}-step`]: step, onclick: () => { U[kind] = step; renderUrdf(); } }, text)));
  const row = ([name, unit], i) => el("div", { class: "pad" }, el("span", { class: "padlabel" }, name),
    el("button", { "data-nudge": `${name}-`, "aria-label": `${name} minus`, onclick: () => nudge(i, -1) }, "−"),
    el("input", { type: "number", step: "any", "data-dim": name, "aria-label": `${name}, ${unit}`, onchange: (e) => setDim(i, parseFloat(e.target.value)) }),
    el("button", { "data-nudge": `${name}+`, "aria-label": `${name} plus`, onclick: () => nudge(i, 1) }, "+"),
    el("span", { class: "unit" }, unit));
  $("urdfPad").replaceChildren(
    el("div", {}, steps("move", "Move step", MOVE_STEPS), DIMS.slice(0, 3).map((d, i) => row(d, i))),
    el("div", {}, steps("turn", "Turn step", TURN_STEPS), DIMS.slice(3).map((d, i) => row(d, i + 3))));
}

function renderUrdf() {
  $("urdfEnd").classList.toggle("hidden", !U.tree);
  if (!U.tree) return;
  const name = (link) => U.tree.links[link];
  const links = chain();
  const joints = links && links.filter((link) => U.tree.moving[link]).length;
  const extra = links ? spare(links) : [];
  $("urdfAsk").textContent = U.ok ? "Place the marker where your gripper's fingertips meet, turned like the YAM's." : "Is this the end of your arm?";
  $("urdfLinks").classList.toggle("hidden", U.ok);
  $("urdfLinks").textContent = U.picking === "base" ? "Click the link your arm is mounted on."
    : !links ? `${name(U.end)} isn't below ${name(U.base)} in your arm. Click the end of your arm.`
    : joints > U.tree.most_joints ? `${name(U.end)} is ${joints} joints from ${name(U.base)}, but your arm has at most ${U.tree.most_joints}. Click the link your gripper is mounted on.`
    // A chain with no joint that moves is not an arm: building it fails only after the tool point is set (a YAM + UMI, say).
    : !joints ? `No joint moves between ${name(U.base)} and ${name(U.end)}. Click the link your gripper is mounted on, or "It's mounted on another link" if your arm isn't mounted on ${name(U.base)}.`
    : `Your arm is mounted on ${name(U.base)} (blue) and ends at ${name(U.end)} (green), ${joints} joints out. If that isn't the end, click the right link in the drawing.`
      + (!extra.length ? "" : U.tree.joint_map.joints ? ` Your embodiment ${U.tree.joint_map.embodiment} doesn't drive ${extra.length} of them. Answer for each below.` : " Answer for each joint below.");
  $("urdfYes").disabled = !links || !joints || joints > U.tree.most_joints || extra.some((joint) => !(joint in U.held));
  renderSpare(extra);
  $("urdfEndActions").classList.toggle("hidden", U.ok);
  $("urdfTool").classList.toggle("hidden", !U.ok);
  $("yamView").classList.toggle("hidden", !U.ok);
  for (const button of $("urdfPad").querySelectorAll("button[data-move-step]")) button.classList.toggle("on", Number(button.dataset.moveStep) === U.move);
  for (const button of $("urdfPad").querySelectorAll("button[data-turn-step]")) button.classList.toggle("on", Number(button.dataset.turnStep) === U.turn);
  const values = [...U.tcp, ...U.rpy];
  for (const input of $("urdfPad").querySelectorAll("input[data-dim]")) {
    input.value = String(Math.round(values[DIMS.findIndex(([dim]) => dim === input.dataset.dim)] * 100) / 100 + 0);
  }
  const pose = U.tree.poses[U.end], turn = turnOf(pose);
  U.view.paint(U.base, U.end, !U.ok);
  U.view.label(U.ok ? "Your arm: drag the marker here" : "Your arm");
  U.view.setMarker(U.ok ? apply(turn, U.tcp.map((v) => v / 1000)).map((v, k) => v + pose[4 + k]) : null, mul(turn, rpyTurn(U.rpy)), { values });
}

// calibration: "calibrated" (Set tool center point) or "skipped" (Skip calibration, confirmed), which the arm file records.
async function addUrdf(calibration) {
  const held = spare(chain()).filter((joint) => U.held[joint]);
  await guard("pickErr", async () => {
    $("urdfNote").textContent = "Building the arm model...";
    let reply;
    try {
      await busy([$("urdfAdd"), $("urdfSkip")], async () => {
        reply = await api("POST", "/api/arm_file", {
          filename: S.urdf.name, urdf: S.urdf.text, base: U.tree.links[U.base], flange: U.tree.links[U.end], ...markerTool(),
          tcp_calibration: calibration, fixed: held,
        });
      });
    } finally {
      $("urdfNote").textContent = "";
    }
    S.urdfIk = reply.ik;
    const skipped = calibration === "skipped" ? " Tool point skipped, so every start shows a warning." : "";
    const still = held.length ? ` Held still: ${held.join(", ")}.` : "";
    $("urdfNote").textContent = `Added ${S.urdf.name}: ${reply.joints.length} joints (${reply.joints.join(", ")}).${still}${skipped}`;
    if (S.state.tool !== "tcp") renderRobots();
    $("tcpIk").textContent = reply.ik;
    $("tcpSaved").classList.toggle("hidden", S.state.tool !== "tcp");
  });
}

// The SO-101's calibration files: the ones lerobot-calibrate left on this box, none ticked until the operator ticks
// their arms' (each says when it last changed, to tell an old one from today's), and any the operator adds, ticked.
// Each arm picks its own file: a Left arm and a Right arm button on every file; one file may serve both. The server
// is sent the files picked and matches each arm's motors to one; the arms step says when a match is not the pick.
function renderSo101Files() {
  const pickFor = (side, file) => el("button", {
    class: "sidepick" + (S.so101Side[side] === file.path ? " on" : ""), "aria-pressed": String(S.so101Side[side] === file.path),
    onclick: () => { S.so101Side[side] = S.so101Side[side] === file.path ? null : file.path; renderRobots(); },
  }, side === "left" ? "Left arm" : "Right arm");
  // With the check: the files the plugged-in arms hold first, then the rest, then what is no calibration at all.
  const check = (file) => (S.so101Match ? S.so101Match.files.find((f) => f.path === file.path) : null);
  const rank = (file) => { const m = check(file); return !m ? 1 : !m.valid ? 2 : m.matches.length ? 0 : 1; };
  const files = [...S.so101Files].sort((a, b) => rank(a) - rank(b));
  const adapterName = (id) => { const a = S.so101Match.adapters.find((x) => x.id === id); return a ? a.label : id; };
  $("so101Files").replaceChildren(...files.map((file) => {
    const m = check(file);
    const status = !m ? null
      : !m.valid ? el("span", { class: "pill" }, "Not a calibration file")
      : m.matches.length ? el("span", { class: "pill good" }, `Matches the arm on ${m.matches.map(adapterName).join(" and ")}`)
      : el("span", { class: "muted" }, "No plugged-in arm holds this");
    return el("div", { class: "device sidefile" + (m && !m.valid ? " off" : "") },
      el("div", {}, el("div", { class: "name" }, file.path.split("/").pop()), el("div", { class: "sub2" }, fileWhen(file)), status),
      m && !m.valid ? null : el("div", { class: "sidepicks" }, pickFor("left", file), pickFor("right", file)));
  }));
  $("so101Files").classList.toggle("hidden", !S.so101Files.length);
  $("so101None").classList.toggle("hidden", !!S.so101Files.length);
  $("so101Dir").textContent = S.state.so101_calibration_dir;
  $("so101Check").classList.toggle("hidden", !S.state.so101_match || !S.so101Files.length);
  $("so101CheckText").textContent = so101CheckLine();
}

// Whether both picked files are ones the check found on the plugged-in arms.
function bothMatched() {
  const matched = new Set(S.so101Match.files.filter((f) => f.matches.length).map((f) => f.path));
  return SIDES.every((side) => matched.has(S.so101Side[side]));
}

// What the check found, in one line: how many files match the plugged-in arms, and any arm it could not read.
function so101CheckLine() {
  if (S.so101Checking) return "Checking which file each plugged-in arm holds...";
  const m = S.so101Match;
  if (!m) return "";
  const matching = m.files.filter((f) => f.matches.length).length;
  const unread = m.adapters.filter((a) => a.error).map((a) => `Couldn't read ${a.label}: ${a.error}.`);
  const found = !m.adapters.length ? "No SO-101 arm is plugged in, so no file could be checked. Plug in both arms and power them on."
    : matching === 2 ? `These two files match the arms that are plugged in${bothMatched() ? ", and are ticked" : ""}. Find arms tells which is left and which is right.`
    : matching ? `${matching} file${matching === 1 ? "" : "s"} match${matching === 1 ? "es" : ""} the arms that are plugged in.`
    : "No file matches the arms that are plugged in. Calibrate each arm with lerobot-calibrate, then check again.";
  return [found, ...unread].join(" ");
}

// The check: which listed file each plugged-in SO-101 arm holds, read from its servos (GET /api/so101/match). Two
// matching files are ticked for the operator, one per arm; Find arms then says which arm each is.
async function checkSo101() {
  if (!S.state.so101_match || S.so101Checking) return;
  S.so101Checking = true;
  renderSo101Files();
  try {
    S.so101Match = await api("GET", "/api/so101/match");
    const matching = S.so101Match.files.filter((f) => f.matches.length).map((f) => f.path);
    // A pick of a file that turned out to be no calibration is dropped: it can't be seen once its buttons go.
    for (const side of SIDES) if (S.so101Match.files.some((f) => !f.valid && f.path === S.so101Side[side])) S.so101Side[side] = null;
    if (matching.length === 2 && !S.so101Side.left && !S.so101Side.right) [S.so101Side.left, S.so101Side.right] = matching;
  } catch (exc) {
    S.so101Match = null;
    $("pickErr").textContent = exc.message;
  } finally {
    S.so101Checking = false;
    renderRobots();
  }
}

// A file the operator points to for one arm: listed with the others, and picked for that arm.
function addSo101File(side) {
  const input = $(side === "left" ? "so101Left" : "so101Right");
  const path = input.value.trim();
  if (!path) return;
  if (!S.so101Files.some((f) => f.path === path)) S.so101Files.push({ path });
  S.so101Side[side] = path;
  input.value = "";
  renderRobots();
}

// Look again: the files lerobot-calibrate wrote since the page opened, kept with any the operator added.
async function lookAgainSo101() {
  const found = (await api("GET", "/api/state")).so101_calibrations;
  for (const file of found) if (!S.so101Files.some((f) => f.path === file.path)) S.so101Files.push({ ...file });
  renderRobots();
}

function fileWhen(file) {
  if (file.modified === undefined) return file.path;
  if (file.modified === null) return `${file.path} · not on this station`;
  const when = new Date(file.modified * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  return `${file.path} · changed ${when}`;
}

async function pickRobot() {
  renderRobots();  // what is still needed now, not at the last render
  if (refused("pickRobot")) return;
  await guard("pickErr", async () => {
    $("pickRobot").disabled = true;
    $("pickNote").textContent = "Loading the arm model. This takes a few seconds...";
    let reply;
    try {
      const fresh = $("embodiment").value === NEW;  // registered by this pick, so the list is asked again after
      const which = { embodiment: fresh ? null : $("embodiment").value };
      const body = S.pick === "custom" ? { custom: { ik_arm: ownArm(), driver: S.driver, ...which } }
        : S.pick === "so101" ? { body: S.pick, calibration: [...new Set([S.so101Side.left, S.so101Side.right])], ...which }
        : { body: S.pick };
      reply = await api("POST", "/api/robot", body);
      if (fresh) S.embodiments = null;
    } finally {
      $("pickNote").textContent = "";
      $("pickRobot").disabled = false;
    }
    applyPick(reply);
    // A new embodiment that turned out to be the account's own from an earlier run is said, not left to a guess.
    S.embodimentNote = reply.embodiment_note || "";
    $("embodimentNote").textContent = S.embodimentNote;
  });
}

function applyPick(reply) {
  S.robot = reply.robot;
  S.reach = reply.reach;
  S.reachSpan = {};
  numberSteps();
  S.joints = reply.joints;  // the arm's own joints by index: an SO-101 has five (B16)
  S.labels = Object.fromEntries(reply.cameras.map((cam) => [cam.role, cam]));  // the server's labels per role
  S.prefill = reply.prefill;
  $("setLeft").value = S.prefill.settings.left;
  $("setRight").value = S.prefill.settings.right;
  $("settingsBox").classList.toggle("hidden", S.robot.body === "custom" || S.robot.body === "so101");
  // A body with no brakes (the server's note): warned where it drops, as the arms are let go and at home.
  for (const id of ["limpNote", "homeLimpNote"]) $(id).hidden = !S.robot.note;
  if (!S.dataset) $("datasetNote").textContent = "";
  S.cams = {};
  for (const row of S.prefill.cameras) S.cams[row.name] = { ...row };
  $("shootAll").textContent = "Take pictures";
  $("shootAll").classList.add("primary");
  for (const id of ["camMissing", "typedCamBox"]) $(id).classList.add("hidden");
  // A new robot: the camera step starts again, its old pictures, roles and live views gone with the old robot's roles.
  S.shots = [];
  S.liveOn = false;
  $("gallery").replaceChildren();
  for (const id of ["viewBar", "camAdvanced"]) $(id).classList.add("hidden");
  $("liveViews").classList.add("hidden");
  $("liveViews").textContent = "Show live views";
  markRoles();
  armsNotFound();
  refreshReferences();
  if (S.dataset) showDatasetInUse();
  V.drawing = null;
  V.home = null;
  V.power = null;
  S.opened = false;
  S.homeSet = false;
  S.scan = null;
  $("findBox").classList.remove("hidden");
  $("findAdapters").classList.remove("hidden");
  $("adapterList").classList.add("hidden");
  $("untickNote").classList.add("hidden");
  $("openArms").disabled = true;
  $("wiggle").classList.add("hidden");
  $("udev").classList.add("hidden");
  $("releaseBox").classList.remove("hidden");
  $("openArms").classList.remove("hidden");
  $("lookAgain").classList.add("hidden");
  go(1);
}

async function resume(picked) {
  S.pick = picked.robot.body;
  renderRobots();
  applyPick(picked);
  await scanArms();
  const st = S.status = await api("GET", "/api/status");
  if (st.wiggle) {  // the arms are open and limp (under power after the checks, the server holds them otherwise)
    S.ticked = new Set(Object.keys(st.wiggle.devices));
    S.opened = true;
    showOpened();
  }
  if (!(st.assigned.left && st.assigned.right)) return;
  summary(1, armsLine(st.assigned));
  markRoles();
  if (!S.robot.roles.every((role) => st.cameras.open.includes(role))) { go(2); return; }
  summary(2, camsLine());
  if (!st.home.left || gripperMissing(st.gripper).length) { go(3); return; }
  summary(3, homeLine(st.home, st.gripper));
  S.homeSet = true;  // a page opened again past this step: the home was set by a click before
  if (!S.state.saved) { go(S.reach ? 4 : 5); return; }
  // Saved before the page was opened again: back on Save station, saved, with Continue to finish.
  if (S.reach) summary(4, "Set.");
  summary(5, "Done before this page was opened again.");
  go(6);
  $("saveWhere").classList.add("hidden");
  $("savedLine").replaceChildren("Saved to ", el("code", {}, S.state.config_path), `. This station is ${S.state.saved.robot} on ${S.state.saved.platform}.`);
  $("savedLine").classList.remove("hidden");
  $("finishActions").classList.remove("hidden");
}

// Step 1: the adapters, found by Find adapters, then the arms, found by moving them.
async function scanArms() {
  await guard("armsErr", async () => {
    S.scan = await api("GET", "/api/scan");
    $("findBox").classList.add("hidden");
    $("findAdapters").classList.add("hidden");
    $("adapterList").classList.remove("hidden");
    $("untickNote").classList.remove("hidden");
    S.ticked = new Set(S.scan.arms.map((d) => d.id));
    S.openErrors = {};
    showScanNote();
    renderDevices(null);
    renderOthers();
    $("openArms").disabled = !S.scan.arms.length;
  });
}

function showScanNote() {
  $("scanNote").textContent = S.scan.arms.length ? "" : "No adapters found. Plug in both arms' adapters and turn on the arms.";
  $("scanNote").classList.toggle("hidden", !!S.scan.arms.length);
}

// Before the arms are released the list follows what is plugged in: a scan every few seconds, the ticks kept for
// the adapters still there and new ones ticked.
// Called from poll(), at most every three seconds.
async function refreshArms() {
  if (S.step !== 1 || S.opened || !S.scan || document.hidden || Date.now() - (S.scannedAt || 0) < 3000) return;
  S.scannedAt = Date.now();
  try {
    const scan = await api("GET", "/api/scan");
    if (S.step !== 1 || S.opened) return;
    const seen = new Set(S.scan.arms.map((d) => d.id));
    if (JSON.stringify(scan.arms) === JSON.stringify(S.scan.arms) && JSON.stringify(scan.others) === JSON.stringify(S.scan.others)) return;
    S.scan = scan;
    S.ticked = new Set(scan.arms.map((d) => d.id).filter((id) => S.ticked.has(id) || !seen.has(id)));
    showScanNote();
    renderDevices(null);
    renderOthers();
    $("openArms").disabled = !S.ticked.size;
  } catch (exc) {
    // a missed refresh is retried on the next one
  }
}

// Switch gripper to MIT, beside Swap arms: every found arm's gripper not in MIT mode, one after the other.
// Setup lets go of the arms to switch, so the page shows them released, and Start over finds them again after the
// power cycle.
async function switchGrippers() {
  const button = $("gripSwitch");
  const go = await ask("Switch the gripper to MIT mode?",
    "Both arms will go limp, so hold them or rest them on their stands first. When the switch is done, turn the arm off and back on, then click Start over.",
    "Switch gripper");
  if (!go) return;
  button.disabled = true;
  try {
    await guard("armsErr", async () => {
      for (const device of S.wrongGrippers) await api("POST", "/api/motors/gripper_mit", { device });
      S.opened = false;
      button.classList.add("hidden");
      $("swapSides").classList.add("hidden");
      $("udev").classList.add("hidden");
      $("lookAgain").classList.remove("hidden");
      armsNotFound();
      $("wiggleAsk").replaceChildren(el("p", { class: "lead warn" },
        "The gripper motor is switched to MIT mode. Turn the arm off, wait a few seconds, and turn it back on. Then click Start over."));
    });
  } finally {
    button.disabled = false;
  }
}

// What the arm cards say about an adapter (setup_page.js): its serial number or its USB port, and its gripper's mode.
globalThis.setupDevice = (id) => (S.scan ? S.scan.arms.find((d) => d.id === id) : null) || null;

// "Your arm is not listed?": the address its driver opens, ticked like an arm the scan found.
async function typeArm() {
  await guard("armsErr", async () => {
    const before = new Set(S.scan.arms.map((device) => device.id));
    S.scan = await api("POST", "/api/arms/typed", { address: $("typedArm").value });
    for (const device of S.scan.arms) if (!before.has(device.id)) S.ticked.add(device.id);
    $("typedArm").value = "";
    showScanNote();
    renderDevices(null);
    renderOthers();
    $("openArms").disabled = S.opened || !S.ticked.size;
  });
}

function linkProblem(device) {
  if (device.kind === "slcan" && device.via && device.state !== "up") {
    return `The slcand connection ${device.via} is ${device.state || "missing"}. Run sudo ip link set ${device.via} up, or stop slcand so setup can open the adapter directly.`;
  }
  if (device.kind === "can" && device.state && device.state !== "up") {
    return `This CAN connection is ${device.state}. Run sudo ip link set ${device.open} up type can bitrate 1000000; the list refreshes on its own.`;
  }
  return null;
}

// Once both arms are found, every other port listed is not an arm: setup closed it, and the row says so rather than
// "Couldn't open this adapter." (a third port beside the two arms otherwise reads as one more arm).
function renderDevices(wiggle, assigned) {
  const rows = wiggle ? wiggle.devices : {};
  const opened = S.opened;
  const arms = assigned && assigned.left && assigned.right ? Object.values(assigned) : null;
  $("devices").replaceChildren(...S.scan.arms.map((device) => {
    const live = rows[device.id];
    const notArm = arms && !arms.includes(device.id);
    const box = el("input", { type: "checkbox", checked: S.ticked.has(device.id), disabled: opened, "aria-label": `Use ${device.label}`, onchange: (e) => {
      if (e.target.checked) S.ticked.add(device.id); else S.ticked.delete(device.id);
      $("openArms").disabled = !S.ticked.size;
      renderDevices(wiggle, assigned);
    } });
    const problem = linkProblem(device);
    const info = el("div", {},
      el("div", { class: "name" }, device.label),
      el("div", { class: "path" }, device.open),
      notArm ? el("div", { class: "sub2" }, "Not an arm. Neither arm is on this port, so setup closed it and left it out of your robot.") : null,
      !notArm && opened && !live && !(S.openErrors || {})[device.id] && S.ticked.has(device.id) ? el("div", { class: "sub2" }, "Couldn't open this adapter.") : null,
      problem ? el("div", { class: "err" }, problem) : null,
      live && live.calibration ? el("div", { class: "sub2" }, `Motors match ${live.calibration}`) : null,
      // Find arms tells which picked file is which arm; only a file picked for neither arm is worth a word.
      live && live.side && live.calibration && S.so101Side[live.side] && !Object.values(S.so101Side).some((path) => path && path.split("/").pop() === live.calibration)
        ? el("div", { class: "err" }, `You chose ${S.so101Side[live.side].split("/").pop()} for the ${live.side} arm, but this arm's motors match ${live.calibration}. Check which file belongs to which arm, then choose again in Select robot.`) : null,
      live && live.error ? el("div", { class: "err" }, live.error) : null,
      S.openErrors && S.openErrors[device.id] ? el("div", { class: "err" }, S.openErrors[device.id]) : null,
    );
    const off = !S.ticked.has(device.id) || notArm;
    // Once Find arms has named an arm, its adapter carries the side.
    const side = SIDES.find((name) => assigned && assigned[name] === device.id) || (live && live.side);
    const pill = side ? el("span", { class: "pill good" }, side === "left" ? "Left" : "Right") : null;
    return el("label", { class: "arow" + (live && live.moved ? " moved" : "") + (off ? " off" : "") }, box, info, pill);
  }));
  $("swapSides").classList.toggle("hidden", !(assigned && assigned.left && assigned.right));
}

// Swap arms: both arms named as the other side and opened again with their own settings, in one request.
async function swapSides() {
  const go = await ask("Swap the left and right arms?",
    "The arm named left becomes the right arm, and the right becomes the left. Setup reconnects both arms, and they stay limp, so keep holding them or keep them on their stands.",
    "Swap arms");
  if (!go) return;
  $("crossed").classList.add("hidden");
  await guard("armsErr", async () => {
    const reply = await api("POST", "/api/arms/swap", {});  // one step on the station: no arm is ever named twice
    if (reply.crossed) {
      $("crossed").textContent = reply.crossed;
      $("crossed").classList.remove("hidden");
    }
  });
}

function renderOthers() {
  const others = S.scan.others || [];
  $("otherDevices").classList.toggle("hidden", !others.length);
  $("otherCount").textContent = String(others.length);
  $("otherList").replaceChildren(...others.map((device) => el("li", {}, el("b", {}, device.label), ` ${device.detail}`)));
}

async function openArms() {
  const button = $("openArms");
  const text = button.textContent;
  button.disabled = true;
  button.textContent = "Finding the arms...";
  $("openNote").textContent = "Opening each adapter takes a few seconds. Hold the arms.";
  try {
    await guard("armsErr", async () => {
      const reply = await api("POST", "/api/arms/open", {
        devices: [...S.ticked], settings: { left: $("setLeft").value, right: $("setRight").value },
      });
      S.openErrors = Object.fromEntries(Object.entries(reply.opened).filter(([, why]) => why));
      S.opened = true;
      showOpened();
    });
  } finally {
    button.textContent = text;
    button.disabled = S.opened || !S.ticked.size;
    $("openNote").textContent = "";
  }
}

// Connect arms' Continue waits again: after Start over, Switch gripper to MIT or a new robot, no arm is named any more.
function armsNotFound() {
  $("armsDone").classList.add("hidden");
  gate("armsDone", [["both arms found", "armsCard"]]);
  $("armsDoneNeed").textContent = "";  // Continue itself is hidden until both arms are found
}

function showOpened() {
  S.askKey = null;
  $("releaseBox").classList.add("hidden");
  $("openArms").classList.add("hidden");
  $("wiggle").classList.remove("hidden");
  $("crossed").classList.add("hidden");
  $("udev").classList.add("hidden");
  $("armsDone").classList.add("hidden");
  $("gripSwitch").classList.add("hidden");
  $("lookAgain").classList.remove("hidden");
  $("untickNote").classList.add("hidden");  // the boxes are locked while the arms are open
  renderDevices(null);
}

async function lookAgain() {
  const go = await ask("Start over finding the arms?",
    "Setup lets go of both arms and forgets which is left and which is right. You'll find the adapters and the arms again. Keep holding the arms or keep them on their stands.",
    "Start over");
  if (!go) return;
  await guard("armsErr", async () => {
    $("lookAgain").disabled = true;
    try {
      await api("POST", "/api/arms/close");
    } finally {
      $("lookAgain").disabled = false;
    }
    S.opened = false;
    S.askKey = null;
    $("releaseBox").classList.remove("hidden");
    $("openArms").classList.remove("hidden");
    $("wiggle").classList.add("hidden");
    $("udev").classList.add("hidden");
    $("lookAgain").classList.add("hidden");
    armsNotFound();
    await scanArms();
  });
}

function renderWiggle(status) {
  const wiggle = status.wiggle;
  if (!wiggle) return;
  const assigned = status.assigned;
  renderDevices(wiggle, assigned);
  const done = assigned.left && assigned.right;
  const key = JSON.stringify([wiggle.moved, wiggle.several, assigned]);
  if (key === S.askKey) return;
  S.askKey = key;
  const ask = $("wiggleAsk");
  if (done) {
    // A gripper motor in another mode is the one thing left to fix: the bar says so, the card under that arm fixes it.
    const wrong = SIDES.map((side) => [side, (globalThis.setupDevice(assigned[side]) || {}).gripper_mode])
      .filter(([, mode]) => mode && mode !== "mit");
    ask.replaceChildren(wrong.length
      ? el("p", { class: "lead warn" }, wrong.map(([side, mode]) => `The ${side} arm's gripper motor is in ${mode} mode, so it won't move. The station drives motors in mit mode only.`).join(" "))
      : el("p", { class: "lead found" }, "Both arms found"));
    S.wrongGrippers = wrong.map(([side]) => assigned[side]);
    $("gripSwitch").classList.toggle("hidden", !(wrong.length && S.state.gripper_switch));
    $("armsDone").disabled = false;
    gate("armsDone", []);
    showRule();
    $("armsDone").classList.remove("hidden");
    return;
  }
  $("udev").classList.add("hidden");  // the permanent names follow the arms, so only once both are found
  // The prompt names the arm to move, so the arm that moves is that side: no question, nothing to answer wrong.
  const waiting = SIDES.find((side) => !assigned[side]);
  if (wiggle.several) {
    ask.replaceChildren(el("p", { class: "lead" }, "More than one arm moved"),
      el("p", { class: "help" }, `Rest each arm where it can't move, like on its stand or the table. Then click Start over and move only the ${waiting} arm.`),
      el("button", { onclick: () => guard("armsErr", () => api("POST", "/api/arms/rebase")) }, "Start over"));
  } else if (wiggle.moved && !Object.values(assigned).includes(wiggle.moved)) {  // the arm already named, moved again, is not the other
    assign(wiggle.moved, waiting);  // the prompt stays until the next one: a line in between would only flash
  } else {
    const lead = `Move any joint on the ${waiting} arm.`;
    ask.replaceChildren(el("p", { class: "lead" }, el("span", { class: "pulse" }), lead));
  }
}

async function assign(device, side) {
  await guard("armsErr", async () => {
    const reply = await api("POST", "/api/arms/assign", { device, side });
    if (reply.crossed) {
      $("crossed").textContent = reply.crossed;
      $("crossed").classList.remove("hidden");
    }
  });
}

async function showRule() {
  const anyCan = S.scan.arms.some((d) => d.kind === "can" && d.usb && Object.values(S.status.assigned).includes(d.id));
  if (!anyCan) return;
  try {
    const rule = await api("GET", "/api/udev_rule");
    showCommand("udevText", rule.install);
    const named = Object.entries(rule.in_effect).filter(([, ok]) => ok).map(([s]) => s);
    const all = named.length === Object.keys(rule.in_effect).length;
    const port = rule.by_port.length ? ` The ${rule.by_port.join(" and ")} adapter has no serial number, so it's named by its USB port. Keep it in that port.` : "";
    $("udevNote").textContent = ((all ? "" : "The arm names take effect at the next reboot. Reboot this computer before you run dreamcli robot up.") + port).trim();
    const found = S.status && S.status.assigned.left && S.status.assigned.right;
    $("udev").classList.toggle("hidden", !found || !S.opened);
  } catch (exc) { $("armsErr").textContent = exc.message; }
}

// Step 2: the cameras.
async function shootAll() {
  await guard("camErr", async () => {
    const scan = await api("GET", "/api/scan");
    S.shots = scan.cameras;
    // Once taken: the button retakes them, and "A camera is missing" follows the pictures.
    $("shootAll").textContent = "Retake all pictures";
    $("shootAll").classList.remove("primary");
    $("camMissing").classList.remove("hidden");
    $("liveViews").classList.remove("hidden");
    // The reference toggle, its opacity and the settings have something to act on once there are pictures.
    for (const id of ["viewBar", "camAdvanced"]) $(id).classList.remove("hidden");
    $("shootNote").textContent = S.shots.length ? "" : "No cameras found. Check that each camera is plugged in, then try again.";
    $("gallery").replaceChildren(...S.shots.map(cameraRow));
    turnTiles();
    for (const cam of S.shots) {
      const q = new URLSearchParams({ source: cam.source, value: cam.value, width: 640, height: 480, fps: 30 });
      try {
        const reply = await fetch(`/api/preview?${q}`);
        if (!reply.ok) throw new Error((await reply.json()).error);
        cam.img.src = URL.createObjectURL(await reply.blob());
        cam.frame.replaceChildren(cam.img);
      } catch (exc) { cam.frame.replaceChildren(el("span", { class: "msg" }, exc.message)); }
    }
    markRoles();
  });
}

// One camera's row: its picture, what it is (a serial number when it has one) and its address and capture size, the
// role it is chosen for, and a quarter turn.
function cameraRow(cam) {
  cam.rotate = 0;
  // A camera's own settings start as those of the role the station file gives it, else the first role's.
  const { width, height, fps } = S.cams[S.robot.roles.find((r) => holds(S.cams[r], cam)) || S.robot.roles[0]];
  cam.capture = { width, height, fps };
  cam.img = el("img", { alt: cam.label });
  cam.frame = el("div", { class: "thumb" }, el("span", { class: "msg" }, "Taking a picture..."));
  const name = cam.source === "serial" ? `SN ${cam.value}` : cam.label;
  const where = cam.source === "serial" ? cam.label.replace(cam.value, "").trim() : cam.value;
  cam.sub = el("div", { class: "camsub" });
  cam.where = where;
  // The role as a toggle, Head / Left / Right: a click takes that role, a click on the one it has frees the camera.
  cam.roleButtons = S.robot.roles.map((role) => el("button", {
    class: "rolepick", "data-role": role, title: S.labels[role].label,
    onclick: () => chooseRole(cam, holds(S.cams[role], cam) ? "" : role),
  }, S.labels[role].short));
  cam.settings = el("div", { class: "shotset" });
  cam.name = name;
  cam.pics = el("div", { class: "campics" });
  cam.turn = el("button", { class: "turnicon", title: "Rotate 90° clockwise", "aria-label": `Rotate ${name} 90 degrees clockwise`,
    onclick: () => turnCamera(cam) }, "↻");
  cam.check = el("div", { class: "camcheck" });
  cam.tile = el("div", { class: "camrow" }, cam.pics,
    el("div", { class: "caminfo" },
      cam.sub,
      el("div", { class: "camrole" }, el("span", {}, "Role"), el("div", { class: "segmented", role: "group", "aria-label": `Role of ${name}` }, cam.roleButtons)),
      cam.check, cam.settings));
  return cam.tile;
}

// One camera's picture: its own (the snapshot, or its live feed once the live views are on) and, once it has a role,
// that role's reference over it at the slider's opacity; then how to line it up.
function showCamPics(cam, role) {
  const live = S.liveOn && role;
  let own = live ? liveImg(role, "live") : cam.img.cloneNode();
  if (!live) own.style.transform = TURN_CSS[role ? Number(S.cams[role].rotate || 0) : cam.rotate] || "";
  if (!cam.img.src) own = el("span", { class: "msg" }, "Taking a picture...");
  const liveTag = live ? el("span", { class: "livetag" }, el("span", { class: "dot", "aria-hidden": "true" }), "Live") : null;
  const ref = role ? refFrame(role, Date.now()).querySelector("img") : null;
  // The reference's opacity, under this camera's picture: down to see your camera through it, up to see the reference.
  cam.opacity ??= 50;
  if (ref) ref.style.opacity = String(cam.opacity / 100);
  const value = el("span", { class: "opacityvalue" }, `${cam.opacity}%`);
  const slider = !ref ? null : el("label", { class: "opacity" }, "Reference",
    el("input", { type: "range", min: "0", max: "100", value: String(cam.opacity), "aria-label": `Reference opacity, ${cam.name}`,
      oninput: (e) => { cam.opacity = Number(e.target.value); ref.style.opacity = String(cam.opacity / 100); value.textContent = `${cam.opacity}%`; } }),
    value);
  cam.pics.replaceChildren(
    el("div", { class: "panehead" }, el("span", {}, cam.name), liveTag),
    el("div", { class: "frame overlay" }, own, ...(ref ? [ref] : []), cam.turn),
    ...(slider ? [slider] : []),
    ...(role ? [el("span", { class: "err", "data-camstate": role })] : []));
  const wrist = role && ((S.labels[role].kind || "").startsWith("wrist") || /wrist/i.test(S.labels[role].label));
  cam.check.replaceChildren(...(!role ? [] : [
    el("p", { class: "note" }, wrist ? "Adjust the mount until the grippers land where the reference shows them." : "Adjust the camera until its view matches the reference.")]));
}

// The line under a camera's name: its address, then the size and rate it opens at (its role's, else its own).
function showCamSub(cam) {
  const role = S.robot.roles.find((r) => holds(S.cams[r], cam));
  const row = role ? S.cams[role] : cam.capture;
  cam.sub.textContent = [cam.where, `${row.width}×${row.height}`, `${row.fps} fps`].filter(Boolean).join(" · ");
}

// The role list: a role takes this camera (and leaves the camera it had), or "Not used" frees the camera.
function chooseRole(cam, role) {
  if (role) { setCamera(role, cam); return; }
  for (const r of S.robot.roles) if (holds(S.cams[r], cam)) S.cams[r].value = "";
  markRoles();
}

// "Your camera is not listed?": an address OpenCV opens or a camera class, then every picture again with it.
async function typeCamera() {
  await guard("camErr", async () => {
    await api("POST", "/api/cameras/typed", { address: $("typedCam").value });
    $("typedCam").value = "";
    await shootAll();
  });
}

const holds = (row, cam) => row.value === cam.value && row.source === cam.source;

function setCamera(role, cam) {
  for (const other of Object.values(S.cams)) if (holds(other, cam)) other.value = "";
  if (S.liveOn) setTimeout(liveViews);  // the live views follow the new roles
  const { width, height, fps } = cam.capture;
  S.cams[role] = { ...S.cams[role], width, height, fps, source: cam.source, value: cam.value, rotate: cam.rotate };
  markRoles();
}

// Each picture of the role step turned as its camera's `rotate` says. A camera that holds a role takes that role's
// value, so the picture, the text field, the live views and the saved file agree; the turn stays with the camera when
// it changes role. The frame is 16:9, so a quarter turn also shrinks the picture to 9/16 to keep it inside.
const TURN_CSS = { 0: "", 90: "rotate(90deg) scale(0.5625)", 180: "rotate(180deg)", 270: "rotate(270deg) scale(0.5625)" };
function turnTiles() {
  for (const cam of S.shots) {
    const role = S.robot.roles.find((r) => holds(S.cams[r], cam));
    if (role) cam.rotate = Number(S.cams[role].rotate || 0);
    cam.img.style.transform = TURN_CSS[cam.rotate] || "";
  }
}

// The small rotate button on a camera's picture: a quarter turn clockwise, kept as the camera's own `rotate` and the
// role's it holds; live views already open open again turned.
function turnCamera(cam) {
  cam.rotate = (cam.rotate + 90) % 360;
  cam.capture.rotate = cam.rotate;
  for (const role of S.robot.roles) if (holds(S.cams[role], cam)) S.cams[role].rotate = cam.rotate;
  markRoles();
  if (S.liveOn) return liveViews();
}

function markRoles() {
  for (const cam of S.shots) {
    const role = S.robot.roles.find((r) => holds(S.cams[r], cam));
    for (const button of cam.roleButtons) {
      const on = button.dataset.role === role;
      button.classList.toggle("on", on);
      button.setAttribute("aria-pressed", String(on));
    }
    showCamSub(cam);
    showCamPics(cam, role);
  }
  showShotSettings();
  turnTiles();
  const missing = S.robot.roles.filter((role) => !S.cams[role].value).map((role) => [`the ${S.labels[role].short} camera`, "gallery"]);
  gate("liveViews", missing);
  gate("camsDone", missing);
}

// The comparison's reference pictures (live views) loaded again, and the heading saying what they come from.
function refreshReferences() {
  showReferenceSource();
  const t = Date.now();
  document.querySelectorAll("img[data-ref]").forEach((img) => { img.src = `/api/reference?role=${img.dataset.ref}&episode=0&t=${t}`; });
}

// What the references are, by what each role shows: the dataset only for the roles given one of its cameras,
// and the link to switch: to the dataset when none is in use, back to the YAM pictures when one is.
function showReferenceSource() {
  const inUse = !!(S.dataset && S.robot.roles.some((role) => S.dataset.keys[role]));
  const mine = inUse || S.refChoice === "mine";  // My dataset shows as picked as soon as it is clicked, before a folder
  $("refYam").textContent = S.robot.body === "yam" ? "YAM training data" : "YAM example";
  $("refYam").setAttribute("aria-checked", String(!mine));
  $("refMine").setAttribute("aria-checked", String(mine));
  // The fine-tune's own data could not be read: the step says so, with the way on, rather than a line of error.
  const failure = S.status && S.status.cameras.reference_failure;
  $("refWarn").classList.toggle("hidden", !failure || inUse);
  if (failure) {
    $("refWarnModel").textContent = failure.model;
    $("refWarnDetail").textContent = failure.error;
  }
}

// YAM example: drop the dataset in use for the YAM pictures again. My dataset: open the dataset field.
async function useYamReferences() {
  S.refChoice = "yam";
  showReferenceSource();
  $("datasetBox").classList.add("hidden");
  if (!S.dataset) return;
  await guard("refErr", async () => {
    await api("POST", "/api/dataset", { path: "" });
    S.dataset = null;
    $("datasetPath").value = "";
    $("datasetNote").textContent = "";
    $("datasetBox").classList.add("hidden");
    renderDatasetKeys();
    refreshReferences();
    if (S.liveOn) markRoles();
  });
}

async function retryReferences() {
  await guard("refErr", async () => {
    $("refRetry").disabled = true;
    try { await api("POST", "/api/references/retry", {}); } finally { $("refRetry").disabled = false; }
    S.status = await api("GET", "/api/status");
    refreshReferences();
    markRoles();
  });
}

// Under each camera's name, its capture settings, shown for every camera at once by the card's Advanced settings: those
// of the role it is chosen for, or its own until it has one.
function showShotSettings() {
  for (const cam of S.shots) {
    if (!cam.settings) continue;
    const role = S.robot.roles.find((r) => holds(S.cams[r], cam));
    // The camera keeps its own settings, so they can be set before a role; the role it holds takes them too.
    const row = role ? S.cams[role] : { ...cam.capture, rotate: cam.rotate };
    const field = (key) => el("div", {}, el("label", { class: "field" }, key), el("input", {
      "aria-label": `${cam.label} ${key}`, value: row[key], oninput: (e) => {
        row[key] = cam.capture[key] = e.target.value;
        showCamSub(cam);
        if (key !== "rotate") return;
        if (!role) cam.rotate = Number(e.target.value) || 0;
        turnTiles();
      },
    }));
    cam.settings.replaceChildren(el("div", { class: "row c2" }, field("width"), field("height"), field("fps"), field("rotate")));
  }
}

function cameraRows() { return S.robot.roles.map((role) => ({ ...S.cams[role], name: role })); }

// A live feed: the role's latest frame, asked for again as each one loads (one request at a time per feed, so six
// feeds never hold the browser's six connections), paused while hidden, until the image leaves the page or feeds again.
const LIVE_FRAME_MS = 100;
let feeds = 0;
function feed(img, role, view) {
  const key = String(++feeds);
  img.dataset.feed = key;
  const ask = () => { img.src = `/api/cameras/frame?role=${role}&view=${view}&t=${Date.now()}`; };
  const next = (wait) => setTimeout(() => {
    if (!img.isConnected || img.dataset.feed !== key) return;
    if (img.offsetParent === null || S.gone) next(S.gone ? 3000 : 500); else ask();
  }, wait);
  img.onload = () => next(LIVE_FRAME_MS);
  img.onerror = () => next(1000);
  ask();
  return img;
}
function liveImg(role, view) { return feed(el("img", { alt: `${S.labels[role].label}, ${view}` }), role, view); }

async function liveViews() {
  await guard("camErr", async () => {
    $("liveViews").disabled = true;
    try { S.camOpened = (await api("POST", "/api/cameras/open", { cameras: cameraRows() })).opened; } finally { $("liveViews").disabled = false; }
    S.liveOn = true;
    $("liveViews").textContent = "Reopen live views";
    showReferenceSource();
    markRoles();
    setTimeout(checkSwap, 1500);
  });
}

function refKind(role) {
  const ref = S.status && S.status.cameras.references[role];
  if (ref && (ref.source === "dataset" || ref.source === "none")) return ref.source;
  return S.robot.body === "yam" ? "yam" : "example";
}

// A role's reference picture; a camera of no kind the server knows, with no dataset key, has none.
function refFrame(role, t) {
  if (refKind(role) === "none") return el("div", { class: "frame" }, el("span", { class: "msg" }, "No reference picture"));
  return el("div", { class: "frame" }, el("img", { "data-ref": role, alt: `Reference picture, ${S.labels[role].label}`, src: `/api/reference?role=${role}&episode=0&t=${t}` }),
    el("span", { class: "tag" }, refKind(role) === "example" ? "Example (YAM)" : "Training data"));
}

async function checkSwap() {
  const match = await api("GET", "/api/cameras/match").catch(() => ({ swapped: [] }));
  const box = $("swap");
  if (!match.swapped.length) { box.classList.add("hidden"); return; }
  const [a, b] = match.swapped[0].roles;
  box.replaceChildren(el("span", {}, `The ${S.labels[a].label} and ${S.labels[b].label} look swapped. Each one looks more like the other's ${refKind(a) === "example" ? "example" : "training"} picture. `),
    el("button", { onclick: () => { const keep = S.cams[a]; S.cams[a] = { ...S.cams[b], name: a }; S.cams[b] = { ...keep, name: b }; markRoles(); liveViews(); } }, "Swap cameras"));
  box.classList.remove("hidden");
}

async function useDataset() {
  await guard("datasetErr", async () => {
    const reply = await api("POST", "/api/dataset", { path: $("datasetPath").value });
    S.dataset = reply.path ? reply : null;
    showDatasetInUse();
    refreshReferences();
  });
}

// The dataset section as the dataset in use says: its folder, its summary and each role's camera.
function showDatasetInUse() {
  if (S.dataset && !S.dataset.fetched) $("datasetPath").value = S.dataset.path;
  $("datasetNote").textContent = S.dataset ? `${S.dataset.episodes} episodes. Cameras: ${S.dataset.video_keys.join(", ")}` : "";
  renderDatasetKeys();
}

function renderDatasetKeys() {
  const box = $("datasetKeys");
  box.classList.toggle("hidden", !S.dataset);
  if (!S.dataset) return;
  box.replaceChildren(...S.robot.roles.map((role) => el("div", {}, el("label", { class: "field" }, S.labels[role].label),
    el("select", { "aria-label": `Dataset camera for ${S.labels[role].label}`, onchange: async (e) => {
      const key = e.target.value || null;
      try {
        await api("POST", "/api/dataset/keys", { keys: { [role]: key } });
      } catch (exc) {
        $("datasetErr").textContent = exc.message;
        e.target.value = S.dataset.keys[role] || "";  // the station kept its key: so does the list
        return;
      }
      $("datasetErr").textContent = "";
      S.dataset.keys[role] = key;
      showReferenceSource();
      document.querySelectorAll(`img[data-ref="${role}"]`).forEach((img) => { img.src = `/api/reference?role=${role}&episode=0&t=${Date.now()}`; });
    } }, el("option", { value: "" }, S.robot.body === "yam" ? "(our YAM data)" : "(our YAM example)"), S.dataset.video_keys.map((key) => el("option", { value: key, selected: S.dataset.keys[role] === key }, key))))));
}

function showHome() {
  const now = S.status.now;
  if (!SIDES.every((side) => now[side] && now[side].frame)) return;
  if (V.home) {
    for (const side of SIDES) V.home[side].live(now[side].frame, now[side].grip);
    return;
  }
  if (S.homeBusy) return;
  S.homeBusy = true;
  armViews({ left: "homeViewLeft", right: "homeViewRight" }, { left: now.left.frame, right: now.right.frame })
    .then((views) => { V.home = views; })
    .catch((exc) => { $("homeErr").textContent = exc.message; })
    .finally(() => { S.homeBusy = false; });
}

// "Set" in the top right of each arm's drawing once Set home pose has saved it; Continue waits for that click, so a
// home left from an earlier pass never stands in for one the person chose.
function showHomeSet() {
  const set = (side) => S.homeSet && S.status.home && S.status.home[side];
  for (const side of SIDES) {
    const view = $(side === "left" ? "homeViewLeft" : "homeViewRight");
    const chip = view.querySelector(".setchip") || view.appendChild(el("span", { class: "setchip" }, "Set"));
    chip.hidden = !set(side);
  }
  gate("homeNext", SIDES.every(set) ? [] : [["the home pose", "homeOk"]]);
}

function drawing() {
  if (!V.drawing) V.drawing = api("GET", "/api/view");
  return V.drawing;
}

// Both arms drawn on dream's model, each framed on its pose in `frames`.
async function armViews(ids, frames) {
  const views = {};
  for (const side of SIDES) {
    $(ids[side]).replaceChildren();
    views[side] = new ArmView($(ids[side]));
    await views[side].load(await drawing(), frames[side]);
    views[side].aim(frames[side], 0);
    views[side].label(`${side} arm`);
    cadNavigation(views[side], 0);
  }
  return views;
}

function renderGripper(gripper) {
  $("gripBox").classList.toggle("hidden", !gripper);
  if (!gripper) return;
  // A robot of your own's driver that reports no gripper motor angle: the page says why the step is skipped.
  for (const id of ["gripHelp", "gripActions"]) $(id).classList.toggle("hidden", Boolean(gripper.skipped));
  $("gripStop").classList.toggle("hidden", S.robot.body === "custom");
  if (gripper.skipped) { $("gripNote").textContent = gripper.skipped; return; }
  $("gripNote").textContent = SIDES.map((side) => {
    const got = ["open", "closed"].filter((end) => end in gripper.recorded[side]).map((end) => `${end} at ${gripper.recorded[side][end].toFixed(0)}°`);
    const ends = gripper.ends[side];
    const kept = ends.error ? "" : `, range ${ends.gripper_closed_deg.toFixed(0)}° to ${ends.gripper_open_deg.toFixed(0)}°`;
    return `${side === "left" ? "Left" : "Right"} gripper: ${got.join(", ") || "not recorded yet"}${kept}.`;
  }).join(" ");
}

// Why each gripper's range cannot be kept yet, where the page measures it (none when the driver skips the step).
function gripperMissing(gripper) {
  return gripper && gripper.ends ? SIDES.map((side) => gripper.ends[side].error).filter(Boolean) : [];
}

async function recordGripper(end) {
  await guard("homeErr", async () => {
    S.status.gripper = await api("POST", "/api/gripper", { end });
    renderGripper(S.status.gripper);
  });
}

function armsLine(assigned) {
  const label = (id) => (S.scan.arms.find((d) => d.id === id) || { label: id }).label;
  return `Left: ${label(assigned.left)}. Right: ${label(assigned.right)}.`;
}
function camsLine() { return S.robot.roles.map((role) => `${S.labels[role].label}: ${S.cams[role].value}`).join(". "); }
function homeLine(home, gripper) {
  const fmt = (s) => S.joints.map((i) => home[s][i].toFixed(0)).join(", ");
  const ends = (s) => `${gripper.ends[s].gripper_closed_deg.toFixed(0)} to ${gripper.ends[s].gripper_open_deg.toFixed(0)}`;
  const grips = gripper && gripper.ends ? ` Grippers: left ${ends("left")}, right ${ends("right")} degrees.` : "";
  return `Home pose: left [${fmt("left")}], right [${fmt("right")}] degrees.${grips}`;
}

// A home joint past the model's range (POST /api/home's out_of_range), as one sentence: its reading, the range and
// which way to move it.
function outsideLine(o) {
  const joint = `${o.side === "left" ? "Left" : "Right"} joint ${o.joint}`, deg = (v) => `${Math.round(v)}°`;
  if (o.deg === null) return `${joint} reads nothing usable.`;
  const back = o.deg > o.upper_deg ? o.upper_deg : o.lower_deg;
  return `${joint} reads ${deg(o.deg)}, the model allows ${deg(o.lower_deg)} to ${deg(o.upper_deg)}: move it back toward ${deg(back)}, or your driver reads it in other units or with another zero.`;
}

async function homeOk() {
  await guard("homeErr", async () => {
    const gripper = S.status.gripper;
    const missing = gripperMissing(gripper);
    if (missing.length) throw new Error(missing[0]);
    const reply = await api("POST", "/api/home", {});
    summary(3, homeLine(reply.home, gripper));
    S.status.home = reply.home;
    S.homeSet = true;
    showHomeSet();
    const out = reply.out_of_range.map(outsideLine);
    if (out.length) $("homeErr").textContent = `${out.join(" ")} Fix that and click Set home pose again, or click Continue to keep this home anyway.`;
  });
}

// Step 4: the reach area, painted while the page is on it (go()). Each arm seen from above, where its first joint turns
// at the centre, x up and y to the left: the painted disks, a line from the base to the hand, and the hand. Nothing here can
// stop the wizard: a drawing that fails says so, and Skip and Continue still work.
const REACH_MIN_SPAN_M = 0.3;
function renderReach(reach) {
  if (!reach) return;
  for (const side of SIDES) {
    const points = reach.points[side];
    $(`reachArea-${side}`).textContent = points.length ? `${areaCm2(points, reach).toLocaleString()} cm² painted`
      : reach.hands[side] ? "Nothing painted yet." : "Can't read this arm.";
    $(`reachClear-${side}`).disabled = !points.length;
    try { drawReach(side, reach); } catch (exc) {
      $("reachWarn").textContent = `Couldn't draw the area (${exc.message}). You can still continue or skip this step.`;
    }
  }
  // The hand at home alone would be a 4 cm disk the arm can barely move in: Continue waits for a real area.
  const least = Math.round(reach.min_area_m2 * 1e4);
  const enough = SIDES.every((side) => areaCm2(reach.points[side], reach) >= least);
  gate("reachDone", SIDES.filter((side) => areaCm2(reach.points[side], reach) < least).map((side) => [`${least} cm² for the ${side} arm`, `reachCanvas-${side}`]));
  $("reachHint").textContent = enough ? "" : `Paint at least ${least} cm² for each arm to continue, or skip this step.`;
  if (reach.error) $("reachWarn").textContent = `${reach.error}. You can still continue or skip this step.`;
}

// The area the client keeps a hand in: every grid cell within the brush of a painted point.
function areaCm2(points, reach) {
  const grid = reach.grid_m;
  const n = Math.round(reach.radius_m / grid);
  const cells = new Set();
  for (const [x, y] of points) {
    const i = Math.round(x / grid);
    const j = Math.round(y / grid);
    for (let di = -n; di <= n; di += 1) for (let dj = -n; dj <= n; dj += 1) if (di * di + dj * dj <= n * n) cells.add(`${i + di},${j + dj}`);
  }
  return Math.round(cells.size * grid * grid * 1e4);
}

function drawReach(side, reach) {
  const canvas = $(`reachCanvas-${side}`);
  const size = canvas.clientWidth;
  if (!size) return;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(size * ratio)) canvas.width = canvas.height = Math.round(size * ratio);
  const points = reach.points[side];
  const hand = reach.hands[side];
  const [ox, oy] = reach.bases[side] || [0, 0];
  let far = REACH_MIN_SPAN_M;
  for (const [x, y] of hand ? [...points, hand] : points) far = Math.max(far, Math.abs(x - ox), Math.abs(y - oy));
  S.reachSpan[side] = Math.max(S.reachSpan[side] || 0, far + 2 * reach.radius_m);  // only grows, so it never jumps back
  const scale = size / 2 / S.reachSpan[side];
  const at = ([x, y]) => [size / 2 - (y - oy) * scale, size / 2 - (x - ox) * scale];
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(name).trim();
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, size, size);
  ctx.strokeStyle = token("--line");
  ctx.lineWidth = 1;
  for (let r = 0.1; r < S.reachSpan[side] * 1.42; r += 0.1) {
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, r * scale, 0, 2 * Math.PI);
    ctx.stroke();
  }
  const brush = reach.radius_m * scale;
  ctx.fillStyle = token("--accent");
  ctx.globalAlpha = 0.32;
  ctx.beginPath();  // one path, so overlapping disks fill once
  for (const point of points) {
    const [u, v] = at(point);
    ctx.moveTo(u + brush, v);
    ctx.arc(u, v, brush, 0, 2 * Math.PI);
  }
  ctx.fill();
  ctx.globalAlpha = 1;
  const [bu, bv] = at([ox, oy]);
  const ring = $(`reachHand-${side}`);
  ring.classList.toggle("hidden", !hand);
  if (hand) {
    const [hu, hv] = at(hand);
    ctx.strokeStyle = token("--ink-2");
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(bu, bv);
    ctx.lineTo(hu, hv);
    ctx.stroke();
    ring.style.left = `${(hu / size) * 100}%`;
    ring.style.top = `${(hv / size) * 100}%`;
  }
  ctx.fillStyle = token("--ink");
  ctx.fillRect(bu - 6, bv - 6, 12, 12);
}

function reachLine(reach) {
  const one = (side) => (reach.points[side].length ? `${areaCm2(reach.points[side], reach).toLocaleString()} cm²` : "nothing");
  return `Reach area: left arm ${one("left")}, right arm ${one("right")}.`;
}

// Step 5: the powered checks. Each drawing follows its arm's readings, the joint being nudged highlighted.
async function powerViews(now, nudging) {
  if (V.busy || !SIDES.every((side) => now[side] && now[side].frame)) return;
  if (!V.power) {
    V.busy = true;
    try {
      V.power = await armViews({ left: "powerViewLeft", right: "powerViewRight" }, { left: now.left.frame, right: now.right.frame });
    } catch (exc) { $("motionErr").textContent = exc.message; return; } finally { V.busy = false; }
  }
  for (const side of SIDES) {
    V.power[side].live(now[side].frame, now[side].grip);
    V.power[side].highlight(nudging && nudging.side === side ? nudging.joint : null);
  }
}

function renderMotion(status) {
  const motion = status.motion;
  const running = motion && motion.state === "running";
  const parking = motion && motion.state === "parking";
  const active = running || parking;
  const ending = !active && status.checks_on;
  powerViews(status.now, running ? motion.nudging : null);
  if (!motion) {
    $("motionGo").classList.remove("hidden");
    $("motionGo").textContent = "Start test";
    for (const id of ["motionBanner", "release", "motionCause", "motionRetry"]) $(id).classList.add("hidden");
    $("motionDone").classList.toggle("hidden", !S.motionStartFailed);
    $("motionDone").classList.remove("primary");
    $("motionDone").textContent = "Skip this step";
    for (const id of ["results", "resultsLeft", "resultsRight"]) setKids($(id), "", () => []);
    return;
  }
  const groups = { hold: "Hold", nudge: "Joint", sweep: "Hand", watchdog: "Follow", limit: "Limit" };
  // Each arm's checks under its own drawing; a check of no one arm, and the notes, under both.
  const resultRow = (row, side) => el("div", { class: "result" },
    el("span", { class: row.finding.state === "pass" ? "ok" : "no", "aria-label": row.finding.state === "pass" ? "Passed" : "Failed" }, row.finding.state === "pass" ? "✓" : "!"),
    el("span", { class: "grp" }, side ? groups[row.group] : `${groups[row.group]} ${row.side}`),
    el("span", {}, row.finding.message));
  // An arm's notes follow its checks: a hand move made smaller or skipped, then once how to avoid that.
  const HOME_HINT = "A more relaxed home pose, with the arm away from its limits, lets every hand move run at full size.";
  for (const side of SIDES) {
    const rows = motion.results.filter((row) => row.side === side);
    const notes = motion.notes.filter((note) => note.side === side);
    setKids($(side === "left" ? "resultsLeft" : "resultsRight"), JSON.stringify([rows, notes]), () => [
      ...rows.map((row) => resultRow(row, side)),
      ...notes.map((note) => el("div", { class: "note armnote" }, note.text)),
      notes.some((note) => note.move) ? el("div", { class: "note armnote" }, HOME_HINT) : null].filter(Boolean));
  }
  // The person's own answer about the drawings is not asked for; a run that holds one anyway does not show it.
  const rest = motion.results.filter((row) => !SIDES.includes(row.side) && row.group !== "eyes");
  const loose = motion.notes.filter((note) => !SIDES.includes(note.side));
  setKids($("results"), JSON.stringify([rest, loose]), () => [
    ...rest.map((row) => resultRow(row, null)), ...loose.map((note) => el("div", { class: "note" }, note.text))]);
  const fail = renderCause(motion.results);
  // The card names a failed check; the banner then says only what the arms do.
  const why = fail ? "" : `The test stopped because a check failed: ${motion.error}. `;
  const banner = $("motionBanner");
  const limp = "The motors are off.";
  const failed = motion.let_go ? `${why}${limp} If an arm still moves, press your emergency stop.`
    : driven(status) ? (fail ? null : `${why}The arms are holding where they stopped.`) : `${why}${limp}`;
  const lost = Object.keys(status.not_answering);
  const stuck = stillHoldingLine(status);
  const parked = Object.keys(status.parked);
  const end = running ? null
    : parking ? ["banner bad", `${why}${PARKING}`]
    : ending ? ["banner bad", `${motion.error ? why : ""}Ending the test. Stay clear for a moment.`]
    : stuck ? ["banner bad", `${stuck} Then click Start over in Connect arms.`]
    : Object.keys(status.off_failed).length ? ["banner bad", offFailedLine(status)]
    : lost.length
    ? ["banner bad", `The ${lost.join(" and ")} arm stopped responding under power. ${S.state.not_answering_help} Then click Start over in Connect arms.`]
    : parked.length ? ["banner bad", `${motion.stop_cut ? `${STOP_CUT} ` : ""}${parkedLine(status)} Then click Start over in Connect arms.`]
    : {
      done: ["banner good", `All checks passed. ${driven(status) ? "Both arms are holding at home." : limp}`],
      failed: ["banner bad", failed],
      stopped: ["banner", `Test stopped. ${driven(status) ? "Both arms are holding where they are." : limp}`],
    }[motion.state];
  banner.classList.toggle("hidden", !(end && end[1]));
  if (end && end[1]) { banner.className = end[0]; setText(banner, end[1]); }
  // Once a run has settled the button stays: Turn off motors while an arm is powered, then "Motors are off", greyed out.
  $("release").classList.toggle("hidden", active || ending);
  const powered = driven(status);
  setText($("releaseBtn"), powered ? "Turn off motors" : "Motors are off");
  $("releaseBtn").disabled = !powered;
  if (powered) {
    const both = status.holding.length === 2;
    setText($("releaseText"), both ? "Support the arms first. They'll drop when the motors turn off." : `Support the ${status.holding[0]} arm first. It'll drop when its motors turn off.`);
  } else setText($("releaseText"), "");
  // After a run that did not pass, running again is the way on; continuing asks first (skipAsk).
  const settled = !active && !ending;
  const retry = settled && motion.state !== "done";
  $("motionGo").classList.toggle("hidden", !settled || retry);
  $("motionGo").textContent = "Run test again";
  $("motionRetry").classList.toggle("hidden", !retry);
  $("motionDone").classList.toggle("hidden", !settled);
  $("motionDone").classList.toggle("primary", !retry);
  $("motionDone").textContent = retry ? "Continue anyway" : "Continue";
}

// The failed check's one cause, if a check failed: which arm and joint, what it is, what it does to the robot, and
// the fix.
// Which arm and joint a result is about: "Left arm, joint 3".
function failedWhere(row) {
  const arm = row.side === "both" ? "Both arms" : `${row.side[0].toUpperCase()}${row.side.slice(1)} arm`;
  return `${arm}${row.finding.joint == null ? "" : `, joint ${row.finding.joint + 1}`}`;
}

function renderCause(results) {
  const fail = results.find((row) => row.finding.state === "fail");
  $("motionCause").classList.toggle("hidden", !fail);
  if (!fail) return null;
  setText($("causeWhere"), failedWhere(fail));
  setText($("causeTitle"), fail.finding.title);
  setText($("causeImpact"), fail.finding.impact);
  setText($("causeFix"), fail.finding.fix);
  setText($("causeGoOn"), fail.finding.go_on);
  return fail;
}

// What the test left undone, for Continue anyway and Save, so nobody continues without knowing what failed and
// whether that is safe: each failed check as "where: what was seen (cause)", and
// whether continuing is safe, which it is only when every failed check says so.
function motionShortfall(motion) {
  const fails = motion ? motion.results.filter((row) => row.finding.state === "fail") : [];
  const lines = fails.map((row) => `${failedWhere(row)}: ${row.finding.message} (${row.finding.title.replace(/\.$/, "")})`);
  if (!fails.length) {
    const why = !motion ? "The test didn't run." : motion.error ? `The test stopped: ${motion.error}.` : "The test didn't finish.";
    return { lines: [why], goOn: "Run it before you continue. Until it passes, nothing confirms your robot moves the way the model expects.", safe: false };
  }
  const unsafe = fails.find((row) => !row.finding.safe);
  return { lines, goOn: (unsafe || fails[0]).finding.go_on, safe: !unsafe };
}

// Step 6: save.
async function save() {
  await busy([$("saveBtn")], () => guard("saveErr", async () => {
    const reply = await api("POST", "/api/save", { cameras: cameraRows() });
    const missing = reply.live_missing.length ? ` Live view stays off until you install the live extra (${reply.live_missing.join(", ")}) and save again.` : "";
    $("savedLine").replaceChildren("Saved to ", el("code", {}, reply.config_path), `. This station is ${reply.robot.name} (${reply.robot.id}) on ${reply.platform}.${missing}`);
    $("savedYaml").replaceChildren(...codeLines(reply.yaml, false));
    $("saved").classList.remove("hidden");
    // The confirmation takes the place of "Saves to ...", at the top of the step.
    $("saveWhere").classList.add("hidden");
    $("savedLine").classList.remove("hidden");
    $("finishActions").classList.remove("hidden");
  }));
}

function renderCamStates(states) {
  for (const box of document.querySelectorAll("[data-camstate]")) {
    const state = states[box.dataset.camstate];
    box.textContent = state && state.state === "reconnecting"
      ? `This camera stopped sending pictures (${state.error}). Reconnecting${state.tries ? ` (attempt ${state.tries})` : ""}. Check its cable.`
      : "";
  }
}

function renderPower(status) {
  const running = !!status.motion && ["running", "parking"].includes(status.motion.state);
  $("stopbar").classList.toggle("hidden", !running);
  if (running) setText($("stopPhase"), status.motion.state === "parking" ? `${status.motion.phase}. ${PARKING}` : status.motion.phase);
  const parked = Object.keys(status.parked).length > 0;
  $("holdbar").classList.toggle("hidden", status.checks_on || !(status.powered || parked) || S.step === 5);
  const [arms, them] = driven(status) ? holding(status) : [];
  setText($("holdText"), [stillHoldingLine(status), offFailedLine(status), parked ? parkedLine(status) : "",
    driven(status) ? `${arms} from the test. Support ${them} with your hands, then` : "Then click Start over in Connect arms."].join(" ").trim());
  $("holdRelease").classList.toggle("hidden", !driven(status));
}

const PARKING = "Stay clear while it moves. STOP turns off both arms' motors at once, so they drop unless you support them.";
const STOP_CUT = "STOP turned off both arms' motors where they were, so they drop unless supported. Support them now.";

function parkedLine(status) { return Object.values(status.parked).join(" "); }

function offFailedLine(status) { return Object.values(status.off_failed).join(" "); }

function stillHoldingLine(status) {
  return Object.values(status.still_holding).map((words) => `${words[0].toUpperCase()}${words.slice(1)}.`).join(" ");
}

function driven(status) { return status.holding.length > 0; }

function holding(status) {
  return status.holding.length === 2 ? ["Both arms are still powered on", "them"] : [`The ${status.holding[0]} arm is still powered on`, "it"];
}

function showGone(gone) {
  S.gone = gone;
  $("gonebar").classList.toggle("hidden", !gone);
  if (gone) {
    for (const id of ["stopbar", "holdbar", "release", "motionBanner", "motionGo", "motionRetry"]) $(id).classList.add("hidden");  // B14b, B38
  }
}

async function poll() {
  if (S.finished) return;
  try {
    S.status = await api("GET", "/api/status");
    S.missedPolls = 0;
    showGone(false);
    renderPower(S.status);
    if (S.robot && [1, 2, 3, 4, 5].includes(S.step)) {
      if (S.step === 1) { renderWiggle(S.status); refreshArms(); }
      if (S.step === 2) { renderCamStates(S.status.cameras.states); showReferenceSource(); }
      if (S.step === 3) { showHome(); showHomeSet(); renderGripper(S.status.gripper); }
      if (S.step === 4) renderReach(S.status.reach);
      if (S.step === 5) renderMotion(S.status);
    }
  } catch (exc) {
    if (!(exc instanceof Gone)) console.error(exc);
    else if (++S.missedPolls >= 3) showGone(true);
  }
  setTimeout(poll, S.gone ? 3000 : S.step >= 3 && S.step <= 5 ? 150 : 300);
}

async function start() {
  S.state = await api("GET", "/api/state");
  if (S.state.dataset.path) S.dataset = S.state.dataset;  // a page opened again finds the dataset the station kept
  globalThis.setupGripperSwitch = !!S.state.gripper_switch;  // the arm cards (setup_page.js) then leave the fix here
  $("agent").classList.toggle("hidden", !S.state.agent_running);
  const c = S.state.connection;
  $("account").replaceChildren(...[["Account", S.state.credentials_path], ["Platform", c.platform_url], ["Gateway", c.gateway_url]]
    .map(([name, value]) => el("div", {}, `${name}: ${value}`)));
  $("version").textContent = `Version ${S.state.version}`;  // the commit the wheel was built from, after its "+g"
  if (S.state.tool === "tcp") return startTcp();
  // A robot of your own: the file's driver and arm on a re-run, else none.
  const ik = S.state.existing_ik;
  S.armWay = !ik ? null : ik.endsWith(".json") ? "urdf" : "ours";
  if (S.armWay === "urdf") {
    S.urdfIk = ik;
    $("urdfDrawn").textContent = `Using the arm from your station file, ${ik.split("/").pop()}. To change it, upload your URDF again.`;
  }
  if (S.armWay === "ours") S.ourIk = ik;
  S.driver = S.state.existing_driver || "";
  if (!/\.py:\w+$/.test(S.driver)) $("customDriver").value = S.driver;
  $("exampleLink").href = `${c.platform_url}/downloads/examples/seeed_rebot/seeed_rebot.py`;
  $("docsLink").href = `${c.platform_url}/docs/robot-of-your-own/`;
  $("customRun").textContent = `dreamcli robot run --config ${S.state.config_path} --checkpoint "<your model>" --task "..." --dry-run`;
  if (S.state.ik_error) $("pickErr").textContent = S.state.ik_error;
  $("saveWhere").classList.remove("hidden");
  $("savedLine").classList.add("hidden");
  $("saveWhere").replaceChildren("Saves to ", el("code", {}, S.state.config_path), `${S.state.exists ? " and replaces the file that's there" : ""}.`);
  S.pick = S.state.existing_body;
  if (S.pick === "so101") checkSo101();  // a station file for an SO-101: check its files against the arms at once
  S.so101Files = S.state.so101_calibrations.map((file) => ({ ...file }));
  showPick();
  numberSteps();
  go(0);
  if (S.state.picked) await guard("pickErr", () => resume(S.state.picked));
  poll();
}

// dreamcli tcp (state's tool): the URDF step alone, for a station file written by hand (Set it up myself); Set tool
// center point writes the arm file whose path is that file's custom.ik. Nothing else is asked and nothing polls.
function startTcp() {
  document.querySelector("main").classList.add("tcp-only");
  $("title").textContent = "Your arm's tool point";
  $("intro").textContent = "Upload your arm's URDF, confirm where the arm ends, and mark where your gripper's fingertips meet. Set tool center point saves the arm file that custom.ik in your station file points to.";
  document.querySelector('section.card[data-step="0"] h2').textContent = "Your arm's URDF";
  $("custom").classList.remove("hidden");
  $("urdfBox").classList.remove("hidden");
  go(0);
}

$("pickRobot").onclick = pickRobot;
$("wayWizard").onclick = () => pickWay("wizard");
$("waySelf").onclick = () => pickWay("self");
$("embodiment").onchange = () => { renderEmbodimentCameras(); renderRobots(); };
$("driverChoose").onclick = () => $("driverDialog").showModal();
$("driverDialogClose").onclick = () => $("driverDialog").close();
$("driverUp").onclick = () => guard("pickErr", () => openFolder(D.listing.parent));
$("driverUse").onclick = () => { if ($("customDriver").value.trim()) pickDriver($("customDriver").value.trim()); };
$("armUrdf").onclick = () => { S.armWay = "urdf"; renderRobots(); };
$("armOurs").onclick = () => { S.armWay = "ours"; renderRobots(); };
$("armSelect").onchange = (e) => { S.ourIk = e.target.value; renderRobots(); };
$("driverTypedOpen").onclick = () => { $("driverTypedBox").classList.toggle("hidden"); $("customDriver").focus(); };
$("urdfDrop").ondragover = (e) => { e.preventDefault(); $("urdfDrop").classList.add("over"); };
$("urdfDrop").ondragleave = () => $("urdfDrop").classList.remove("over");
$("urdfDrop").ondrop = (e) => {
  e.preventDefault();
  $("urdfDrop").classList.remove("over");
  const entry = e.dataTransfer.items[0]?.webkitGetAsEntry();
  if (entry?.isDirectory) { readUrdfFolder(entry); return; }
  $("urdfFile").files = e.dataTransfer.files;
  readUrdf();
};
$("urdfFlip").onclick = flipUp;
buildPad();
$("addArm").onclick = typeArm;
$("addCam").onclick = typeCamera;
$("urdfFile").onchange = readUrdf;
$("urdfFolder").onchange = () => readUrdfFolder($("urdfFolder").files);
$("urdfAdd").onclick = () => addUrdf("calibrated");
$("urdfSkip").onclick = () => { $("tcpSkipTick").checked = false; $("tcpSkipGo").disabled = true; $("tcpSkip").showModal(); };
$("tcpSkipTick").onchange = (e) => { $("tcpSkipGo").disabled = !e.target.checked; };
$("tcpSkipBack").onclick = () => $("tcpSkip").close();
$("tcpSkipGo").onclick = () => { $("tcpSkip").close(); addUrdf("skipped"); };
$("urdfYes").onclick = () => guard("pickErr", confirmEnd);
$("urdfBaseBtn").onclick = () => { U.picking = "base"; renderUrdf(); };
$("urdfReset").onclick = resetMarker;
$("urdfBack").onclick = () => { forgetArm(); U.ok = false; renderUrdf(); };
$("swapSides").onclick = () => swapSides();
$("findAdapters").onclick = scanArms;
$("openArms").onclick = openArms;
$("lookAgain").onclick = lookAgain;
$("udevScan").onclick = lookAgain;
$("gripSwitch").onclick = switchGrippers;
$("armsDone").onclick = () => {
  if (refused("armsDone")) return;
  summary(1, armsLine(S.status.assigned));
  go(2);
};
$("shootAll").onclick = shootAll;
$("camAdvanced").onclick = () => {
  const open = $("gallery").classList.toggle("adv");
  $("camAdvanced").setAttribute("aria-expanded", String(open));
};
// Choice buttons show their pick with .on; screen readers hear it as pressed (4.1.2).
const PICK_GROUPS = "#ways button, #robots button, #armOurs, #armUrdf, #urdfPad button[data-turn-step], #urdfPad button[data-move-step]";
const syncPressed = () => document.querySelectorAll(PICK_GROUPS).forEach((b) => b.setAttribute("aria-pressed", String(b.classList.contains("on"))));
new MutationObserver(syncPressed).observe(document.querySelector("main"), { subtree: true, attributes: true, attributeFilter: ["class"], childList: true });
syncPressed();
$("camMissing").onclick = () => $("typedCamBox").classList.toggle("hidden");
$("refYam").onclick = useYamReferences;
$("refMine").onclick = $("refUseMine").onclick = () => {
  S.refChoice = "mine";
  showReferenceSource();
  $("datasetBox").classList.remove("hidden");
  $("datasetPath").focus();
};
$("refRetry").onclick = retryReferences;
$("refDetails").onclick = () => {
  const open = $("refWarnDetail").classList.toggle("hidden") === false;
  $("refDetails").setAttribute("aria-expanded", String(open));
  $("refDetails").textContent = open ? "Hide details" : "Show details";
};
$("liveViews").onclick = () => { if (!refused("liveViews")) liveViews(); };
$("useDataset").onclick = useDataset;
// Continue, with or without the live views: the station opens each role's camera either way, as later steps read them.
$("camsDone").onclick = () => {
  if (refused("camsDone")) return;
  return guard("camErr", async () => {
    await api("POST", "/api/cameras/open", { cameras: cameraRows() });  // as they are now, whatever changed since the live views
    S.status = await api("GET", "/api/status");
    summary(2, camsLine());
    go(3);
  });
};
$("homeOk").onclick = homeOk;
$("homeNext").onclick = () => { if (!refused("homeNext")) go(S.reach ? 4 : 5); };
for (const side of SIDES) {
  $(`reachClear-${side}`).onclick = () => guard("reachErr", async () => {
    await api("POST", "/api/reach/clear", { side });
    S.reachSpan[side] = 0;
  });
}
$("reachDone").onclick = () => {
  if (refused("reachDone")) return;
  summary(4, reachLine(S.status.reach));
  go(5);
};
$("reachSkip").onclick = () => $("reachAsk").showModal();
$("reachBack").onclick = () => $("reachAsk").close();
$("reachSkipGo").onclick = () => guard("reachErr", async () => {
  $("reachAsk").close();
  await api("POST", "/api/reach/skip", {});
  summary(4, "Skipped. The arms can move anywhere.");
  go(5);
});
$("gripOpen").onclick = () => recordGripper("open");
$("gripClosed").onclick = () => recordGripper("closed");
$("so101LeftAdd").onclick = () => addSo101File("left");
$("so101RightAdd").onclick = () => addSo101File("right");
$("so101Look").onclick = () => guard("pickErr", async () => { await lookAgainSo101(); await checkSo101(); });
$("so101CheckAgain").onclick = checkSo101;
$("motionGo").onclick = $("motionRetry").onclick = () => { $("safeTick").checked = false; $("safeGo").disabled = true; $("safety").showModal(); };
$("safeTick").onchange = (e) => { $("safeGo").disabled = !e.target.checked; };
$("safeCancel").onclick = () => $("safety").close();
$("safeGo").onclick = async () => {
  $("safety").close();
  await guard("motionErr", async () => {
    try {
      await api("POST", "/api/motion/start", { confirm: true });
      S.motionStartFailed = false;
    } catch (exc) {
      S.motionStartFailed = true;  // a test that can't start can still be skipped, after the skip dialog says what it costs
      throw exc;
    }
  });
};
$("stopBtn").onclick = () => api("POST", "/api/motion/stop").catch((exc) => {
  if (exc instanceof Gone) showGone(true); else $("stopPhase").textContent = exc.message;
});
$("releaseBtn").onclick = () => guard("motionErr", () => api("POST", "/api/motion/release"));
$("holdRelease").onclick = () => guard("holdErr", async () => {
  await api("POST", "/api/motion/release");
  $("homeErr").textContent = "";
});
function continueToSave() {
  const m = S.status.motion;
  const ok = m && m.state === "done";
  summary(5, ok ? "All automatic checks passed." : "Some checks didn't pass.");
  const warn = [];
  if (!ok) {
    const short = motionShortfall(m);
    warn.push(`Some checks didn't pass. ${short.lines.join(" ")} ${short.goOn}`);
  }
  if (S.status.reach && S.status.reach.skipped) warn.push("You skipped the reach area, so the arms can move anywhere, including into your camera mount.");
  $("saveWarn").textContent = warn.join(" ");
  $("saveWarn").classList.toggle("hidden", !warn.length);
  $("saveCustom").classList.toggle("hidden", S.robot.body !== "custom");
  go(6);
}
$("motionDone").onclick = () => {
  const motion = S.status.motion;
  if (motion && motion.state === "done") { continueToSave(); return; }
  const short = motionShortfall(motion);
  setKids($("skipFailed"), JSON.stringify(short), () => short.lines.map((line) => el("div", { class: "result" }, el("span", { class: "no" }, "!"), el("span", {}, line))));
  setText($("skipGoOn"), short.goOn);
  $("skipAsk").showModal();
};
$("skipBack").onclick = () => $("skipAsk").close();
$("skipGo").onclick = () => { $("skipAsk").close(); continueToSave(); };
$("saveBtn").onclick = save;
const FINISH_POWERED = {
  makers: "Finishing turns off the arms. Each Maker Arm moves to its rest pose (every joint at zero), opens its gripper, then turns off its motors and drops. Stay clear while it moves, then support both arms. Finish now?",
  metal: "Finishing turns off the arms. Each arm moves back to where it was when the test started, opens its gripper, then turns off its motors and drops. Stay clear while it moves, then support both arms. Finish now?",
  so101: "Finishing turns off the arms. Each arm moves back to where it was when the test started, opens its gripper, then turns off its motors and drops. Stay clear while it moves, then support both arms. Finish now?",
};
// Code in a Terminal or file block, coloured as the platform's Code block colours it: a comment line in the comment
// colour and, for a command, its first word in the command colour.
function codeLines(text, command) {
  return text.split(/(?<=\n)/).flatMap((line, i) => {
    if (line.trimStart().startsWith("#")) return [el("span", { class: "tk-comment" }, line)];
    if (i > 0 || !command) return [line];
    const [, first, rest] = line.match(/^(\S*)([\s\S]*)$/);
    return [el("span", { class: "tk-cmd" }, first), rest];
  });
}

const commandParts = (text) => codeLines(text, true);

function showCommand(id, text) {
  $(id).replaceChildren(...commandParts(text));
}

function terminal(id, text) {
  return el("div", { class: "term" },
    el("div", { class: "termhead" }, el("span", { class: "termlabel" }, "Terminal"),
      el("button", { type: "button", class: "copy", "data-copy": id, "aria-label": "Copy to the clipboard", title: "Copy" })),
    el("pre", {}, el("code", { id }, commandParts(text))));
}

// The Setup complete page both ways of setting up end on, the step list's last step (Open Live view, setup_page.js).
function showComplete(banners) {
  const main = document.querySelector("main");
  main.replaceChildren(el("div", {}, el("p", { class: "stepof", id: "stepOf" }), el("h1", { id: "title" }, "Setup complete")),
    ...banners, ...nextSteps());
  main.classList.add("finished");
}

// What to do once setup is done: the command to copy, and Live view, where the robot is driven from.
function nextSteps() {
  const live = `${S.state.connection.platform_url}/live`;
  return [
    el("p", {}, "Next, start your robot. Run this command on the station:"),
    terminal("nextCmd", "dreamcli robot up"),
    el("p", {}, "Then click Start in Live view:"),
    el("div", { class: "actions" }, el("a", { class: "go", href: live, target: "_blank", rel: "noopener" }, "Open Live view")),
    el("p", { class: "note" }, "To run a single task instead, use dreamcli robot run --task \"<what to do>\" on the station. Add --dry-run to check it without moving the arms. You can close this page now."),
  ];
}
// Finish on Save station: setup stops and the Setup complete page shows; arms still powered are asked about first, in
// the page's dialog, and said to be moving to rest; an error shows on the step.
const finishSetup = (button, errId) => () => busy([button], () => guard(errId, async () => {
  const status = await api("GET", "/api/status");
  const question = driven(status) ? FINISH_POWERED[S.robot.body] || "Finishing turns off the powered arms. Support them first. Finish now?" : null;
  if (question && !(await ask("Finish setup?", question, "Finish"))) return;
  await api("POST", "/api/exit");
  S.finished = true;
  const moving = driven(status) ? el("p", { class: "banner bad" }, "The arms are moving to rest and turning off. Stay clear until they stop.") : null;
  const stuck = stillHoldingLine(status) ? el("p", { class: "banner bad" }, stillHoldingLine(status)) : null;
  showComplete([moving, stuck].filter(Boolean));
}));
// Set up the station myself: Finish just closes the wizard; the station is set up from its terminal from here.
$("selfClose").onclick = () => busy([$("selfClose")], () => guard("pickErr", async () => {
  await api("POST", "/api/exit");
  S.finished = true;
  document.querySelector("main").replaceChildren(el("h1", {}, "Setup wizard closed"),
    el("p", { class: "help" }, "The wizard has stopped on the station. You can close this tab."));
  window.close();  // closes the tab when the browser allows it; otherwise the line above says to
}));
$("finish").onclick = finishSetup($("finish"), "saveErr");
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-copy]");
  if (!target) return;
  const text = $(target.dataset.copy);
  if (window.isSecureContext) { navigator.clipboard.writeText(text.textContent); return; }
  // Opened over plain http by the box's tailnet name, the browser has no clipboard API: select the text and copy.
  getSelection().selectAllChildren(text);
  document.execCommand("copy");
});
start();
