// The setup wizard page: one step at a time, each a card, talking to the routes in robots/setup_wizard.py.
import { ArmView, MARKER_ROOM, UrdfView, folderFiles, unzip } from "/setup_viewer.js";

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
  so101Files: [], urdf: null, homeBusy: false, embodiments: null, finished: false, missedPolls: 0, gone: false,
  reach: false, reachSpan: {}, way: "wizard",
};
// The 3D views: how the arm is drawn (GET /api/view), and both arms in the home step and in the powered checks.
const V = { drawing: null, home: null, power: null, busy: false };

const GONE = "The setup program on the robot's computer stopped. The arms are in whatever state it left them: support them, then run dreamcli setup again.";
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

async function busy(buttons, fn) {
  for (const button of buttons) button.disabled = true;
  try { await fn(); } finally { for (const button of buttons) button.disabled = false; }
}

// The cards: the one being done is open, those before are done with a summary, those after wait.
function go(step) {
  S.step = step;
  document.querySelectorAll("section.card").forEach((card) => {
    const i = Number(card.dataset.step);
    card.classList.toggle("open", i === step);
    card.classList.toggle("done", i < step);
    card.classList.toggle("later", i > step);
    const pill = card.querySelector("[data-pill]");
    pill.textContent = i < step ? "Done" : i === step ? "Now" : "";
    pill.className = "pill" + (i < step ? " good" : i > step ? " hidden" : "");  // none on a step not reached (B13)
  });
  document.querySelector(`section.card[data-step="${step}"]`).scrollIntoView({ behavior: "smooth", block: "start" });
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

function summary(step, text, back) {
  const box = document.querySelector(`section.card[data-step="${step}"] [data-summary]`);
  box.replaceChildren(el("span", {}, text));
  if (back) box.append(el("button", { onclick: () => go(step) }, "Change"));
}

// Step 0: the robot.
function renderRobots() {
  const box = $("robots");
  const choices = S.state.robots.map((robot) => ({
    key: robot.body, title: robot.label, sub: robot.unavailable || "", off: !!robot.unavailable, note: robot.note,
  }));
  choices.push({ key: "custom", title: "My own robot", sub: "your driver, dream's model of your arm", off: false, note: null });
  box.replaceChildren(...choices.map((c) => el("button", {
    class: "robot" + (S.pick === c.key ? " on" : ""), disabled: c.off,
    onclick: () => { S.pick = c.key; showPick(); },
  }, el("b", {}, c.title), el("span", {}, c.sub))));
  const picked = choices.find((c) => c.key === S.pick);
  const self = S.way === "self";
  $("robotNote").textContent = !self && picked && picked.note ? picked.note : "";  // nothing lets an arm go
  $("custom").classList.toggle("hidden", self || S.pick !== "custom");
  $("so101").classList.toggle("hidden", self || S.pick !== "so101");
  $("zeroBanner").classList.toggle("hidden", !self && S.pick === "so101");  // the wizard calibrates an SO-101 itself
  $("embodimentBox").classList.toggle("hidden", self || (S.pick !== "so101" && S.pick !== "custom"));
  $("newBox").classList.toggle("hidden", self || (S.pick !== "so101" && S.pick !== "custom") || $("embodiment").value !== NEW);
  // New reuses a listed `<user>/<arm>`, the arm's name as _zero_shot_embodiment() writes it.
  const ik = (S.pick === "so101" ? S.state.robots.find((r) => r.body === "so101").ik_arm : ownArm()) || "";
  const arm = (ik.endsWith(".json") ? ik.split("/").pop().slice(0, -5) : ik).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64);
  const reuses = arm && (S.embodiments || []).find((e) => e.name.includes("/") && e.name.slice(e.name.indexOf("/") + 1) === arm);
  $("newBox").textContent = reuses
    ? `Your account already has ${reuses.name}, named after this arm, so Continue reuses it instead of registering a new one. You upload your robot's recorded data under it for a fine-tune.`
    : "Continue registers the embodiment on your account, named after your arm, with your arm's joints and the three cameras Dream Labs' base models read: one that sees the whole scene and one on each wrist. You give each camera its role in the camera step. You upload your robot's recorded data under it for a fine-tune.";
  $("pickActions").classList.toggle("hidden", self);
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
    needs.push(S.armWay === "urdf" ? ["your arm's tool center point (Set tool center point, or skip it)", "urdfBox"]
      : S.armWay === "ours" ? ["one of the arms Dream Labs has a model of", "armList"] : ["dream's model of your arm", "armUrdf"]);
  }
  if (own && !S.driver) needs.push(["your driver", "driverBox"]);
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
// Set it up myself: the file Save writes for the picked robot, every value the steps measure left as REPLACE_
// (GET /api/template), and the run the wizard's file gets: a dry run first, then the run.
async function showTemplate() {
  const path = S.state.config_path, model = S.pick === "yam" ? "" : ' --checkpoint "<your model>"';
  const run = `dreamcli robot run --config ${path}${model} --task "<what to do>"`;
  $("selfPath").textContent = path;
  $("selfYaml").textContent = (await api("GET", `/api/template?body=${encodeURIComponent(S.pick)}`)).yaml;
  $("selfDry").textContent = `${run} --dry-run`;
  $("selfTcp").classList.toggle("hidden", S.pick !== "custom");
  $("selfRun").textContent = run;
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
  $("driverChoose").textContent = S.driver ? "Change driver file" : "Choose your driver file (.py)";
  $("driverChoose").classList.toggle("primary", !S.driver);
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
  $("armList").replaceChildren(...S.state.ik_arms.map((arm) => el("button", {
    class: "robot" + (S.ourIk === arm.name ? " on" : ""), onclick: () => { S.ourIk = arm.name; renderRobots(); },
  }, el("b", {}, arm.label), el("span", {}, arm.name))));
}

// The embodiment list's last option: a new one, which Continue registers from the arm.
const NEW = "new";
async function loadEmbodiments() {
  if (S.embodiments) return;
  S.embodiments = (await api("GET", "/api/embodiments")).embodiments;
  // An embodiment with no settings has no cameras; it is listed as such, not a crash.
  const cameras = (e) => ((e.cameras || []).length ? `cameras: ${e.cameras.join(", ")}` : "no cameras");
  $("embodiment").replaceChildren(...S.embodiments.map((e) => el("option", { value: e.name }, `${e.name} (${cameras(e)})`)),
    el("option", { value: NEW }, "New: one for Dream Labs' base models, from your arm and cameras"));
  if (S.embodiments.some((e) => e.name === S.state.existing_embodiment)) $("embodiment").value = S.state.existing_embodiment;
  $("embodimentNote").textContent = S.embodiments.length ? "" : "Your account has no embodiment yet, so Continue registers a new one.";
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

// The URDF the operator gives: a .zip of its URDF and meshes, or a .urdf alone (urdfFile), or its folder (urdfFolder),
// which the drawing then shows.
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

// The URDF's folder (urdfFolder): the files in it, read as a .zip's are (folderFiles()).
async function readUrdfFolder() {
  const list = $("urdfFolder").files;
  $("urdfPick").classList.add("hidden");
  if (!list.length) return;
  await guard("pickErr", async () => pickUrdf(list[0].webkitRelativePath.split("/")[0], await folderFiles(list)));
}

// The URDFs among the files of a .zip or folder (`name`): one is drawn; several are listed to pick one,
// and nothing is drawn until then.
async function pickUrdf(name, files) {
  const urdfs = [...files.keys()].filter((path) => path.toLowerCase().endsWith(".urdf"));
  if (!urdfs.length) throw new Error(`${name} holds no .urdf file. Give your arm's URDF as a .zip with its meshes, or its folder.`);
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
  $("urdfPickAsk").textContent = `${name} holds ${urdfs.length} URDFs. Pick your arm's:`;
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
    U.view ||= new UrdfView($("urdfView"), pickLink, dragMarker);
    $("urdfDrawn").textContent = drawnLine(tree, await U.view.tree(tree, files, path), files);
    renderUrdf();
  });
}

// What the drawing shows of how the arm looks (UrdfView.tree()'s answer): the links its meshes draw, and the ones
// drawn plain and why.
function drawnLine(tree, drawn, files) {
  if (!files) return drawn.plain.length ? "Your URDF's meshes are files of their own, so its links are drawn plain. Give the URDF as a .zip with its meshes, or its folder, to see your arm as it looks." : "";
  const plain = drawn.plain.map((link) => tree.links[link]).join(", ");
  return `${drawn.shown.length} links drawn from your meshes.` + (plain ? ` Drawn plain: ${plain} (not in the zip, not STL, OBJ, COLLADA or glTF, or a compressed glTF: ${drawn.missing.join(", ")}).` : "");
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
  U.yam.label("Reference: YAM, for comparison, not editable");
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
  $("urdfAsk").textContent = U.ok ? "Put the marker where your gripper's fingertips close, turned like the YAM's." : "Is this the end of your arm?";
  $("urdfLinks").classList.toggle("hidden", U.ok);
  $("urdfLinks").textContent = U.picking === "base" ? "Click the link your arm stands on."
    : !links ? `${name(U.end)} does not hang below ${name(U.base)}. Click the end of your arm.`
    : joints > U.tree.most_joints ? `${name(U.end)} is ${joints} joints out from ${name(U.base)}; an arm has at most ${U.tree.most_joints}. Click the link your gripper is mounted on.`
    // A chain with no joint that moves is not an arm: building it fails only after the tool point is set (a YAM + UMI, say).
    : !joints ? `No joint moves between ${name(U.base)} and ${name(U.end)}. Click the link your gripper is mounted on, or "It stands on another link" if ${name(U.base)} is not what your arm stands on.`
    : `Your arm stands on ${name(U.base)} (blue) and ends at ${name(U.end)} (green), ${joints} joints out. Not the end? Click the right link on the drawing.`
      + (!extra.length ? "" : U.tree.joint_map.joints ? ` ${U.tree.joint_map.embodiment} does not drive ${extra.length} of them; answer for each below.` : " Answer for each joint below.");
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
    $("urdfNote").textContent = "Building dream's model of your arm...";
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
    const skipped = calibration === "skipped" ? " Tool point skipped: every start warns." : "";
    const still = held.length ? ` Held still: ${held.join(", ")}.` : "";
    $("urdfNote").textContent = `Using ${S.urdf.name}: ${reply.joints.length} joints (${reply.joints.join(", ")}).${still}${skipped}`;
    if (S.state.tool !== "tcp") renderRobots();
    $("tcpIk").textContent = reply.ik;
    $("tcpSaved").classList.toggle("hidden", S.state.tool !== "tcp");
  });
}

// The SO-101's calibration files: the ones lerobot-calibrate left on this box, ticked when a plugged-in arm's servos
// hold it (state's `matches`, which each names) and otherwise left for the operator to tick (each says when it last
// changed, to tell an old one from today's), and any the operator adds, ticked.
function renderSo101Files() {
  $("so101Files").replaceChildren(...(S.so101Files.length ? S.so101Files.map((file) => el("label", { class: "device" },
    el("input", { type: "checkbox", checked: file.on, onchange: (e) => { file.on = e.target.checked; renderRobots(); } }),
    el("div", {}, el("div", { class: "name" }, file.path.split("/").pop()), el("div", { class: "sub2" }, fileWhen(file)),
      ...(file.matches || []).map((label) => el("div", { class: "sub2" }, `Matches the arm on ${label}`))),
    el("span"))) : [el("p", { class: "note" }, "No calibration file found on this box. Add the path of yours below.")]));
}

function fileWhen(file) {
  if (file.modified === undefined) return file.path;
  if (file.modified === null) return `${file.path} · not on this box`;
  const when = new Date(file.modified * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  return `${file.path} · last changed ${when}`;
}

async function pickRobot() {
  renderRobots();  // what is still needed now, not at the last render
  if (refused("pickRobot")) return;
  await guard("pickErr", async () => {
    $("pickRobot").disabled = true;
    $("pickNote").textContent = "Loading dream's model of the arm (a few seconds)...";
    let reply;
    try {
      const fresh = $("embodiment").value === NEW;  // registered by this pick, so the list is asked again after
      const which = { embodiment: fresh ? null : $("embodiment").value };
      const body = S.pick === "custom" ? { custom: { ik_arm: ownArm(), driver: S.driver, ...which } }
        : S.pick === "so101" ? { body: S.pick, calibration: S.so101Files.filter((f) => f.on).map((f) => f.path), ...which }
        : { body: S.pick };
      reply = await api("POST", "/api/robot", body);
      if (fresh) S.embodiments = null;
    } finally {
      $("pickNote").textContent = "";
      $("pickRobot").disabled = false;
    }
    applyPick(reply);
    scanArms();
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
  // MakerMods' arms and cameras are always found by the scan: no address to type.
  const makermods = ["so101", "makers", "metal"].includes(S.robot.body);
  $("typedArmBox").classList.toggle("hidden", makermods);
  $("typedCamBox").classList.toggle("hidden", makermods);
  $("limpNote").textContent = S.robot.note || "";
  $("limpNote").hidden = !S.robot.note;
  if (!S.dataset) $("datasetNote").textContent = noDatasetNote();
  S.cams = {};
  for (const row of S.prefill.cameras) S.cams[row.name] = { ...row };
  V.drawing = null;
  V.home = null;
  V.homeTarget = false;
  V.power = null;
  S.opened = false;
  $("wiggle").classList.add("hidden");
  $("lookAgain").classList.add("hidden");
  const arm = S.robot.ik_arm.endsWith(".json") ? `your URDF's ${S.robot.ik_arm.split("/").pop()}` : `dream's ${S.robot.ik_arm}`;
  const drawn = S.robot.body === "custom" ? `, drawn as ${arm} arm` : "";
  const embodiment = S.robot.custom ? S.robot.custom.embodiment : S.robot.embodiment;
  const reused = reply.reused ? " (already on your account; reused, not registered again)" : "";
  summary(0, `${S.robot.label}${drawn}. ${embodiment ? `Embodiment ${embodiment}${reused}. ` : ""}Cameras: ${S.robot.roles.join(", ")}.`, true);
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
  const cal = st.calibration;  // an SO-101 arm being calibrated, or still to calibrate, keeps the page on step 1
  if (cal && (cal.active || SIDES.some((side) => cal.needed[side]))) return;
  summary(1, armsLine(st.assigned), true);
  markRoles();
  if (!S.robot.roles.every((role) => st.cameras.open.includes(role))) { go(2); return; }
  summary(2, camsLine(), true);
  if (!st.home.left || gripperMissing(st.gripper).length) { go(3); return; }
  summary(3, homeLine(st.home, st.gripper), true);
  go(S.reach ? 4 : 5);
}

// Step 1: the arms, found by moving them.
async function scanArms() {
  await guard("armsErr", async () => {
    S.scan = await api("GET", "/api/scan");
    S.ticked = new Set(S.scan.arms.map((d) => d.id));
    S.openErrors = {};
    $("scanNote").textContent = S.scan.arms.length ? `${S.scan.arms.length} found` : `None found. Plug in both arms' adapters, switch the arms on, and look again.`;
    renderDevices(null);
    renderOthers();
    $("openArms").disabled = !S.scan.arms.length;
  });
}

// "Your arm is not listed?": the address its driver opens, ticked like an arm the scan found.
async function typeArm() {
  await guard("armsErr", async () => {
    const before = new Set(S.scan.arms.map((device) => device.id));
    S.scan = await api("POST", "/api/arms/typed", { address: $("typedArm").value });
    for (const device of S.scan.arms) if (!before.has(device.id)) S.ticked.add(device.id);
    $("typedArm").value = "";
    $("scanNote").textContent = `${S.scan.arms.length} to open`;
    renderDevices(null);
    renderOthers();
    $("openArms").disabled = S.opened || !S.ticked.size;
  });
}

function linkProblem(device) {
  if (device.kind === "slcan" && device.via && device.state !== "up") {
    return `slcand's link ${device.via} is ${device.state || "missing"}: run sudo ip link set ${device.via} up, or stop that slcand and the adapter is opened directly.`;
  }
  // Down first: a link stuck after a bus fault keeps its UP flag and refuses the up alone as busy.
  if (device.kind === "can" && device.state && device.state !== "up") {
    return `This CAN link is ${device.state}: run sudo ip link set ${device.open} down, then sudo ip link set ${device.open} up type can bitrate 1000000, then look again.`;
  }
  return null;
}

// Once both arms are found, every other port listed is not an arm: the wizard closed it, and the row says so rather
// than "Not opened." (a third port listed beside the two arms otherwise reads as one more arm).
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
    } });
    const side = live && live.side ? el("span", { class: "pill good" }, `${live.side} arm`) : null;
    const meter = el("div", { class: "meter" }, el("i", { style: `width:${live ? Math.min(100, live.score_deg / 20 * 100) : 0}%` }));
    const problem = linkProblem(device);
    const info = el("div", {},
      el("div", { class: "name" }, device.label, " ", side),
      el("div", { class: "sub2" }, device.detail),
      notArm ? el("div", { class: "sub2" }, "Not an arm: neither arm is on this port, so it is closed and left out of the robot.") : null,
      !notArm && opened && !live && !(S.openErrors || {})[device.id] && S.ticked.has(device.id) ? el("div", { class: "sub2" }, "Not opened.") : null,
      !notArm && !S.ticked.has(device.id) ? el("div", { class: "sub2" }, "Unticked: left alone.") : null,
      problem ? el("div", { class: "err" }, problem) : null,
      live && live.calibration ? el("div", { class: "sub2" }, `Motors agree with ${live.calibration}`) : null,
      live && live.uncalibrated ? el("div", { class: "sub2" }, "Not calibrated yet: once both arms are found, the wizard calibrates it.") : null,
      live && live.error ? el("div", { class: "err" }, live.error) : null,
      live ? el("div", { class: "jrow" }, S.joints.map((i) => el("span", {}, `J${i + 1} ${live.joints_deg[i].toFixed(1)}`)), el("span", {}, `grip ${live.gripper === null ? "?" : live.gripper.toFixed(2)}`)) : null,
      S.openErrors && S.openErrors[device.id] ? el("div", { class: "err" }, S.openErrors[device.id]) : null,
    );
    return el("div", { class: "device" + (live && live.moved ? " moved" : "") + (notArm ? " not-arm" : "") }, box, info, opened && !notArm ? meter : el("span"));
  }));
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

function showOpened() {
  S.askKey = null;
  $("wiggle").classList.remove("hidden");
  $("crossed").classList.add("hidden");
  $("udev").classList.add("hidden");
  $("armsDone").classList.add("hidden");
  $("lookAgain").classList.remove("hidden");
  renderDevices(null);
}

async function lookAgain() {
  await guard("armsErr", async () => {
    $("lookAgain").disabled = true;
    try {
      await api("POST", "/api/arms/close");
    } finally {
      $("lookAgain").disabled = false;
    }
    S.opened = false;
    S.askKey = null;
    $("wiggle").classList.add("hidden");
    $("lookAgain").classList.add("hidden");
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
  const label = (id) => (S.scan.arms.find((d) => d.id === id) || { label: id }).label;
  const ask = $("wiggleAsk");
  if (done) {
    ask.replaceChildren(el("p", { class: "lead" }, "Both arms found."),
      el("p", { class: "help" }, `Left: ${label(assigned.left)}. Right: ${label(assigned.right)}.`));
    $("armsDone").disabled = false;
    gate("armsDone", []);
    showRule();
    $("armsDone").classList.remove("hidden");
    return;
  }
  const found = SIDES.filter((s) => assigned[s]).map((s) => `${s} arm found`).join(", ");
  if (wiggle.several) {
    ask.replaceChildren(el("p", { class: "lead" }, "More than one arm moved, so we cannot tell which one you meant."),
      el("p", { class: "help" }, "A limp arm that sags or slides as it goes limp counts as moving too. Rest every arm where it cannot move (on its stand or the table), press Start over, then move only one arm."),
      el("button", { onclick: () => api("POST", "/api/arms/rebase") }, "Start over"));
  } else if (wiggle.moved) {
    ask.replaceChildren(el("p", { class: "lead" }, `You moved: ${label(wiggle.moved)}. Which arm is it?`),
      el("div", { class: "actions" },
        el("button", { class: "primary", onclick: () => assign(wiggle.moved, "left") }, "Left arm"),
        el("button", { class: "primary", onclick: () => assign(wiggle.moved, "right") }, "Right arm"),
        el("button", { onclick: () => assign(wiggle.moved, "none") }, "Not part of this robot")));
  } else {
    ask.replaceChildren(el("p", { class: "lead" }, el("span", { class: "pulse" }), "Move one arm by hand, a little, any joint."),
      el("p", { class: "help" }, found ? `${found}. Now move the other arm.` : "We watch every adapter and show you which one moved."));
  }
}

// An SO-101's calibration, once both arms are found: each arm not calibrated yet is put in its middle pose (drawn on
// dream's model at its zero), then every joint is moved end to end while a bar fills per joint. The server moves each
// joint's zero to where the model's arm had it (robots/so101_calibration.py), so a middle pose put by eye is enough.
const JOINT_WORDS = { shoulder_pan: "Base turn", shoulder_lift: "Shoulder", elbow_flex: "Elbow", wrist_flex: "Wrist bend", gripper: "Gripper" };

function renderCalibration(status) {
  const cal = status.calibration;
  const box = $("calibrate");
  const found = status.assigned.left && status.assigned.right;
  if (!cal || !found) { box.classList.add("hidden"); return; }
  const active = cal.active;
  const pending = SIDES.filter((side) => cal.needed[side]);
  box.classList.toggle("hidden", !active && !pending.length && !Object.keys(cal.done).length);
  $("armsDone").classList.toggle("hidden", Boolean(active || pending.length));
  setText($("calErr"), cal.error || $("calErr").textContent);
  const key = JSON.stringify([active && [active.side, active.stage], pending, Object.keys(cal.done)]);
  if (active && active.stage === "range") drawCalJoints(active.joints);
  if (box.dataset.drawn === key) return;
  box.dataset.drawn = key;
  $("calJoints").replaceChildren();
  $("calViews").classList.add("hidden");
  const button = (label, path, primary) => el("button", { class: primary ? "primary" : "", onclick: (e) => busy([e.target], () => guard("calErr", () => api("POST", path, path.endsWith("start") ? { side: label.side } : {}))) }, label.text);
  if (!active) {
    const lines = SIDES.map((side) => {
      const done = cal.done[side];
      if (cal.needed[side]) return el("p", { class: "help" }, `The ${side} arm is not calibrated yet.`);
      if (!done) return el("p", { class: "help" }, `The ${side} arm is calibrated already.`);
      const moved = Object.entries(done.moved_deg).map(([joint, deg]) => `${(JOINT_WORDS[joint] || joint).toLowerCase()} ${Math.abs(deg).toFixed(1)} degrees`);
      return el("p", { class: "help" }, `The ${side} arm is calibrated. `,
        moved.length ? `Your middle pose was corrected to the model's: ${moved.join(", ")}. ` : "", ...done.notes.map((n) => `Note: ${n}. `));
    });
    setText($("calLead"), pending.length ? `Calibrate your arms (${pending.length === 2 ? "both" : `the ${pending[0]} one`})` : "Both arms are calibrated.");
    $("calHelp").replaceChildren(...lines,
      ...(pending.length ? [el("p", { class: "note" }, "This tells each motor where its zero is. It takes about a minute per arm, with the arm limp in your hands.")] : []));
    $("calActions").replaceChildren(...SIDES.map((side) => cal.needed[side]
      ? button({ side, text: `Calibrate the ${side} arm` }, "/api/calibrate/start", side === pending[0])
      : el("button", { class: "quiet", onclick: (e) => busy([e.target], () => guard("calErr", () => api("POST", "/api/calibrate/start", { side }))) }, `Calibrate the ${side} arm again`)));
    return;
  }
  const cancel = el("button", { class: "quiet", onclick: (e) => busy([e.target], () => guard("calErr", () => api("POST", "/api/calibrate/cancel"))) }, "Cancel");
  if (active.stage === "middle") {
    setText($("calLead"), `${active.side[0].toUpperCase()}${active.side.slice(1)} arm, 1 of 2: put it in the middle pose`);
    $("calHelp").replaceChildren(el("p", { class: "help" }, "Its motors are off now: hold it. Bend it into the pose in the drawing: upper arm straight up, forearm straight forward, wrist in line with the forearm, gripper pointing forward and level, jaws closed. Close is enough; the wizard corrects it in the next step. Then press Set middle pose."));
    $("calActions").replaceChildren(el("button", { class: "primary", onclick: (e) => busy([e.target], () => guard("calErr", () => api("POST", "/api/calibrate/middle"))) }, "Set middle pose"), cancel);
    if (active.middle_frame) {
      $("calViews").classList.remove("hidden");
      $("calView").replaceChildren();
      const view = new ArmView($("calView"));
      drawing().then((d) => view.load(d, active.middle_frame)).then(() => { view.live(active.middle_frame, 0); view.aim(active.middle_frame, 0); view.label("middle pose"); }).catch(() => $("calViews").classList.add("hidden"));
    }
    return;
  }
  setText($("calLead"), `${active.side[0].toUpperCase()}${active.side.slice(1)} arm, 2 of 2: move each joint from one end to the other`);
  $("calHelp").replaceChildren(el("p", { class: "help" }, "One joint at a time, slowly, all the way to where it stops at each end, then back. Open and close the gripper fully. The wrist roll is not needed. Each bar fills when its joint has gone far enough; then press Finish."));
  $("calActions").replaceChildren(el("button", { class: "primary", onclick: (e) => busy([e.target], () => guard("calErr", () => api("POST", "/api/calibrate/finish"))) }, "Finish"), cancel);
  drawCalJoints(active.joints);
}

function drawCalJoints(joints) {
  $("calJoints").replaceChildren(...joints.map((j) => {
    const span = j.high_deg - j.low_deg;
    const need = j.target_deg === null ? 30 : j.target_deg - 10;
    return el("div", { class: "device" + (j.done ? " moved" : "") }, el("span"),
      el("div", {}, el("div", { class: "name" }, JOINT_WORDS[j.name] || j.name, " ", j.done ? el("span", { class: "pill good" }, "done") : null),
        el("div", { class: "sub2" }, `${span.toFixed(0)} of about ${need.toFixed(0)} degrees`)),
      el("div", { class: "meter" }, el("i", { style: `width:${Math.min(100, (span / need) * 100)}%` })));
  }));
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
    $("udevText").textContent = rule.install;
    const named = Object.entries(rule.in_effect).filter(([, ok]) => ok).map(([s]) => s);
    const all = named.length === Object.keys(rule.in_effect).length;
    const port = rule.by_port.length ? ` The ${rule.by_port.join(" and ")} adapter reports no serial number of its own, so it is named by its USB port: keep it in that port.` : "";
    $("udevNote").textContent = (all ? "The names are in effect now." : "The names take effect at the next reboot: reboot this computer before you run dreamcli robot up.") + port;
    $("udev").classList.remove("hidden");
  } catch (exc) { $("armsErr").textContent = exc.message; }
}

// Step 2: the cameras.
async function shootAll() {
  await guard("camErr", async () => {
    const scan = await api("GET", "/api/scan");
    S.shots = scan.cameras;
    $("shootNote").textContent = S.shots.length ? "" : "No cameras found.";
    const tiles = S.shots.map((cam) => {
      const img = el("img", { alt: cam.label });
      cam.rotate = 0;
      cam.turn = turnButton(() => turnCamera(cam), cam.rotate);
      const frame = el("div", { class: "frame" }, el("span", { class: "msg" }, "Taking a picture..."));
      const roles = el("div", { class: "roles" }, S.robot.roles.map((role) => el("button", {
        "data-role": role, onclick: () => setCamera(role, cam),
      }, S.labels[role].short)));
      const how = { port: "Named by its USB port: keep it in this port.", serial: "Named by its serial number.", typed: "Typed: opened as written." }[cam.bound];
      // A RealSense says the USB link it is on; one on USB2 cannot carry its colour stream, so it says what to do.
      const usb = cam.usb_warning ? el("div", { class: "err" }, cam.usb_warning) : cam.usb ? el("div", { class: "how" }, `USB ${cam.usb}`) : null;
      cam.tile = el("div", { class: "shot" }, el("div", { class: "turnbar" }, cam.turn), frame, el("div", { class: "meta" }, el("div", { class: "name" }, cam.label), el("div", { class: "how" }, how), usb, roles));
      cam.frame = frame;
      cam.img = img;
      return cam.tile;
    });
    $("gallery").replaceChildren(...tiles);
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

// "Your camera is not listed?": an address OpenCV opens or a camera class, then every picture again with it.
async function typeCamera() {
  await guard("camErr", async () => {
    await api("POST", "/api/cameras/typed", { address: $("typedCam").value });
    $("typedCam").value = "";
    await shootAll();
  });
}

const holds = (row, cam) => row.value === cam.value && row.source === cam.source;

// The Rotate button in the bar right above a camera's picture (not inside it), the same in the role step and the live views: a quarter turn clockwise,
// with the picture's turn now on its chip.
function turnButton(onclick, degrees) {
  const button = el("button", { class: "turn", onclick, title: "Turn this camera's picture a quarter turn clockwise" });
  showTurn(button, degrees);
  return button;
}
function showTurn(button, degrees) {
  button.replaceChildren(el("span", { class: "icon", "aria-hidden": "true" }, "↻"), "Rotate 90°", el("span", { class: "chip" }, `now ${degrees}°`));
}

function setCamera(role, cam) {
  for (const other of Object.values(S.cams)) if (holds(other, cam)) other.value = "";
  S.cams[role] = { ...S.cams[role], source: cam.source, value: cam.value, rotate: cam.rotate };
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
    showTurn(cam.turn, cam.rotate);
  }
}

// A quarter turn clockwise of one camera's picture, kept as the `rotate` of the role holding it (or the one it takes
// next); live views already open open again turned.
function turnCamera(cam) {
  cam.rotate = (cam.rotate + 90) % 360;
  for (const role of S.robot.roles) if (holds(S.cams[role], cam)) S.cams[role].rotate = cam.rotate;
  markRoles();
  if (!$("compare").classList.contains("hidden")) return liveViews();
}

function markRoles() {
  for (const cam of S.shots) {
    let taken = false;
    cam.tile.querySelectorAll("[data-role]").forEach((button) => {
      const on = holds(S.cams[button.dataset.role], cam);
      button.classList.toggle("on", on);
      taken = taken || on;
    });
    cam.tile.classList.toggle("taken", taken);
  }
  $("camRows").replaceChildren(...S.robot.roles.map((role) => {
    const row = S.cams[role];
    const field = (key) => el("div", {}, el("label", { class: "field" }, key), el("input", { value: row[key], oninput: (e) => { row[key] = e.target.value; if (key === "rotate") turnTiles(); else reopenSoon(e.target.value); } }));
    return el("div", { style: "margin-top:12px" }, el("div", { class: "note" }, `${S.labels[role].label}: ${row.value || "no camera yet"}`),
      el("div", { class: "row c4" }, field("width"), field("height"), field("fps"), field("rotate")));
  }));
  turnTiles();
  gate("liveViews", S.robot.roles.filter((role) => !S.cams[role].value).map((role) => [`a camera for ${S.labels[role].label}`, "gallery"]));
}

function cameraRows() { return S.robot.roles.map((role) => ({ ...S.cams[role], name: role })); }

// A width, height or frame rate typed while the live views are shown opens them again with it once typing stops for
// REOPEN_MS, so the feed shows what the camera gives for it; only a whole number does, never a half-typed "1280x".
const REOPEN_MS = 500;
let reopening = 0;
function reopenSoon(value) {
  clearTimeout(reopening);
  if (!/^[1-9]\d*$/.test(String(value).trim()) || $("compare").classList.contains("hidden")) return;
  reopening = setTimeout(liveViews, REOPEN_MS);
}

// The size a camera delivers, from the open's answer (the frame after `rotate`, turned back to the camera's own width
// and height), and a warning when it is not the size typed: a camera that cannot give the mode asked keeps one of its
// own, and the station opens it the same way.
function deliveredSize(row, opened) {
  const [width, height] = Number(row.rotate || 0) % 180 === 90 ? [opened.height, opened.width] : [opened.width, opened.height];
  const same = width === Number(row.width) && height === Number(row.height);
  return [el("div", { class: "cap" }, `Your camera delivers ${width}x${height}.`),
    same ? null : el("div", { class: "err" }, `You typed ${row.width}x${row.height}, but this camera gives ${width}x${height}: a camera that cannot give the size asked keeps one of its own, and the station opens it the same way.`)];
}

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
    const rows = cameraRows();
    let opened;
    try { opened = (await api("POST", "/api/cameras/open", { cameras: rows })).opened; } finally { $("liveViews").disabled = false; }
    const t = Date.now();
    const kinds = S.robot.roles.map(refKind);
    const note = S.status && S.status.cameras.reference_note;
    $("compareNote").textContent = (note ? note + " " : "") + (kinds.includes("example") ? COMPARE_NOTE.example : kinds.includes("yam") ? COMPARE_NOTE.yam : kinds.includes("dataset") ? COMPARE_NOTE.dataset : COMPARE_NOTE.none);
    $("compareRows").replaceChildren(...S.robot.roles.map((role) => el("div", {},
      el("div", { class: "rolehead" }, el("h3", {}, S.labels[role].label), el("span", { class: "err", "data-camstate": role })),
      el("div", { class: "compare" },
        el("div", {}, el("div", { class: "turnbar" }), refFrame(role, t), el("div", { class: "cap" }, refCaption(role))),
        el("div", {}, el("div", { class: "turnbar" }, turnButton(() => rotateCamera(role), Number(S.cams[role].rotate || 0))),
          el("div", { class: "frame" }, liveImg(role, "live"), el("span", { class: "tag" }, "Your camera")),
          deliveredSize(rows.find((row) => row.name === role), opened[role])),
        el("div", {}, el("div", { class: "turnbar" }), el("div", { class: "frame" }, liveImg(role, "model"), el("span", { class: "tag" }, "What the model sees")),
          el("div", { class: "cap" }, "Your frame resized exactly as the dream client sends it."),
          opened[role].warning ? el("div", { class: "cap" }, opened[role].warning) : null)))));
    $("compare").classList.remove("hidden");
    setTimeout(checkSwap, 1500);
  });
}

// A quarter turn clockwise: the same `rotate` the capture settings and the saved file keep, then the views again.
function rotateCamera(role) {
  const row = S.cams[role];
  row.rotate = (Number(row.rotate || 0) + 90) % 360;
  markRoles();
  return liveViews();
}

function refKind(role) {
  const ref = S.status && S.status.cameras.references[role];
  if (ref && (ref.source === "dataset" || ref.source === "none")) return ref.source;
  return S.robot.body === "yam" ? "yam" : "example";
}

// A role's reference picture; a camera of no kind the server knows, with no dataset key, has none.
function refFrame(role, t) {
  if (refKind(role) === "none") return el("div", { class: "frame" }, el("span", { class: "msg" }, "No reference picture"));
  return el("div", { class: "frame" }, el("img", { "data-ref": role, src: `/api/reference?role=${role}&episode=0&t=${t}` }),
    el("span", { class: "tag" }, refKind(role) === "example" ? "Example (YAM)" : "Training data"));
}
const COMPARE_NOTE = {
  dataset: "The left picture is the first frame of your dataset. The closer your camera view is to it, the better your model works.",
  yam: "The left picture is from our YAM training data, which the YAM's model learned from. The closer your camera view is to it, the better it works.",
  example: "The left picture is from our YAM data: only an example of where such a camera sits, not what your model learned from. To compare with your model's own views, give your dataset above.",
  none: "We have no reference pictures for cameras with these names. To compare with your model's own views, give your dataset above and pick each camera's key.",
};

function noDatasetNote() {
  return S.robot.body === "yam" ? "Without one we show our YAM training views." : "Without one we show our YAM views, as an example only.";
}

function refCaption(role) {
  const ref = S.status && S.status.cameras.references[role];
  return { dataset: `First frame of your dataset (${ref && ref.key}).`, yam: "Our YAM training data.", example: "Our YAM data, an example only.",
    none: `We have no picture of a camera named ${role}: give your dataset above, pick its key, then Show live views again.` }[refKind(role)];
}

async function checkSwap() {
  const match = await api("GET", "/api/cameras/match").catch(() => ({ swapped: [] }));
  const box = $("swap");
  if (!match.swapped.length) { box.classList.add("hidden"); return; }
  const [a, b] = match.swapped[0].roles;
  box.replaceChildren(el("span", {}, `The ${S.labels[a].label} and ${S.labels[b].label} look swapped: each looks more like the other's ${refKind(a) === "example" ? "example" : "training"} picture. `),
    el("button", { onclick: () => { const keep = S.cams[a]; S.cams[a] = { ...S.cams[b], name: a }; S.cams[b] = { ...keep, name: b }; markRoles(); liveViews(); } }, "Swap them"));
  box.classList.remove("hidden");
}

async function useDataset() {
  await guard("camErr", async () => {
    const reply = await api("POST", "/api/dataset", { path: $("datasetPath").value });
    S.dataset = reply.path ? reply : null;
    $("datasetNote").textContent = S.dataset ? `${reply.episodes} episodes, cameras: ${reply.video_keys.join(", ")}` : noDatasetNote();
    renderDatasetKeys();
  });
}

function renderDatasetKeys() {
  const box = $("datasetKeys");
  box.classList.toggle("hidden", !S.dataset);
  if (!S.dataset) return;
  box.replaceChildren(...S.robot.roles.map((role) => el("div", {}, el("label", { class: "field" }, S.labels[role].label),
    el("select", { onchange: async (e) => {
      await guard("camErr", () => api("POST", "/api/dataset/keys", { keys: { [role]: e.target.value || null } }));
      document.querySelectorAll(`img[data-ref="${role}"]`).forEach((img) => { img.src = `/api/reference?role=${role}&episode=0&t=${Date.now()}`; });
    } }, el("option", { value: "" }, S.robot.body === "yam" ? "(our YAM data)" : "(our YAM example)"), S.dataset.video_keys.map((key) => el("option", { value: key, selected: S.dataset.keys[role] === key }, key))))));
}

// An SO-101's home pose, drawn once on dream's model, for the operator to seat the arms in (status `home_target`).
function showHomeTarget() {
  const frame = S.status.home_target;
  $("homeTargetBox").classList.toggle("hidden", !frame);
  if (!frame || V.homeTarget) return;
  V.homeTarget = true;
  $("homeTargetView").replaceChildren();
  const view = new ArmView($("homeTargetView"));
  drawing().then((d) => view.load(d, frame)).then(() => { view.live(frame, 0); view.aim(frame, 0); view.label("home pose"); })
    .catch(() => { $("homeTargetBox").classList.add("hidden"); V.homeTarget = false; });
}

function showHome() {
  showHomeTarget();
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
    const got = ["open", "closed"].filter((end) => end in gripper.recorded[side]).map((end) => `${end} at ${gripper.recorded[side][end].toFixed(0)}`);
    const ends = gripper.ends[side];
    const kept = ends.error ? "" : `, runs ${ends.gripper_closed_deg.toFixed(0)} to ${ends.gripper_open_deg.toFixed(0)} degrees`;
    return `${side === "left" ? "Left" : "Right"} gripper: ${got.join(", ") || "nothing recorded yet"}${kept}.`;
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
  return `Home: left [${fmt("left")}], right [${fmt("right")}] degrees.${grips}`;
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
    summary(3, homeLine(reply.home, gripper), true);
    const out = reply.out_of_range.map(outsideLine);
    if (!out.length) { go(S.reach ? 4 : 5); return; }
    $("homeErr").replaceChildren(`${out.join(" ")} Fix that and press OK again, or `,
      el("button", { onclick: () => go(S.reach ? 4 : 5) }, "Continue anyway"));
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
      : reach.hands[side] ? "Nothing painted yet." : "No reading from this arm.";
    $(`reachClear-${side}`).disabled = !points.length;
    try { drawReach(side, reach); } catch (exc) {
      $("reachWarn").textContent = `The drawing failed (${exc.message}). You can still continue or skip this step.`;
    }
  }
  // The hand at home alone would be a 4 cm disk the arm can barely move in: Continue waits for a real area.
  const least = Math.round(reach.min_area_m2 * 1e4);
  const enough = SIDES.every((side) => areaCm2(reach.points[side], reach) >= least);
  gate("reachDone", SIDES.filter((side) => areaCm2(reach.points[side], reach) < least).map((side) => [`${least} cm² for the ${side} arm`, `reachCanvas-${side}`]));
  $("reachHint").textContent = enough ? "" : `Paint at least ${least} cm² per arm to continue, or skip this step.`;
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
  return `Painted: left arm ${one("left")}, right arm ${one("right")}.`;
}

// Step 5: the powered checks. Each drawing follows its arm's readings, the joint being nudged highlighted; at the end
// you say whether each joint and the hand moved like the drawings.
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
    $("motionPhase").textContent = "";
    $("motionGo").classList.remove("hidden");
    $("motionGo").textContent = "Start";
    for (const id of ["motionBanner", "release", "eyes", "motionCause", "motionRetry", "motionDone"]) $(id).classList.add("hidden");
    setKids($("results"), "", () => []);
    return;
  }
  setText($("motionPhase"), active ? `Now: ${motion.phase}` : "");
  const groups = { hold: "Hold", nudge: "Nudge", sweep: "Sweep", watchdog: "Follow", limit: "Limit", eyes: "Eyes" };
  setKids($("results"), JSON.stringify([motion.results, motion.notes]), () => [
    ...motion.results.map((row) => el("div", { class: "result" },
      el("span", { class: row.finding.state === "pass" ? "ok" : "no" }, row.finding.state === "pass" ? "✓" : "!"),
      el("span", { class: "grp" }, `${groups[row.group]} ${row.side}`),
      el("span", {}, row.finding.message))),
    // Each note is `{"side", "text"}` (robots.setup_motion), said with the arm it is about.
    ...motion.notes.map((note) => el("div", { class: "note" }, `${note.side ? `${note.side[0].toUpperCase()}${note.side.slice(1)} arm: ` : ""}${note.text}`))]);
  const fail = renderCause(motion.results);
  // The card names a failed check; the banner then says only what the arms do.
  const why = fail ? "" : `Stopped on a failed check: ${motion.error}. `;
  const banner = $("motionBanner");
  const limp = "The motors are off.";
  const failed = motion.let_go ? `${why}${limp} If an arm still moves, press your emergency stop.`
    : driven(status) ? (fail ? null : `${why}The arms hold where they stopped.`) : `${why}${limp}`;
  const lost = Object.keys(status.not_answering);
  const stuck = stillHoldingLine(status);
  const parked = Object.keys(status.parked);
  const end = running ? null
    : parking ? ["banner bad", `${why}${PARKING}`]
    : ending ? ["banner bad", `${motion.error ? why : ""}Ending the powered checks; keep clear for a moment.`]
    : stuck ? ["banner bad", `${stuck} Then find the arms again on step 1.`]
    : Object.keys(status.off_failed).length ? ["banner bad", offFailedLine(status)]
    : lost.length
    ? ["banner bad", `The ${lost.join(" and ")} arm stopped answering under power: ${S.state.not_answering_help} Then find the arms again on step 1 (Close the arms and look again).`]
    : parked.length ? ["banner bad", `${motion.stop_cut ? `${STOP_CUT} ` : ""}${parkedLine(status)} Then find the arms again on step 1 (Close the arms and look again).`]
    : {
      done: ["banner good", `Every powered check passed. ${driven(status) ? "Both arms hold at home." : limp}`],
      failed: ["banner bad", failed],
      stopped: ["banner", `Stopped. ${driven(status) ? "Both arms hold where they are." : limp}`],
    }[motion.state];
  banner.classList.toggle("hidden", !(end && end[1]));
  if (end && end[1]) { banner.className = end[0]; setText(banner, end[1]); }
  $("release").classList.toggle("hidden", active || ending || !driven(status));
  if (driven(status)) {
    const both = status.holding.length === 2;
    // With the station file's home_before_motors_off the arms go home under power first.
    const limp = status.home_before_off ? (both ? "They go home, then limp." : "It goes home, then limp.") : (both ? "They go limp." : "It goes limp.");
    setText($("releaseText"), both ? `Hold the arms first. ${limp}` : `Hold the ${status.holding[0]} arm first. ${limp}`);
  }
  const asking = motion.state === "done" && !motion.results.some((row) => row.group === "eyes");
  $("eyes").classList.toggle("hidden", ending || !asking);
  // After a run that did not pass, running again is the way on; continuing asks first (skipAsk).
  const settled = !active && !ending;
  const retry = settled && motion.state !== "done";
  $("motionGo").classList.toggle("hidden", !settled || retry);
  $("motionGo").textContent = "Run again";
  $("motionRetry").classList.toggle("hidden", !retry);
  $("motionDone").classList.toggle("hidden", !settled || asking);
  $("motionDone").classList.toggle("primary", !retry);
  $("motionDone").textContent = retry ? "Continue anyway" : "Continue";
}

// Which arm and joint a result is about: "Left arm, joint 3".
function failedWhere(row) {
  const arm = row.side === "both" ? "Both arms" : `${row.side[0].toUpperCase()}${row.side.slice(1)} arm`;
  return `${arm}${row.finding.joint == null ? "" : `, joint ${row.finding.joint + 1}`}`;
}

// The failed check's one cause, if a check failed: which arm and joint, what it is, what it does to the robot, the
// fix, and whether continuing past it is safe.
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

// What the powered checks left undone, for Continue anyway and Save, so nobody continues without knowing what
// failed and whether that is safe: each failed check as "where: what was seen
// (cause)", and whether continuing is safe, which it is only when every failed check says so.
function motionShortfall(motion) {
  const fails = motion ? motion.results.filter((row) => row.finding.state === "fail") : [];
  const lines = fails.map((row) => `${failedWhere(row)}: ${row.finding.message} (${row.finding.title.replace(/\.$/, "")})`);
  if (!fails.length) {
    const why = !motion ? "The powered checks did not run." : motion.error ? `The powered checks stopped: ${motion.error}.` : "The powered checks did not finish.";
    return { lines: [why], goOn: "Run them before you continue: until they pass, nothing says your robot moves the way the model sends it.", safe: false };
  }
  const unsafe = fails.find((row) => !row.finding.safe);
  return { lines, goOn: (unsafe || fails[0]).finding.go_on, safe: !unsafe };
}

// Step 6: save.
async function save() {
  await busy([$("saveBtn")], () => guard("saveErr", async () => {
    const reply = await api("POST", "/api/save", { cameras: cameraRows() });
    const missing = reply.live_missing.length ? ` The live view is off until you install the live extra (${reply.live_missing.join(", ")}) and save again.` : "";
    $("savedLine").textContent = `Saved ${reply.config_path}. This station is ${reply.robot.name} (${reply.robot.id}) on ${reply.platform}.${missing}`;
    $("savedYaml").textContent = reply.yaml;
    $("saved").classList.remove("hidden");
  }));
}

function renderCamStates(states) {
  for (const box of document.querySelectorAll("[data-camstate]")) {
    const state = states[box.dataset.camstate];
    box.textContent = state && state.state === "reconnecting"
      ? `Stopped sending pictures (${state.error}). Opening it again${state.tries ? `, try ${state.tries}` : ""}; check its cable.`
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
    driven(status) ? `${arms} under power from the powered checks. Hold ${them} up with your hands, then` : "Then find the arms again on step 1."].join(" ").trim());
  $("holdRelease").classList.toggle("hidden", !driven(status));
}

const PARKING = "Keep clear while it moves. STOP lets both arms go limp at once, so they drop unless you hold them.";
const STOP_CUT = "STOP let both arms go limp where they were (motors off), so they drop unless held: support them.";

function parkedLine(status) { return Object.values(status.parked).join(" "); }

function offFailedLine(status) { return Object.values(status.off_failed).join(" "); }

function stillHoldingLine(status) {
  return Object.values(status.still_holding).map((words) => `${words[0].toUpperCase()}${words.slice(1)}.`).join(" ");
}

function driven(status) { return status.holding.length > 0; }

function holding(status) {
  return status.holding.length === 2 ? ["Both arms still hold", "them"] : [`The ${status.holding[0]} arm still holds`, "it"];
}

function showGone(gone) {
  S.gone = gone;
  $("gonebar").textContent = GONE;
  $("gonebar").classList.toggle("hidden", !gone);
  if (gone) {
    for (const id of ["stopbar", "holdbar", "release", "motionBanner", "motionGo", "motionRetry"]) $(id).classList.add("hidden");  // B14b, B38
    $("motionPhase").textContent = "";
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
      if (S.step === 1) { renderWiggle(S.status); renderCalibration(S.status); }
      if (S.step === 2) renderCamStates(S.status.cameras.states);
      if (S.step === 3) { showHome(); renderGripper(S.status.gripper); }
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
  $("agent").classList.toggle("hidden", !S.state.agent_running);
  const c = S.state.connection;
  $("account").textContent = `Account: ${S.state.credentials_path}. Platform ${c.platform_url}, gateway ${c.gateway_url}.`;
  $("version").textContent = `Version ${S.state.version}`;  // the commit the wheel was built from, after its "+g"
  if (S.state.tool === "tcp") return startTcp();
  // A robot of your own: the file's driver and arm on a re-run, else none.
  const ik = S.state.existing_ik;
  S.armWay = !ik ? null : ik.endsWith(".json") ? "urdf" : "ours";
  if (S.armWay === "urdf") {
    S.urdfIk = ik;
    $("urdfDrawn").textContent = `Using your station file's arm, ${ik.split("/").pop()}. Give your URDF again to change it.`;
  }
  if (S.armWay === "ours") S.ourIk = ik;
  S.driver = S.state.existing_driver || "";
  if (!/\.py:\w+$/.test(S.driver)) $("customDriver").value = S.driver;
  const arms = S.state.ik_arms.map((arm) => arm.label);
  $("armOursSub").textContent = arms.length > 3 ? `${arms.slice(0, 3).join(", ")} and ${arms.length - 3} more` : arms.join(", ");
  $("exampleLink").href = `${c.platform_url}/downloads/examples/seeed_rebot/seeed_rebot.py`;
  $("docsLink").href = `${c.platform_url}/docs/robot-of-your-own/`;
  $("customRun").textContent = `dreamcli robot run --config ${S.state.config_path} --checkpoint "<your model>" --task "..." --dry-run`;
  if (S.state.ik_error) $("pickErr").textContent = S.state.ik_error;
  $("saveWhere").textContent = `Writes ${S.state.config_path}${S.state.exists ? ", replacing the file there" : ""}.`;
  S.pick = S.state.existing_body;
  S.so101Files = S.state.so101_calibrations.map((file) => ({ ...file, on: file.matches.length > 0 }));
  // A file a plugged-in arm already holds is shown, ticked: that arm keeps it and is not calibrated again.
  if (S.so101Files.some((f) => f.on)) $("so101Own").open = true;
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
  $("intro").textContent = "Give your arm's URDF, say where it ends, and mark where your gripper's fingertips close. Set tool center point writes the arm file your station file's custom.ik names.";
  document.querySelector('section.card[data-step="0"] h2').textContent = "Your arm's URDF";
  $("custom").classList.remove("hidden");
  $("urdfBox").classList.remove("hidden");
  go(0);
}

$("pickRobot").onclick = pickRobot;
$("wayWizard").onclick = () => pickWay("wizard");
$("waySelf").onclick = () => pickWay("self");
$("embodiment").onchange = renderRobots;
$("driverChoose").onclick = () => $("driverDialog").showModal();
$("driverDialogClose").onclick = () => $("driverDialog").close();
$("driverUp").onclick = () => guard("pickErr", () => openFolder(D.listing.parent));
$("driverUse").onclick = () => { if ($("customDriver").value.trim()) pickDriver($("customDriver").value.trim()); };
$("armUrdf").onclick = () => { S.armWay = "urdf"; renderRobots(); };
$("armOurs").onclick = () => { S.armWay = "ours"; renderRobots(); };
$("urdfFlip").onclick = flipUp;
buildPad();
$("addArm").onclick = typeArm;
$("addCam").onclick = typeCamera;
$("urdfFile").onchange = readUrdf;
$("urdfFolder").onchange = readUrdfFolder;
$("urdfAdd").onclick = () => addUrdf("calibrated");
$("urdfSkip").onclick = () => { $("tcpSkipTick").checked = false; $("tcpSkipGo").disabled = true; $("tcpSkip").showModal(); };
$("tcpSkipTick").onchange = (e) => { $("tcpSkipGo").disabled = !e.target.checked; };
$("tcpSkipBack").onclick = () => $("tcpSkip").close();
$("tcpSkipGo").onclick = () => { $("tcpSkip").close(); addUrdf("skipped"); };
$("urdfYes").onclick = () => guard("pickErr", confirmEnd);
$("urdfBaseBtn").onclick = () => { U.picking = "base"; renderUrdf(); };
$("urdfReset").onclick = resetMarker;
$("urdfBack").onclick = () => { forgetArm(); U.ok = false; renderUrdf(); };
$("scanArms").onclick = () => (S.opened ? lookAgain() : scanArms());
$("openArms").onclick = openArms;
$("lookAgain").onclick = lookAgain;
$("armsDone").onclick = () => {
  if (refused("armsDone")) return;
  summary(1, armsLine(S.status.assigned), true);
  go(2);
};
$("shootAll").onclick = shootAll;
$("liveViews").onclick = () => { if (!refused("liveViews")) liveViews(); };
$("useDataset").onclick = useDataset;
$("camsDone").onclick = async () => {
  S.status = await api("GET", "/api/status");
  summary(2, camsLine(), true);
  go(3);
};
$("homeOk").onclick = homeOk;
for (const side of SIDES) {
  $(`reachClear-${side}`).onclick = () => guard("reachErr", async () => {
    await api("POST", "/api/reach/clear", { side });
    S.reachSpan[side] = 0;
  });
}
$("reachDone").onclick = () => {
  if (refused("reachDone")) return;
  summary(4, reachLine(S.status.reach), true);
  go(5);
};
$("reachSkip").onclick = () => $("reachAsk").showModal();
$("reachBack").onclick = () => $("reachAsk").close();
$("reachSkipGo").onclick = () => guard("reachErr", async () => {
  $("reachAsk").close();
  await api("POST", "/api/reach/skip", {});
  summary(4, "Skipped: the arms can move anywhere.", true);
  go(5);
});
$("gripOpen").onclick = () => recordGripper("open");
$("gripClosed").onclick = () => recordGripper("closed");
$("so101AddBtn").onclick = () => {
  const path = $("so101Add").value.trim();
  if (path && !S.so101Files.some((f) => f.path === path)) S.so101Files.push({ path, on: true });
  $("so101Add").value = "";
  renderRobots();
};
$("motionGo").onclick = $("motionRetry").onclick = () => { $("safeTick").checked = false; $("safeGo").disabled = true; $("safety").showModal(); };
$("safeTick").onchange = (e) => { $("safeGo").disabled = !e.target.checked; };
$("safeCancel").onclick = () => $("safety").close();
$("safeGo").onclick = async () => {
  $("safety").close();
  await guard("motionErr", () => api("POST", "/api/motion/start", { confirm: true }));
};
$("stopBtn").onclick = () => api("POST", "/api/motion/stop").catch((exc) => {
  if (exc instanceof Gone) showGone(true); else $("stopPhase").textContent = exc.message;
});
$("releaseBtn").onclick = () => guard("motionErr", () => api("POST", "/api/motion/release"));
$("holdRelease").onclick = () => guard("holdErr", async () => {
  await api("POST", "/api/motion/release");
  $("homeErr").textContent = "";
});
for (const [id, same] of [["eyesYes", true], ["eyesNo", false]]) {
  $(id).onclick = () => busy([$("eyesYes"), $("eyesNo")], () => guard("motionErr", () => api("POST", "/api/motion/eyes", { same })));
}
function continueToSave() {
  const m = S.status.motion;
  const ok = m && m.state === "done";
  summary(5, ok ? "Every nudge and sweep move passed, and each joint and the hand moved like the drawings." : "The powered checks did not all pass.", true);
  const warn = [];
  if (!ok) {
    const short = motionShortfall(m);
    warn.push(`The powered checks did not all pass. ${short.lines.join(" ")} ${short.goOn}`);
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
  makers: "Finishing switches the arms off. Each Maker Arm first moves to its rest pose (every joint at zero) and opens its gripper, then goes limp and drops. Keep clear while it moves, then hold both arms up. Finish now?",
  metal: "Finishing switches the arms off. Each arm first moves back to where it was when the powered checks started and opens its gripper, then goes limp and drops. Keep clear while it moves, then hold both arms up. Finish now?",
  so101: "Finishing switches the arms off. Each arm first moves back to where it was when the powered checks started and opens its gripper, then goes limp and drops. Keep clear while it moves, then hold both arms up. Finish now?",
};
// What to do once setup is done: the command to copy, and Live view, where the robot is driven from.
function nextSteps() {
  const live = `${S.state.connection.platform_url}/live`;
  return [
    el("p", {}, "Next, start the station on this computer:"),
    el("div", { class: "codeblock" }, el("pre", {}, el("code", { id: "nextCmd" }, "dreamcli robot up")),
      el("button", { class: "copy", "data-copy": "nextCmd" }, "Copy")),
    el("p", {}, "Then press Start in Live view:"),
    el("div", { class: "actions" }, el("a", { class: "go", href: live, target: "_blank", rel: "noopener" }, "Open Live view")),
    el("p", { class: "note" }, "Or one run from the terminal: dreamcli robot run --task \"<what to do>\" (--dry-run first to check without moving). You can close this page."),
  ];
}
$("finish").onclick = () => busy([$("finish")], async () => {
  const status = await api("GET", "/api/status");
  const ask = driven(status) ? FINISH_POWERED[S.robot.body] || "Finishing switches the powered arms off. Hold them up first. Finish now?" : null;
  if (ask && !window.confirm(ask)) return;
  await api("POST", "/api/exit");
  S.finished = true;
  const moving = driven(status) ? el("p", { class: "banner bad" }, "The arms are moving to rest and switching off: keep clear until they stop.") : null;
  const stuck = stillHoldingLine(status) ? el("p", { class: "banner bad" }, stillHoldingLine(status)) : null;
  document.querySelector("main").replaceChildren(...[el("h1", {}, "Done"), moving, stuck, ...nextSteps()].filter(Boolean));
});
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
