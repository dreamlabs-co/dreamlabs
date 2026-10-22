// Runs the setup page's own script (src/robots/setup_wizard.js, or the redesigned page's setup_wizard_v2.js) under
// node on a stub page, for tests/test_setup_wizard.py.
//
//   node tests/setup_page.mjs scenario.json
//
// The scenario names the script (optional "script", setup_wizard.js when left out) and gives the routes' replies by "METHOD /path" (the path without its query), or "base", the address of a
// running wizard that answers every request with no scripted reply (each step then waits for its answers); state to set on the page and field values by id
// (optional "values") before the call, the page function to call (optional; "args" to pass it), the buttons the page
// drew to press after it (optional "press", each [id, selector, n]: the n-th match of the selector inside that
// element), the buttons to click after those (optional "click", by id, in order), page functions to call last
// (optional "after", each a name or [name, ...args], or {"press": [id, selector, n]}, or {"type": [id, selector, n,
// value]} to type a value into a field the page drew, or {"input": [id, selector, n, value]} to type it keystroke-wise
// (its input event, not its change), or {"wait": ms} to run once each timer the page set for exactly ms, or
// {"click": id}, or {"tick": id} to tick a checkbox by id), and
// the element ids to report. A file input's file is optional "files", {id: name}, and the .zip or folder it is, its
// files, is optional "zip", {path in the zip or folder: text}. Every element id exists and starts empty; nothing is
// laid out and no timer fires but by "wait", so a feed or a poll runs once. Without "base" a request with no scripted reply answers
// 404, which the page shows as an error; a request without options is a GET, as in a browser (the camera pictures),
// and a scripted reply is also a blob. Prints `{"ids": {id: {text, children, hidden, disabled, blocked, missing, ticked, value, turned
// (the CSS transforms set inside it), open (a dialog shown)}}, "warnings": {id: [the texts of its class "err" descendants]}, "requests":
// [{key, body}], "errors": [what the page logged with console.error, promises it left rejected, and each "<id>: text"
// it wrote to an element whose id ends in Err, the page's error lines, though a later step cleared it], "summaries": {step: its
// summary line}}`.
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const scenario = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const errors = [];
console.error = (...parts) => errors.push(parts.map((part) => (part instanceof Error ? part.stack : String(part))).join(" "));
process.on("unhandledRejection", (reason) => errors.push(reason instanceof Error ? reason.stack : String(reason)));

class FakeNode {}
class FakeText extends FakeNode {
  constructor(text) { super(); this.textContent = String(text); }
}
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.id = "";
    this.kids = [];
    this.attrs = {};
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.open = false;
    this.found = {};
    this.on = {};
    const classes = () => new Set(this.className.split(" ").filter(Boolean));
    const write = (set) => { this.className = [...set].join(" "); };
    this.classList = {
      contains: (c) => classes().has(c),
      add: (c) => { const s = classes(); s.add(c); write(s); },
      remove: (c) => { const s = classes(); s.delete(c); write(s); },
      toggle: (c, on) => { const s = classes(); if (on ?? !s.has(c)) s.add(c); else s.delete(c); write(s); },
    };
  }
  get textContent() { return this.kids.map((kid) => kid.textContent).join(""); }
  set textContent(text) {
    this.kids = text === "" ? [] : [new FakeText(text)];
    if (text !== "" && /Err$/.test(this.id)) errors.push(`${this.id}: ${text}`);
  }
  get isConnected() { return false; }
  setAttribute(name, value) {
    this.attrs[name] = String(value);
    if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase())] = String(value);
    if (name === "src") this.src = String(value);
  }
  getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
  addEventListener(type, handler) { this.on[type] = handler; }
  append(...kids) { for (const kid of kids) this.kids.push(kid instanceof FakeNode ? kid : new FakeText(kid)); }
  replaceChildren(...kids) { this.kids = []; this.append(...kids); }
  remove() {}
  scrollIntoView() {}
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
  // The page's static markup is not here: a selector answers an empty element of its own, the same one each time.
  querySelector(selector) { return (this.found[selector] ||= new FakeElement("div")); }
  querySelectorAll(selector) { return descendants(this).filter(matcher(selector)); }
}

function descendants(node) {
  return node.kids.filter((kid) => kid instanceof FakeElement).flatMap((kid) => [kid, ...descendants(kid)]);
}

// `tag`, `.class`, `[attr]` and `[attr="value"]`, joined: the selectors the page's querySelectorAll calls use.
function matcher(selector) {
  const m = /^(\w+)?(?:\.([\w-]+))?(?:\[([\w-]+)(?:="([^"]*)")?\])?$/.exec(selector);
  if (!m) throw new Error(`stub page: no selector ${selector}`);
  const [, tag, cls, attr, value] = m;
  return (node) => (!tag || node.tagName === tag.toUpperCase()) && (!cls || node.classList.contains(cls))
    && (!attr || (attr in node.attrs && (value === undefined || node.attrs[attr] === value)));
}

// On the document also `#id`, `#id <selector>` and lists of them joined by commas: v2 marks its pressed buttons over
// such a list (PICK_GROUPS).
function select(selector) {
  return selector.split(",").map((part) => part.trim()).flatMap((part) => {
    const m = /^#([\w-]+)(?: (.+))?$/.exec(part);
    if (!m) return root.querySelectorAll(part);
    const found = globalThis.document.getElementById(m[1]);
    return m[2] ? descendants(found).filter(matcher(m[2])) : [found];
  });
}

const byId = {};
const root = new FakeElement("html");
globalThis.Node = FakeNode;
globalThis.document = {
  getElementById: (id) => {
    if (!byId[id]) { byId[id] = new FakeElement("div"); byId[id].id = id; root.append(byId[id]); }
    return byId[id];
  },
  createElement: (tag) => new FakeElement(tag),
  querySelector: (selector) => root.querySelector(selector),
  querySelectorAll: (selector) => select(selector),
  addEventListener() {},
};
// Timers are kept, never fired on their own: a "wait" step runs those set for its delay. They are the page's alone
// (its module declares setTimeout and clearTimeout as these, below): node's own fetch sets timers of its own and
// calls their unref(), so the global ones stay node's.
const timers = [];
globalThis.pageTimers = {
  set: (fn, ms) => timers.push({ fn, ms }),
  clear: (id) => { if (timers[id - 1]) timers[id - 1].fn = null; },
};
globalThis.requestAnimationFrame = (fn) => fn();
// Nothing is laid out or restyled, so no change is observed (v2 mirrors its pressed buttons' aria-pressed this way).
globalThis.MutationObserver = class { observe() {} };

const requests = [];
// The running wizard's answers, read whole before the page sees them, and those still on their way.
const served = globalThis.fetch;
const inflight = new Set();
async function answer(where, init) {
  const reply = await served(scenario.base + where, init);
  const body = await reply.arrayBuffer();
  const type = reply.headers.get("content-type");
  return { ok: reply.ok, status: reply.status, json: async () => JSON.parse(new TextDecoder().decode(body)), blob: async () => new Blob([body], { type }) };
}
globalThis.fetch = async (where, init = { method: "GET" }) => {
  const key = `${init.method} ${where.split("?")[0]}`;
  requests.push({ key, body: init.body === undefined ? null : JSON.parse(init.body) });
  const scripted = key in scenario.replies;
  if (!scripted && scenario.base) {
    const asked = answer(where, init);
    inflight.add(asked);
    try { return await asked; } finally { inflight.delete(asked); }
  }
  const reply = scripted ? scenario.replies[key] : { error: `stub page: no reply scripted for ${key}` };
  return { ok: scripted, status: scripted ? 200 : 404, json: async () => reply, blob: async () => new Blob([JSON.stringify(reply)]) };
};

const here = path.dirname(url.fileURLToPath(import.meta.url));
const script = scenario.script ?? "setup_wizard.js";
const source = fs.readFileSync(path.join(here, "..", "src", "robots", script), "utf8");
const viewer = {
  "setup_wizard.js": 'import { ArmView, MARKER_ROOM, UrdfView, folderFiles, unzip } from "/setup_viewer.js";',
  "setup_wizard_v2.js": 'import { ArmView, MARKER_ROOM, UrdfView, cadNavigation, folderFiles, unzip } from "/setup_viewer.js";',
}[script];
if (!source.includes(viewer)) throw new Error(`${script} no longer imports the 3D view as this stub expects`);
const after = (scenario.after ?? []).map((step) => (Array.isArray(step) || typeof step === "object" ? step : [step]));
const called = [scenario.call, ...after.filter(Array.isArray).map(([name]) => name)].filter(Boolean);
// The 3D views draw nothing here: each takes the calls the page makes and does nothing.
const view = "const setTimeout = globalThis.pageTimers.set, clearTimeout = globalThis.pageTimers.clear;\n"
  + "const MARKER_ROOM = null;\nclass ArmView { async load() {} aim() {} label() {} live() {} highlight() {} mark() {} }\n"
  + "class UrdfView extends ArmView { async tree() { return { shown: [], plain: [], missing: [] }; } paint() {} setMarker() {} }\n"
  + "const cadNavigation = () => {};\n"
  + "const unzip = () => new Map(Object.entries(globalThis.zipped).map(([p, t]) => [p, async () => new TextEncoder().encode(t)]));\n"
  + "const folderFiles = async () => unzip();";
globalThis.zipped = scenario.zip ?? {};
const page = `${source.replace(viewer, view)}\nexport { S, ${[...new Set(called)].join(", ")} };\n`;
const loaded = await import(`data:text/javascript;base64,${Buffer.from(page).toString("base64")}`);
const settle = async () => {
  do {
    await Promise.allSettled([...inflight]);
    for (let i = 0; i < 20; i += 1) await new Promise((done) => setImmediate(done));
  } while (inflight.size);
};
await settle();  // start(): GET /api/state, then the first poll
Object.assign(loaded.S, scenario.set);
for (const [id, value] of Object.entries(scenario.values ?? {})) globalThis.document.getElementById(id).value = value;
for (const [id, name] of Object.entries(scenario.files ?? {})) {
  const file = { name, webkitRelativePath: `${name}/`, text: async () => "", arrayBuffer: async () => new ArrayBuffer(0) };
  globalThis.document.getElementById(id).files = [file];
}
if (scenario.call) await loaded[scenario.call](...(scenario.args ?? []));
await settle();
const drawn = (id, selector, n) => descendants(globalThis.document.getElementById(id)).filter(matcher(selector))[n];
for (const [id, selector, n] of scenario.press ?? []) {
  await drawn(id, selector, n).on.click();
  await settle();
}
for (const id of scenario.click ?? []) {
  await globalThis.document.getElementById(id).onclick();
  await settle();
}
for (const step of after) {
  if (step.press) await drawn(...step.press).on.click();
  else if (step.click) await globalThis.document.getElementById(step.click).onclick();
  else if (step.tick) {
    const box = globalThis.document.getElementById(step.tick);
    box.checked = true;
    await box.onchange({ target: box });
  } else if (step.input) {
    const [id, selector, n, value] = step.input;
    const field = drawn(id, selector, n);
    field.value = value;
    await field.on.input({ target: field });
  } else if (step.wait) {
    for (const timer of timers.filter((t) => t.fn && t.ms === step.wait)) {
      const { fn } = timer;
      timer.fn = null;
      await fn();
    }
  } else if (step.type) {
    const [id, selector, n, value] = step.type;
    const field = drawn(id, selector, n);
    field.value = value;
    await field.on.change({ target: field });
  } else await loaded[step[0]](...step.slice(1));
  await settle();
}

const ids = Object.fromEntries(scenario.show.map((id) => {
  const node = globalThis.document.getElementById(id);
  const ticked = descendants(node).filter((kid) => kid.attrs.type === "checkbox" && "checked" in kid.attrs).length;
  return [id, {
    text: node.textContent, children: node.kids.length, hidden: node.classList.contains("hidden"), disabled: node.disabled,
    blocked: node.classList.contains("blocked"), missing: node.classList.contains("missing"),
    ticked, value: node.value, turned: descendants(node).map((kid) => kid.style.transform).filter(Boolean),
    open: node.open,
  }];
}));
const warnings = Object.fromEntries(Object.entries(byId).map(([id, node]) => [
  id, descendants(node).filter((kid) => kid.classList.contains("err")).map((kid) => kid.textContent).filter(Boolean),
]).filter(([, texts]) => texts.length));
const summaries = Object.fromEntries(Object.entries(root.found).map(([sel, node]) => [/data-step="(\d+)"/.exec(sel)?.[1], node.textContent]).filter(([step]) => step));
process.stdout.write(JSON.stringify({ ids, warnings, requests, errors, summaries }));
