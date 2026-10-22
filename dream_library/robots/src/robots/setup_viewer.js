// The setup wizard's 3D view of an arm: dream's model of it, with the maker's meshes on each link or simple shapes.
// three.js comes from the CDN the page's import map pins (setup_wizard.html).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Meshes load once for the whole page; every view shares them.
const loader = new STLLoader();
const meshes = new Map();
const mesh = (url) => meshes.get(url) || meshes.set(url, loader.loadAsync(url)).get(url);

// A pose from the server is [qw, qx, qy, qz, x, y, z].
const matrix = (p) => new THREE.Matrix4().compose(new THREE.Vector3(p[4], p[5], p[6]), new THREE.Quaternion(p[1], p[2], p[3], p[0]), new THREE.Vector3(1, 1, 1));
const clamp01 = (g) => Math.min(1, Math.max(0, g));
const UP = new THREE.Vector3(0, 1, 0);
// The room a view with a tool marker leaves around the arm for the marker and its labels, a fraction of the arm's size.
export const MARKER_ROOM = 0.65;

// The view every robot opens at: from the side, along the base's y axis, 15 degrees from above, so the base is on the
// left and forward (+x) on the right. Orbit stays free.
const SIDE_VIEW = new THREE.Vector3(0, -Math.cos(THREE.MathUtils.degToRad(15)), Math.sin(THREE.MathUtils.degToRad(15)));

// One arm: a group per link, posed per frame; meshes or shapes hang on the links, the jaws move with the grip.
class Arm {
  constructor(drawing, body) {
    this.drawing = drawing;
    this.body = body;
    this.root = new THREE.Group();
    this.links = drawing.links.map(() => { const g = new THREE.Group(); this.root.add(g); return g; });
    this.pieces = [];
    this.grip = 1;
  }

  async build(frame) {
    const d = this.drawing;
    if (d.kind === "meshes") {
      const parts = d.parts.concat(d.jaws);
      const geometries = await Promise.all(parts.map((part) => mesh(part.url)));
      parts.forEach((part, i) => {
        const object = new THREE.Mesh(geometries[i], this.body);
        object.matrixAutoUpdate = false;
        object.matrix.copy(matrix(part.pose));
        this.add(object, part.link, part.move ? { ...part, base: object.matrix.clone() } : null);
      });
    } else this.shapes(frame);
  }

  add(object, link, jaw) {
    object.castShadow = true;
    this.links[link].add(object);
    this.pieces.push({ object, link, jaw });
  }

  // The arm from dream's model alone: a capsule from each link to the next, a disc on each joint, two jaws at the tool.
  shapes(frame) {
    const d = this.drawing;
    const local = (parent, child) => new THREE.Vector3(...frame[child].slice(4)).applyMatrix4(matrix(frame[parent]).invert());
    const bones = d.segments.map(([p, c]) => [p, local(p, c)]).filter(([, v]) => v.length() > 0.002);
    const reach = bones.reduce((sum, [, v]) => sum + v.length(), 0);
    const r = Math.min(0.025, Math.max(0.008, reach * 0.03));
    const fixed = (object, link) => { object.matrixAutoUpdate = false; object.updateMatrix(); this.add(object, link, null); };
    for (const [p, v] of bones) {
      const bone = new THREE.Mesh(new THREE.CapsuleGeometry(r, v.length(), 6, 16), this.body);
      bone.position.copy(v).multiplyScalar(0.5);
      bone.quaternion.setFromUnitVectors(UP, v.clone().normalize());
      fixed(bone, p);
    }
    for (const joint of d.joints) {
      const disc = new THREE.Mesh(new THREE.CylinderGeometry(r * 1.7, r * 1.7, r * 1.4, 24), this.body);
      disc.quaternion.setFromUnitVectors(UP, new THREE.Vector3(...joint.axis));
      fixed(disc, joint.link);
    }
    // dream's tool frame: z along the fingers, x across the jaws.
    for (const side of [1, -1]) {
      const finger = new THREE.Mesh(new THREE.BoxGeometry(r * 0.6, r * 1.2, r * 3), this.body);
      finger.position.set(side * r * 0.7, 0, -r * 1.5);
      finger.matrixAutoUpdate = false;
      finger.updateMatrix();
      this.add(finger, d.tool_link, { move: "slide", axis: [side, 0, 0], pivot: [0, 0, 0], travel: r * 2, base: finger.matrix.clone() });
    }
  }

  pose(frame, grip) {
    frame.forEach((p, i) => { this.links[i].quaternion.set(p[1], p[2], p[3], p[0]); this.links[i].position.set(p[4], p[5], p[6]); });
    this.setGrip(grip);
  }

  setGrip(grip) {
    this.grip = clamp01(grip);
    for (const { object, jaw } of this.pieces) {
      if (!jaw) continue;
      const d = jaw.travel * this.grip;
      const axis = new THREE.Vector3(...jaw.axis);
      const move = jaw.move === "slide"
        ? new THREE.Matrix4().makeTranslation(axis.multiplyScalar(d))
        : new THREE.Matrix4().makeTranslation(...jaw.pivot).multiply(new THREE.Matrix4().makeRotationAxis(axis.normalize(), d)).multiply(new THREE.Matrix4().makeTranslation(-jaw.pivot[0], -jaw.pivot[1], -jaw.pivot[2]));
      object.matrix.multiplyMatrices(move, jaw.base);
    }
  }

  // Eases every link toward a frame: `f` 1 lands on it.
  toward(frame, grip, f) {
    const q = new THREE.Quaternion(), p = new THREE.Vector3();
    frame.forEach((pb, i) => {
      this.links[i].quaternion.slerp(q.set(pb[1], pb[2], pb[3], pb[0]), f);
      this.links[i].position.lerp(p.set(pb[4], pb[5], pb[6]), f);
    });
    this.setGrip(this.grip + (clamp01(grip) - this.grip) * f);
  }
}

export class ArmView {
  // box: an empty element with a size; the view fills it.
  constructor(box) {
    this.box = box;
    this.canvas = box.appendChild(document.createElement("canvas"));
    this.chip = box.appendChild(document.createElement("span"));
    this.chip.className = "chip hidden";
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 4 / 3, 0.01, 20);
    this.camera.up.set(0, 0, 1);
    this.camera.position.copy(SIDE_VIEW);
    this.controls = new OrbitControls(this.camera, this.canvas);
    Object.assign(this.controls, { enableZoom: false, enablePan: false, enableDamping: true });
    const sky = new THREE.HemisphereLight(0xffffff, 0x8a7d85, 1.7);
    sky.position.set(0, 0, 1);
    this.key = new THREE.DirectionalLight(0xffffff, 2.2);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(1024, 1024);
    this.key.shadow.bias = -0.0005;
    this.fill = new THREE.DirectionalLight(0xffffff, 0.8);
    this.ground = new THREE.Mesh(new THREE.CircleGeometry(1.5, 64), new THREE.ShadowMaterial({ opacity: 0.16 }));
    this.ground.receiveShadow = true;
    this.grid = new THREE.GridHelper(1.0, 10);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.5;
    this.scene.add(sky, this.key, this.key.target, this.fill, this.ground, this.grid);
    this.colors = {
      body: new THREE.MeshStandardMaterial({ metalness: 0.05, roughness: 0.55 }),
      lit: new THREE.MeshStandardMaterial({ metalness: 0.05, roughness: 0.45 }),
      base: new THREE.MeshStandardMaterial({ metalness: 0.05, roughness: 0.45 }),
      end: new THREE.MeshStandardMaterial({ metalness: 0.05, roughness: 0.45 }),
      faded: new THREE.MeshStandardMaterial({ metalness: 0.05, roughness: 0.55, transparent: true, opacity: 0.3 }),
    };
    this.mode = "still";
    this.clock = new THREE.Clock();
    this.visible = true;
    new IntersectionObserver(([entry]) => { this.visible = entry.isIntersecting; }).observe(this.canvas);
    new ResizeObserver(() => this.resize()).observe(box);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.theme());
    this.theme();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  // drawing: GET /api/view. frame: any pose of the arm, for the shapes' sizes.
  async load(drawing, frame) {
    this.drawing = drawing;
    this.solid = new Arm(drawing, this.colors.body);
    await this.solid.build(frame);
    this.solid.pose(frame, 1);
    this.scene.add(this.solid.root);
    this.drawn = this.solid.root;
  }

  theme() {
    const dark = css("color-scheme") === "dark";
    this.colors.body.color.set(css("--arm"));
    this.colors.lit.color.set(css("--arm-lit"));
    this.colors.base.color.set(css("--sky"));
    this.colors.end.color.set(css("--teal"));
    this.colors.faded.color.set(css("--arm"));
    this.ground.material.opacity = dark ? 0.75 : 0.16;
    this.grid.material.color.set(css(dark ? "--muted" : "--line"));
  }

  label(text) {
    this.chip.textContent = text || "";
    this.chip.classList.toggle("hidden", !text);
  }

  // The part of the solid arm that `joint` (an index into the drawing's joints, or null for none) turns, highlighted.
  highlight(joint) {
    const link = joint === null ? null : this.drawing.joints[joint].link;
    for (const { object, link: on } of this.solid.pieces) object.material = on === link ? this.colors.lit : this.colors.body;
  }

  // The solid arm follows the readings, easing between polls.
  live(frame, grip) {
    if (!frame) return;
    if (this.mode !== "live") this.solid.pose(frame, grip);
    this.mode = "live";
    this.target = { frame, grip };
  }

  // Frames the arm at `frame` from the side view (SIDE_VIEW), with `room` (a fraction of its size) around it.
  aim(frame, room) {
    this.solid.pose(frame, this.solid.grip);
    this.frameObject(this.solid.root, room);
  }

  // The tool point's triad (triad()), not draggable, on a link of the solid arm at its origin: a YAM's tool link is
  // its tool point.
  mark(link) {
    this.solid.links[link].add(triad(this.radius * TRIAD_SHARE, false));
  }

  // Frames `object` from the side view (SIDE_VIEW), with `room` (a fraction of its size) around it for a marker and
  // its labels, the lights and the ground under it.
  frameObject(object, room) {
    object.updateMatrixWorld(true);
    const sphere = new THREE.Box3().expandByObject(object).getBoundingSphere(new THREE.Sphere());
    this.radius = sphere.radius;
    const distance = ((sphere.radius * (1 + room)) / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 0.85;
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(SIDE_VIEW, distance);
    const s = sphere.radius;
    this.key.position.copy(sphere.center).add(new THREE.Vector3(s * 1.2, -s * 1.6, s * 3.5));
    this.key.target.position.copy(sphere.center);
    Object.assign(this.key.shadow.camera, { left: -s * 2, right: s * 2, top: s * 2, bottom: -s * 2, near: 0.05, far: s * 10 });
    this.key.shadow.camera.updateProjectionMatrix();
    this.fill.position.copy(sphere.center).add(new THREE.Vector3(-s * 2, s, s));
    this.ground.position.set(sphere.center.x, sphere.center.y, 0);
    this.grid.position.set(sphere.center.x, sphere.center.y, 0.0005);
  }

  resize() {
    const w = this.box.clientWidth, h = this.box.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.1) * 1000;
    if (!this.canvas.isConnected) { this.renderer.setAnimationLoop(null); this.renderer.dispose(); return; }
    if (!this.visible || !this.box.offsetParent || !this.drawn) return;
    if (this.mode === "live" && this.target) this.solid.toward(this.target.frame, this.target.grip, 1 - Math.exp(-dt / 70));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

// An arrow from the origin along `way` (a unit THREE.Vector3), `size` metres long, in `color`, drawn over the arm.
function arrow(way, size, color) {
  const paint = new THREE.MeshBasicMaterial({ color, depthTest: false });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.0175, size * 0.0175, size * 0.75, 12), paint);
  shaft.position.copy(way).multiplyScalar(size * 0.375);
  const head = new THREE.Mesh(new THREE.ConeGeometry(size * 0.05, size * 0.25, 16), paint);
  head.position.copy(way).multiplyScalar(size * 0.875);
  for (const part of [shaft, head]) { part.quaternion.setFromUnitVectors(UP, way); part.renderOrder = 2; }
  return [shaft, head];
}

// The files of a .zip by their path in it, each a function that inflates it: the URDF step reads a URDF and its
// meshes from one. Stored and deflated files, which every zip tool writes, through the browser's DecompressionStream;
// a Mac's __MACOSX copies are left out.
export function unzip(buffer) {
  const view = new DataView(buffer);
  let end = -1;  // the end of central directory record, in the last 64 KiB
  for (let i = buffer.byteLength - 22; i >= Math.max(0, buffer.byteLength - 65557) && end < 0; i -= 1) {
    if (view.getUint32(i, true) === 0x06054b50) end = i;
  }
  if (end < 0) throw new Error("This file is not a .zip.");
  let at = view.getUint32(end + 16, true);
  if (at === 0xffffffff) throw new Error("This .zip is a ZIP64 archive. Zip your URDF and its meshes again, under 4 GB.");
  const files = new Map();
  for (let n = view.getUint16(end + 10, true); n > 0; n -= 1) {
    if (view.getUint32(at, true) !== 0x02014b50) throw new Error("This .zip is damaged: its list of files does not read.");
    const method = view.getUint16(at + 10, true), packed = view.getUint32(at + 20, true), local = view.getUint32(at + 42, true);
    const length = view.getUint16(at + 28, true);
    const name = new TextDecoder().decode(new Uint8Array(buffer, at + 46, length));
    at += 46 + length + view.getUint16(at + 30, true) + view.getUint16(at + 32, true);
    if (name.endsWith("/") || name.startsWith("__MACOSX/")) continue;
    const bytes = new Uint8Array(buffer, local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true), packed);
    files.set(name, async () => {
      if (method === 0) return bytes.slice().buffer;
      if (method !== 8) throw new Error(`This .zip packs ${name} in a way the page cannot open (method ${method}). Zip it again with your system's zip.`);
      return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer();
    });
  }
  return files;
}

// The files of a folder the operator gave, by their path from its parent (`arm/meshes/a.stl`), each a function that
// reads it, as unzip() gives a .zip's, so its URDF finds its meshes the same way: the FileList of an
// <input webkitdirectory>, or a folder dropped on the page (its FileSystemDirectoryEntry), read through.
export async function folderFiles(given) {
  const files = given.isDirectory ? await entryFiles(given) : [...given].map((file) => [file.webkitRelativePath, file]);
  return new Map(files.map(([path, file]) => [path, () => file.arrayBuffer()]));
}

// Every file under a dropped folder's entry, with its path from the folder's parent.
async function entryFiles(entry) {
  if (entry.isFile) return [[entry.fullPath.slice(1), await new Promise((done, fail) => entry.file(done, fail))]];
  const reader = entry.createReader(), entries = [];
  for (let batch; (batch = await new Promise((done, fail) => reader.readEntries(done, fail))).length;) entries.push(...batch);
  return (await Promise.all(entries.map(entryFiles))).flat();
}

// The file of a .zip a URDF's mesh names: `package://pkg/meshes/a.stl`, `file:///home/me/meshes/a.stl` or a path from
// the URDF's own folder (`urdfPath`). The zip's file at that path, else the one whose path ends with the most of the
// name's folders (the name's file at least, any case), the first such in the zip; null for none.
function meshFile(files, name, urdfPath) {
  const parts = name.replace(/^[a-z]+:\/\//i, "").split("/").filter((part) => part && part !== ".");
  const here = urdfPath.split("/").slice(0, -1);
  for (const part of parts) if (part === "..") here.pop(); else here.push(part);
  if (!/^[a-z]+:/i.test(name) && files.has(here.join("/"))) return here.join("/");
  const want = parts.filter((part) => part !== "..").map((part) => part.toLowerCase()).reverse();
  let best = null, most = 0;
  for (const path of files.keys()) {
    const have = path.toLowerCase().split("/").reverse();
    let same = 0;
    while (same < want.length && same < have.length && have[same] === want[same]) same += 1;
    if (same > most) { best = path; most = same; }
  }
  return best;
}

// A blank picture for any texture a mesh file names: the view paints every part in its own colours, and asks nothing
// of the server.
const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
const quiet = new THREE.LoadingManager();
quiet.setURLModifier(() => BLANK);

// A glTF file's scene (.glb, or .gltf with its buffers the zip's files it names from its own folder), each node where
// the file puts it: its vertices in the link's frame, as ROS draws it, with no turn for glTF's y up. Its pictures are
// blank, as a COLLADA file's. null for a .gltf whose buffer the zip lacks, or a file the loader cannot read, such as a
// compressed one (Draco or meshopt, whose decoders the page does not load), so the link is drawn plain and named.
async function gltfObject(path, files, material) {
  const data = await files.get(path)();
  const buffers = new Map();
  const named = path.toLowerCase().endsWith(".gltf") ? JSON.parse(new TextDecoder().decode(data)).buffers || [] : [];
  for (const { uri } of named) {
    if (!uri || uri.startsWith("data:")) continue;
    const file = meshFile(files, decodeURIComponent(uri), path);
    if (!file) return null;
    buffers.set(uri, URL.createObjectURL(new Blob([await files.get(file)()])));
  }
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => buffers.get(url) || (url.startsWith("data:") ? url : BLANK));
  try {
    const object = (await new GLTFLoader(manager).parseAsync(data, "")).scene;
    object.traverse((part) => { if (part.isMesh) part.material = material; });
    return object;
  } catch (error) {
    console.warn(`${path}: ${error.message}`);
    return null;
  } finally {
    for (const url of buffers.values()) URL.revokeObjectURL(url);
  }
}

// A mesh file's parts as one object, by its kind, any case: STL, OBJ, COLLADA (.dae, drawn as ROS draws it: its
// vertices in the link's frame whichever way it says is up, at its unit) or glTF (.glb, .gltf); null for another kind.
async function meshObject(path, files, material) {
  const kind = path.toLowerCase().split(".").pop();
  const read = files.get(path);
  if (kind === "stl") return new THREE.Mesh(new STLLoader().parse(await read()), material);
  if (kind === "glb" || kind === "gltf") return gltfObject(path, files, material);
  if (kind !== "obj" && kind !== "dae") return null;
  const text = new TextDecoder().decode(await read());
  const object = kind === "obj" ? new OBJLoader().parse(text) : new ColladaLoader(quiet).parse(text, "").scene;
  object.rotation.set(0, 0, 0);
  object.traverse((part) => { if (part.isMesh) part.material = material; });
  return object;
}

// The URDF step's view: every link of the operator's URDF at its zero (POST /api/urdf), a click on one picks it, and
// the tool point's triad (triad()), which a drag or the step's buttons move. The page computes the triad's place;
// this view draws it and says what a drag or a typed number asks for.
export class UrdfView extends ArmView {
  // onPick: called with the index of the link a click (not a drag) lands on. onMove: called with (k, amount, from)
  // when a drag or the number box at a handle moves the tool point `amount` mm along the triad's axis k (0, 1, 2 for
  // its x, y, z) or turns it `amount` degrees about axis k - 3, from setMarker()'s `values` when it started.
  constructor(box, onPick, onMove) {
    super(box);
    this.controls.enableZoom = true;
    this.onMove = onMove;
    this.hint = box.appendChild(document.createElement("span"));
    this.hint.className = "hint hidden";
    this.typer = box.appendChild(document.createElement("label"));
    this.typer.className = "typer hidden";
    this.typerName = this.typer.appendChild(document.createElement("span"));
    this.typerField = this.typer.appendChild(document.createElement("input"));
    Object.assign(this.typerField, { type: "number", step: "any" });
    this.typerField.addEventListener("keydown", (e) => {
      if (e.key === "Escape") this.closeTyper();
      if (e.key !== "Enter") return;
      const k = this.typing, amount = parseFloat(this.typerField.value);
      this.closeTyper();
      if (Number.isFinite(amount)) this.onMove(k, amount, this.handles.values);
    });
    this.typerField.addEventListener("blur", () => this.closeTyper());
    let down = null;
    // Before OrbitControls' own listener, so a grabbed handle does not also turn the view.
    this.canvas.addEventListener("pointerdown", (e) => {
      // With cadNavigation() on, any press but a plain left one moves the view: it grabs no handle and picks no link.
      if (this.cad && (e.button !== 0 || e.shiftKey || e.ctrlKey || e.metaKey)) { down = null; return; }
      down = [e.clientX, e.clientY];
      const k = this.handleAt(e);
      if (k === null) return;
      this.controls.enabled = false;
      this.canvas.setPointerCapture(e.pointerId);
      this.grab = this.startDrag(k, e);
    }, { capture: true });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!this.grab) { this.hover(this.handleAt(e)); return; }
      const g = this.grab;
      if (!g.moved) {
        if (Math.hypot(e.clientX - g.from[0], e.clientY - g.from[1]) <= 4) return;
        Object.assign(g, { moved: true, from: [e.clientX, e.clientY] }, g.k < 3 ? {} : { last: this.angle(g, e) });
      }
      const amount = Math.round(this.dragged(g, e));
      if (amount !== g.sent) { g.sent = amount; this.onMove(g.k, amount, g.values); }
    });
    const release = (e) => {
      const g = this.grab;
      this.grab = null;
      this.controls.enabled = true;
      if (g) { if (!g.moved && e.type === "pointerup") this.ask(g.k, e); return; }
      if (e.type !== "pointerup" || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 || !this.pieces) return;
      const hit = this.ray(e).intersectObjects(this.pieces.map((piece) => piece.object))[0];
      if (hit) onPick(hit.object.userData.link);
    };
    this.canvas.addEventListener("pointerup", release);
    this.canvas.addEventListener("pointercancel", release);
  }

  // A ray from the camera through the pointer.
  ray(e) {
    const rect = this.canvas.getBoundingClientRect();
    const at = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, 1 - ((e.clientY - rect.top) / rect.height) * 2);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(at, this.camera);
    return ray;
  }

  // The triad handle under the pointer: 0 to 2 an arrow, 3 to 5 a ring; or null.
  handleAt(e) {
    if (!this.triad || !this.triad.visible || !this.handles) return null;
    const hit = this.ray(e).intersectObjects(this.triad.userData.picks)[0];
    return hit ? hit.object.userData.k : null;
  }

  // A point in the URDF's root frame on the screen, in client pixels.
  screen(p) {
    const rect = this.canvas.getBoundingClientRect(), v = new THREE.Vector3(...p).project(this.camera);
    return [rect.left + ((v.x + 1) / 2) * rect.width, rect.top + ((1 - v.y) / 2) * rect.height];
  }

  // What a drag needs, fixed when it starts: the numbers then, an arrow's axis on the screen and its pixels per mm (at
  // least 0.5, so an arrow pointing at the camera moves at most 2 mm a pixel), or a ring's centre on the screen and
  // which way it turns as seen.
  startDrag(k, e) {
    const h = this.handles, o = h.origin, c = this.screen(o), axis = h.axes[k % 3];
    const g = { k, from: [e.clientX, e.clientY], values: h.values, sent: 0, moved: false };
    if (k < 3) {
      const tip = this.screen(o.map((v, i) => v + axis[i] * 0.01)), way = [tip[0] - c[0], tip[1] - c[1]];
      const length = Math.hypot(...way);
      return { ...g, way: way.map((v) => v / (length || 1)), px: Math.max(length / 10, 0.5) };
    }
    const toward = new THREE.Vector3(...axis).dot(this.camera.position.clone().sub(new THREE.Vector3(...o)));
    return { ...g, centre: c, sign: toward >= 0 ? 1 : -1, turned: 0 };
  }

  // The pointer's angle about a ring's centre on the screen, counter-clockwise.
  angle(g, e) {
    return Math.atan2(g.centre[1] - e.clientY, e.clientX - g.centre[0]);
  }

  // How far the drag has gone: mm along an arrow, or degrees about a ring, right-handed about its arrow.
  dragged(g, e) {
    if (g.k < 3) return ((e.clientX - g.from[0]) * g.way[0] + (e.clientY - g.from[1]) * g.way[1]) / g.px;
    const now = this.angle(g, e);
    g.turned += Math.atan2(Math.sin(now - g.last), Math.cos(now - g.last));
    g.last = now;
    return (g.sign * g.turned * 180) / Math.PI;
  }

  // A handle's name: "red arrow (fingers close)".
  handleName(k) {
    return `${TRIAD_AXES[k % 3][1]} ${k < 3 ? "arrow" : "ring"} (${TRIAD_AXES[k % 3][2]})`;
  }

  // The handle under the pointer lit, what it does in the corner.
  hover(k) {
    for (const [i, paint] of this.triad ? this.triad.userData.paints.entries() : []) paint.opacity = i === k ? 1 : 0.8;
    this.box.style.cursor = k === null ? "" : "grab";
    this.hint.classList.toggle("hidden", k === null);
    if (k !== null) this.hint.textContent = `${this.handleName(k)}: drag to ${k < 3 ? "move" : "turn"}, or click to type`;
  }

  // The number box at the pointer for handle k: how far to move along it (mm) or turn about it (degrees); Enter
  // applies it (onMove), Escape or a click away closes it.
  ask(k, e) {
    const rect = this.box.getBoundingClientRect();
    this.typing = k;
    this.typerName.textContent = k < 3 ? `move along ${TRIAD_AXES[k][1]}, mm` : `turn about ${TRIAD_AXES[k - 3][1]}, °`;
    this.typerField.value = "0";
    this.typer.style.left = `${Math.min(e.clientX - rect.left, rect.width - 210)}px`;
    this.typer.style.top = `${Math.min(e.clientY - rect.top + 12, rect.height - 44)}px`;
    this.typer.classList.remove("hidden");
    this.typerField.focus();
    this.typerField.select();
  }

  closeTyper() {
    this.typer.classList.add("hidden");
    this.typing = null;
  }

  // tree: POST /api/urdf. files: unzip() of the .zip the URDF came in, at urdfPath in it, or null for a bare URDF.
  // Each link as its visuals draw it (meshes from the zip, a URDF's boxes, cylinders and spheres), or plain where none
  // draws: a ball on the link's origin and a capsule from its parent. A click on any of them picks the link.
  // Returns { shown, plain, missing }: the links drawn from their visuals, the links drawn plain whose URDF names a
  // mesh, and the mesh files the zip lacks or the page cannot draw.
  async tree(tree, files, urdfPath) {
    if (this.drawn) this.scene.remove(this.drawn);
    this.parents = tree.parents;
    this.poses = tree.poses;
    const at = tree.poses.map((p) => new THREE.Vector3(p[4], p[5], p[6]));
    const reach = Math.max(0.05, ...at.map((v) => v.length()));
    const r = Math.min(0.02, Math.max(0.005, reach * 0.022));
    const group = new THREE.Group();
    this.pieces = [];
    const add = (object, link) => {
      object.traverse((part) => {
        if (!part.isMesh) return;
        part.userData.link = link;
        part.castShadow = true;
        this.pieces.push({ object: part, link });
      });
      group.add(object);
    };
    const loaded = new Map(), missing = new Set(), shown = [], plain = [];
    const visual = async (v) => {
      if (v.box) return new THREE.Mesh(new THREE.BoxGeometry(...v.box), this.colors.body);
      if (v.sphere) return new THREE.Mesh(new THREE.SphereGeometry(v.sphere, 24, 16), this.colors.body);
      if (v.cylinder) {  // a URDF cylinder stands on its z, a three.js one on its y
        const cylinder = new THREE.Mesh(new THREE.CylinderGeometry(v.cylinder[0], v.cylinder[0], v.cylinder[1], 24), this.colors.body);
        cylinder.rotation.x = Math.PI / 2;
        return new THREE.Group().add(cylinder);
      }
      const path = files && meshFile(files, v.mesh, urdfPath);
      if (!path) { missing.add(v.mesh); return null; }
      if (!loaded.has(path)) loaded.set(path, meshObject(path, files, this.colors.body));
      const object = await loaded.get(path);
      if (!object) { missing.add(v.mesh); return null; }
      const copy = object.clone();
      copy.scale.multiply(new THREE.Vector3(...v.scale));
      return copy;
    };
    for (const [link, parent] of tree.parents.entries()) {
      const drawn = (await Promise.all(tree.visuals[link].map(visual))).filter(Boolean);
      for (const [i, object] of drawn.entries()) {
        const holder = new THREE.Group().add(object);
        holder.matrixAutoUpdate = false;
        holder.matrix.copy(matrix(tree.poses[link])).multiply(matrix(tree.visuals[link][i].pose));
        add(holder, link);
      }
      if (drawn.length) { shown.push(link); continue; }
      if (tree.visuals[link].some((v) => v.mesh)) plain.push(link);
      const ball = new THREE.Mesh(new THREE.SphereGeometry(r * 1.4, 20, 14), this.colors.body);
      ball.position.copy(at[link]);
      add(ball, link);
      const bone = parent < 0 ? null : at[link].clone().sub(at[parent]);
      if (!bone || bone.length() < 1e-4) continue;
      const capsule = new THREE.Mesh(new THREE.CapsuleGeometry(r, bone.length(), 6, 16), this.colors.body);
      capsule.position.copy(at[parent]).addScaledVector(bone, 0.5);
      capsule.quaternion.setFromUnitVectors(UP, bone.clone().normalize());
      add(capsule, link);
    }
    this.scene.add(group);
    this.drawn = group;
    this.frameObject(group, MARKER_ROOM);
    if (this.triad) { this.scene.remove(this.triad); this.triad = null; }
    return { shown, plain, missing: [...missing] };
  }

  // While the end is picked (lit), the link the arm stands on in blue and the one it ends at in green, the links between
  // plain; once it is, all of them plain, so the triad's red, green and blue are the drawing's only colours. Any other
  // link faded.
  paint(base, end, lit) {
    const chain = new Set();
    for (let link = end; link >= 0 && link !== base; link = this.parents[link]) chain.add(link);
    for (const { object, link } of this.pieces) {
      object.material = lit && link === base ? this.colors.base : lit && link === end ? this.colors.end
        : link === base || link === end || chain.has(link) ? this.colors.body : this.colors.faded;
    }
  }

  // The triad at `p` ([x, y, z] in the URDF's root frame), turned by `turn` (rows of the (3, 3) whose columns are its
  // x, y and z), or hidden for a null `p`. handles: what a drag starts from, `values`, the page's six numbers now; its
  // axes in the root frame are `turn`'s columns.
  setMarker(p, turn, handles) {
    if (!this.triad) {
      this.triad = triad(this.radius * TRIAD_SHARE, true);
      this.triad.matrixAutoUpdate = false;
      this.scene.add(this.triad);
    }
    this.triad.visible = p !== null;
    this.handles = p === null ? null : { ...handles, origin: p, axes: [0, 1, 2].map((j) => turn.map((row) => row[j])) };
    if (p === null) return;
    const [a, b, c] = turn;
    this.triad.matrix.set(a[0], a[1], a[2], p[0], b[0], b[1], b[2], p[1], c[0], c[1], c[2], p[2], 0, 0, 0, 1);
    this.triad.matrixWorldNeedsUpdate = true;
    this.triad.updateMatrixWorld(true);
  }
}

// The triad's arrows, a fifth of the arm's size (ArmView.radius), so they reach not much past a gripper.
const TRIAD_SHARE = 0.2;

// The triad's axes in dream's tool frame (setup_view.reference()), each in its frame colour (--axis-x, -y, -z) and
// named once, in the page's legend too: x the way the fingers close, y up (on a YAM at rest, the gripper's top), z out
// of the gripper.
const TRIAD_AXES = [["--axis-x", "red", "fingers close"], ["--axis-y", "green", "up"], ["--axis-z", "blue", "out of the gripper"]];

// The tool point's triad, one only (the selected tool point's), red-green-blue, with arrows to drag and rings to
// rotate as a CAD triad has:
// a dot at the tool point, and per axis (TRIAD_AXES) an arrow `size` metres long and a ring about it, drawn over the
// arm. A draggable one gives each arrow and ring a thicker unseen twin the pointer hits; userData: `picks` (the twins,
// userData.k: 0 to 2 the arrows, 3 to 5 the rings) and `paints` (by k, for UrdfView.hover()).
function triad(size, draggable) {
  const group = new THREE.Group(), unseen = new THREE.MeshBasicMaterial({ visible: false });
  const picks = [], paints = [];
  const add = (k, shapes, pick, paint) => {
    for (const shape of shapes) { shape.renderOrder = 2; group.add(shape); }
    paints[k] = paint;
    if (!draggable) return;
    pick.userData.k = k;
    picks.push(pick);
    group.add(pick);
  };
  for (const [i, [token]] of TRIAD_AXES.entries()) {
    const way = new THREE.Vector3(...[0, 1, 2].map((j) => (j === i ? 1 : 0)));
    const color = css(token);
    const [shaft, head] = arrow(way, size, color);
    const reach = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.12, size * 0.12, size, 8), unseen);
    reach.position.copy(way).multiplyScalar(size * 0.5);
    reach.quaternion.setFromUnitVectors(UP, way);
    add(i, [shaft, head], reach, shaft.material);
    const paint = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.8 });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(size * 0.6, size * 0.012, 8, 64), paint);
    const grip = new THREE.Mesh(new THREE.TorusGeometry(size * 0.6, size * 0.08, 8, 48), unseen);
    for (const shape of [ring, grip]) shape.quaternion.setFromUnitVectors(OUT, way);
    add(3 + i, [ring], grip, paint);
  }
  const dot = new THREE.Mesh(new THREE.SphereGeometry(size * 0.06, 16, 12), new THREE.MeshBasicMaterial({ color: css("--ink"), depthTest: false }));
  dot.renderOrder = 2;
  group.add(dot);
  group.userData = { picks, paints };
  return group;
}

const OUT = new THREE.Vector3(0, 0, 1);

// The standard views a CAD tool offers, by where the camera looks from in the base's frame (z up, x forward): Front
// along x, Side along -y (SIDE_VIEW without its tilt), Top straight down (a hair off, so "up" on the screen stays y),
// and Iso from the front, the side and above at once. Fit is the view the page opens at (frameObject()).
const STANDARD_VIEWS = [
  ["Fit", null, "Fit the arm in the view (F)"],
  ["Front", new THREE.Vector3(1, 0, 0), "Look from the front"],
  ["Side", new THREE.Vector3(0, -1, 0), "Look from the side"],
  ["Top", new THREE.Vector3(0, -1e-4, 1).normalize(), "Look from above"],
  ["Iso", new THREE.Vector3(1, -1, 1).normalize(), "Look from the front, the side and above"],
];

// CAD-style moves for a view, as SolidWorks, Fusion and Onshape share them: left-drag turns, as on every view, so a
// left click still picks a link or grabs the marker; right-drag, middle-drag, or a Shift, Ctrl or Cmd drag (for a
// trackpad) moves the view; the wheel or a pinch zooms toward the pointer, so pointing at a tool point zooms in on it;
// Fit, Front, Side, Top and Iso in the bottom right corner fly to a standard view, and F fits the focused view. The
// redesigned page (DREAM_SETUP_WIZARD_V2) turns it on for each view; the current page's views only turn.
// view: an ArmView or UrdfView. room: what frameObject() leaves around the arm (MARKER_ROOM with a marker, else 0).
export function cadNavigation(view, room) {
  const controls = view.controls;
  Object.assign(controls, { enablePan: true, screenSpacePanning: true, enableZoom: true, zoomToCursor: true });
  controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.PAN };
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  view.cad = true;

  // Frames the arm as the page does, then turns the camera to look from `from` (null: the page's own view), over a
  // third of a second unless the system asks for less motion.
  const fly = (from) => {
    if (!view.drawn) return;
    const startPos = view.camera.position.clone(), startTarget = controls.target.clone();
    view.frameObject(view.drawn, room);
    const target = controls.target.clone(), framed = view.camera.position.clone(), distance = framed.distanceTo(target);
    const endPos = from ? target.clone().addScaledVector(from, distance) : framed;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      view.camera.position.copy(endPos);
      controls.update();
      return;
    }
    view.camera.position.copy(startPos);
    controls.target.copy(startTarget);
    const away = startPos.clone().sub(startTarget).normalize(), startDistance = startPos.distanceTo(startTarget);
    const swing = new THREE.Quaternion().setFromUnitVectors(away, endPos.clone().sub(target).normalize());
    const begin = performance.now(), turn = new THREE.Quaternion();
    const step = (now) => {
      const t = Math.min(1, (now - begin) / 350), eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      controls.target.lerpVectors(startTarget, target, eased);
      turn.identity().slerp(swing, eased);
      view.camera.position.copy(controls.target).addScaledVector(away.clone().applyQuaternion(turn), startDistance + (distance - startDistance) * eased);
      controls.update();
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  const bar = view.box.appendChild(document.createElement("div"));
  bar.className = "viewnav";
  bar.setAttribute("role", "toolbar");
  bar.setAttribute("aria-label", "View");
  for (const [name, from, title] of STANDARD_VIEWS) {
    const button = bar.appendChild(document.createElement("button"));
    Object.assign(button, { type: "button", textContent: name, title });
    button.addEventListener("pointerdown", (e) => e.stopPropagation());  // a click here is not a drag on the view
    button.addEventListener("click", () => fly(from));
  }
  view.box.tabIndex = 0;
  view.box.addEventListener("keydown", (e) => {
    if ((e.key === "f" || e.key === "F") && e.target === view.box) fly(null);
  });
}
