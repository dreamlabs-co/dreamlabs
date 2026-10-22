// Loads mesh files with the 3D view's own loader (meshObject() in src/robots/setup_viewer.js) under node, for
// tests/test_setup_view.py.
//
//   node tests/setup_meshes.mjs <folder> <zip|list|drop> <path in the zip>...
//
// <folder> holds node_modules/three (the files the page's import map pins) and zip/, the zip's files by their paths,
// given to the view as unzip() gives a .zip's (zip), or through folderFiles() as a folder: an <input webkitdirectory>'s
// FileList (list), or each top folder dropped on the page, its entries read one at a time (drop).
// The view's script is copied into <folder> so its bare "three" imports resolve there. Prints, per path, null when the
// view cannot draw it, else {"meshes": how many, "painted": whether each wears the view's own material, "box": [min,
// max] of its vertices where the view puts them}.
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

// three.js's file loader reports progress with the browser's ProgressEvent, which node lacks.
globalThis.ProgressEvent = class extends Event {
  constructor(type, init) { super(type); Object.assign(this, init); }
};
const [folder, given, ...paths] = process.argv.slice(2);
const here = path.dirname(url.fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "..", "src", "robots", "setup_viewer.js"), "utf8");
fs.writeFileSync(path.join(folder, "viewer.mjs"), `${source}\nexport { meshObject };\n`);
const { meshObject, folderFiles } = await import(url.pathToFileURL(path.join(folder, "viewer.mjs")).href);
const THREE = await import(url.pathToFileURL(path.join(folder, "node_modules", "three", "build", "three.module.min.js")).href);

const root = path.join(folder, "zip");
const read = (name) => async () => new Uint8Array(fs.readFileSync(path.join(root, name))).buffer;
const names = fs.readdirSync(root, { recursive: true }).map((name) => name.split(path.sep).join("/"));
const isFile = (name) => fs.statSync(path.join(root, name)).isFile();
// A dropped folder's FileSystemEntry, as the browser gives it: its directory reader hands out one entry per call.
const entry = (name) => isFile(name)
  ? { isFile: true, isDirectory: false, fullPath: `/${name}`, file: (done) => done({ arrayBuffer: read(name) }) }
  : { isFile: false, isDirectory: true, name: path.basename(name), fullPath: `/${name}`, createReader: () => {
      const left = fs.readdirSync(path.join(root, name)).map((child) => entry(`${name}/${child}`));
      return { readEntries: (done) => done(left.splice(0, 1)) };
    } };
const files = await {
  zip: async () => new Map(names.filter(isFile).map((name) => [name, read(name)])),
  list: () => folderFiles(names.filter(isFile).map((name) => ({ webkitRelativePath: name, arrayBuffer: read(name) }))),
  drop: async () => new Map((await Promise.all(fs.readdirSync(root).map(async (top) => [...await folderFiles(entry(top))]))).flat()),
}[given]();
const material = new THREE.MeshStandardMaterial();
const shown = {};
for (const name of paths) {
  const object = await meshObject(name, files, material);
  if (!object) { shown[name] = null; continue; }
  object.updateMatrixWorld(true);
  const meshes = [];
  object.traverse((part) => { if (part.isMesh) meshes.push(part); });
  const box = new THREE.Box3().setFromObject(object);
  shown[name] = { meshes: meshes.length, painted: meshes.every((part) => part.material === material), box: [box.min.toArray(), box.max.toArray()] };
}
console.log(JSON.stringify(shown));
