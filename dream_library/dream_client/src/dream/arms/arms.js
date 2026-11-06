// The arms at the top of the page of what the model sees, while python -m dream.check runs (DebugView.show_arms): a
// ghost of each arm where the check commands it, a solid arm at the pose the loop reported at its last call, each
// hand's commanded path (dashed) and where each call's pose put the hand (dots). Live while the arms move, then
// replayed. A YAM is drawn from its model; any other arm the check knows as a chain of its joints, from the chain the
// check sends (StandIn.arms `chains`); an arm it does not know as its tool point alone. three.js comes from the page's
// import map; the YAM's model and its joints from the view's own server.
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {Line2} from 'three/addons/lines/Line2.js';
import {LineGeometry} from 'three/addons/lines/LineGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';

const SIDES = ['left', 'right'];   // arm 0 of the YAM model is the left arm
const YAM_PARTS = {left_arm: 6, right_arm: 6, left_gripper: 1, right_gripper: 1};   // the YAM's state parts drawn
const LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'gripper'];
const APART = 0.5;   // metres between the bases of two arms drawn from their chains or tool points
const HOLD_S = 1.5;   // the replay rests this long on the last pose before it starts again
const RISE = 0.45, MARGIN = 1.3;   // the view: this much up per 1 across, and this much room round the arms' joints
const UP = new THREE.Matrix4().makeRotationX(-Math.PI / 2);   // a robot's z up to the page's y up
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const m4 = (o) => new THREE.Matrix4().set(...o);
const M = new THREE.Matrix4();

export async function start(box, warn) {
  box.innerHTML = '<h1>The arms</h1><div class="card"><div class="stage3d"></div><div class="armsbar">'
    + '<button type="button" disabled>Pause</button><input type="range" min="0" max="1000" value="0" disabled '
    + 'aria-label="Time in the motion"><span class="clock">Waiting for your loop\'s first call.</span></div>'
    + '<div class="key"><b class="g">blue ghost</b> = where the check commands each arm · <b>solid</b> = the pose your '
    + 'loop reported at its last call · <b class="g">dashed</b> = each hand\'s commanded path · <b class="o">orange '
    + 'dots</b> = where each call\'s pose put the hand. Drag to turn the view, scroll to zoom.</div><div class="note"></div></div>';
  const stage = box.querySelector('.stage3d'), note = box.querySelector('.note');
  const [btn, scrub, clock] = box.querySelector('.armsbar').children;

  // ---------- the scene ----------
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); stage.append(renderer.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
  const controls = new OrbitControls(camera, renderer.domElement);
  const VIEW = new THREE.Vector3(1, RISE, 0).normalize();   // set from the arms once the plan comes
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(-1, 2, 1.5); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7); fill.position.set(1.5, 0.5, -1); scene.add(fill);
  const grid = new THREE.GridHelper(1.6, 16, 0x888888, 0x888888);
  grid.material.transparent = true; grid.material.opacity = 0.25; scene.add(grid);
  const onTop = (o, order) => { o.material.depthTest = false; o.renderOrder = order; o.frustumCulled = false; scene.add(o); return o; };
  const lines = [];   // every LineMaterial, whose width is in pixels of the stage
  const line = (width, dashed, order) => {
    const l = onTop(new Line2(new LineGeometry(), new LineMaterial({linewidth: width, dashed, dashSize: 0.008, gapSize: 0.005,
      transparent: true})), order);
    lines.push(l.material); return l;
  };

  // ---------- the motion: the check's targets per row and each call's state (StandIn.arms) ----------
  let m = await first();
  const body = m.arm === 'yam' ? await yamBody() : Object.keys(m.chains).length ? chainBody() : toolBody();
  const sides = body.sides;
  const labels = sides.map((s) => { const l = document.createElement('span'); l.className = 'side'; l.textContent = s; stage.append(l); return l; });
  const paths = sides.map(() => line(2.5, true, 6));
  const dots = [];   // {t, mesh}: per call and arm, where the reported pose put the hand
  const ball = new THREE.SphereGeometry(0.009, 16, 12);
  let got = 0, plan = '', end = 0, t = 0, playing = true, last = 0, drawn = 0, shown = false;
  const middles = sides.map(() => new THREE.Vector3());   // per arm, the middle of its hand's commanded path
  const rings = sides.map(() => []);   // per arm, its hand's commanded circle (the plan's `together` rows)
  let fit = [];   // the arms at the start and both circles: the view keeps them in frame
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  function colours() {
    body.colours();
    for (const p of paths) p.material.color.set(css('--acc'));
    for (const d of dots) d.mesh.material.color.set(css('--meas'));
  }
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', colours);

  function target(part, r) {   // a part's target at row r, between the rows either side
    const T = m.targets[part], i = Math.max(0, Math.min(T.length - 2, Math.floor(r))), u = Math.max(0, Math.min(1, r - i));
    return T[i].map((v, k) => v + (T[i + 1][k] - v) * u);
  }
  const targets = (r) => Object.fromEntries(Object.keys(m.targets).map((p) => [p, target(p, r)]));
  function take(j) {
    const ended = shown && m.done;
    m = j; got = performance.now(); shown = true;
    const segments = JSON.stringify(m.segments);
    if (segments !== plan) {   // a new plan, or one the check extended to move a part at fault alone
      plan = segments; end = Math.max(...m.segments.map((s) => s.start + s.rows + s.rest)) / m.rate;
      sides.forEach((s, a) => {
        const points = [];
        for (let r = 0; r <= end * m.rate; r++) { body.place('ghost', a, targets(r)); points.push(body.tool('ghost', a)); }
        paths[a].geometry = new LineGeometry().setPositions(points.flatMap((v) => [v.x, v.y, v.z]));
        paths[a].computeLineDistances();
        middles[a].set(0, 0, 0); for (const v of points) middles[a].addScaledVector(v, 1 / points.length);
        const ring = m.segments.find((g) => g.kind === 'together');
        rings[a] = points.slice(ring.start, ring.start + ring.rows);
      });
      if (dots.length === 0) aim();   // the first plan only: a view the person turned stays
    }
    for (; drawn < m.calls.length; drawn++) {
      sides.forEach((s, a) => {
        body.place('ghost', a, m.calls[drawn].state);
        const mesh = onTop(new THREE.Mesh(ball, new THREE.MeshBasicMaterial()), 7);
        mesh.position.copy(body.tool('ghost', a)); dots.push({t: m.calls[drawn].t, mesh});
      });
    }
    colours();
    if (m.done && !ended) { t = reduce ? end : 0; playing = !reduce; btn.disabled = scrub.disabled = false; }
    note.replaceChildren();
    if (m.stopped) warn(note, `The check stopped the motion: ${m.stopped}`);
  }
  async function fetchArms() {
    const r = await fetch('arms');
    if (!r.ok) throw new Error(`arms: ${r.status}`);
    return r.json();
  }
  async function first() {   // the motion once the loop's first call has come: it names the arm
    for (;;) {
      const j = await fetchArms();
      if (j !== null) return j;
      await new Promise((done) => setTimeout(done, 250));
    }
  }
  async function poll() {
    const j = await fetchArms();
    take(j);
    if (!j.done) setTimeout(() => poll().catch(lost), 250);
  }
  const lost = (e) => warn(note, `The arms stopped: ${e.message}`);

  // ---------- drawing: live while the arms move, then the whole motion again ----------
  function draw() {
    let c = m.calls[0]; for (const x of m.calls) if (x.t <= t) c = x;
    sides.forEach((s, a) => { body.place('solid', a, c.state); body.place('ghost', a, targets(t * m.rate)); });
    for (const d of dots) d.mesh.visible = d.t <= t;
    renderer.render(scene, camera);
    labels.forEach((l, a) => {   // each arm's name above its hand's path
      const v = middles[a].clone().setY(middles[a].y + 0.09).project(camera);
      l.style.left = `${(v.x + 1) / 2 * stage.clientWidth}px`; l.style.top = `${(1 - v.y) / 2 * stage.clientHeight}px`;
    });
    const seen = m.calls.filter((x) => x.t <= t).length;
    clock.textContent = `${m.done ? 'Replay' : 'Live'}: ${Math.min(t, end).toFixed(1)} of ${end.toFixed(0)} s · the `
      + `solid arms are at call ${seen} of ${m.calls.length}`;
    scrub.value = String(Math.round(1000 * Math.min(t, end) / end));
  }
  function frame(ms) {
    const dt = last ? Math.min(0.1, (ms - last) / 1000) : 0; last = ms;
    if (!m.done) t = Math.min(end, m.now + (performance.now() - got) / 1000);
    else if (playing) { t += dt; if (t > end + HOLD_S) t = 0; }
    draw();
    requestAnimationFrame(frame);
  }
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? 'Pause' : 'Play'; };
  scrub.oninput = () => { t = end * (+scrub.value) / 1000; playing = false; btn.textContent = 'Play'; };
  function facing(ring) {   // an upright circle's level normal: across its widest level spread
    const c = ring.reduce((sum, v) => sum.add(v), new THREE.Vector3()).divideScalar(ring.length);
    let xx = 0, zz = 0, xz = 0;
    for (const v of ring) { const dx = v.x - c.x, dz = v.z - c.z; xx += dx * dx; zz += dz * dz; xz += dx * dz; }
    const th = 0.5 * Math.atan2(2 * xz, xx - zz);   // the circle's level axis; the normal is level and across it
    return new THREE.Vector3(-Math.sin(th), 0, Math.cos(th));
  }
  function aim() {   // facing the circles, from in front of the arms and a little above, the arms and paths in frame
    sides.forEach((s, a) => body.place('solid', a, m.calls[0].state));
    const feet = body.feet().reduce((x, y) => x.add(y), new THREE.Vector3()).divideScalar(sides.length);
    const out = middles.reduce((x, y) => x.add(y), new THREE.Vector3()).divideScalar(sides.length).sub(feet).setY(0);
    const normals = rings.map(facing), n = normals[0].clone();
    for (const k of normals.slice(1)) n.add(k.dot(n) < 0 ? k.clone().negate() : k);
    if (n.dot(out) < 0) n.negate();
    VIEW.copy(n.normalize()).setY(RISE).normalize();
    fit = [...body.fit(), ...rings.flat()];
    new THREE.Box3().setFromPoints(fit).getCenter(controls.target);
    resize();
  }
  function resize() {   // the nearest distance along VIEW that keeps every point of `fit` in the window
    const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const up = new THREE.Vector3(0, 1, 0), right = new THREE.Vector3().crossVectors(up, VIEW).normalize();
    const top = new THREE.Vector3().crossVectors(VIEW, right), tv = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    let d = 1.6;
    if (fit.length) d = Math.max(...fit.map((p) => {
      const v = p.clone().sub(controls.target);
      return v.dot(VIEW) + MARGIN * Math.max(Math.abs(v.dot(right)) / (tv * camera.aspect), Math.abs(v.dot(top)) / tv);
    }));
    camera.position.copy(VIEW).multiplyScalar(d).add(controls.target);
    controls.update();
    for (const l of lines) l.resolution.set(w, h);
  }

  // ---------- the bodies: place(which, arm, state) poses the solid or ghost arm; tool() is its hand afterwards ----------
  async function yamBody() {   // the YAM pair from its model, posed by the joints
    const missing = Object.entries(YAM_PARTS).filter(([p, n]) => !(p in m.targets) || m.targets[p][0].length !== n);
    if (missing.length) {
      throw new Error('the arms are drawn for a YAM pair: state.left_arm and state.right_arm with 6 joints each, '
        + `state.left_gripper and state.right_gripper with one; your loop sent ${Object.keys(m.targets).join(', ')}`);
    }
    const [gltf, jt] = await Promise.all([
      new GLTFLoader().loadAsync('yam.glb'),
      fetch('yam.joints.json').then((r) => { if (!r.ok) throw new Error(`yam.joints.json: ${r.status}`); return r.json(); }),
    ]);
    const jointsOf = jt.bodies[0];
    const ORIG = LINKS.map((_, i) => m4(jointsOf.joints[`joint${i + 1}`].origin));
    const SIGN = LINKS.map((_, i) => jointsOf.joints[`joint${i + 1}`].sign ?? 1);   // the table names only a reversed joint
    const TIPS = jointsOf.groups.find((g) => g.name === 'gripper').joints.map((j) => ({o: m4(j.origin), a: j.a, b: j.b}));
    const solidRoot = gltf.scene, ghostRoot = gltf.scene.clone(true), solidNode = new Map(), ghostNode = new Map();
    const all = [], copies = []; solidRoot.traverse((o) => all.push(o)); ghostRoot.traverse((o) => copies.push(o));
    all.forEach((o, i) => {
      const found = gltf.parser.associations.get(o);
      if (found && found.nodes !== undefined) { solidNode.set(o, found.nodes); ghostNode.set(copies[i], found.nodes); }
    });
    function bind(root, nodeOf) {   // per arm, its six moving links and two finger tips, posed by their matrices
      const byName = {};
      root.traverse((o) => { if (nodeOf.has(o)) byName[gltf.parser.json.nodes[nodeOf.get(o)].name] = o; });
      return [0, 1].map((a) => {
        const links = LINKS.map((n) => byName[`yam_st_${a}/${n}`]);
        const tips = [byName[`yam_st_${a}/tip_left`], byName[`yam_st_${a}/tip_right`]];
        const base = byName[`yam_st_${a}/base`];
        for (const o of [base, ...links, ...tips]) { if (!o) throw new Error(`yam.glb has no arm ${a}`); o.matrixAutoUpdate = false; }
        return {base, links, tips};
      });
    }
    solidRoot.traverse((o) => { if (o.isMesh) { o.userData.light = o.material.color.r > 0.5;
      o.material = new THREE.MeshStandardMaterial({roughness: 0.6, metalness: 0.05, flatShading: true}); } });
    // The ghost: see-through, giving way to the solid arm where they meet, and only its nearest surface drawn (a
    // depth-only copy of it first), so its insides never show through it.
    const lift = {polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 8};
    const shell = new THREE.MeshBasicMaterial({colorWrite: false, ...lift});
    const meshes = []; ghostRoot.traverse((o) => { if (o.isMesh) meshes.push(o); });
    for (const o of meshes) {
      o.renderOrder = 2; o.material = new THREE.MeshBasicMaterial({transparent: true, opacity: 0.3, depthWrite: false, ...lift});
      const near = new THREE.Mesh(o.geometry, shell); near.renderOrder = 1; o.add(near);
    }
    const arms = {solid: bind(solidRoot, solidNode), ghost: bind(ghostRoot, ghostNode)};
    scene.add(solidRoot, ghostRoot);
    function pose(arm, q, g) {   // q: 6 joints, radians; g: the gripper, 0 closed to 1 open
      for (let i = 0; i < 6; i++) arm.links[i].matrix.copy(ORIG[i]).multiply(M.makeRotationZ(q[i] * SIGN[i]));
      for (const [k, T] of TIPS.entries()) arm.tips[k].matrix.copy(T.o).multiply(M.makeTranslation(0, 0, T.a + g * (T.b - T.a)));
    }
    const world = (o) => { o.updateWorldMatrix(true, false); return o.getWorldPosition(new THREE.Vector3()); };
    return {
      sides: SIDES,
      place: (which, a, state) => pose(arms[which][a], state[`${SIDES[a]}_arm`], state[`${SIDES[a]}_gripper`][0]),
      tool: (which, a) => {   // the check's tool point: tcp_m along the gripper's roll axis, between the fingertips
        const g = arms[which][a].links[5]; g.updateWorldMatrix(true, false);
        return new THREE.Vector3(0, 0, -m.tcp_m).applyMatrix4(g.matrixWorld);
      },
      feet: () => arms.solid.map((arm) => world(arm.base)),
      fit: () => arms.solid.flatMap((arm) => [arm.base, ...arm.links, ...arm.tips].map(world)),
      colours: () => {
        const dark = new THREE.Color(css('--arm-dark')), light = new THREE.Color(css('--arm-light'));
        solidRoot.traverse((o) => { if (o.isMesh) o.material.color.copy(o.userData.light ? light : dark); });
        ghostRoot.traverse((o) => { if (o.isMesh) o.material.color.set(css('--acc')); });
      },
    };
  }

  function apart(n) {   // each arm's base in the page, the arms side by side, the left one first
    return Array.from({length: n}, (_, a) => new THREE.Matrix4().makeTranslation(0, 0, (a - (n - 1) / 2) * APART));
  }
  function chainBody() {   // any other arm the check knows: its joints' chain, as sticks
    const known = SIDES.filter((s) => s in m.chains);
    const bases = apart(known.length);
    const chains = known.map((s) => ({
      links: m.chains[s].links.map((l) => ({...l, O: m4(l.origin), axis: new THREE.Vector3(...l.axis)})),
      T: m4(m.chains[s].tool),
    }));
    const sticks = {solid: known.map(() => line(6, false, 4)), ghost: known.map(() => line(6, false, 3))};
    const at = {solid: known.map(() => []), ghost: known.map(() => [])};   // per arm, its joints and hand, placed
    function place(which, a, state) {
      const q = state[`${known[a]}_arm`], T = bases[a].clone().multiply(UP), points = [new THREE.Vector3().setFromMatrixPosition(T)];
      for (const l of chains[a].links) {
        T.multiply(l.O);
        if (l.col !== null) {
          T.multiply(l.kind === 'prismatic' ? M.makeTranslation(l.axis.x * q[l.col], l.axis.y * q[l.col], l.axis.z * q[l.col])
            : M.makeRotationAxis(l.axis, q[l.col]));
        }
        points.push(new THREE.Vector3().setFromMatrixPosition(T));
      }
      points.push(new THREE.Vector3().setFromMatrixPosition(T.multiply(chains[a].T)));
      at[which][a] = points;
      sticks[which][a].geometry.setPositions(points.flatMap((v) => [v.x, v.y, v.z]));
    }
    return {
      sides: known,
      place,
      tool: (which, a) => at[which][a][at[which][a].length - 1].clone(),
      feet: () => at.solid.map((points) => points[0].clone()),
      fit: () => at.solid.flat(),
      colours: () => {
        for (const s of sticks.solid) { s.material.color.set(css('--fg')); s.material.opacity = 1; }
        for (const s of sticks.ghost) { s.material.color.set(css('--acc')); s.material.opacity = 0.45; }
      },
    };
  }

  function toolBody() {   // an arm the check does not know: each hand's tool point alone, where its pose puts it
    const known = SIDES.filter((s) => `${s}_eef_xyz` in m.targets);
    const bases = apart(known.length);
    const hand = new THREE.SphereGeometry(0.016, 20, 14);
    const balls = {solid: known.map(() => onTop(new THREE.Mesh(hand, new THREE.MeshBasicMaterial()), 5)),
      ghost: known.map(() => onTop(new THREE.Mesh(hand, new THREE.MeshBasicMaterial({transparent: true, opacity: 0.45})), 4))};
    const where = (a, xyz) => new THREE.Vector3(...xyz).applyMatrix4(bases[a].clone().multiply(UP));
    return {
      sides: known,
      place: (which, a, state) => balls[which][a].position.copy(where(a, state[`${known[a]}_eef_xyz`])),
      tool: (which, a) => balls[which][a].position.clone(),
      feet: () => known.map((_, a) => where(a, [0, 0, 0])),
      fit: () => [...balls.solid.map((b) => b.position.clone()), ...known.map((_, a) => where(a, [0, 0, 0]))],
      colours: () => {
        for (const b of balls.solid) b.material.color.set(css('--fg'));
        for (const b of balls.ghost) b.material.color.set(css('--acc'));
      },
    };
  }

  new ResizeObserver(resize).observe(stage); resize();
  take(m);
  await poll();
  requestAnimationFrame(frame);
}
