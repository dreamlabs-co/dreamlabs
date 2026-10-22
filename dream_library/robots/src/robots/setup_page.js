// Both setup pages' own part around the steps: the step list that follows the cards, and the station line. The steps
// themselves are setup_wizard.js's.
(function () {
  const $ = (id) => document.getElementById(id);
  const NAMES = ["Select robot", "Connect arms", "Set up cameras", "Move to home", "Set reach area", "Test joints", "Save station"];
  const cards = [...document.querySelectorAll("section.card")];
  const wayCard = $("wayCard");
  const steps = $("steps");
  const main = document.querySelector("main");

  // Step 1, Choose setup, is this page's own card before the script's: the wizard or Set it up myself. It shows until
  // Continue, and the script's cards wait behind it; a page that opens on a later step (a rerun) has chosen already,
  // and the tool point page alone (`dreamcli tcp`, main.tcp-only) has nothing to choose.
  let choosing = !!wayCard;
  let rerun = true;  // until the operator answers it, a page that opens on a later step skips the choice
  const later = () => cards.some((card) => card.dataset.step !== "0" && card.classList.contains("open"));
  const showWay = (on) => { rerun = false; choosing = on; steps.classList.toggle("choosing", on); follow(); };

  const item = (name, onclick) => {
    const li = document.createElement("li");
    const link = document.createElement("button");
    link.type = "button";
    link.innerHTML = '<span class="dot"></span><span class="name"></span>';
    link.querySelector(".name").textContent = name;
    link.onclick = onclick;
    li.append(link);
    return li;
  };
  const wayItem = wayCard ? [item("Choose setup method", () => { if (!choosing) showWay(true); })] : [];
  // Gated: only a done step can be gone back to; the current one and those after it do nothing.
  $("progress").replaceChildren(...wayItem, ...cards.map((card, i) => item(NAMES[i], () => {
    if (choosing || !card.classList.contains("done") || !window.setupGo) return;
    window.setupGo(Number(card.dataset.step));
  })));
  // The last step is no card: Open Live view, the Setup complete page either way of setting up ends on.
  const doneLink = $("progress").appendChild(item("Open Live view", () => {})).querySelector("button");
  const links = [...document.querySelectorAll("#progress button")];
  const wayLink = wayCard ? links[0] : null;
  const cardLinks = wayCard ? links.slice(1, -1) : links.slice(0, -1);
  const finished = () => main.classList.contains("finished");

  // Every number on the page counts from 1, Choose setup first: the list's circles and each card's badge alike.
  function number() {
    let n = wayCard ? 1 : 0;
    cards.forEach((card, i) => {
      const shown = !card.classList.contains("hidden") && !(main.classList.contains("self-way") && card.dataset.step !== "0");
      cardLinks[i].parentElement.hidden = !shown;
      if (!shown) return;
      n += 1;
      const badge = card.querySelector(".step");
      if (badge.textContent !== String(n)) badge.textContent = String(n);
      cardLinks[i].querySelector(".dot").textContent = String(n);
    });
    if (wayLink) wayLink.querySelector(".dot").textContent = "1";
    doneLink.querySelector(".dot").textContent = String(n + 1);
  }

  // The page heading is the current step's name, with "Step n of m" above it; the tool point page keeps its own, with
  // no step count.
  function heading() {
    if (!$("stepOf")) return;
    if (main.classList.contains("tcp-only")) {
      $("stepOf").textContent = "";
      return;
    }
    const shown = links.filter((link) => !link.parentElement.hidden);
    if (finished()) {  // the Setup complete page keeps its own heading
      $("stepOf").textContent = `Step ${shown.length} of ${shown.length}`;
      return;
    }
    let current = choosing ? wayLink : cardLinks.find((link) => link.getAttribute("aria-current") === "step");
    if (!current) return;
    // Set up the station myself shows Select robot alone, whatever step the wizard was on.
    if (current.parentElement.hidden) current = shown[shown.length - 1];
    const line = `Step ${shown.indexOf(current) + 1} of ${shown.length}`;
    const name = current.querySelector(".name").textContent;
    if ($("stepOf").textContent !== line) $("stepOf").textContent = line;
    if ($("title").textContent !== name) $("title").textContent = name;
  }

  function follow() {
    if (choosing && ((rerun && later()) || main.classList.contains("tcp-only"))) { choosing = false; steps.classList.remove("choosing"); }
    number();
    if (wayLink) {
      wayLink.classList.toggle("done", !choosing);
      if (choosing) wayLink.setAttribute("aria-current", "step"); else wayLink.removeAttribute("aria-current");
    }
    cards.forEach((card, i) => {
      const link = cardLinks[i];
      link.classList.toggle("done", finished() || card.classList.contains("done"));
      link.disabled = finished() || choosing || card.classList.contains("later");
      if (!finished() && !choosing && card.classList.contains("open")) link.setAttribute("aria-current", "step");
      else link.removeAttribute("aria-current");
    });
    doneLink.disabled = !finished();
    if (finished()) doneLink.setAttribute("aria-current", "step"); else doneLink.removeAttribute("aria-current");
    heading();
  }
  if (wayCard) {
    steps.classList.add("choosing");
    $("wayNext").onclick = () => showWay(false);
  }
  new MutationObserver(follow).observe(steps, { subtree: true, attributes: true, attributeFilter: ["class"], childList: true, characterData: true });
  new MutationObserver(follow).observe(main, { attributes: true, attributeFilter: ["class"] });
  follow();

  fetch("/api/state").then((r) => r.json()).then((state) => {
    if (state.station_name) $("stationName").textContent = state.station_name;
    $("stationPath").textContent = state.config_path;
    $("stationLoaded").hidden = !state.exists;
    // The logo opens the platform this station's credentials name; platform.dreamlabs.ai until the state says.
    const platform = state.connection && state.connection.platform_url;
    if (platform) document.querySelector("header.top .brand").href = platform;
  }).catch(() => {});

  if ($("armCards")) watchArms();
})();

// The joint cards: while the arms are let go, a Left arm and a Right arm card, read from GET /api/status every
// second. A side's card says it is not found until setup_wizard.js has asked which arm moved; then it names its
// adapter and shows a tile per joint, lit while the joint is more than 5 degrees (a gripper, a tenth of its travel)
// from the arm's first reading.
function watchArms() {
  const $ = (id) => document.getElementById(id);
  const DEG = "\u00b0";
  // Joint names by the arm's joint count: a YAM's six, an SO-101's five; any other arm is numbered only.
  const NAMES = {
    6: [["Base", "turns the whole arm left and right"], ["Shoulder", "lifts the arm up and down"], ["Elbow", "bends the arm"],
      ["Wrist tilt", "tips the hand up and down"], ["Wrist swing", "swings the hand side to side"], ["Wrist twist", "twists the hand"]],
    5: [["Shoulder pan", "turns the whole arm left and right"], ["Shoulder lift", "lifts the arm up and down"],
      ["Elbow flex", "bends the arm"], ["Wrist flex", "tips the hand up and down"], ["Wrist roll", "twists the hand"]],
  };
  const SIDES = ["left", "right"];
  const first = {};
  const cards = {};

  function tile(id, name, tip) {
    const node = document.createElement("div");
    node.className = "tile";
    node.title = tip;
    node.innerHTML = '<div class="jid"></div><div class="jname"></div><div class="jval"></div>';
    node.querySelector(".jid").textContent = id;
    node.querySelector(".jname").textContent = name;
    return node;
  }

  function tilesFor(count) {
    const names = NAMES[count] || [];
    const tiles = [];
    for (let i = 0; i < count; i++) {
      const [name, tip] = names[i] || ["", ""];
      tiles.push(tile("J" + (i + 1), name, "J" + (i + 1) + (name ? ", " + name.toLowerCase() + ": " + tip + "." : ".")));
    }
    tiles.push(tile("Gripper", "0 closed, 1 open", "The gripper: 0 is fully closed, 1 is fully open."));
    return tiles;
  }

  // One card per side, made once: its name, a status pill, the adapter it is on, then its joints once it is found.
  function card(side) {
    const holder = document.createElement("div");
    holder.className = "arm";
    holder.innerHTML = '<div class="armhead"><h3></h3><span class="armstate"></span></div><p class="armid"></p>'
      + '<p class="armsub"></p><div class="tiles"></div>';
    holder.querySelector("h3").textContent = side[0].toUpperCase() + side.slice(1) + " arm";
    const fix = document.createElement("div");  // the gripper fix: a card of its own, under the arm's
    fix.className = "gripfix";
    fix.hidden = true;
    return { holder, fix, tiles: [], count: -1 };
  }

  function draw(side, device, row) {
    const c = cards[side] || (cards[side] = card(side));
    const state = c.holder.querySelector(".armstate");
    const id = c.holder.querySelector(".armid");
    const sub = c.holder.querySelector(".armsub");
    const box = c.holder.querySelector(".tiles");
    const adapter = (window.setupDevice && device && window.setupDevice(device)) || {};
    const port = (adapter.usb || "").split(":")[0];
    const portOnly = !!(device && row && !row.error && !adapter.serial && port);
    c.holder.classList.toggle("found", !!(device && row) && !row.error && !portOnly);
    c.holder.classList.toggle("check", portOnly);
    c.holder.classList.toggle("bad", !!(device && row && row.error));
    showGripperFix(c.fix, device && row ? adapter.gripper_mode : null);
    if (!device || !row) {
      state.textContent = "Not found yet";
      id.textContent = "";
      sub.textContent = "Move any joint on the " + side + " arm when asked.";
      box.replaceChildren();
      c.count = -1;
      return c.holder;
    }
    // The card's corner names the adapter the arm is on: its serial number, else its USB port, else its address.
    const named = adapter.serial ? "Serial " + adapter.serial : port ? "USB port " + port + " \u00b7 no serial" : adapter.open || device;
    state.textContent = row.error ? "Can't read" : named;
    id.textContent = "";
    sub.textContent = portOnly ? "This adapter has no serial number. Keep it in USB port " + port + " or it may swap sides." : "";
    if (c.count !== row.joints_deg.length) {
      c.count = row.joints_deg.length;
      c.tiles = tilesFor(c.count);
      box.replaceChildren(...c.tiles);
    }
    if (!first[device]) first[device] = { joints: row.joints_deg.slice(), gripper: row.gripper };
    const from = first[device];
    row.joints_deg.forEach((deg, i) => {
      c.tiles[i].classList.toggle("hot", Math.abs(deg - from.joints[i]) > 5);
      c.tiles[i].querySelector(".jval").textContent = (Math.abs(deg) < 0.05 ? "0.0" : deg.toFixed(1)) + DEG;
    });
    const grip = c.tiles[c.count];
    grip.classList.toggle("hot", row.gripper !== null && from.gripper !== null && Math.abs(row.gripper - from.gripper) > 0.1);
    grip.querySelector(".jval").textContent = row.gripper === null ? "-" : row.gripper.toFixed(2);
    return c.holder;
  }

  // A YAM gripper motor not in MIT mode moves for nothing the station sends: the status bar says so (setup_wizard.js),
  // and this card under the arm gives the steps, unless the wizard offers its own button (`DREAM_SETUP_GRIPPER_MIT`).
  function showGripperFix(holder, mode) {
    const wrong = !!mode && mode !== "mit" && !window.setupGripperSwitch;
    holder.hidden = !wrong;
    if (!wrong || holder.dataset.mode === mode) return;
    holder.dataset.mode = mode;
    holder.innerHTML = '<ol class="steps">'
      + "<li><span>Stop setup: press Control-C in the terminal or SSH session running <code>dreamcli setup</code>.</span></li>"
      + "<li><span>Switch the gripper to MIT mode: <code>dreamcli motors set-mit</code></span></li>"
      + "<li><span>Turn the arm off, wait a few seconds, and turn it back on.</span></li>"
      + "<li><span>Run <code>dreamcli setup</code> again.</span></li></ol>";
  }

  async function poll() {
    if (!$("wiggle") || $("wiggle").classList.contains("hidden") || document.hidden) {
      Object.keys(first).forEach((key) => { delete first[key]; });
      return;
    }
    let status;
    try { status = await (await fetch("/api/status")).json(); } catch (exc) { return; }
    if (!status.wiggle) return;
    const assigned = status.assigned || {};
    $("armCards").replaceChildren(...SIDES.flatMap((side) => [
      draw(side, assigned[side], status.wiggle.devices[assigned[side]]), cards[side].fix,
    ]));
  }
  setInterval(poll, 1000);
}
