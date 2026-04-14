const state = {
  tabs: [],
  activeTabId: null,
  damageMode: false,
  weaponMode: false,
  systemMode: false,
  refitMode: false,
  addSquareMode: false,
  convertContext: null,
  convertMenuHidden: false,
  jsonEditAddBoxSsdKey: "",
  jsonEditAddBoxSpec: null,
  addGroupMode: false,
  addGroupSeedId: null,
  arcHighlight: null
}

let APP_VERSION = null

let energyTemplate = null
let weaponDamageCache = null
let turnMovementTableCache = null
let externalLabelSetCache = null
let jsonEditBoxTypeOptionsCache = null
let jsonEditBoxEntryOptionsCache = null
let jsonEditDatalistCounter = 0
const VEIL_FOLDER_MARKER = "#"
const ANALYSIS_EXTERNAL_KEYWORD_MAP = [
  { key: "boarding", terms: ["boarding parties", "boarding party"] },
  { key: "crew", terms: ["crew units"] },
  { key: "tbomb", terms: ["transporter bombs", "transport bombs", "transporter bomb"] },
  {
    key: "dbomb",
    terms: [
      "transporter dummies",
      "transporter dummy",
      "transporter dummie",
      "transport dummies",
      "transport dummy",
      "transport dummie"
    ]
  },
  { key: "flagbridge", terms: ["flag bridge"] },
  { key: "bridge", terms: ["bridge"] },
  { key: "emer", terms: ["emergency bridge"] },
  { key: "cloak", terms: ["cloak", "cloaking device"] },
  { key: "tran", terms: ["transporter"] },
  { key: "trac", terms: ["tractor"] },
  { key: "fhull", terms: ["forward hull"] },
  { key: "ahull", terms: ["aft hull"] },
  { key: "lwarp", terms: ["left warp"] },
  { key: "rwarp", terms: ["right warp"] },
  { key: "cwarp", terms: ["center warp"] },
  { key: "imp", terms: ["impulse"] },
  { key: "btty", terms: ["battery"] },
  { key: "lab", terms: ["lab"] },
  { key: "adminshuttle", terms: ["admin shuttles", "admin shuttle"] },
  { key: "shuttle", terms: ["shuttle"] },
  { key: "aux", terms: ["auxiliary control"] },
  { key: "apr", terms: ["apr"] },
  { key: "probeammo", terms: ["probes (ammo)", "probe ammo", "probes ammo"] },
  { key: "probe", terms: ["probe"] },
  { key: "damcon", terms: ["damage control"] },
  { key: "excessdamage", terms: ["excess damage"] },
  { key: "sensor", terms: ["sensor"] },
  { key: "scanner", terms: ["scanner"] }
]
const ENERGY_TEMPLATE_LINES = [
  "*WARP ENGINE POWER",
  "*IMPULSE ENGINE POWER",
  "*REACTOR POWER",
  "*TOTAL POWER AVAILABLE",
  "",
  "*BATTERY POWER AVAILABLE",
  "TOTAL BATTERIES",
  "BATTERY POWER RECHARGED",
  "*BATTERY POWER DISCHARGED",
  "",
  "Vuldar Ionization",
  "",
  "LIFE SUPPORT",
  "ACTIVE FIRE CONTROL",
  "",
  "PHASERS",
  "*HELD PHASER CAPACITORS",
  "*MAX PHASER CAPACITORS",
  "CHARGE PHASER CAPACITORS",
  "*CURRENT PHASER CHARGE",
  "",
  "HEAVY WEAPONS",
  "",
  "ACTIVATE SHIELDS",
  "GENERAL REINFORCEMENT",
  "SPECIFIC REINFORCEMENT",
  "",
  "Breaking Energy",
  "ENERGY FOR MOVEMENT",
  "MOVEMENT COST",
  "*SPEED",
  "HET",
  "EM / BRAKING",
  "",
  "DAMAGE CONTROL",
  "RESERVE WARP",
  "TRACTOR / NEGATIVE TRACTOR",
  "Anti-Transporter Field",
  "TRANSPORTERS",
  "",
  "ECM - Defense",
  "ECCM - Offence",
  "LABS",
  "CHARGE WILD WEASEL / SUICIDE SHUTTLE",
  "CLOAKING DEVICE",
  "",
  "*TOTAL POWER USED",
  "*PHASER CAPACITORS USED"
]

let toastTimer = null
let xxviRunning = false

function stripVeilMarkerName(name) {
  const raw = String(name || "").trimEnd()
  if (!raw) return ""
  const marker = VEIL_FOLDER_MARKER
  if (!marker) return raw.trim()
  const rawLower = raw.toLowerCase()
  const markerLower = marker.toLowerCase()
  if (rawLower.endsWith(markerLower)) {
    return raw.slice(0, raw.length - marker.length).trimEnd()
  }
  return raw.trim()
}

function ensureVeilMarkerName(name) {
  const base = stripVeilMarkerName(name)
  if (!base) return ""
  const maxBaseLength = Math.max(1, 80 - VEIL_FOLDER_MARKER.length)
  const clipped = base.slice(0, maxBaseLength).trimEnd()
  return `${clipped}${VEIL_FOLDER_MARKER}`
}

function showTempMessage(text, durationMs = 1600) {
  if (!text) return
  let el = document.getElementById("tempToast")
  if (!el) {
    el = document.createElement("div")
    el.id = "tempToast"
    el.className = "tempToast"
    document.body.appendChild(el)
  }
  el.textContent = text
  el.classList.add("show")

  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => {
    el.classList.remove("show")
  }, Math.max(500, Number(durationMs) || 0))
}

function renderAppVersion() {
  const topBar = document.getElementById("contentTopBar")
  if (!topBar) return

  let el = document.getElementById("appVersion")
  if (!el) {
    el = document.createElement("div")
    el.id = "appVersion"
    el.style.fontSize = "12px"
    el.style.color = "rgba(255, 255, 255, 0.65)"
    el.style.border = "1px solid rgba(255, 255, 255, 0.12)"
    el.style.padding = "4px 8px"
    el.style.borderRadius = "10px"
    el.style.background = "rgba(255, 255, 255, 0.06)"
    el.style.whiteSpace = "nowrap"
    el.style.userSelect = "none"
    el.style.cursor = "pointer"
    el.title = "Check for updates"
    el.onclick = async () => {
      const ok = confirm("Check for updates now?")
      if (!ok) return
      if (window.api && typeof window.api.checkForUpdates === "function") {
        try {
          const res = await window.api.checkForUpdates()
          if (!res || !res.ok) {
            alert(res?.error || "Failed to check for updates.")
          } else {
            showTempMessage("Checking for updates...")
          }
        } catch {
          alert("Failed to check for updates.")
        }
      } else if (window.api && typeof window.api.openUpdates === "function") {
        await window.api.openUpdates()
      } else {
        alert("Update checking isn't configured.")
      }
    }
    topBar.appendChild(el)
  }
  if (APP_VERSION) {
    el.textContent = `v${APP_VERSION}`
    el.style.display = ""
  } else {
    el.textContent = ""
    el.style.display = "none"
  }
}

async function loadAppVersion() {
  if (!window.api || typeof window.api.getAppVersion !== "function") return
  try {
    const v = await window.api.getAppVersion()
    if (v) {
      APP_VERSION = String(v)
      const el = document.getElementById("appVersion")
      if (el) {
        el.textContent = `v${APP_VERSION}`
        el.style.display = ""
      }
    }
  } catch {}
}

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16)
}

function getActiveTab() {
  return state.tabs.find(t => t.id === state.activeTabId) || null
}

function setupXXVIUI() {
  const btn = document.getElementById("btnRunXXVI")
  const panel = document.getElementById("xxviPanel")
  const label = document.getElementById("xxviProgressLabel")
  const fill = document.getElementById("xxviProgressFill")
  const output = document.getElementById("xxviOutput")

  if (!btn || !label || !fill || !output) return

  const updateVisibility = () => {
    if (!panel) return
    const labelText = String(label.textContent || "").trim().toLowerCase()
    const hasOutput = String(output.textContent || "").trim().length > 0
    const isComplete = labelText.startsWith("complete")
    const active = xxviRunning || (!isComplete && (hasOutput || (labelText && labelText !== "idle")))
    panel.classList.toggle("hidden", !active)
    syncLeftPanelScrollbar()
  }

  const appendOutput = (text) => {
    if (!text) return
    output.textContent += (output.textContent ? "\n" : "") + text
    output.scrollTop = output.scrollHeight
    updateVisibility()
  }

  const setRunning = (running) => {
    xxviRunning = running
    btn.disabled = running
    updateVisibility()
  }

  btn.onclick = async () => {
    if (xxviRunning) return
    output.textContent = ""
    fill.style.width = "0%"
    label.textContent = "Starting..."
    setRunning(true)
    updateVisibility()

    if (!window.api || typeof window.api.runShipyardXXVI !== "function") {
      setRunning(false)
      label.textContent = "Error"
      appendOutput("Shipyard XXVI runner is not available.")
      return
    }

    try {
      const res = await window.api.runShipyardXXVI()
      if (!res || !res.ok) {
        setRunning(false)
        label.textContent = "Error"
        appendOutput(res?.error || "Failed to start Shipyard XXVI.")
        updateVisibility()
      }
    } catch (err) {
      setRunning(false)
      label.textContent = "Error"
      appendOutput(err?.message || "Failed to start Shipyard XXVI.")
      updateVisibility()
    }
  }

  window.api.onShipyardXXVIProgress?.((payload) => {
    if (!payload) return
    const line = payload.line || ""
    const percent = payload.percent
    if (typeof percent === "number") {
      fill.style.width = `${Math.max(0, Math.min(100, percent))}%`
      if (line) label.textContent = line
    } else if (line) {
      label.textContent = line
      appendOutput(line)
    }
    updateVisibility()
  })

  window.api.onShipyardXXVIDone?.((payload) => {
    setRunning(false)
    const code = payload?.code
    if (typeof code === "number") {
      label.textContent = code === 0 ? "Complete" : `Exited with code ${code}`
    } else {
      label.textContent = "Complete"
    }
    updateVisibility()

    if (code === 0) {
      const ok = confirm("Read SSD complete. Run Shipyard now?")
      if (ok) {
        runShipyard()
      }
    }
  })

  window.api.onShipyardXXVIError?.((payload) => {
    setRunning(false)
    label.textContent = "Error"
    appendOutput(payload?.error || "Shipyard XXVI failed.")
    updateVisibility()
  })

  updateVisibility()
}

function cleanupSquareTimers(squares) {
  if (!Array.isArray(squares)) return
  for (const s of squares) {
    if (s.phaserGTimer) {
      clearTimeout(s.phaserGTimer)
      delete s.phaserGTimer
    }
  }
}

function setActiveTab(tabId) {
  const oldTab = getActiveTab()
  if (oldTab && oldTab.id !== tabId && oldTab.doc?.squares) {
    cleanupSquareTimers(oldTab.doc.squares)
  }
  state.activeTabId = tabId
  renderTabs()
  renderCanvas()
  renderCurrentProps()
}

function syncLeftPanelScrollbar() {
  const el = document.getElementById("leftPanel")
  if (!el) return
  const needsScroll = el.scrollHeight > el.clientHeight + 1
  el.classList.toggle("noScroll", !needsScroll)
}

function findEnergyKey(obj, targetLower) {
  if (!obj) return null
  for (const k of Object.keys(obj)) {
    if (k.toLowerCase() === targetLower) return k
  }
  return null
}

function syncPhaserRemainingWithMax(tab, maxCaps) {
  if (!tab) return
  if (!tab.energyValues) tab.energyValues = {}
  const key =
    findEnergyKey(tab.energyValues, "held phaser capacitors") ||
    findEnergyKey(tab.energyValues, "phaser capacitors remaining") ||
    "Held phaser capacitors"
  const current = Number(tab.energyValues[key])
  const safeVal = Number.isFinite(current) ? Math.min(current, maxCaps) : maxCaps
  tab.energyValues[key] = safeVal

  tab.energyDisplay = tab.energyDisplay || {}
  const dispKey =
    findEnergyKey(tab.energyDisplay, "held phaser capacitors") ||
    findEnergyKey(tab.energyDisplay, "phaser capacitors remaining") ||
    key
  tab.energyDisplay[dispKey] = tab.energyValues[key]
}

function addEmptyTab() {
  const id = uid()
  state.tabs.push({
    id,
    title: "Untitled",
    affiliation: null,
    shipName: null,

    // UI mode for conditional panels in the left explorer
    uiState: "empty",

    imageDataUrl: null,
    imageObj: null,

    jsonPath: null,
    doc: null,
    selected: null,
    energyValues: {},
    energyDisplay: {},
    turnLocked: false,
    turnNumber: 1,
    view: { scale: 1, ox: 0, oy: 0 },
    customShieldLabels: new Set(),
    gameSavePath: null
  })
  setActiveTab(id)
}

function closeActiveTab() {
  if (!state.activeTabId) return
  closeTabById(state.activeTabId)
}

function closeTabById(tabId) {
  const idx = state.tabs.findIndex(t => t.id === tabId)
  if (idx === -1) return

  const closingTab = state.tabs[idx]
  if (closingTab.doc?.squares) {
    cleanupSquareTimers(closingTab.doc.squares)
  }

  const closingActive = state.activeTabId === tabId
  state.tabs.splice(idx, 1)

  if (state.tabs.length === 0) {
    addEmptyTab()
    return
  }

  if (closingActive) {
    const nextIdx = Math.max(0, idx - 1)
    setActiveTab(state.tabs[nextIdx].id)
  } else {
    renderTabs()
  }
}

function closeAllTabsForNewTask() {
  if (Array.isArray(state.tabs)) {
    for (const tab of state.tabs) {
      if (tab?.doc?.squares) cleanupSquareTimers(tab.doc.squares)
    }
  }
  state.tabs = []
  state.activeTabId = null
  state.convertContext = null
  state.jsonEditAddBoxSsdKey = ""
  state.jsonEditAddBoxSpec = null
  setAddSquareMode(false)
  setAddGroupMode(false)
}

function ensureEmptyActiveTab() {
  const current = getActiveTab()
  if (!current || current.doc) {
    addEmptyTab()
    return getActiveTab()
  }
  return current
}

function renderTabs() {
  const tabsEl = document.getElementById("tabs")
  if (!tabsEl) return
  tabsEl.innerHTML = ""

  for (const tab of state.tabs) {
    const el = document.createElement("div")
    el.className = "tab" + (tab.id === state.activeTabId ? " active" : "")
    el.onclick = () => setActiveTab(tab.id)

    const title = document.createElement("span")
    title.className = "tabTitle"
    title.textContent = tab.title
    if (tab.affiliation === "friendly") title.style.color = "#2edb73"
    else if (tab.affiliation === "enemy") title.style.color = "#ff5c5c"
    el.appendChild(title)

    const closeBtn = document.createElement("button")
    closeBtn.className = "tabClose"
    closeBtn.textContent = "x"
    closeBtn.onclick = (e) => {
      e.stopPropagation()
      closeTabById(tab.id)
    }
    el.appendChild(closeBtn)

    tabsEl.appendChild(el)
  }
}

function normText(s) {
  if (s === null || s === undefined) return ""
  return String(s).trim()
}

function getGroupDisplayName(g) {
  return normText(g.name) || normText(g.label) || ""
}

function getSquareDisplayName(s) {
  return normText(s.label) || normText(s.name) || ""
}

function isSquareRemovedOrDamaged(s) {
  return !!s?.refitRemoved || !!s?.damaged
}

function isBatterySquare(s) {
  const name = getSquareDisplayName(s).toLowerCase()
  return name === "battery" || name.startsWith("battery ")
}

function isGroupLabeled(g) {
  return getGroupDisplayName(g).length > 0
}

function isSquareLabeled(s) {
  return getSquareDisplayName(s).length > 0
}

function baseNameFromPath(p) {
  if (!p) return ""
  const parts = String(p).split(/[/\\]+/g).filter(Boolean)
  if (parts.length === 0) return ""
  return parts[parts.length - 1]
}

function fileStemFromPath(p) {
  const name = baseNameFromPath(p)
  const i = name.lastIndexOf(".")
  if (i <= 0) return name
  return name.slice(0, i)
}

/* -----------------------------
   Shipyard visibility helpers
----------------------------- */
function isShipyardDoneVisible() {
  const panel = document.getElementById("shipyardPanel")
  const done = document.getElementById("shipyardDoneStep")
  if (!panel || !done) return false
  return !panel.classList.contains("hidden") && !done.classList.contains("hidden")
}

function isShipyardLabelVisible() {
  const panel = document.getElementById("shipyardPanel")
  const labelStep = document.getElementById("shipyardLabelStep")
  if (!panel || !labelStep) return false
  return !panel.classList.contains("hidden") && !labelStep.classList.contains("hidden")
}

function setAddSquareMode(active) {
  state.addSquareMode = Boolean(active)
  const btnIds = ["shipyardAddSquare", "jsonEditAddBox"]
  for (const id of btnIds) {
    const btn = document.getElementById(id)
    if (!btn) continue
    btn.classList.toggle("addSquareActive", state.addSquareMode)
  }
}

function setAddGroupMode(active, options = {}) {
  const resetSeed = options.resetSeed !== false
  state.addGroupMode = Boolean(active)
  if (resetSeed) state.addGroupSeedId = null
  const btn = document.getElementById("shipyardAddGroup")
  if (btn) btn.classList.toggle("addGroupActive", state.addGroupMode)
}

/* -----------------------------
   Close tab from Shipyard exit paths
----------------------------- */
function closeTabFromShipyard() {
  hideShipyardPanel()
  closeActiveTab()
}

/* -----------------------------
   Image caching
----------------------------- */
function ensureTabImageLoaded(tab) {
  if (!tab || !tab.imageDataUrl) return Promise.resolve(false)

  if (tab.imageObj && tab.imageObj._src === tab.imageDataUrl) {
    return Promise.resolve(true)
  }

  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      img._src = tab.imageDataUrl
      tab.imageObj = img
      resolve(true)
    }
    img.onerror = () => resolve(false)
    img.src = tab.imageDataUrl
  })
}


/* -----------------------------
   Canvas
----------------------------- */
function fitImageToCanvas(imgW, imgH, canvasW, canvasH) {
  const scale = Math.min(canvasW / imgW, canvasH / imgH)
  const drawW = imgW * scale
  const drawH = imgH * scale
  const ox = (canvasW - drawW) / 2
  const oy = (canvasH - drawH) / 2
  return { scale, ox, oy }
}

function renderCanvas() {
  const tab = getActiveTab()
  const canvas = document.getElementById("canvas")
  const ctx = canvas.getContext("2d")

  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.floor(rect.width * dpr)
  canvas.height = Math.floor(rect.height * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  ctx.clearRect(0, 0, rect.width, rect.height)

  if (!tab || !tab.doc) {
    hideAnalysisTooltip()
    return
  }

  if (tab.uiState === "analysis") {
    const primarySsdHighlight = getAnalysisSsdOnlyHighlightWeapon(tab)
    const hasCompare = Boolean(tab.analysisCompare)
    if (!hasCompare) {
      if (primarySsdHighlight && drawAnalysisSsdPreview(ctx, tab, rect, { weapon: primarySsdHighlight })) {
        hideAnalysisTooltip()
        return
      }
      drawAnalysisHexMap(ctx, tab, rect)
      return
    }

    const layout = getAnalysisCompareLayout(rect, true)
    const compareMode = tab.analysisCompareMode || "normal"
    const diffScale = compareMode === "difference"
      ? computeAnalysisDiffScale(tab.analysis, tab.analysisCompare, tab.analysisRadius)
      : 0
    const showCompareTooltip = tab.analysisHoverSide === "compare"
    const showPrimaryTooltip = tab.analysisHoverSide !== "compare"
    if (!tab.analysisHoverSide) hideAnalysisTooltip()

    let primaryRenderedAsSsd = false
    if (primarySsdHighlight) {
      primaryRenderedAsSsd = drawAnalysisSsdPreview(ctx, tab, layout.primary, {
        weapon: primarySsdHighlight,
        offsetX: layout.primary.x,
        offsetY: layout.primary.y
      })
      if (primaryRenderedAsSsd && tab.analysisHoverSide !== "compare") hideAnalysisTooltip()
    }
    if (!primaryRenderedAsSsd) {
      drawAnalysisHexMap(ctx, tab, layout.primary, {
        analysis: tab.analysis,
        compareAnalysis: tab.analysisCompare,
        compareMode,
        diffScale,
        highlight: tab.analysisHighlight,
        labelKeys: tab.analysisHexLabels,
        hover: tab.analysisHover,
        offsetX: layout.primary.x,
        offsetY: layout.primary.y,
        radius: tab.analysis?.radius,
        showTooltip: showPrimaryTooltip
      })
    }

    drawAnalysisHexMap(ctx, tab, layout.compare, {
      analysis: tab.analysisCompare,
      compareAnalysis: tab.analysis,
      compareMode,
      diffScale,
      highlight: null,
      labelKeys: tab.analysisCompareHexLabels,
      hover: tab.analysisCompareHover,
      offsetX: layout.compare.x,
      offsetY: layout.compare.y,
      radius: tab.analysisCompare?.radius,
      shipLabel: tab.analysisCompareLabel,
      doc: tab.analysisCompareDoc,
      showTooltip: showCompareTooltip
    })

    ctx.save()
    ctx.strokeStyle = "rgba(255, 255, 255, 0.18)"
    ctx.lineWidth = 1
    const dividerX = layout.compare.x - layout.gap / 2
    ctx.beginPath()
    ctx.moveTo(dividerX, 0)
    ctx.lineTo(dividerX, rect.height)
    ctx.stroke()
    ctx.restore()
    return
  }

  if (tab.uiState === "movementAnalysis") {
    hideAnalysisTooltip()
    drawMovementEndpointMap(ctx, tab, rect)
    return
  }

  hideAnalysisTooltip()

  if (!tab.imageDataUrl) {
    return
  }

  if (!tab.imageObj || tab.imageObj._src !== tab.imageDataUrl) {
    ensureTabImageLoaded(tab).then((ok) => {
      if (!ok) return
      renderCanvas()
    })
    return
  }

  const img = tab.imageObj
  const fit = fitImageToCanvas(img.width, img.height, rect.width, rect.height)
  tab.view.scale = fit.scale
  tab.view.ox = fit.ox
  tab.view.oy = fit.oy

  ctx.drawImage(img, fit.ox, fit.oy, img.width * fit.scale, img.height * fit.scale)

  drawDamagedFill(ctx, tab, fit)
  drawSystemLines(ctx, tab, fit)
  drawRefitSquares(ctx, tab, fit)
  drawDamagedOutlines(ctx, tab, fit)
  drawFiredSquares(ctx, tab, fit)

  const hideOverlays = false
  if (!hideOverlays) {
    drawGroups(ctx, tab, fit)
    drawSquares(ctx, tab, fit)
    drawSelection(ctx, tab, fit)
    drawArcHighlight(ctx, fit)
  }
}

function drawGroups(ctx, tab, fit) {
  const groups = tab.doc.groups || []
  ctx.save()
  ctx.lineWidth = 1
  ctx.globalAlpha = 0.9
  ctx.strokeStyle = "#ffd166"

  for (const g of groups) {
    if (!g.bbox) continue
    const x = fit.ox + g.bbox.x1 * fit.scale
    const y = fit.oy + g.bbox.y1 * fit.scale
    const w = (g.bbox.x2 - g.bbox.x1) * fit.scale
    const h = (g.bbox.y2 - g.bbox.y1) * fit.scale
    ctx.strokeRect(x, y, w, h)
  }

  ctx.restore()
}

function drawSquares(ctx, tab, fit) {
  const squares = tab.doc.squares || []
  ctx.save()
  ctx.lineWidth = 1
  ctx.globalAlpha = 0.9

  for (const s of squares) {
    const isUngrouped = s.groupId === null
    ctx.strokeStyle = isUngrouped ? "#8ecae6" : "#90ee90"

    const pts = s.points || []
    if (pts.length < 4) continue

    ctx.beginPath()
    for (let i = 0; i < pts.length; i++) {
      const px = fit.ox + pts[i][0] * fit.scale
      const py = fit.oy + pts[i][1] * fit.scale
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.stroke()
  }

  ctx.restore()
}

function drawDamagedFill(ctx, tab, fit) {
  const squares = tab.doc.squares || []
  ctx.save()
  ctx.globalAlpha = 1
  ctx.fillStyle = "#d70000ff"

  for (const s of squares) {
    if (!s.damaged) continue
    const pts = s.points || []
    if (pts.length < 4) continue

    ctx.beginPath()
    for (let i = 0; i < pts.length; i++) {
      const px = fit.ox + pts[i][0] * fit.scale
      const py = fit.oy + pts[i][1] * fit.scale
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.fill()
  }

  ctx.restore()
}

function drawRefitSquares(ctx, tab, fit) {
  const squares = tab.doc.squares || []
  ctx.save()
  ctx.globalAlpha = 0.85
  ctx.fillStyle = "#000"
  ctx.strokeStyle = "#000"
  ctx.lineWidth = 1.5

  for (const s of squares) {
    if (!s.refitRemoved) continue
    const pts = s.points || []
    if (pts.length < 4) continue

    ctx.beginPath()
    for (let i = 0; i < pts.length; i++) {
      const px = fit.ox + pts[i][0] * fit.scale
      const py = fit.oy + pts[i][1] * fit.scale
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
  }

  ctx.restore()
}

function drawSystemLines(ctx, tab, fit) {
  const squares = tab.doc.squares || []
  ctx.save()
  ctx.globalAlpha = 0.85
  ctx.strokeStyle = "#2de37f"
  ctx.lineWidth = 3

  for (const s of squares) {
    if (!s.systemDeployed) continue
    const b = s.bbox
    if (!b) continue

    const x1 = fit.ox + b.x1 * fit.scale
    const y1 = fit.oy + b.y1 * fit.scale
    const x2 = fit.ox + b.x2 * fit.scale
    const y2 = fit.oy + b.y2 * fit.scale

    // Draw from top right to bottom left
    ctx.beginPath()
    ctx.moveTo(x2, y1)
    ctx.lineTo(x1, y2)
    ctx.stroke()
  }

  ctx.restore()
}

function drawDamagedOutlines(ctx, tab, fit) {
  const squares = tab.doc.squares || []
  ctx.save()
  ctx.globalAlpha = 0.9
  ctx.strokeStyle = "#b83232"
  ctx.lineWidth = 1.5

  for (const s of squares) {
    if (!s.damaged) continue
    if (s.refitRemoved) continue
    if (s.fired) continue
    const pts = s.points || []
    if (pts.length < 4) continue

    ctx.beginPath()
    for (let i = 0; i < pts.length; i++) {
      const px = fit.ox + pts[i][0] * fit.scale
      const py = fit.oy + pts[i][1] * fit.scale
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.stroke()
  }

  ctx.restore()
}

function drawFiredSquares(ctx, tab, fit) {
  const squares = tab.doc.squares || []
  ctx.save()
  ctx.lineWidth = 2
  ctx.globalAlpha = 0.9
  ctx.strokeStyle = "#66b3ff"

  for (const s of squares) {
    if (!s.fired) continue
    const pts = s.points || []
    if (pts.length < 4) continue

    ctx.beginPath()
    for (let i = 0; i < pts.length; i++) {
      const px = fit.ox + pts[i][0] * fit.scale
      const py = fit.oy + pts[i][1] * fit.scale
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.stroke()
  }

  ctx.restore()
}

function drawSelection(ctx, tab, fit) {
  if (!tab.selected) return

  ctx.save()
  ctx.lineWidth = 3
  ctx.globalAlpha = 1
  ctx.strokeStyle = "#ff4d6d"

  if (tab.selected.kind === "square") {
    const s = (tab.doc.squares || []).find(x => x.id === tab.selected.id)
    if (!s) return
    const pts = s.points || []
    ctx.beginPath()
    for (let i = 0; i < pts.length; i++) {
      const px = fit.ox + pts[i][0] * fit.scale
      const py = fit.oy + pts[i][1] * fit.scale
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.stroke()
  } else {
    const g = (tab.doc.groups || []).find(x => x.id === tab.selected.id)
    if (!g || !g.bbox) return
    const x = fit.ox + g.bbox.x1 * fit.scale
    const y = fit.oy + g.bbox.y1 * fit.scale
    const w = (g.bbox.x2 - g.bbox.x1) * fit.scale
    const h = (g.bbox.y2 - g.bbox.y1) * fit.scale
    ctx.strokeRect(x, y, w, h)
  }

  ctx.restore()
}

function drawAnalysisSsdPreview(ctx, tab, rect, options = {}) {
  if (!ctx || !tab || tab.uiState !== "analysis") return false
  const weapon = options.weapon || getAnalysisSsdOnlyHighlightWeapon(tab)
  if (!weapon || !analysisWeaponHasSsdGeometry(weapon)) return false
  const img = tab.imageObj
  if (!img) return false

  const width = Number(rect?.width) || 0
  const height = Number(rect?.height) || 0
  if (width <= 0 || height <= 0) return false

  const offsetX = Number(options.offsetX) || 0
  const offsetY = Number(options.offsetY) || 0
  const fit = fitImageToCanvas(img.width, img.height, width, height)

  ctx.save()
  ctx.translate(offsetX, offsetY)
  ctx.fillStyle = "rgba(18, 18, 18, 0.94)"
  ctx.fillRect(0, 0, width, height)

  ctx.drawImage(img, fit.ox, fit.oy, img.width * fit.scale, img.height * fit.scale)

  ctx.fillStyle = "rgba(0, 0, 0, 0.35)"
  ctx.fillRect(0, 0, width, height)
  ctx.drawImage(img, fit.ox, fit.oy, img.width * fit.scale, img.height * fit.scale)

  drawSingleArcHighlight(ctx, fit, {
    points: Array.isArray(weapon.points) ? weapon.points : null,
    bbox: weapon?.bbox && typeof weapon.bbox === "object" ? weapon.bbox : null,
    strokeStyle: "#f59e0b",
    lineDash: [7, 4],
    lineWidth: 4,
    alpha: 0.98
  })

  const titleLines = [
    getShipLabelFromDoc(tab.doc, tab.title || "Ship"),
    "Selected weapon does not bear on the map.",
    "Highlighted on SSD."
  ]
  ctx.font = '12px "Segoe UI", sans-serif'
  let maxTextWidth = 0
  for (const line of titleLines) {
    const textWidth = ctx.measureText(line).width
    if (textWidth > maxTextWidth) maxTextWidth = textWidth
  }
  const boxX = 10
  const boxY = 10
  const lineHeight = 16
  const boxW = Math.min(width - 20, maxTextWidth + 16)
  const boxH = titleLines.length * lineHeight + 8
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)"
  ctx.strokeStyle = "rgba(255, 255, 255, 0.18)"
  ctx.lineWidth = 1
  ctx.fillRect(boxX, boxY, boxW, boxH)
  ctx.strokeRect(boxX, boxY, boxW, boxH)
  let y = boxY + 14
  for (let i = 0; i < titleLines.length; i++) {
    ctx.fillStyle = i === 0 ? "rgba(255, 255, 255, 0.95)" : "rgba(245, 245, 245, 0.92)"
    ctx.fillText(titleLines[i], boxX + 8, y)
    y += lineHeight
  }

  ctx.restore()
  return true
}

const ANALYSIS_HEX_RADIUS_DEFAULT = 15
const ANALYSIS_HEX_RADIUS_MIN = 5
const ANALYSIS_HEX_RADIUS_MAX = 30
const ANALYSIS_HEX_WEDGE = (Math.PI * 2) / 6
const ANALYSIS_HEX_HEMISPHERE_EPS = 1e-6
const ANALYSIS_HEX_ARC_EPS = 1e-6
const MOVEMENT_ENDPOINT_MAX_MOVES = 64
const MOVEMENT_SPEED_MAX = 32
const MOVEMENT_ASSUMED_FORWARD_HEADING = 0 // up on the hex map
const MOVEMENT_SIDESLIP_REQUIRED_STRAIGHT = 1
const MOVEMENT_HEX_DIRS = [
  { q: 0, r: -1 }, // up
  { q: 1, r: -1 }, // up-right
  { q: 1, r: 0 },  // down-right
  { q: 0, r: 1 },  // down
  { q: -1, r: 1 }, // down-left
  { q: -1, r: 0 }  // up-left
]
const MOVEMENT_HEADING_OPTION_LABELS = [
  "Forward (Up)",
  "Forward-Right",
  "Rear-Right",
  "Rear (Down)",
  "Rear-Left",
  "Forward-Left"
]

function normalizeAnalysisHexRadius(value) {
  const raw = Math.round(Number(value))
  if (!Number.isFinite(raw)) return ANALYSIS_HEX_RADIUS_DEFAULT
  if (raw < ANALYSIS_HEX_RADIUS_MIN) return ANALYSIS_HEX_RADIUS_MIN
  if (raw > ANALYSIS_HEX_RADIUS_MAX) return ANALYSIS_HEX_RADIUS_MAX
  return raw
}

function getAnalysisHexRadius(tab) {
  return normalizeAnalysisHexRadius(tab?.analysisRadius)
}

function getAnalysisSectorIndex(dx, dy) {
  const angle = Math.atan2(dy, dx)
  const normalized = normalizeAnalysisAngle(angle)
  // Sector boundaries align with the arrows; each arc spans between arrows.
  return Math.floor(normalized / ANALYSIS_HEX_WEDGE) % 6
}

function normalizeAnalysisAngle(angle) {
  return (angle + Math.PI / 2 + Math.PI * 2) % (Math.PI * 2)
}

function getAnalysisHexAngle(q, r) {
  const dx = 1.5 * q
  const dy = Math.sqrt(3) * (r + q / 2)
  return normalizeAnalysisAngle(Math.atan2(dy, dx))
}

function getAnalysisHexSectors(q, r) {
  if (q === 0 && r === 0) return []
  const s = -q - r
  if (q === 0 && r !== 0) return r < 0 ? [5, 0] : [2, 3]
  if (r === 0 && q !== 0) return q > 0 ? [1, 2] : [4, 5]
  if (s === 0 && q !== 0) return q > 0 ? [0, 1] : [3, 4]

  const dx = 1.5 * q
  const dy = Math.sqrt(3) * (r + q / 2)
  return [getAnalysisSectorIndex(dx, dy)]
}

function getAnalysisHexRange(q, r) {
  const s = -q - r
  return Math.max(Math.abs(q), Math.abs(r), Math.abs(s))
}

function getAnalysisHemisphereValue(q, r) {
  return r + q / 2
}

function isAnalysisFrontHemisphere(q, r) {
  return getAnalysisHemisphereValue(q, r) <= ANALYSIS_HEX_HEMISPHERE_EPS
}

function isAnalysisRearHemisphere(q, r) {
  return getAnalysisHemisphereValue(q, r) >= -ANALYSIS_HEX_HEMISPHERE_EPS
}

function weaponCoversHex(weapon, sectors, isFrontHemisphere, isRearHemisphere, hexAngle) {
  if (!weapon) return false
  if (weapon.hasAll) return true
  if (weapon.hasFrontHemisphere && isFrontHemisphere) return true
  if (weapon.hasRearHemisphere && isRearHemisphere) return true
  if (weapon.plasmaParts && weapon.plasmaParts.size > 0 && Number.isFinite(hexAngle)) {
    for (const part of weapon.plasmaParts) {
      if (isAngleInPlasmaArc(hexAngle, part)) return true
    }
  }
  if (weapon.sectors && weapon.sectors.size > 0) {
    for (const idx of sectors) {
      if (weapon.sectors.has(idx)) return true
    }
  }
  return false
}

function getAnalysisHexGeometry(rect, radius) {
  if (!rect) return null
  const width = rect.width
  const height = rect.height
  const padding = Math.min(width, height) * 0.06
  const usableW = Math.max(10, width - padding * 2)
  const usableH = Math.max(10, height - padding * 2)
  const safeRadius = Number.isFinite(radius) && radius > 0 ? radius : ANALYSIS_HEX_RADIUS_DEFAULT
  const size = Math.min(
    usableW / (safeRadius * 3 + 2),
    usableH / (Math.sqrt(3) * (safeRadius * 2 + 1))
  )
  if (!Number.isFinite(size) || size <= 0) return null

  const center = { x: width / 2, y: height / 2 }
  const axialToPixel = (q, r) => ({
    x: center.x + size * 1.5 * q,
    y: center.y + size * Math.sqrt(3) * (r + q / 2)
  })

  return { width, height, padding, radius: safeRadius, size, center, axialToPixel }
}

function axialRound(q, r) {
  let x = q
  let z = r
  let y = -x - z

  let rx = Math.round(x)
  let ry = Math.round(y)
  let rz = Math.round(z)

  const xDiff = Math.abs(rx - x)
  const yDiff = Math.abs(ry - y)
  const zDiff = Math.abs(rz - z)

  if (xDiff > yDiff && xDiff > zDiff) {
    rx = -ry - rz
  } else if (yDiff > zDiff) {
    ry = -rx - rz
  } else {
    rz = -rx - ry
  }

  return { q: rx, r: rz }
}

function analysisPointToHex(rect, x, y, radius) {
  const geom = getAnalysisHexGeometry(rect, radius)
  if (!geom) return null

  const dx = x - geom.center.x
  const dy = y - geom.center.y
  const q = (2 / 3) * (dx / geom.size)
  const r = (-1 / 3) * (dx / geom.size) + (1 / Math.sqrt(3)) * (dy / geom.size)
  const rounded = axialRound(q, r)
  const s = -rounded.q - rounded.r
  const dist = Math.max(Math.abs(rounded.q), Math.abs(rounded.r), Math.abs(s))
  if (dist > geom.radius) return null

  return rounded
}

function getShipLabelFromDoc(doc, fallback) {
  const shipData = doc?.shipData || {}
  const shipName = String(doc?.shipName || "").trim()
  const empire = String(shipData.empire || "").trim()
  const type = String(shipData.type || "").trim()
  return shipName || [empire, type].filter(Boolean).join(" ") || fallback || "Ship"
}

function getAnalysisCompareLayout(rect, hasCompare) {
  const width = rect.width
  const height = rect.height
  if (!hasCompare) {
    return {
      primary: { x: 0, y: 0, width, height },
      compare: null,
      gap: 0
    }
  }
  const gap = Math.max(14, Math.round(width * 0.03))
  const paneWidth = Math.max(10, (width - gap) / 2)
  return {
    primary: { x: 0, y: 0, width: paneWidth, height },
    compare: { x: paneWidth + gap, y: 0, width: paneWidth, height },
    gap
  }
}

function getAnalysisPaneAtPoint(tab, rect, x, y) {
  const layout = getAnalysisCompareLayout(rect, Boolean(tab?.analysisCompare))
  if (!layout.compare) {
    return { side: "primary", rect: layout.primary, layout }
  }
  if (x >= layout.compare.x && x <= layout.compare.x + layout.compare.width) {
    return { side: "compare", rect: layout.compare, layout }
  }
  if (x >= layout.primary.x && x <= layout.primary.x + layout.primary.width) {
    return { side: "primary", rect: layout.primary, layout }
  }
  return { side: null, rect: null, layout }
}

function drawAnalysisHexMap(ctx, tab, rect, options = {}) {
  if (!tab || tab.uiState !== "analysis") return
  const analysis = options.analysis || tab.analysis
  if (!analysis) return

  const { maxHeat, avgHeat } = analysis
  const highlightValue = options.highlight !== undefined ? options.highlight : tab.analysisHighlight
  const highlightWeapon = highlightValue && typeof highlightValue === "object"
    ? highlightValue
    : null
  const highlightPart = typeof highlightValue === "string"
    ? highlightValue.trim().toUpperCase()
    : ""
  const highlightHemisphere =
    highlightPart === "FH"
      ? "front"
      : highlightPart === "RH"
        ? "rear"
        : null
  const highlightPlasma = highlightPart && PLASMA_ARC_RANGES[highlightPart]
    ? highlightPart
    : null
  const highlightSectors = !highlightHemisphere && !highlightPlasma && highlightPart && ARC_SECTOR_MAP[highlightPart]
    ? new Set(ARC_SECTOR_MAP[highlightPart])
    : null
  const highlightMode = highlightWeapon
    ? "weapon"
    : highlightHemisphere
      ? "hemisphere"
      : highlightPlasma || highlightSectors
        ? "arc"
        : null
  const compareAnalysis = options.compareAnalysis || null
  const compareMode = options.compareMode || "normal"
  const diffMode = Boolean(compareAnalysis && compareMode === "difference")
  const diffScale = Number.isFinite(options.diffScale) ? options.diffScale : 0
  const width = rect.width
  const height = rect.height
  const offsetX = Number(options.offsetX) || 0
  const offsetY = Number(options.offsetY) || 0
  const radius = normalizeAnalysisHexRadius(options.radius ?? analysis.radius ?? tab.analysisRadius)
  const showTooltip = options.showTooltip !== false
  const labelKeys = options.labelKeys instanceof Set
    ? options.labelKeys
    : tab.analysisHexLabels instanceof Set
      ? tab.analysisHexLabels
      : null
  const hover = options.hover !== undefined ? options.hover : tab.analysisHover

  const geom = getAnalysisHexGeometry(rect, radius)
  if (!geom) return
  const { padding, size, center, axialToPixel } = geom
  const uiScale = Math.max(0.75, Math.min(1.35, Math.min(width, height) / 700))
  const labelFontSize = Math.max(10, Math.round(size * 0.55))

  const drawHex = (cx, cy, fill, stroke, strokeWidth) => {
    ctx.beginPath()
    for (let i = 0; i < 6; i++) {
      const angle = i * (Math.PI / 3)
      const px = cx + size * Math.cos(angle)
      const py = cy + size * Math.sin(angle)
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    if (fill) {
      ctx.fillStyle = fill
      ctx.fill()
    }
    if (stroke) {
      ctx.strokeStyle = stroke
      ctx.lineWidth = Number.isFinite(strokeWidth) ? strokeWidth : 1
      ctx.stroke()
    }
  }

  ctx.save()
    ctx.translate(offsetX, offsetY)
    ctx.fillStyle = "rgba(26, 26, 26, 0.85)"
    ctx.fillRect(0, 0, width, height)

  const avgValue = Number(avgHeat)
  const maxValue = Number(maxHeat)
  const maxScale = Number.isFinite(maxValue) && maxValue > 0
    ? maxValue
    : Math.max(0, avgValue || 0)
  for (let q = -radius; q <= radius; q++) {
    for (let r = -radius; r <= radius; r++) {
      const s = -q - r
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) > radius) continue

      const pos = axialToPixel(q, r)
      const hexKey = `${q},${r}`
      if (q === 0 && r === 0) {
        drawHex(pos.x, pos.y, "rgba(77, 77, 77, 0.25)", null)
        if (labelKeys && labelKeys.has(hexKey)) {
          ctx.font = `${labelFontSize}px "Segoe UI", sans-serif`
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.lineWidth = Math.max(2, size * 0.12)
          ctx.strokeStyle = "rgba(0, 0, 0, 0.65)"
          ctx.fillStyle = "rgba(255, 255, 255, 0.95)"
          ctx.strokeText("0", pos.x, pos.y)
          ctx.fillText("0", pos.x, pos.y)
        }
        continue
      }

      const sectors = getAnalysisHexSectors(q, r)
      const hexAngle = getAnalysisHexAngle(q, r)
      const range = getAnalysisHexRange(q, r)
      const isFrontHemisphere = isAnalysisFrontHemisphere(q, r)
      const isRearHemisphere = isAnalysisRearHemisphere(q, r)
      let count = 0
      if (sectors.length > 0) {
        count = getAnalysisHexDamage(analysis, sectors, hexAngle, range, isFrontHemisphere, isRearHemisphere)
      }
      let compareCount = 0
      let delta = 0
      if (diffMode && sectors.length > 0) {
        compareCount = getAnalysisHexDamage(compareAnalysis, sectors, hexAngle, range, isFrontHemisphere, isRearHemisphere)
        delta = count - compareCount
      }
      const isHighlighted = highlightWeapon
        ? weaponCoversHex(highlightWeapon, sectors, isFrontHemisphere, isRearHemisphere, hexAngle)
        : highlightHemisphere
          ? (highlightHemisphere === "front" ? isFrontHemisphere : isRearHemisphere)
          : highlightPlasma
            ? isAngleInPlasmaArc(hexAngle, highlightPlasma)
            : highlightSectors
              ? sectors.some(idx => highlightSectors.has(idx))
              : false
      const highlightActive = Boolean(highlightMode)
      const highlightStrong = Boolean(isHighlighted && (highlightMode === "hemisphere" || highlightMode === "weapon"))
      const dimFactor = highlightActive && !isHighlighted
        ? (highlightMode === "hemisphere" ? 0.18 : 0.25)
        : 1
      let fill = ""
      if (diffMode) {
        const absDelta = Math.abs(delta)
        const t = diffScale > 0 ? Math.min(1, absDelta / diffScale) : 0
        const baseFillAlpha = delta === 0 ? 0.08 : 0.2 + t * 0.6
        const fillAlpha = Math.min(
          0.95,
          baseFillAlpha * dimFactor + (highlightStrong ? 0.12 : 0)
        )
        if (absDelta <= 0.02) {
          fill = `rgba(0, 0, 0, ${fillAlpha})`
        } else if (delta > 0) {
          fill = `hsla(210, 85%, 60%, ${fillAlpha})`
        } else {
          fill = `hsla(320, 90%, 58%, ${fillAlpha})`
        }
      } else {
        const t = maxScale > 0 ? Math.min(1, Math.max(0, count) / maxScale) : 0
        let hue = 0
        let saturation = 90
        let lightness = 46
        if (count >= 151) {
          hue = 0
          saturation = 90
          lightness = 46
        } else if (count >= 101) {
          hue = 28
          saturation = 90
          lightness = 50
        } else if (count >= 51) {
          hue = 50
          saturation = 90
          lightness = 55
        } else {
          hue = 120
          saturation = 70
          lightness = 45
        }
        const alpha = 0.18 + t * 0.62
        const baseFillAlpha = count > 0 ? alpha : 0.12
        const fillAlpha = Math.min(
          0.95,
          baseFillAlpha * dimFactor + (highlightStrong ? 0.12 : 0)
        )
        fill = `hsla(${hue}, ${saturation}%, ${lightness}%, ${fillAlpha})`
      }
      drawHex(pos.x, pos.y, fill, null, 0)

      if (labelKeys && labelKeys.has(hexKey)) {
        const label = diffMode ? formatSignedAnalysisCount(delta) : formatAnalysisCount(count)
        ctx.font = `${labelFontSize}px "Segoe UI", sans-serif`
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.lineWidth = Math.max(2, size * 0.12)
        ctx.strokeStyle = "rgba(0, 0, 0, 0.65)"
        ctx.fillStyle = "rgba(255, 255, 255, 0.95)"
        ctx.strokeText(label, pos.x, pos.y)
        ctx.fillText(label, pos.x, pos.y)
      }
    }
  }

  const triSide = size * 0.85
  const triHeight = (Math.sqrt(3) / 2) * triSide
  const triTop = { x: center.x, y: center.y - (2 * triHeight) / 3 }
  const triLeft = { x: center.x - triSide / 2, y: center.y + triHeight / 3 }
  const triRight = { x: center.x + triSide / 2, y: center.y + triHeight / 3 }

  ctx.fillStyle = "#ffcc00"
  ctx.strokeStyle = "rgba(0, 0, 0, 0.35)"
  ctx.lineWidth = Math.max(1, size * 0.08)
  ctx.beginPath()
  ctx.moveTo(triTop.x, triTop.y)
  ctx.lineTo(triLeft.x, triLeft.y)
  ctx.lineTo(triRight.x, triRight.y)
  ctx.closePath()
  ctx.fill()
  ctx.stroke()

  const shipLabel = options.shipLabel || getShipLabelFromDoc(options.doc || tab?.doc, tab?.title || "Ship")
  const shipLabelSize = Math.max(14, Math.round(18 * uiScale))
  ctx.font = `600 ${shipLabelSize}px "Segoe UI", sans-serif`
  ctx.textAlign = "left"
  ctx.textBaseline = "top"
  ctx.lineWidth = Math.max(2, size * 0.12)
  ctx.strokeStyle = "rgba(0, 0, 0, 0.65)"
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)"
  ctx.strokeText(shipLabel, padding, Math.max(4, padding * 0.3))
  ctx.fillText(shipLabel, padding, Math.max(4, padding * 0.3))

  const legendItems = diffMode
    ? [
        { label: "Lower", h: 320, s: 90, l: 58 },
        { label: "Higher", h: 210, s: 85, l: 60 }
      ]
    : [
        { label: "0-50", h: 120, s: 70, l: 45 },
        { label: "51-100", h: 50, s: 90, l: 55 },
        { label: "101-150", h: 28, s: 90, l: 50 },
        { label: "151+", h: 0, s: 90, l: 46 }
      ]
  const legendWidth = Math.round(160 * uiScale)
  const legendItemHeight = Math.round(16 * uiScale)
  const legendGap = Math.round(5 * uiScale)
  const legendHeight = legendItemHeight * legendItems.length + legendGap * (legendItems.length - 1)
  const legendX = width - padding - legendWidth
  const legendY = height - padding - legendHeight
  const labelSize = Math.max(10, Math.round(12 * uiScale))

  for (let i = 0; i < legendItems.length; i++) {
    const item = legendItems[i]
    const y = legendY + i * (legendItemHeight + legendGap)
    const gradient = ctx.createLinearGradient(legendX, y, legendX + legendWidth, y)
    gradient.addColorStop(0, `hsla(${item.h}, ${item.s}%, ${item.l}%, 0.2)`)
    gradient.addColorStop(0.55, `hsla(${item.h}, ${item.s}%, ${item.l}%, 0.6)`)
    gradient.addColorStop(1, `hsla(${item.h}, ${item.s}%, ${item.l}%, 0.95)`)
    ctx.fillStyle = gradient
    ctx.fillRect(legendX, y, legendWidth, legendItemHeight)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)"
    ctx.lineWidth = 1
    ctx.strokeRect(legendX, y, legendWidth, legendItemHeight)

    ctx.font = `600 ${labelSize}px "Segoe UI", sans-serif`
    ctx.fillStyle = "rgba(255, 255, 255, 0.85)"
    ctx.textBaseline = "middle"
    ctx.textAlign = "center"
    ctx.fillText(item.label, legendX + legendWidth / 2, y + legendItemHeight / 2)
  }

  if (showTooltip) {
    updateAnalysisTooltip(tab, geom, hover, labelKeys, {
      analysis,
      compareAnalysis: diffMode ? compareAnalysis : null,
      compareMode,
      offsetX,
      offsetY,
      width,
      height
    })
  }

  ctx.restore()
}

function analysisWeaponHasMapHighlight(weapon) {
  return Boolean(
    weapon?.hasAll ||
    weapon?.hasFrontHemisphere ||
    weapon?.hasRearHemisphere ||
    (weapon?.sectors && weapon.sectors.size > 0) ||
    (weapon?.plasmaParts && weapon.plasmaParts.size > 0)
  )
}

function analysisWeaponHasSsdGeometry(weapon) {
  if (!weapon || typeof weapon !== "object") return false
  const points = Array.isArray(weapon.points) ? weapon.points : null
  const bbox = weapon?.bbox && typeof weapon.bbox === "object" ? weapon.bbox : null
  return Boolean((points && points.length >= 4) || bbox)
}

function getAnalysisSsdOnlyHighlightWeapon(tab) {
  const weapon = tab?.analysisHighlight && typeof tab.analysisHighlight === "object"
    ? tab.analysisHighlight
    : null
  if (!weapon) return null
  if (analysisWeaponHasMapHighlight(weapon)) return null
  if (!analysisWeaponHasSsdGeometry(weapon)) return null
  return weapon
}

function analysisPaneUsesSsdPreview(tab, side) {
  if (!tab || tab.uiState !== "analysis") return false
  if (side !== "primary") return false
  if (!tab.imageObj) return false
  return Boolean(getAnalysisSsdOnlyHighlightWeapon(tab))
}

function drawMovementEndpointMap(ctx, tab, rect) {
  if (!tab || tab.uiState !== "movementAnalysis") return
  const width = rect.width
  const height = rect.height
  const result = getMovementEndpointAnalysis(tab)
  const selectedEndpointKey = String(tab?.movementSelectedEndpointKey || "")
  const selectedPath = Array.isArray(tab?.movementSelectedPath) ? tab.movementSelectedPath : null

  const radius = getMovementEndpointMapRadius(tab, result)
  const geom = getAnalysisHexGeometry(rect, radius)
  if (!geom) return
  const { size, axialToPixel, padding } = geom
  const endpointKeys = result?.ok ? result.endpointKeySet : null

  const drawHex = (cx, cy, fill, stroke, strokeWidth = 1) => {
    ctx.beginPath()
    for (let i = 0; i < 6; i++) {
      const angle = i * (Math.PI / 3)
      const px = cx + size * Math.cos(angle)
      const py = cy + size * Math.sin(angle)
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    if (fill) {
      ctx.fillStyle = fill
      ctx.fill()
    }
    if (stroke) {
      ctx.strokeStyle = stroke
      ctx.lineWidth = strokeWidth
      ctx.stroke()
    }
  }

  ctx.save()
  ctx.fillStyle = "rgba(20, 24, 28, 0.95)"
  ctx.fillRect(0, 0, width, height)

  for (let q = -radius; q <= radius; q++) {
    for (let r = -radius; r <= radius; r++) {
      const s = -q - r
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) > radius) continue
      const key = `${q},${r}`
      const pos = axialToPixel(q, r)
      const isOrigin = q === 0 && r === 0
      const isEndpoint = Boolean(endpointKeys && endpointKeys.has(key))
      const isSelectedEndpoint = selectedEndpointKey && key === selectedEndpointKey
      let fill = "rgba(255, 255, 255, 0.02)"
      let stroke = "rgba(255, 255, 255, 0.12)"
      let strokeWidth = 1

      if (isEndpoint) {
        fill = "rgba(76, 217, 255, 0.52)"
        stroke = "rgba(160, 245, 255, 0.95)"
        strokeWidth = Math.max(1.2, size * 0.08)
      }
      if (isSelectedEndpoint) {
        fill = "rgba(255, 82, 82, 0.55)"
        stroke = "rgba(255, 138, 138, 0.98)"
        strokeWidth = Math.max(strokeWidth, Math.max(1.8, size * 0.12))
      }
      if (isOrigin) {
        if (isEndpoint) {
          fill = "rgba(255, 204, 0, 0.55)"
          stroke = "rgba(255, 230, 128, 0.95)"
        } else {
          fill = "rgba(255, 204, 0, 0.16)"
          stroke = "rgba(255, 204, 0, 0.65)"
        }
        strokeWidth = Math.max(strokeWidth, Math.max(1.2, size * 0.09))
      }

      drawHex(pos.x, pos.y, fill, stroke, strokeWidth)
    }
  }

  if (selectedPath && selectedPath.length > 0) {
    ctx.save()
    ctx.lineJoin = "round"
    ctx.lineCap = "round"
    ctx.strokeStyle = "rgba(255, 64, 64, 0.95)"
    ctx.lineWidth = Math.max(2.2, size * 0.16)
    ctx.beginPath()
    for (let i = 0; i < selectedPath.length; i++) {
      const step = selectedPath[i]
      const pos = axialToPixel(Number(step.q) || 0, Number(step.r) || 0)
      if (i === 0) ctx.moveTo(pos.x, pos.y)
      else ctx.lineTo(pos.x, pos.y)
    }
    ctx.stroke()

    for (let i = 0; i < selectedPath.length; i++) {
      const step = selectedPath[i]
      const pos = axialToPixel(Number(step.q) || 0, Number(step.r) || 0)
      const isLast = i === selectedPath.length - 1
      const markerHeading = Number.isFinite(Number(step.heading))
        ? Number(step.heading)
        : (i > 0 && Number.isFinite(Number(selectedPath[i - 1]?.heading))
          ? Number(selectedPath[i - 1].heading)
          : MOVEMENT_ASSUMED_FORWARD_HEADING)
      drawMovementFacingTriangle(
        ctx,
        pos.x,
        pos.y,
        markerHeading,
        isLast ? Math.max(9, size * 0.62) : Math.max(7, size * 0.46),
        isLast ? "rgba(255, 153, 153, 0.98)" : "rgba(255, 92, 92, 0.9)",
        "rgba(80, 0, 0, 0.75)",
        Math.max(1, size * 0.05)
      )
    }
    ctx.restore()
  }

  // Assumed forward facing indicator (same triangle style as Weapon Analysis).
  const originPos = axialToPixel(0, 0)
  drawMovementFacingTriangle(
    ctx,
    originPos.x,
    originPos.y,
    MOVEMENT_ASSUMED_FORWARD_HEADING,
    Math.max(6, size * 0.85),
    "#ffcc00",
    "rgba(0, 0, 0, 0.35)",
    Math.max(1, size * 0.08)
  )

  const titleLines = []
  titleLines.push("Movement Endpoints (Endpoints Only)")
  if (result?.ok) {
    titleLines.push(
      `Turn ${result.turnModeLetter} | Speed ${String(result.speedRaw)} | Straight before turn: ${result.requiredStraight}`
    )
    if (normalizeMovementHeadingIndex(result.endingHeading) !== null) {
      titleLines.push(`Ending direction filter: ${getMovementHeadingLabel(result.endingHeading)}`)
    }
    titleLines.push(
      `Endpoints: ${result.endpointCount} unique hexes | Final states: ${result.finalStateCount} | Moves simulated: ${result.plottedMoves}`
    )
    titleLines.push("Assumptions: standard turns + sideslips, turn available at start, assumed forward facing = up (yellow triangle).")
    titleLines.push("Sideslip mode 1 is used and is assumed available at start.")
    if (result.capped && result.note) titleLines.push(result.note)
    titleLines.push("Click a reachable hex to show one possible path in red.")
  } else if (result?.status === "needsSpeed") {
    titleLines.push("Enter a speed to generate endpoint hexes.")
  } else if (result?.status === "invalidSpeed") {
    titleLines.push("Enter a valid non-negative speed.")
  } else if (result?.status === "noMatch") {
    titleLines.push("Selected turn mode + speed does not match a row in the Turn & Movement table.")
  } else {
    titleLines.push("Turn & Movement data is unavailable.")
  }

  const lineHeight = Math.max(13, Math.round(size * 0.6))
  ctx.font = `600 ${Math.max(11, Math.round(size * 0.58))}px "Segoe UI", sans-serif`
  ctx.textAlign = "left"
  ctx.textBaseline = "top"
  let maxTextWidth = 0
  for (const line of titleLines) {
    const w = ctx.measureText(line).width
    if (w > maxTextWidth) maxTextWidth = w
  }
  const boxX = padding * 0.55
  const boxY = padding * 0.45
  const boxW = Math.min(width - boxX * 2, maxTextWidth + 16)
  const boxH = titleLines.length * lineHeight + 12
  ctx.fillStyle = "rgba(0, 0, 0, 0.45)"
  ctx.strokeStyle = "rgba(255, 255, 255, 0.16)"
  ctx.lineWidth = 1
  ctx.fillRect(boxX, boxY, boxW, boxH)
  ctx.strokeRect(boxX, boxY, boxW, boxH)

  let textY = boxY + 6
  for (let i = 0; i < titleLines.length; i++) {
    const line = titleLines[i]
    ctx.fillStyle = i === 0 ? "rgba(255, 255, 255, 0.95)" : "rgba(235, 245, 255, 0.9)"
    ctx.fillText(line, boxX + 8, textY)
    textY += lineHeight
  }

  ctx.restore()
}

function drawSingleArcHighlight(ctx, fit, highlight) {
  if (!highlight || typeof highlight !== "object") return

  const strokeStyle = String(highlight.strokeStyle || "#ffcc00")
  const lineWidth = Number(highlight.lineWidth)
  const alpha = Number(highlight.alpha)
  const lineDash = Array.isArray(highlight.lineDash) ? highlight.lineDash : [6, 4]

  ctx.save()
  ctx.lineWidth = Number.isFinite(lineWidth) && lineWidth > 0 ? lineWidth : 4
  ctx.globalAlpha = Number.isFinite(alpha) && alpha >= 0 && alpha <= 1 ? alpha : 0.95
  ctx.strokeStyle = strokeStyle
  ctx.setLineDash(lineDash)

  const pts = Array.isArray(highlight.points) ? highlight.points : []
  if (pts.length >= 4) {
    ctx.beginPath()
    for (let i = 0; i < pts.length; i++) {
      const px = fit.ox + pts[i][0] * fit.scale
      const py = fit.oy + pts[i][1] * fit.scale
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.stroke()
    ctx.restore()
    return
  }

  const b = highlight.bbox
  if (b && typeof b === "object") {
    const x = fit.ox + b.x1 * fit.scale
    const y = fit.oy + b.y1 * fit.scale
    const w = (b.x2 - b.x1) * fit.scale
    const h = (b.y2 - b.y1) * fit.scale
    ctx.strokeRect(x, y, w, h)
  }

  ctx.restore()
}

function drawArcHighlight(ctx, fit) {
  const highlight = state.arcHighlight
  if (!highlight) return

  const entries = Array.isArray(highlight.entries) ? highlight.entries : [highlight]
  for (const entry of entries) {
    drawSingleArcHighlight(ctx, fit, entry)
  }
}

function canvasToImagePoint(tab, clientX, clientY) {
  const canvas = document.getElementById("canvas")
  const rect = canvas.getBoundingClientRect()
  const x = clientX - rect.left
  const y = clientY - rect.top

  const ix = (x - tab.view.ox) / tab.view.scale
  const iy = (y - tab.view.oy) / tab.view.scale
  return { ix, iy }
}

function hitTestSquare(tab, ix, iy) {
  for (const s of tab.doc.squares || []) {
    const b = s.bbox
    if (!b) continue
    if (ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) return s
  }
  return null
}

function hitTestGroup(tab, ix, iy) {
  for (const g of tab.doc.groups || []) {
    const b = g.bbox
    if (!b) continue
    if (ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) return g
  }
  return null
}

/* -----------------------------
   Remove helpers (Shipyard)
----------------------------- */
function removeGroupById(tab, groupId) {
  if (!tab || !tab.doc) return
  tab.doc.squares = (tab.doc.squares || []).filter(s => s.groupId !== groupId)
  tab.doc.groups = (tab.doc.groups || []).filter(g => g.id !== groupId)
  tab.selected = null
  renderCanvas()
  renderCurrentProps()
}

function removeSquareById(tab, squareId) {
  if (!tab || !tab.doc) return
  tab.doc.squares = (tab.doc.squares || []).filter(s => s.id !== squareId)
  tab.selected = null
  renderCanvas()
  renderCurrentProps()
}

async function editSelectedLabelShipyard() {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "shipyard" || !tab.doc) {
    alert("Editing labels is only available in Shipyard mode.")
    return
  }

  const sel = tab.selected
  if (!sel) {
    alert("Select a box or group on the image first.")
    return
  }

  if (sel.kind === "square") {
    const s = (tab.doc.squares || []).find(x => x.id === sel.id)
    if (!s) return
    const current = getSquareDisplayName(s)
    const next = await promptLabelDropdown("Edit Box Label", current, tab.boxLabelOptions || [])
    if (next === null) return
    applySquareLabel(s, next, tab)
    renderCanvas()
    renderCurrentProps(sel)
    return
  }

  if (sel.kind === "group") {
    const g = (tab.doc.groups || []).find(x => x.id === sel.id)
    if (!g) return
    const current = getGroupDisplayName(g)
    const next = await promptLabelDropdown("Edit Group Name", current, tab.boxLabelOptions || [])
    if (next === null) return
    applyGroupLabel(g, next)
    for (const s of tab.doc.squares || []) {
      if (s.groupId === g.id) {
        applySquareLabel(s, next, tab)
        s._inheritedFromGroupLabel = true
      }
    }
    renderCanvas()
    renderCurrentProps(sel)
    return
  }
}

function promptLabelDropdown(titleText, currentValue, options) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div")
    overlay.className = "modalOverlay"

    const card = document.createElement("div")
    card.className = "modalCard"

    const title = document.createElement("div")
    title.className = "modalTitle"
    title.textContent = titleText

    const filter = document.createElement("input")
    filter.type = "text"
    filter.placeholder = "type to filter options..."
    filter.style.width = "100%"
    filter.style.margin = "10px 0"

    const select = document.createElement("select")
    select.size = 10
    select.style.width = "100%"
    select.style.marginBottom = "12px"

    function fill(list) {
      select.innerHTML = ""
      for (const opt of list) {
        const o = document.createElement("option")
        o.value = opt
        o.textContent = opt
        if (opt === currentValue) o.selected = true
        select.appendChild(o)
      }
      if (select.options.length > 0 && select.selectedIndex === -1) select.selectedIndex = 0
    }
    fill(options || [])

    filter.oninput = () => {
      const q = filter.value.trim().toLowerCase()
      if (!q) return fill(options)
      fill((options || []).filter(x => x.toLowerCase().includes(q)))
    }

    select.ondblclick = () => finish(select.value)

    const actions = document.createElement("div")
    actions.className = "modalActions"

    const btnCancel = document.createElement("button")
    btnCancel.textContent = "Cancel"
    const btnSave = document.createElement("button")
    btnSave.textContent = "Apply"

    actions.appendChild(btnCancel)
    actions.appendChild(btnSave)

    card.appendChild(title)
    card.appendChild(filter)
    card.appendChild(select)
    card.appendChild(actions)
    overlay.appendChild(card)
    document.body.appendChild(overlay)

    filter.focus()

    function cleanup(val) {
      document.body.removeChild(overlay)
      resolve(val)
    }

    function finish(val) {
      if (val === undefined || val === null) {
        cleanup(null)
        return
      }
      cleanup(String(val).trim())
    }

    btnCancel.onclick = () => cleanup(null)
    btnSave.onclick = () => finish(select.value || "")

    card.onkeydown = (e) => {
      if (e.key === "Escape") cleanup(null)
      if (e.key === "Enter") finish(select.value || "")
    }
  })
}

function promptJsonEditAddBoxText(defaultValue, entries) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div")
    overlay.className = "modalOverlay"

    const card = document.createElement("div")
    card.className = "modalCard"
    card.style.width = "420px"

    const title = document.createElement("div")
    title.className = "modalTitle"
    title.textContent = "Add JSON Box"

    const desc = document.createElement("div")
    desc.className = "modalDesc"
    desc.textContent = "Type or choose a box label from Data.json. Use section:<key> only for a raw SSD section."

    const input = document.createElement("input")
    input.type = "text"
    input.value = String(defaultValue || "")
    input.placeholder = "Bridge, Phaser 1, Photon Torpedo..."
    input.style.width = "100%"
    input.style.margin = "0 0 10px"

    const select = document.createElement("select")
    select.size = 12
    select.style.width = "100%"
    select.style.marginBottom = "12px"

    const options = (Array.isArray(entries) ? entries : [])
      .map(entry => String(entry?.name || entry?.label || "").trim())
      .filter(Boolean)
    const uniqueOptions = Array.from(new Set(options))

    function fill() {
      const q = input.value.trim().toLowerCase()
      const list = q
        ? uniqueOptions.filter(option => option.toLowerCase().includes(q))
        : uniqueOptions
      select.innerHTML = ""
      for (const opt of list) {
        const o = document.createElement("option")
        o.value = opt
        o.textContent = opt
        select.appendChild(o)
      }
      if (select.options.length > 0) select.selectedIndex = 0
    }

    const actions = document.createElement("div")
    actions.className = "modalActions"

    const btnCancel = document.createElement("button")
    btnCancel.textContent = "Cancel"
    const btnApply = document.createElement("button")
    btnApply.textContent = "Add"

    actions.appendChild(btnCancel)
    actions.appendChild(btnApply)

    card.appendChild(title)
    card.appendChild(desc)
    card.appendChild(input)
    card.appendChild(select)
    card.appendChild(actions)
    overlay.appendChild(card)
    document.body.appendChild(overlay)

    function cleanup(value) {
      window.removeEventListener("keydown", onKey)
      if (overlay.parentElement) overlay.parentElement.removeChild(overlay)
      resolve(value)
    }

    function finish(value) {
      const clean = String(value || "").trim()
      cleanup(clean || null)
    }

    function getChoice() {
      const typed = String(input.value || "").trim()
      if (/^section\s*:/i.test(typed)) return typed
      const typedKey = compactJsonEditKey(typed)
      const exact = uniqueOptions.find(option => compactJsonEditKey(option) === typedKey)
      if (exact) return exact
      return select.value || typed
    }

    function onKey(e) {
      if (e.key === "Escape") {
        cleanup(null)
        return
      }
      if (e.key === "Enter") {
        finish(getChoice())
      }
    }

    input.oninput = () => fill()
    select.onchange = () => {
      if (select.value) input.value = select.value
      fill()
    }
    select.ondblclick = () => finish(select.value)
    btnCancel.onclick = () => cleanup(null)
    btnApply.onclick = () => finish(getChoice())
    window.addEventListener("keydown", onKey)

    fill()
    input.focus()
    input.select()
  })
}

function promptArcDropdown(titleText, currentValue, options, hooks = {}) {
  return new Promise((resolve) => {
    const panel = document.getElementById("arcPromptPanel")
    const title = document.getElementById("arcPromptTitle")
    const filter = document.getElementById("arcPromptFilter")
    const list = document.getElementById("arcPromptOptions")
    const btnApply = document.getElementById("arcPromptApply")
    const btnCancel = document.getElementById("arcPromptCancel")

    if (!panel || !title || !filter || !list || !btnApply || !btnCancel) {
      resolve(null)
      return
    }

    const normalizedOptions = (options || []).map(opt => String(opt || "").trim()).filter(Boolean)
    const optionSet = new Set(normalizedOptions)
    const selected = new Set(
      String(currentValue || "")
        .split("+")
        .map(part => part.trim())
        .filter(part => part && optionSet.has(part))
    )

    const isNoneOption = (opt) => String(opt || "").trim().toLowerCase() === "none"
    const hasNoneSelection = () => {
      for (const opt of selected) {
        if (isNoneOption(opt)) return true
      }
      return false
    }
    const clearNoneSelection = () => {
      for (const opt of selected) {
        if (isNoneOption(opt)) selected.delete(opt)
      }
    }
    const orderedSelection = () => normalizedOptions.filter(opt => selected.has(opt))
    const emitSelectionChange = () => {
      if (typeof hooks?.onSelectionChange !== "function") return
      hooks.onSelectionChange(orderedSelection())
    }

    const renderOptions = (items) => {
      list.innerHTML = ""
      for (const opt of items) {
        const row = document.createElement("label")
        row.className = "arcPromptOption"

        const input = document.createElement("input")
        input.type = "checkbox"
        input.value = opt
        input.checked = selected.has(opt)

        const text = document.createElement("span")
        text.textContent = opt

        input.onchange = () => {
          const hadNone = hasNoneSelection()
          if (input.checked) {
            if (isNoneOption(opt)) {
              selected.clear()
              selected.add(opt)
            } else {
              clearNoneSelection()
              selected.add(opt)
            }
          } else {
            selected.delete(opt)
          }
          const resetFilter = filter.value.trim().length > 0
          if (resetFilter) {
            filter.value = ""
            renderOptions(normalizedOptions)
          } else if (hadNone !== hasNoneSelection()) {
            renderOptions(items)
          }
          emitSelectionChange()
        }

        row.appendChild(input)
        row.appendChild(text)
        list.appendChild(row)
      }
    }

    const applyFilter = () => {
      const q = filter.value.trim().toLowerCase()
      if (!q) return renderOptions(normalizedOptions)
      return renderOptions(normalizedOptions.filter(x => x.toLowerCase().includes(q)))
    }

    const finish = (val) => {
      cleanup()
      resolve(val)
    }

    const apply = () => {
      const values = orderedSelection()
      if (values.length === 0) {
        showTempMessage("Select at least one arc.")
        return
      }
      const none = values.find(opt => isNoneOption(opt))
      if (none) {
        finish(none)
        return
      }
      finish(values.join("+"))
    }

    const cancel = () => finish(null)

    const onKey = (e) => {
      if (e.key === "Escape") cancel()
      if (e.key === "Enter") apply()
    }

    const cleanup = () => {
      panel.classList.add("hidden")
      filter.oninput = null
      btnApply.onclick = null
      btnCancel.onclick = null
      window.removeEventListener("keydown", onKey)
      syncLeftPanelScrollbar()
    }

    title.textContent = titleText
    filter.value = ""
    panel.classList.remove("hidden")
    applyFilter()
    syncLeftPanelScrollbar()
    filter.focus()

    filter.oninput = () => applyFilter()
    btnApply.onclick = () => apply()
    btnCancel.onclick = () => cancel()
    window.addEventListener("keydown", onKey)
  })
}

function promptDesignationInput(titleText, currentValue) {
  return new Promise((resolve) => {
    const panel = document.getElementById("designationPromptPanel")
    const title = document.getElementById("designationPromptTitle")
    const input = document.getElementById("designationPromptInput")
    const btnApply = document.getElementById("designationPromptApply")
    const btnCancel = document.getElementById("designationPromptCancel")

    if (!panel || !title || !input || !btnApply || !btnCancel) {
      resolve(null)
      return
    }

    const finish = (val) => {
      cleanup()
      resolve(val)
    }

    const apply = () => {
      const val = String(input.value || "").trim()
      if (!val) {
        showTempMessage("Enter a designation.")
        return
      }
      finish(val)
    }

    const cancel = () => finish(null)

    const onKey = (e) => {
      if (e.key === "Escape") cancel()
      if (e.key === "Enter") apply()
    }

    const cleanup = () => {
      panel.classList.add("hidden")
      btnApply.onclick = null
      btnCancel.onclick = null
      window.removeEventListener("keydown", onKey)
    }

    title.textContent = titleText
    input.value = String(currentValue || "")
    panel.classList.remove("hidden")
    input.focus()
    input.select()

    btnApply.onclick = () => apply()
    btnCancel.onclick = () => cancel()
    window.addEventListener("keydown", onKey)
  })
}

function promptRankValueInput(titleText, currentRank, currentValue, options = {}) {
  return new Promise((resolve) => {
    const panel = document.getElementById("rankValuePromptPanel")
    const title = document.getElementById("rankValuePromptTitle")
    const rankInput = document.getElementById("rankValuePromptRank")
    const valueInput = document.getElementById("rankValuePromptValue")
    const btnApply = document.getElementById("rankValuePromptApply")
    const btnCancel = document.getElementById("rankValuePromptCancel")

    if (!panel || !title || !rankInput || !valueInput || !btnApply || !btnCancel) {
      resolve(null)
      return
    }

    const lockRank = !!options?.lockRank

    const finish = (val) => {
      cleanup()
      resolve(val)
    }

    const apply = () => {
      const rankText = String(rankInput.value || "").trim()
      const valueText = String(valueInput.value || "").trim()
      if (!valueText || (!lockRank && !rankText)) {
        showTempMessage(lockRank ? "Enter a value." : "Enter a rank and value.")
        return
      }
      const rank = Number(rankText)
      const value = Number(valueText)
      if (!Number.isFinite(rank)) {
        showTempMessage("Rank is missing. Re-select first and last rank boxes.")
        return
      }
      if (!Number.isFinite(value)) {
        showTempMessage("Value must be a number.")
        return
      }
      finish({ rank, value })
    }

    const cancel = () => finish(null)

    const onKey = (e) => {
      if (e.key === "Escape") cancel()
      if (e.key === "Enter") apply()
    }

    const cleanup = () => {
      panel.classList.add("hidden")
      btnApply.onclick = null
      btnCancel.onclick = null
      window.removeEventListener("keydown", onKey)
      rankInput.readOnly = false
      syncLeftPanelScrollbar()
    }

    title.textContent = titleText
    rankInput.value = currentRank === undefined || currentRank === null ? "" : String(currentRank)
    valueInput.value = currentValue === undefined || currentValue === null ? "" : String(currentValue)
    rankInput.readOnly = lockRank
    panel.classList.remove("hidden")
    syncLeftPanelScrollbar()
    if (lockRank) {
      valueInput.focus()
      valueInput.select()
    } else {
      rankInput.focus()
      rankInput.select()
    }

    btnApply.onclick = () => apply()
    btnCancel.onclick = () => cancel()
    window.addEventListener("keydown", onKey)
  })
}

function promptRankRangeSelection(group, items) {
  return new Promise((resolve) => {
    const panel = document.getElementById("rankRangePromptPanel")
    const title = document.getElementById("rankRangePromptTitle")
    const firstSelect = document.getElementById("rankRangePromptFirst")
    const lastSelect = document.getElementById("rankRangePromptLast")
    const btnApply = document.getElementById("rankRangePromptApply")
    const btnCancel = document.getElementById("rankRangePromptCancel")

    if (!panel || !title || !firstSelect || !lastSelect || !btnApply || !btnCancel) {
      resolve(null)
      return
    }

    const normalizedItems = (Array.isArray(items) ? items : [])
      .map((item, index) => {
        const assignmentKey = getRankAssignmentKey(item)
        if (!assignmentKey) return null
        return {
          item,
          assignmentKey,
          label: formatRankRangeOptionLabel(item, index)
        }
      })
      .filter(Boolean)
      .sort((a, b) => compareRankItemsForDisplay(a.item, b.item))

    if (normalizedItems.length < 2) {
      resolve(null)
      return
    }

    const itemByKey = new Map(normalizedItems.map((entry) => [entry.assignmentKey, entry.item]))

    const fillSelect = (selectEl, preferredKey) => {
      selectEl.innerHTML = ""
      for (const entry of normalizedItems) {
        const opt = document.createElement("option")
        opt.value = entry.assignmentKey
        opt.textContent = entry.label
        selectEl.appendChild(opt)
      }
      const hasPreferred = normalizedItems.some((entry) => entry.assignmentKey === preferredKey)
      selectEl.value = hasPreferred ? preferredKey : normalizedItems[0].assignmentKey
    }

    const firstDefault = normalizedItems[0].assignmentKey
    const lastDefault = normalizedItems[normalizedItems.length - 1].assignmentKey

    const showRangeHighlights = () => {
      const firstKey = String(firstSelect.value || "").trim()
      const lastKey = String(lastSelect.value || "").trim()
      const firstItem = itemByKey.get(firstKey)
      const lastItem = itemByKey.get(lastKey)
      const highlights = []
      if (firstItem) {
        highlights.push({
          item: firstItem,
          style: { strokeStyle: "#22c55e", lineDash: [6, 4], lineWidth: 4, alpha: 0.95 }
        })
      }
      if (lastItem) {
        const sameSelection = firstKey && firstKey === lastKey
        highlights.push({
          item: lastItem,
          style: sameSelection
            ? { strokeStyle: "#f59e0b", lineDash: [2, 3], lineWidth: 4, alpha: 0.95 }
            : { strokeStyle: "#ef4444", lineDash: [2, 3], lineWidth: 4, alpha: 0.95 }
        })
      }
      setArcHighlights(highlights)
      renderCanvas()
    }

    const finish = (val) => {
      cleanup()
      resolve(val)
    }

    const apply = () => {
      const firstKey = String(firstSelect.value || "").trim()
      const lastKey = String(lastSelect.value || "").trim()
      if (!firstKey || !lastKey) {
        showTempMessage("Select both first and last rank boxes.")
        return
      }
      if (firstKey === lastKey) {
        showTempMessage("First and last rank boxes must be different.")
        return
      }
      finish({ firstKey, lastKey })
    }

    const cancel = () => finish(null)

    const onKey = (e) => {
      if (e.key === "Escape") cancel()
      if (e.key === "Enter") apply()
    }

    const cleanup = () => {
      panel.classList.add("hidden")
      firstSelect.onchange = null
      firstSelect.onfocus = null
      lastSelect.onchange = null
      lastSelect.onfocus = null
      btnApply.onclick = null
      btnCancel.onclick = null
      window.removeEventListener("keydown", onKey)
      clearArcHighlight()
      renderCanvas()
      syncLeftPanelScrollbar()
    }

    title.textContent = `Select first (front/green) and last (back/red) rank for ${formatRankValueGroupLabel(group)}`
    fillSelect(firstSelect, firstDefault)
    fillSelect(lastSelect, lastDefault)
    panel.classList.remove("hidden")
    syncLeftPanelScrollbar()

    firstSelect.onchange = () => showRangeHighlights()
    firstSelect.onfocus = () => showRangeHighlights()
    lastSelect.onchange = () => showRangeHighlights()
    lastSelect.onfocus = () => showRangeHighlights()
    btnApply.onclick = () => apply()
    btnCancel.onclick = () => cancel()
    window.addEventListener("keydown", onKey)

    firstSelect.focus()
    showRangeHighlights()
  })
}

/* -----------------------------
   Properties (shared renderer)
----------------------------- */
function renderPropsInto(gridId, titleId, rawId, sel) {
  const tab = getActiveTab()
  const grid = document.getElementById(gridId)
  const title = document.getElementById(titleId)
  const raw = document.getElementById(rawId)

  if (!grid || !title || !raw) return

  grid.innerHTML = ""
  raw.value = ""
  title.textContent = "Nothing selected"

  if (!tab || !tab.doc || !sel) return

  const minimalShipyardDone = gridId === "shipyardPropGrid" && isShipyardDoneVisible()

  if (sel.kind === "square") {
    const s = (tab.doc.squares || []).find(x => x.id === sel.id)
    if (!s) return

    title.textContent = `Square ${s.id}`
    if (minimalShipyardDone) {
      addReadonlyRow(grid, "id", s.id)
      addReadonlyRow(grid, "label", getSquareDisplayName(s) || "(unlabeled)")
      raw.value = JSON.stringify(s, null, 2)
      return
    }
    addReadonlyRow(grid, "id", s.id)
    addEditableSquareLabelRow(grid, s)
    addEditableGroupIdRow(grid, tab, s)
    addReadonlyRow(grid, "sideLengthPx", s.sideLengthPx)
    addReadonlyRow(grid, "center.x", s.center?.x)
    addReadonlyRow(grid, "center.y", s.center?.y)
    raw.value = JSON.stringify(s, null, 2)
    return
  }

  const g = (tab.doc.groups || []).find(x => x.id === sel.id)
  if (!g) return

  title.textContent = `Group ${g.id}`
  if (minimalShipyardDone) {
    addReadonlyRow(grid, "id", g.id)
    addReadonlyRow(grid, "name", getGroupDisplayName(g) || "(unnamed)")
    raw.value = JSON.stringify(g, null, 2)
    return
  }
  addReadonlyRow(grid, "id", g.id)
  addEditableGroupNameRow(grid, g)
  addReadonlyRow(grid, "count", g.count)
  addReadonlyRow(grid, "center.x", g.center?.x)
  addReadonlyRow(grid, "center.y", g.center?.y)
  raw.value = JSON.stringify(g, null, 2)
}

function renderShipyardProps(sel) {
  renderPropsInto("shipyardPropGrid", "shipyardPropTitle", "shipyardRawJson", sel)
}

function renderJsonEditProps(sel) {
  const tab = getActiveTab()
  const grid = document.getElementById("jsonEditPropGrid")
  const title = document.getElementById("jsonEditPropTitle")
  const raw = document.getElementById("jsonEditRawJson")
  if (!grid || !title || !raw) return

  grid.innerHTML = ""
  raw.value = ""
  title.textContent = "Nothing selected"

  if (!tab || tab.uiState !== "jsonEdit" || !tab.doc || !sel) return

  if (sel.kind === "square") {
    const s = (tab.doc.squares || []).find(x => x.id === sel.id)
    if (!s) return
    const meta = s.__jsonEdit && typeof s.__jsonEdit === "object" ? s.__jsonEdit : null
    title.textContent = `Box ${s.id}`
    addReadonlyRow(grid, "id", s.id)
    if (meta?.format === "superluminal") {
      const updateRawEntry = () => {
        raw.value = JSON.stringify(getJsonEditSourceEntry(tab, meta) || meta.entry || {}, null, 2)
      }
      addReadonlyRow(grid, "label", getSquareDisplayName(s) || "(unlabeled)")
      addReadonlyRow(grid, "source", "Superluminal")
      addReadonlyRow(grid, "ssdKey", meta.ssdKey || "")
      addReadonlyRow(grid, "entryIndex", meta.entryIndex)
      addEditableJsonEditEntryChoiceRow(
        grid,
        tab,
        s,
        meta,
        "type",
        "type",
        tab.jsonEditTypeOptions || [],
        (nextType) => {
          const clean = normalizeLabel(nextType).clean
          const fallback = formatJsonEditGroupName(meta.ssdKey)
          s.label = clean || fallback
          s.name = clean || fallback
          renderCanvas()
          updateRawEntry()
        }
      )
      addEditableJsonEditEntryFieldRow(grid, tab, s, meta, "designation", "designation", updateRawEntry)
      addEditableJsonEditEntryChoiceRow(
        grid,
        tab,
        s,
        meta,
        "arc",
        "arc",
        tab.jsonEditArcOptions || [],
        updateRawEntry
      )
      addReadonlyRow(grid, "pos", meta.pos || "")
      updateRawEntry()
      return
    }
    addReadonlyRow(grid, "source", "Veil")
    addEditableSquareLabelRow(grid, s)
    addEditableGroupIdRow(grid, tab, s)
    addReadonlyRow(grid, "sideLengthPx", s.sideLengthPx)
    addReadonlyRow(grid, "center.x", s.center?.x)
    addReadonlyRow(grid, "center.y", s.center?.y)
    raw.value = JSON.stringify(s, null, 2)
    return
  }

  const g = (tab.doc.groups || []).find(x => x.id === sel.id)
  if (!g) return
  const groupMeta = g.__jsonEdit && typeof g.__jsonEdit === "object" ? g.__jsonEdit : null
  title.textContent = `Group ${g.id}`
  addReadonlyRow(grid, "id", g.id)
  addReadonlyRow(grid, "name", getGroupDisplayName(g) || "(unnamed)")
  addReadonlyRow(grid, "count", g.count)
  if (groupMeta?.format === "superluminal") {
    addReadonlyRow(grid, "source", "Superluminal")
    addReadonlyRow(grid, "ssdKey", groupMeta.ssdKey || "")
  } else {
    addReadonlyRow(grid, "source", "Veil")
  }
  raw.value = JSON.stringify(g, null, 2)
}

/* -----------------------------
   Energy allocation table
----------------------------- */
function parseEnergyTemplate(text) {
  const rows = []
  const lines = String(text || "").split(/\r?\n/)
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) {
      rows.push({ type: "gap" })
      continue
    }
    const locked = trimmed.includes("*")
    const label = trimmed.replace(/\*/g, "").trim()
    if (!label) {
      rows.push({ type: "gap" })
      continue
    }
    rows.push({ type: "row", label, locked })
  }
  return rows
}

function ensureEnergyTemplate() {
  if (energyTemplate) return energyTemplate
  energyTemplate = parseEnergyTemplate(ENERGY_TEMPLATE_LINES.join("\n"))
  return energyTemplate
}

function computeWarpCounts(tab) {
  const squares = tab?.doc?.squares || []
  const counts = { left: 0, right: 0, center: 0, total: 0 }
  for (const s of squares) {
    if (isSquareRemovedOrDamaged(s)) continue
    const name = getSquareDisplayName(s).toLowerCase()
    if (!name) continue
    if (name === "left warp") counts.left += 1
    else if (name === "right warp") counts.right += 1
    else if (name === "center warp") counts.center += 1
  }
  counts.total = counts.left + counts.right + counts.center
  return counts
}

function computePowerStats(tab) {
  const squares = tab?.doc?.squares || []
  const warp = computeWarpCounts(tab)
  let impulse = 0
  let reactor = 0
  let battery = 0

  for (const s of squares) {
    if (isSquareRemovedOrDamaged(s)) continue
    const name = getSquareDisplayName(s).toLowerCase()
    if (!name) continue
    if (name === "impulse" || name.startsWith("impulse ")) impulse += 1
    if (name === "apr" || name === "a.p.r" || name.startsWith("apr ")) reactor += 1
    if (name === "battery" || name.startsWith("battery ")) battery += 1
  }

  return {
    warp: warp.total,
    impulse,
    reactor,
    battery,
    total: warp.total + impulse + reactor,
    warpDetail: warp
  }
}

function computeMaxPhaserCaps(tab) {
  const squares = tab?.doc?.squares || []
  let total = 0
  for (const s of squares) {
    if (isSquareRemovedOrDamaged(s)) continue
    const name = getSquareDisplayName(s).toLowerCase()
    if (!name.startsWith("phaser")) continue
    if (name.includes("phaser 1") || name.includes("phaser 2") || name.includes("phaser g") || name.includes("phaser 4")) {
      total += 1
    } else if (name.includes("phaser 3")) {
      total += 0.5
    }
  }
  return total
}

// Compute "Phaser capacitors used" based on deployed phasers.
function recomputePhaserCapsUsed(tab) {
  if (!tab || !tab.doc) return 0
  let total = 0
  const squares = tab.doc.squares || []
  for (const s of squares) {
    if (isSquareRemovedOrDamaged(s)) continue
    const name = getSquareDisplayName(s).toLowerCase()
    if (!name.startsWith("phaser")) continue
    if (name.includes("phaser g")) {
      const count = Number(s.phaserGCount) || 0
      if (count > 0) {
        const use = 0.25 * count
        total += use
        s.phaserCapUsage = use
      }
      continue
    }
    if (!s.systemDeployed) continue
    if (name.includes("phaser 3")) {
      total += 0.5
      if (s.phaserCapUsage !== 0.5) s.phaserCapUsage = 0.5
      continue
    }
    if (name.includes("phaser 4")) {
      const use = typeof s.phaserCapUsage === "number" ? s.phaserCapUsage : 2
      total += use
      continue
    }
    if (name.includes("phaser 1") || name.includes("phaser 2")) {
      const use = typeof s.phaserCapUsage === "number" ? s.phaserCapUsage : 1
      total += use
      continue
    }
  }
  const rounded = Math.round(total * 100) / 100
  tab.energyValues["Phaser capacitors used"] = rounded
  return rounded
}

function getShieldNumbers(tab) {
  const nums = new Set()
  const regex = /sh(?:ie|ei)ld\s*#?\s*(\d+)/i

  for (const s of tab?.doc?.squares || []) {
    const name = getSquareDisplayName(s).toLowerCase()
    const m = regex.exec(name)
    if (m && m[1]) {
      const n = parseInt(m[1], 10)
      if (Number.isFinite(n)) nums.add(n)
    }
  }

  for (const key of Object.keys(tab?.energyValues || {})) {
    const m = regex.exec(key)
    if (m && m[1]) {
      const n = parseInt(m[1], 10)
      if (Number.isFinite(n)) nums.add(n)
    }
  }

  if (tab?.customShieldLabels instanceof Set) {
    for (const n of tab.customShieldLabels) {
      if (Number.isFinite(n)) nums.add(n)
    }
  }
  return Array.from(nums).sort((a, b) => a - b)
}

function computeBatteryDischarged(tab) {
  if (!tab || !tab.doc) return 0
  let total = 0
  for (const s of tab.doc.squares || []) {
    if (isSquareRemovedOrDamaged(s)) continue
    if (!s.systemDeployed) continue
    if (isBatterySquare(s)) total += 1
  }
  return total
}

// Refresh energy allocation UI/calcs after marker changes, if allocation is unlocked.
function refreshEnergyAfterMarkerChange(tab) {
  if (!tab || tab.turnLocked) return
  recomputePhaserCapsUsed(tab)
  renderEnergyTable()
}

function computeCurrentPhaserCharge(tab, { updateValues = true } = {}) {
  if (!tab) return 0
  if (!tab.energyValues) tab.energyValues = {}

  const remainingKey =
    findEnergyKey(tab.energyValues, "held phaser capacitors") ||
    findEnergyKey(tab.energyValues, "phaser capacitors remaining")
  const chargeKey = findEnergyKey(tab.energyValues, "charge phaser capacitors")
  const usedKey = findEnergyKey(tab.energyValues, "phaser capacitors used")

  const remaining = Number(tab.energyValues[remainingKey])
  const charge = Number(tab.energyValues[chargeKey])
  const used = Number(tab.energyValues[usedKey])
  const maxCaps = computeMaxPhaserCaps(tab)

  const safeRemaining = Number.isFinite(remaining) ? remaining : 0
  const safeCharge = Number.isFinite(charge) ? charge : 0
  const safeUsed = Number.isFinite(used) ? used : 0

  let val = safeRemaining + safeCharge - safeUsed
  if (!Number.isFinite(val)) val = 0
  val = Math.max(0, Math.min(val, maxCaps))

  if (updateValues) {
    const label =
      findEnergyKey(tab.energyValues, "current phaser charge") || "Current phaser charge"
    tab.energyValues[label] = val
  }
  return val
}

function updateCurrentPhaserChargeDisplay(tab) {
  const table = document.getElementById("energyTable")
  if (!table) return
  const val = computeCurrentPhaserCharge(tab)
  const labels = table.querySelectorAll(".energyLabel")
  for (const lbl of labels) {
    if (lbl.textContent.trim().toLowerCase() === "current phaser charge") {
      const wrap = lbl.nextElementSibling
      if (!wrap) break
      const input = wrap.querySelector("input.energyInput")
      if (input) input.value = String(val)
      break
    }
  }
}

function formatHeavyEnergyLabel(square) {
  const name = getSquareDisplayName(square) || "Heavy Weapon"
  const words = name.trim().split(/\s+/).filter(Boolean)
  const firstTwo = words.slice(0, 2).join(" ") || name
  const letter = (square.heavyLetter || "").trim()
  return letter ? `${firstTwo}, ${letter}` : firstTwo
}

// Parse a raw fraction string into numeric parts (used by Movement Cost).
// Returns { raw, num, numPart, denPart } where raw is normalized ("a/b").
function parseFractionParts(raw) {
  const t = String(raw ?? "").trim()
  if (!t) return { raw: "0/1", num: 0, numPart: "0", denPart: "1" }

  if (t.includes("/")) {
    const [a, b] = t.split("/")
    const na = Number(a)
    const nb = Number(b)
    const safeA = Number.isFinite(na) ? Math.max(0, Math.floor(na)) : 0
    const safeB = Number.isFinite(nb) && nb !== 0 ? Math.max(1, Math.floor(nb)) : 1
    return {
      raw: `${safeA}/${safeB}`,
      num: safeA / safeB,
      numPart: String(safeA),
      denPart: String(safeB)
    }
  }

  const n = Number(t)
  if (Number.isFinite(n)) {
    const safe = Math.max(0, n)
    const whole = Math.floor(safe)
    return { raw: `${whole}/1`, num: safe, numPart: String(whole), denPart: "1" }
  }

  return { raw: "0/1", num: 0, numPart: "0", denPart: "1" }
}

function renderEnergyTable() {
  const table = document.getElementById("energyTable")
  const tab = getActiveTab()
  if (!table) return

  if (!tab || !tab.doc) {
    table.innerHTML = ""
    return
  }

  const tmpl = ensureEnergyTemplate()
  if (!tmpl) return
  table.innerHTML = ""
  if (tmpl.length === 0) {
    const msg = document.createElement("div")
    msg.className = "hint"
    msg.style.textAlign = "center"
    msg.textContent = "EnergyAllocation has no entries."
    table.appendChild(msg)
    return
  }

  if (!tab.energyValues) tab.energyValues = {}
  if (!tab.energyDisplay) tab.energyDisplay = {}

  const locked = !!tab.turnLocked
  const resetMode = !!tab.energyReset

  const autoLabelsSet = new Set([
    "warp engine power",
    "total warp",
    "impulse engine power",
    "reactor power",
    "total available power",
    "total power available",
    "total power",
    "total power used",
    "movement cost",
    "speed",
    "phaser capacitors used",
    "phasers",
    "held phaser capacitors",
    "current phaser charge",
    "max phaser capacitors",
    "battery power available",
    "battery power discharged",
    "total batteries"
  ])

  let totalUsedInput = null
  let totalAvailableVal = 0
  let speedInput = null

  const getEnergyVal = (labelLower) => {
    for (const [k, v] of Object.entries(tab.energyValues || {})) {
      if (k.toLowerCase() === labelLower) return v
    }
    return undefined
  }

  const updateSpeedInput = () => {
    if (locked) return
    if (!speedInput) return
    const energyRaw = getEnergyVal("energy for movement")
    const costRaw = getEnergyVal("movement cost")
    const energy = Number.isFinite(Number(energyRaw)) ? Number(energyRaw) : 0
    const cost = Number.isFinite(Number(costRaw)) ? Number(costRaw) : 0
    const speedVal = cost > 0 ? energy / cost : 0
    const rounded = Math.round(speedVal) // nearest integer
    tab.energyValues["Speed"] = rounded
    speedInput.value = String(Number.isFinite(rounded) ? rounded : 0)
  }

  const updateTotalUsedInput = () => {
    if (locked) {
      if (totalUsedInput) {
        const stored = Number(tab.energyValues["Total power used"] ?? 0)
        totalUsedInput.value = String(Number.isFinite(stored) ? stored : 0)
        totalUsedInput.classList.remove("powerOk", "powerWarn", "powerOver")
        if (stored > totalAvailableVal) totalUsedInput.classList.add("powerOver")
        else if (stored < totalAvailableVal) totalUsedInput.classList.add("powerWarn")
        else totalUsedInput.classList.add("powerOk")
      }
      return
    }
    if (!totalUsedInput) return
    const totalUsed = Object.entries(tab.energyValues || {}).reduce((acc, [k, v]) => {
      const kl = k.toLowerCase()
      if (autoLabelsSet.has(kl)) return acc
      const n = Number(v)
      return acc + (Number.isFinite(n) ? n : 0)
    }, 0)
    if (!locked) tab.energyValues["Total power used"] = totalUsed
    totalUsedInput.value = String(totalUsed)

    // Apply status colors
    totalUsedInput.classList.remove("powerOk", "powerWarn", "powerOver")
    if (totalUsed > totalAvailableVal) totalUsedInput.classList.add("powerOver")
    else if (totalUsed < totalAvailableVal) totalUsedInput.classList.add("powerWarn")
    else totalUsedInput.classList.add("powerOk")
  }

  for (const row of tmpl) {
    if (row.type === "gap") {
      const gap = document.createElement("div")
      gap.className = "energyGap"
      table.appendChild(gap)
      continue
    }

    const labelLower = row.label.toLowerCase()
    const powerStats = computePowerStats(tab)
    const isEnergyForMovement = labelLower === "energy for movement"

    const label = document.createElement("div")
    label.className = "energyLabel"
    label.textContent = row.label

    const inputWrap = document.createElement("div")
    inputWrap.className = "energyInputWrap"

    const isWarpEnginePower =
      labelLower === "warp engine power" || labelLower === "total warp" || labelLower.includes("warp engine")
    const isImpulsePower = labelLower === "impulse engine power" || labelLower.includes("impulse")
    const isReactorPower = labelLower === "reactor power" || labelLower.includes("apr")
    const isBatteryAvailable = labelLower === "battery power available"
    const isBatteryDischarged = labelLower === "battery power discharged"
    const isTotalAvailable =
      labelLower === "total available power" ||
      labelLower === "total power available" ||
      labelLower === "total power" ||
      labelLower.includes("total available") ||
      labelLower.includes("total power available")
    const isTotalUsed = labelLower === "total power used"
    const isTotalBatteries = labelLower === "total batteries"
    const isPhaserHeader = labelLower === "phasers"
    const isMaxPhaserCaps = labelLower === "max phaser capacitors" || labelLower.includes("max phaser")
    const isPhaserCapsUsed = labelLower === "phaser capacitors used" || labelLower.includes("phaser capacitors used")
    const isPhaserCapsRemaining = labelLower === "held phaser capacitors"
    const isPhaserCharge = labelLower === "current phaser charge"
    const isMovementCost = labelLower === "movement cost"
    const isHeavyWeaponsRow = labelLower === "heavy weapons"
    const isLifeSupport = labelLower === "life support"
    const isTransporters = labelLower === "transporters"
    const isSpecificReinforcement = labelLower === "specific reinforcement"

    if (isTotalAvailable) {
      label.classList.add("energyStickyTotal")
      inputWrap.classList.add("energyStickyTotal")
    }

    if (isPhaserHeader) {
      const header = document.createElement("div")
      header.className = "energyLabel"
      header.textContent = "Phasers"
      header.style.gridColumn = "1 / span 2"
      table.appendChild(header)
      continue
    }

    if (isHeavyWeaponsRow) {
      const heavySquares = (tab.doc?.squares || []).filter(s => {
        const n = getSquareDisplayName(s).toLowerCase()
        if (isSquareRemovedOrDamaged(s)) return false
        return s.additionalLabel === "heavy Weapons" || tab.heavyLabelSet?.has(n)
      })

      // Sort heavy weapons by their letter designation (A, B, AA, etc.)
      const sortedHeavySquares = heavySquares.slice().sort((a, b) => {
        const aLetter = (a.heavyLetter || "").trim().toUpperCase()
        const bLetter = (b.heavyLetter || "").trim().toUpperCase()
        if (aLetter && bLetter) {
          const cmp = aLetter.localeCompare(bLetter)
          if (cmp !== 0) return cmp
        } else if (aLetter || bLetter) {
          return aLetter ? -1 : 1
        }
        // Fallback to label ordering if letters are missing
        return formatHeavyEnergyLabel(a).toLowerCase().localeCompare(
          formatHeavyEnergyLabel(b).toLowerCase()
        )
      })

      // Section header spanning both columns
      const header = document.createElement("div")
      header.className = "energyLabel"
      header.textContent = "Heavy Weapons"
      header.style.gridColumn = "1 / span 2"
      table.appendChild(header)

      if (sortedHeavySquares.length === 0) {
        const placeholderLabel = document.createElement("div")
        placeholderLabel.className = "energyLabel"
        placeholderLabel.textContent = "None"
        const placeholderInputWrap = document.createElement("div")
        placeholderInputWrap.className = "energyInputWrap"
        const placeholderInput = document.createElement("input")
        placeholderInput.type = "number"
        placeholderInput.className = "energyInput"
        placeholderInput.value = "0"
        placeholderInput.disabled = true
        placeholderInput.placeholder = "No heavy weapons"
        placeholderInputWrap.appendChild(placeholderInput)
        table.appendChild(placeholderLabel)
        table.appendChild(placeholderInputWrap)
        continue
      }

      for (const hs of sortedHeavySquares) {
        const customLabel = formatHeavyEnergyLabel(hs)

        const hl = document.createElement("div")
        hl.className = "energyLabel"
        hl.textContent = customLabel
        table.appendChild(hl)

        const hwWrap = document.createElement("div")
        hwWrap.className = "energyInputWrap"
      const hwInput = document.createElement("input")
      hwInput.type = "number"
      hwInput.className = "energyInput"

        const storedVal = tab.energyValues[customLabel]
        const val = Number(storedVal)
        const hasVal = storedVal !== undefined && Number.isFinite(val)
        if (!locked && hasVal) tab.energyValues[customLabel] = val
        hwInput.value = !locked && resetMode && !hasVal ? "" : String(hasVal ? val : 0)
        hwInput.disabled = locked

        if (!locked) {
          hwInput.oninput = () => {
            tab.energyReset = false
            let n = parseInt(hwInput.value, 10)
            if (Number.isNaN(n) || n < 0) n = 0
            tab.energyValues[customLabel] = n
            hwInput.value = String(n)
            updateTotalUsedInput()
          }
        }

        hwWrap.appendChild(hwInput)
        table.appendChild(hwWrap)
      }

      continue
    }

    if (isSpecificReinforcement) {
      const header = document.createElement("div")
      header.className = "energyLabel"
      header.textContent = "Specific Reinforcement"
      header.style.gridColumn = "1 / span 2"
      table.appendChild(header)

      const shields = getShieldNumbers(tab)
      if (shields.length === 0) {
        const placeholderLabel = document.createElement("div")
        placeholderLabel.className = "energyLabel"
        placeholderLabel.textContent = "Shields"
        const placeholderInputWrap = document.createElement("div")
        placeholderInputWrap.className = "energyInputWrap"
        const placeholderInput = document.createElement("input")
        placeholderInput.type = "number"
        placeholderInput.className = "energyInput"
        placeholderInput.disabled = true
        placeholderInput.placeholder = "No shields found"
        placeholderInputWrap.appendChild(placeholderInput)
        table.appendChild(placeholderLabel)
        table.appendChild(placeholderInputWrap)
        continue
      }

      for (const n of shields) {
        const shieldLabel = `Shield #${n}`
        const lbl = document.createElement("div")
        lbl.className = "energyLabel"
        lbl.textContent = shieldLabel
        table.appendChild(lbl)

        const wrap = document.createElement("div")
        wrap.className = "energyInputWrap"
        const inp = document.createElement("input")
        inp.type = "number"
        inp.step = "0.5"
        inp.className = "energyInput"
        const storedVal = tab.energyValues[shieldLabel]
        let val = Number(storedVal)
        const hasVal = storedVal !== undefined && Number.isFinite(val)
        inp.value = !locked && resetMode && !hasVal ? "" : String(hasVal ? val : 0)
        inp.disabled = locked
        if (!locked) {
          inp.oninput = () => {
            tab.energyReset = false
            const parsed = parseFloat(inp.value)
            let nVal = Number.isFinite(parsed) ? parsed : 0
            if (Number.isNaN(nVal) || nVal < 0) nVal = 0
            nVal = Math.round(nVal * 2) / 2
            tab.energyValues[shieldLabel] = nVal
            inp.value = String(nVal)
            updateTotalUsedInput()
          }
        }
        wrap.appendChild(inp)
        table.appendChild(wrap)
      }

      continue
    }

    table.appendChild(label)

    if (labelLower === "speed") {
      const input = document.createElement("input")
      input.type = "number"
      input.step = "0.01"
      input.className = "energyInput"
      input.disabled = true
      input.placeholder = "Auto"
      input.setAttribute("aria-disabled", "true")
      speedInput = input
      const storedSpeed = tab.energyValues["Speed"]
      if (storedSpeed !== undefined) input.value = String(storedSpeed)
      updateSpeedInput()
      inputWrap.appendChild(input)
      table.appendChild(inputWrap)
      continue
    }

    if (isPhaserCapsUsed) {
      const input = document.createElement("input")
      input.type = "number"
      input.step = "0.5"
      input.className = "energyInput"
      input.disabled = true
      input.placeholder = "Auto"
      input.setAttribute("aria-disabled", "true")
      const total = recomputePhaserCapsUsed(tab)
      const showBlank = tab.phaserCapsCleared && total === 0 && !locked
      if (total > 0) tab.phaserCapsCleared = false
      input.value = showBlank ? "" : String(Number.isFinite(total) ? total : 0)
      input.title = "Sum of deployed Phaser 1/2/3 usage"
      inputWrap.appendChild(input)
      table.appendChild(inputWrap)
      continue
    }

    if (isPhaserCharge) {
      const val = computeCurrentPhaserCharge(tab, { updateValues: !locked })

      const input = document.createElement("input")
      input.type = "number"
      input.className = "energyInput"
      input.disabled = true
      input.placeholder = "Auto"
      input.setAttribute("aria-disabled", "true")
      input.value = String(val)
      input.title = "(Remaining + Charge) - Used, capped at Max"
      inputWrap.appendChild(input)
      table.appendChild(inputWrap)
      continue
    }

    if (isMovementCost) {
      const storedRaw = tab.energyDisplay[row.label]
      const storedVal = tab.energyValues[row.label]
      const parsed = parseFractionParts(storedRaw ?? storedVal ?? "1/1")
      if (!locked) {
        tab.energyValues[row.label] = parsed.num
        tab.energyDisplay[row.label] = parsed.raw
      }

      const fracWrap = document.createElement("div")
      fracWrap.className = "fractionWrap"

      const numInput = document.createElement("input")
      numInput.type = "number"
      numInput.min = "0"
      numInput.step = "1"
      numInput.className = "energyInput fractionInput"
      numInput.value = parsed.numPart
      numInput.disabled = locked

      const slash = document.createElement("span")
      slash.className = "fractionSlash"
      slash.textContent = "/"

      const denInput = document.createElement("input")
      denInput.type = "number"
      denInput.min = "1"
      denInput.step = "1"
      denInput.className = "energyInput fractionInput"
      denInput.value = parsed.denPart
      denInput.disabled = locked

      const applyFraction = () => {
        let nVal = parseInt(numInput.value, 10)
        let dVal = parseInt(denInput.value, 10)
        if (!Number.isFinite(nVal) || nVal < 0) nVal = 0
        if (!Number.isFinite(dVal) || dVal <= 0) dVal = 1
        numInput.value = String(nVal)
        denInput.value = String(dVal)
        const raw = `${nVal}/${dVal}`
        const next = parseFractionParts(raw)
        if (!locked) {
          tab.energyValues[row.label] = next.num
          tab.energyDisplay[row.label] = next.raw
          updateTotalUsedInput()
          updateSpeedInput()
        }
      }

      if (!locked) {
        numInput.oninput = applyFraction
        denInput.oninput = applyFraction
      }

      if (row.locked) {
        numInput.disabled = true
        denInput.disabled = true
        numInput.placeholder = "Locked"
        denInput.placeholder = "Locked"
      }

      fracWrap.appendChild(numInput)
      fracWrap.appendChild(slash)
      fracWrap.appendChild(denInput)
      inputWrap.appendChild(fracWrap)
      table.appendChild(inputWrap)
      continue
    }

    const input = document.createElement("input")
    input.type = "number"
    input.className = "energyInput"
    if (isEnergyForMovement) input.step = "0.5"
    if (isLifeSupport) {
      input.step = "0.5"
      input.inputMode = "decimal"
    }
    if (isTransporters) input.step = "0.5"

    if (isWarpEnginePower) {
      const totalWarp = powerStats.warp
      const { left, right, center } = powerStats.warpDetail
      const val = locked ? (tab.energyValues[row.label] ?? totalWarp) : totalWarp
      if (!locked) tab.energyValues[row.label] = totalWarp
      input.value = String(val ?? 0)
      input.disabled = true
      input.title = `Left Warp: ${left} | Right Warp: ${right} | Center Warp: ${center}`
    } else if (isImpulsePower) {
      const val = locked ? (tab.energyValues[row.label] ?? powerStats.impulse) : powerStats.impulse
      if (!locked) tab.energyValues[row.label] = powerStats.impulse
      input.value = String(val ?? 0)
      input.disabled = true
      input.title = "Count of boxes labeled 'Impulse'"
    } else if (isReactorPower) {
      const val = locked ? (tab.energyValues[row.label] ?? powerStats.reactor) : powerStats.reactor
      if (!locked) tab.energyValues[row.label] = powerStats.reactor
      input.value = String(val ?? 0)
      input.disabled = true
      input.title = "Count of boxes labeled 'APR'"
    } else if (isBatteryAvailable) {
      const stored = tab.energyValues[row.label]
      const hasStored = stored !== undefined && Number.isFinite(Number(stored))
      const rawVal = hasStored ? Number(stored) : powerStats.battery
      const clamped = Math.max(0, Math.min(powerStats.battery, rawVal))
      if (!locked) tab.energyValues[row.label] = clamped
      input.value = String(clamped ?? 0)
      input.disabled = true
      input.title = "Count of boxes labeled 'Battery' (or carried over from previous turn)"
    } else if (isTotalBatteries) {
      const val = powerStats.battery
      if (!locked) tab.energyValues[row.label] = val
      input.value = String(val ?? 0)
      input.disabled = true
      input.title = "Total number of boxes labeled 'Battery'"
    } else if (isBatteryDischarged) {
      const val = computeBatteryDischarged(tab)
      if (!locked) tab.energyValues[row.label] = val
      input.value = String(val ?? 0)
      input.disabled = true
      input.setAttribute("aria-disabled", "true")
      input.placeholder = "Auto"
      input.title = "Program-controlled battery discharge"
    } else if (isTotalAvailable) {
      // Total available = warp + impulse + reactor (from computed counts)
      const totalVal = powerStats.total
      const val = locked ? (tab.energyValues[row.label] ?? totalVal) : totalVal
      if (!locked) tab.energyValues[row.label] = totalVal
      input.value = String(val ?? 0)
      input.disabled = true
      input.title = "Warp + Impulse + Reactor power"
      totalAvailableVal = totalVal
    } else if (isTotalUsed) {
      if (!locked) tab.energyValues[row.label] = 0
      const stored = tab.energyValues[row.label]
      input.value = String(stored === undefined ? 0 : stored)
      input.disabled = true
      input.setAttribute("aria-disabled", "true")
      input.title = "Sum of user inputs (excludes Movement Cost and auto fields)"
      totalUsedInput = input
    } else if (isMaxPhaserCaps) {
      const caps = computeMaxPhaserCaps(tab)
      const val = locked ? (tab.energyValues[row.label] ?? caps) : caps
      if (!locked) {
        tab.energyValues[row.label] = caps
        syncPhaserRemainingWithMax(tab, caps)
      }
      input.value = String(val ?? 0)
      input.disabled = true
      input.title = "Auto-calculated from Phaser boxes (1/2 for Phaser 3)"
    } else {
      const storedVal = tab.energyValues[row.label]
      let val = Number(storedVal)
      const hasVal = storedVal !== undefined && Number.isFinite(val)
    if (isPhaserCapsRemaining) {
      const caps = computeMaxPhaserCaps(tab)
      const storedKey =
        findEnergyKey(tab.energyValues, "held phaser capacitors") ||
        findEnergyKey(tab.energyValues, "phaser capacitors remaining") ||
        row.label
      const storedValNorm = Number(tab.energyValues[storedKey])
      const hasStored = Number.isFinite(storedValNorm)
      val = hasStored ? storedValNorm : caps
      val = Math.min(val, caps)
      if (!locked) tab.energyValues[storedKey] = val
    }
    if (!locked && hasVal) tab.energyValues[row.label] = val
      input.value = !locked && resetMode && !hasVal ? "" : String(hasVal ? val : 0)
      input.disabled = locked || !!row.locked
    }

    const lockPlaceholder =
      !isBatteryDischarged &&
      (row.locked || isWarpEnginePower || isImpulsePower || isReactorPower || isTotalAvailable || isTotalUsed)
    input.placeholder = lockPlaceholder ? "Locked" : ""

    if (!locked) {
      input.oninput = () => {
        tab.energyReset = false
        if (isEnergyForMovement) {
          const raw = input.value
          const n = Number(raw)
          if (Number.isFinite(n)) {
            const rounded = Math.max(0, Math.round(n * 2) / 2)
            tab.energyValues[row.label] = rounded
          }
          // Keep the user's raw text so typing "1.5" doesn't get truncated.
          if (raw === "") tab.energyValues[row.label] = 0
        } else if (isLifeSupport) {
          const raw = input.value
          const parsed = Number(raw)
          if (Number.isFinite(parsed)) {
            const n = Math.max(0, parsed)
            tab.energyValues[row.label] = n
            // Keep the user's text so decimals stay visible
            input.value = raw
          } else if (raw === "") {
            tab.energyValues[row.label] = 0
          }
        } else if (isTransporters) {
          const raw = input.value
          const parsed = Number(raw)
          if (Number.isFinite(parsed)) {
            const rounded = Math.max(0, Math.round(parsed * 2) / 2)
            tab.energyValues[row.label] = rounded
            input.value = String(rounded)
          } else if (raw === "") {
            tab.energyValues[row.label] = 0
          }
        } else {
          const parsed = parseInt(input.value, 10)
          let n = Number.isFinite(parsed) ? parsed : 0
          if (Number.isNaN(n) || n < 0) n = 0
          tab.energyValues[row.label] = n
          input.value = String(n)
        }
        if (isEnergyForMovement) updateSpeedInput()
        updateTotalUsedInput()
        if (labelLower === "charge phaser capacitors") {
          updateCurrentPhaserChargeDisplay(tab)
        }
      }
    }

    inputWrap.appendChild(input)
    table.appendChild(inputWrap)
  }

  updateTotalUsedInput()
  updateSpeedInput()
}

function renderCurrentProps() {
  const tab = getActiveTab()

  const shipyardPanel = document.getElementById("shipyardPanel")
  const energyPanel = document.getElementById("energyPanel")
  const analysisPanel = document.getElementById("analysisPanel")
  const movementAnalysisPanel = document.getElementById("movementAnalysisPanel")
  const jsonEditPanel = document.getElementById("jsonEditPanel")

  if (!tab || !tab.doc) {
    if (shipyardPanel) shipyardPanel.classList.add("hidden")
    if (energyPanel) energyPanel.classList.add("hidden")
    const convertPanel = document.getElementById("convertPanel")
    if (convertPanel) convertPanel.classList.add("hidden")
    if (analysisPanel) analysisPanel.classList.add("hidden")
    if (movementAnalysisPanel) movementAnalysisPanel.classList.add("hidden")
    if (jsonEditPanel) jsonEditPanel.classList.add("hidden")
    return
  }

  const isShipyard = tab.uiState === "shipyard"
  const isConvert = tab.uiState === "convert"
  const isAnalysis = tab.uiState === "analysis"
  const isMovementAnalysis = tab.uiState === "movementAnalysis"
  const isJsonEdit = tab.uiState === "jsonEdit"
  const convertPanel = document.getElementById("convertPanel")

  if (shipyardPanel) {
    if (isShipyard) shipyardPanel.classList.remove("hidden")
    else shipyardPanel.classList.add("hidden")
  }

  if (convertPanel) {
    if (isConvert && !state.convertMenuHidden) convertPanel.classList.remove("hidden")
    else convertPanel.classList.add("hidden")
  }

  if (analysisPanel) {
    if (isAnalysis) analysisPanel.classList.remove("hidden")
    else analysisPanel.classList.add("hidden")
  }

  if (movementAnalysisPanel) {
    if (isMovementAnalysis) movementAnalysisPanel.classList.remove("hidden")
    else movementAnalysisPanel.classList.add("hidden")
  }

  if (jsonEditPanel) {
    if (isJsonEdit) jsonEditPanel.classList.remove("hidden")
    else jsonEditPanel.classList.add("hidden")
  }

  if (energyPanel) {
    energyPanel.classList.add("hidden")
  }

  if (isShipyardDoneVisible()) renderShipyardProps(tab?.selected || null)
  if (isAnalysis) renderAnalysisPanel(tab)
  if (isMovementAnalysis) renderMovementAnalysisPanel(tab)
  if (isJsonEdit) renderJsonEditProps(tab?.selected || null)
}

function addReadonlyRow(grid, key, val) {
  const row = document.createElement("div")
  row.className = "gridRow"

  const k = document.createElement("div")
  k.className = "gridKey"
  k.textContent = key

  const v = document.createElement("div")
  v.className = "gridVal"

  const input = document.createElement("input")
  input.value = val === undefined ? "" : String(val)
  input.disabled = true

  v.appendChild(input)
  row.appendChild(k)
  row.appendChild(v)
  grid.appendChild(row)
}

function addEditableGroupNameRow(grid, group) {
  const row = document.createElement("div")
  row.className = "gridRow"

  const k = document.createElement("div")
  k.className = "gridKey"
  k.textContent = "name"

  const v = document.createElement("div")
  v.className = "gridVal"

  const input = document.createElement("input")
  input.value = group.name || group.label || ""
  input.oninput = () => {
    group.name = input.value
    group.label = input.value
    renderCanvas()
    renderCurrentProps()
  }

  v.appendChild(input)
  row.appendChild(k)
  row.appendChild(v)
  grid.appendChild(row)
}

function addEditableSquareLabelRow(grid, square) {
  const row = document.createElement("div")
  row.className = "gridRow"

  const k = document.createElement("div")
  k.className = "gridKey"
  k.textContent = "label"

  const v = document.createElement("div")
  v.className = "gridVal"

  const input = document.createElement("input")
  input.value = square.label || square.name || ""
  input.oninput = () => {
    square.label = input.value
    square.name = input.value
    renderCanvas()
    renderCurrentProps()
  }

  v.appendChild(input)
  row.appendChild(k)
  row.appendChild(v)
  grid.appendChild(row)
}

function addEditableGroupIdRow(grid, tab, square) {
  const row = document.createElement("div")
  row.className = "gridRow"

  const k = document.createElement("div")
  k.className = "gridKey"
  k.textContent = "groupId"

  const v = document.createElement("div")
  v.className = "gridVal"

  const sel = document.createElement("select")

  const optNull = document.createElement("option")
  optNull.value = "null"
  optNull.textContent = "null"
  sel.appendChild(optNull)

  const groupIds = (tab.doc.groups || []).map(g => g.id).sort((a, b) => a - b)
  for (const gid of groupIds) {
    const opt = document.createElement("option")
    opt.value = String(gid)
    opt.textContent = String(gid)
    sel.appendChild(opt)
  }

  sel.value = square.groupId === null ? "null" : String(square.groupId)

  sel.onchange = () => {
    square.groupId = sel.value === "null" ? null : Number(sel.value)
    renderCanvas()
    renderCurrentProps()
  }

  v.appendChild(sel)
  row.appendChild(k)
  row.appendChild(v)
  grid.appendChild(row)
}

function getJsonEditSourceEntry(tab, meta) {
  if (!tab || !meta || meta.format !== "superluminal") return null
  const ssdKey = String(meta.ssdKey || "").trim()
  const entryIndex = Number(meta.entryIndex)
  if (!ssdKey || !Number.isInteger(entryIndex) || entryIndex < 0) return null

  const entries = tab.jsonEditSourceDoc?.ssd?.[ssdKey]
  if (!Array.isArray(entries)) return null
  const entry = entries[entryIndex]
  return entry && typeof entry === "object" ? entry : null
}

function getJsonEditEntryField(tab, meta, fieldName) {
  const key = String(fieldName || "").trim()
  if (!key) return ""

  const sourceEntry = getJsonEditSourceEntry(tab, meta)
  if (sourceEntry && Object.prototype.hasOwnProperty.call(sourceEntry, key)) {
    return sourceEntry[key]
  }

  const metaEntry = meta?.entry && typeof meta.entry === "object" ? meta.entry : null
  return metaEntry && Object.prototype.hasOwnProperty.call(metaEntry, key)
    ? metaEntry[key]
    : ""
}

function setJsonEditEntryField(tab, square, meta, fieldName, value) {
  const key = String(fieldName || "").trim()
  if (!key) return

  const next = String(value ?? "")
  const sourceEntry = getJsonEditSourceEntry(tab, meta)
  if (sourceEntry) {
    sourceEntry[key] = next
  }

  if (meta && typeof meta === "object") {
    if (!meta.entry || typeof meta.entry !== "object") meta.entry = {}
    meta.entry[key] = next
  }

  if (square?.__jsonEdit && typeof square.__jsonEdit === "object") {
    if (!square.__jsonEdit.entry || typeof square.__jsonEdit.entry !== "object") {
      square.__jsonEdit.entry = {}
    }
    square.__jsonEdit.entry[key] = next
  }
}

function addEditableJsonEditEntryFieldRow(grid, tab, square, meta, fieldName, label = fieldName, onAfterInput = null) {
  const row = document.createElement("div")
  row.className = "gridRow"

  const k = document.createElement("div")
  k.className = "gridKey"
  k.textContent = label

  const v = document.createElement("div")
  v.className = "gridVal"

  const input = document.createElement("input")
  input.value = String(getJsonEditEntryField(tab, meta, fieldName) ?? "")
  input.oninput = () => {
    setJsonEditEntryField(tab, square, meta, fieldName, input.value)
    if (typeof onAfterInput === "function") onAfterInput(input.value)
  }

  v.appendChild(input)
  row.appendChild(k)
  row.appendChild(v)
  grid.appendChild(row)
}

function addEditableJsonEditEntryChoiceRow(grid, tab, square, meta, fieldName, label = fieldName, options = [], onAfterInput = null) {
  const row = document.createElement("div")
  row.className = "gridRow"

  const k = document.createElement("div")
  k.className = "gridKey"
  k.textContent = label

  const v = document.createElement("div")
  v.className = "gridVal"

  const input = document.createElement("input")
  const listId = `jsonEditChoiceList${++jsonEditDatalistCounter}`
  input.setAttribute("list", listId)
  input.value = String(getJsonEditEntryField(tab, meta, fieldName) ?? "")

  const datalist = document.createElement("datalist")
  datalist.id = listId
  const seen = new Set()
  for (const rawOption of Array.isArray(options) ? options : []) {
    const optionText = String(rawOption || "").trim()
    if (!optionText || seen.has(optionText.toLowerCase())) continue
    seen.add(optionText.toLowerCase())
    const opt = document.createElement("option")
    opt.value = optionText
    datalist.appendChild(opt)
  }

  input.oninput = () => {
    setJsonEditEntryField(tab, square, meta, fieldName, input.value)
    if (typeof onAfterInput === "function") onAfterInput(input.value)
  }

  v.appendChild(input)
  v.appendChild(datalist)
  row.appendChild(k)
  row.appendChild(v)
  grid.appendChild(row)
}

/* -----------------------------
   Shipyard panel helpers
----------------------------- */
function showShipyardPanel() {
  const shipyardPanel = document.getElementById("shipyardPanel")
  if (shipyardPanel) shipyardPanel.classList.remove("hidden")
}

function hideShipyardPanel() {
  const shipyardPanel = document.getElementById("shipyardPanel")
  if (shipyardPanel) shipyardPanel.classList.add("hidden")
}

function showShipyardNameStep() {
  document.getElementById("shipyardNameStep").classList.remove("hidden")
  document.getElementById("shipyardLabelStep").classList.add("hidden")
  document.getElementById("shipyardDoneStep").classList.add("hidden")
}

function showShipyardLabelStep() {
  document.getElementById("shipyardNameStep").classList.add("hidden")
  document.getElementById("shipyardLabelStep").classList.remove("hidden")
  document.getElementById("shipyardDoneStep").classList.add("hidden")
}

function showShipyardDoneStep(doneText) {
  document.getElementById("shipyardNameStep").classList.add("hidden")
  document.getElementById("shipyardLabelStep").classList.add("hidden")
  document.getElementById("shipyardDoneStep").classList.remove("hidden")
  document.getElementById("shipyardDoneText").textContent = doneText || "Shipyard complete."

  const editBtn = document.getElementById("shipyardEditLabel")
  if (editBtn) {
    editBtn.disabled = false
    editBtn.style.pointerEvents = "auto"
    editBtn.onclick = () => editSelectedLabelShipyard()
  }

  renderShipyardProps(getActiveTab()?.selected || null)
}

function shipyardAskShipName(defaultName) {
  showShipyardPanel()
  showShipyardNameStep()

  const input = document.getElementById("shipyardShipName")
  const btnStart = document.getElementById("shipyardNameStart")
  const btnSkip = document.getElementById("shipyardNameSkip")
  const btnCancel = document.getElementById("shipyardNameCancel")

  input.value = defaultName || ""
  input.focus()
  input.select()

  return new Promise((resolve) => {
    function cleanup() {
      btnStart.onclick = null
      btnSkip.onclick = null
      btnCancel.onclick = null
      input.onkeydown = null
      window.removeEventListener("keydown", onEsc)
    }

    function finishCancel() {
      cleanup()
      resolve({ action: "cancel" })
    }

    function finishSkip() {
      cleanup()
      resolve({ action: "skip" })
    }

    function finishRename() {
      const name = String(input.value || "").trim()
      if (!name) return
      cleanup()
      resolve({ action: "rename", name })
    }

    function onEsc(e) {
      if (e.key === "Escape") finishCancel()
    }

    window.addEventListener("keydown", onEsc)

    btnStart.onclick = () => finishRename()
    btnSkip.onclick = () => finishSkip()
    btnCancel.onclick = () => finishCancel()

    input.onkeydown = (e) => {
      if (e.key === "Enter") finishRename()
      if (e.key === "Escape") finishCancel()
    }
  })
}

function shipyardPickLabel(titleText, options, removeMode) {
  showShipyardPanel()
  showShipyardLabelStep()

  const title = document.getElementById("shipyardTitle")
  const filter = document.getElementById("shipyardFilter")
  const select = document.getElementById("shipyardSelect")
  const btnApply = document.getElementById("shipyardApply")
  const btnSkip = document.getElementById("shipyardSkip")
  const btnCancel = document.getElementById("shipyardCancel")
  const btnRemove = document.getElementById("shipyardRemoveGroup")

  title.textContent = titleText
  filter.value = ""

  const normalizedOptions = (options || []).map(o => String(o || ""))

  const allowRemove = removeMode === "group" || removeMode === "square"
  btnRemove.disabled = !allowRemove
  btnRemove.style.opacity = allowRemove ? "1" : "0.4"

  function fill(list) {
    select.innerHTML = ""
    for (const opt of list) {
      const o = document.createElement("option")
      o.value = opt
      o.textContent = opt
      select.appendChild(o)
    }
    if (select.options.length > 0) select.selectedIndex = 0
  }

  fill(normalizedOptions)
  filter.focus()

  return new Promise((resolve) => {
    function cleanup() {
      btnApply.onclick = null
      btnSkip.onclick = null
      btnCancel.onclick = null
      btnRemove.onclick = null
      filter.oninput = null
      select.ondblclick = null
      window.removeEventListener("keydown", onKey)
    }

    function apply() {
      const val = select.value || ""
      cleanup()
      resolve({ action: "apply", value: val })
    }

    function onKey(e) {
      if (e.key === "Escape") {
        cleanup()
        resolve({ action: "cancel" })
      } else if (e.key === "Enter") {
        apply()
      }
    }

    window.addEventListener("keydown", onKey)

    filter.oninput = () => {
      const q = filter.value.trim().toLowerCase()
      if (!q) return fill(normalizedOptions)
      fill(normalizedOptions.filter(x => x.toLowerCase().includes(q)))
    }

    select.ondblclick = () => apply()
    btnApply.onclick = () => apply()

    btnSkip.onclick = () => {
      cleanup()
      resolve({ action: "skip" })
    }

    btnCancel.onclick = () => {
      cleanup()
      resolve({ action: "cancel" })
    }

    btnRemove.onclick = () => {
      cleanup()
      resolve({ action: "remove", removeMode })
    }
  })
}

function shipyardDoneAskSave() {
  const btnSave = document.getElementById("shipyardDoneSave")
  const doneStep = document.getElementById("shipyardDoneStep")

  // Make sure the done panel and buttons are interactive.
  if (doneStep) {
    doneStep.classList.remove("hidden")
    doneStep.style.pointerEvents = "auto"
  }
  if (btnSave) btnSave.disabled = false

  // If Save button is missing, bail out quietly
  if (!btnSave) {
    return Promise.resolve({ action: "close" })
  }

  return new Promise((resolve) => {
    function cleanup() {
      btnSave.removeEventListener("click", onSave)
    }

    function onSave() {
      cleanup()
      resolve({ action: "save" })
    }

    btnSave.addEventListener("click", onSave, { once: true })
  })
}

const FIRING_ARC_TOKENS = new Set([
  "NONE",
  "360",
  "FA",
  "FX",
  "FP",
  "RF",
  "LF",
  "R",
  "L",
  "RR",
  "LR",
  "RA",
  "RX",
  "RS",
  "LS",
  "FH",
  "RH",
  "LP",
  "RP",
  "AP",
  "LPA",
  "RPA"
])

function splitFiringArcTokens(raw) {
  return String(raw || "")
    .trim()
    .split(/[+\\/,&]/)
    .map(part => part.trim().toUpperCase())
    .filter(Boolean)
}

function isFiringArcText(raw) {
  const tokens = splitFiringArcTokens(raw)
  if (tokens.length === 0) return false
  return tokens.every(token => FIRING_ARC_TOKENS.has(token))
}

function stripFiringArcSuffix(raw) {
  let text = String(raw || "").trim()
  if (!text) return ""

  while (true) {
    const prev = text

    const bracketed = text.match(
      /^(.*?)(?:\s*[-,:]\s*|\s+)?[\(\[\{]\s*([A-Za-z0-9+\\/,&\s]+)\s*[\)\]\}]\s*$/
    )
    if (bracketed && isFiringArcText(bracketed[2])) {
      text = bracketed[1].trim()
      continue
    }

    const delimited = text.match(/^(.*?)(?:\s*[-,:]\s*)([A-Za-z0-9+\\/,&\s]+)\s*$/)
    if (delimited && isFiringArcText(delimited[2])) {
      text = delimited[1].trim()
      continue
    }

    const bareExpression = text.match(
      /^(.*\S)\s+([A-Za-z0-9]+(?:\s*[+\\/,&]\s*[A-Za-z0-9]+)+)\s*$/
    )
    if (bareExpression && isFiringArcText(bareExpression[2])) {
      text = bareExpression[1].trim()
      continue
    }

    const bareToken = text.match(/^(.*\S)\s+([A-Za-z0-9]+)\s*$/)
    if (bareToken) {
      const token = bareToken[2].trim().toUpperCase()
      // Do not strip single-letter tokens from bare suffixes (e.g., "Plasma Torpedo R").
      if (token.length > 1 && FIRING_ARC_TOKENS.has(token)) {
        text = bareToken[1].trim()
        continue
      }
    }

    if (text === prev) break
  }

  return text
}

function normalizeLabel(raw) {
  const text = String(raw ?? "").trim()
  const hasCaret = text.startsWith("^")
  const withoutCaret = hasCaret ? text.slice(1).trimStart() : text
  const clean = stripFiringArcSuffix(withoutCaret)
  return { clean, hasCaret }
}

function analysisKeyFromLabel(raw) {
  const clean = String(raw || "").trim().toLowerCase()
  const key = clean.replace(/[^a-z0-9]+/g, "")
  return key || "misc"
}

function mapLabelToAnalysisKey(labelLower) {
  if (!labelLower) return "misc"
  for (const entry of ANALYSIS_EXTERNAL_KEYWORD_MAP) {
    if (entry.terms.some(term => labelLower.includes(term))) {
      return entry.key
    }
  }
  return analysisKeyFromLabel(labelLower)
}

function buildExternalKeySet(externalLabelSet) {
  const set = new Set()
  if (!externalLabelSet || externalLabelSet.size === 0) return set
  for (const label of externalLabelSet) {
    const key = mapLabelToAnalysisKey(label)
    if (key) set.add(key)
  }
  return set
}

async function readExternalLabelSet() {
  if (externalLabelSetCache) return externalLabelSetCache
  if (!window.api || typeof window.api.readExternalList !== "function") {
    externalLabelSetCache = new Set()
    return externalLabelSetCache
  }
  try {
    const res = await window.api.readExternalList()
    if (!res || !res.ok || !Array.isArray(res.options)) {
      externalLabelSetCache = new Set()
      return externalLabelSetCache
    }
    const set = new Set()
    for (const opt of res.options) {
      const clean = normalizeLabel(opt).clean.toLowerCase()
      if (clean) set.add(clean)
    }
    externalLabelSetCache = set
    return externalLabelSetCache
  } catch {
    externalLabelSetCache = new Set()
    return externalLabelSetCache
  }
}

function computeShieldCountsFromSsd(doc) {
  const counts = [0, 0, 0, 0, 0, 0]
  const shields = doc?.ssd?.shields
  if (!shields || typeof shields !== "object" || Array.isArray(shields)) return counts
  for (const [key, value] of Object.entries(shields)) {
    const match = String(key).match(/sh(?:ie|ei)ld[^0-9]*(\d+)/i)
    if (!match) continue
    const idx = Number(match[1])
    if (idx < 1 || idx > 6) continue
    const numeric = Number(value)
    if (Number.isFinite(numeric)) counts[idx - 1] += numeric
  }
  return counts
}

function isExternalEntry(entry, groupKey, externalLabelSet, externalKeySet) {
  const typeRaw = String(entry?.type || entry?.name || entry?.label || "").trim()
  if (typeRaw) {
    const clean = normalizeLabel(typeRaw).clean.toLowerCase()
    if (clean && externalLabelSet.has(clean)) return true
  }
  const groupRaw = String(groupKey || "").trim()
  const groupClean = normalizeLabel(groupRaw).clean.toLowerCase()
  if (groupClean && externalLabelSet.has(groupClean)) return true
  const mappedKey = mapLabelToAnalysisKey(groupClean)
  if (mappedKey && externalKeySet.has(mappedKey)) return true
  return externalKeySet.has(groupRaw)
}

function computeShipAnalysisStats(doc, externalLabelSet) {
  const stats = {
    shieldCounts: [0, 0, 0, 0, 0, 0],
    nonExternalBoxes: 0,
    totalBoxes: 0,
    externalBoxes: 0
  }

  if (!doc) return stats

  const externalKeySet = buildExternalKeySet(externalLabelSet)
  if (doc?.ssd && typeof doc.ssd === "object") {
    stats.shieldCounts = computeShieldCountsFromSsd(doc)
    for (const [key, value] of Object.entries(doc.ssd)) {
      if (Array.isArray(value)) {
        for (const entry of value) {
          stats.totalBoxes += 1
          if (isExternalEntry(entry, key, externalLabelSet, externalKeySet)) {
            stats.externalBoxes += 1
          }
        }
      } else if (key === "shields" && value && typeof value === "object") {
        const shieldTotal = stats.shieldCounts.reduce((sum, v) => sum + (Number(v) || 0), 0)
        if (shieldTotal > 0) {
          stats.totalBoxes += shieldTotal
          const shieldKeys = Object.keys(value)
          const hasExternalShieldKey = shieldKeys.some((shieldKey) => {
            const clean = normalizeLabel(shieldKey).clean.toLowerCase()
            if (clean && externalLabelSet.has(clean)) return true
            const mappedKey = mapLabelToAnalysisKey(clean)
            return Boolean(mappedKey && externalKeySet.has(mappedKey))
          })
          if (hasExternalShieldKey || externalKeySet.has("shields")) {
            stats.externalBoxes += shieldTotal
          }
        }
      }
    }
  }

  stats.nonExternalBoxes = Math.max(0, stats.totalBoxes - stats.externalBoxes)
  return stats
}

function formatShieldCountsLine(counts) {
  if (!Array.isArray(counts) || counts.length !== 6) return "Shields: n/a"
  const total = counts.reduce((sum, v) => sum + (Number(v) || 0), 0)
  if (total <= 0) return "Shields: none"
  const parts = counts.map((count, idx) => `#${idx + 1} ${Number(count) || 0}`)
  return `Shields: ${parts.join(", ")}`
}

function applyGroupLabel(group, label) {
  const { clean } = normalizeLabel(label)
  group.name = clean
  group.label = clean
}

function applySquareLabel(square, label, tab) {
  const { clean } = normalizeLabel(label)
  square.label = clean
  square.name = clean
  const isHeavy = tab?.heavyLabelSet?.has(clean.toLowerCase())
  if (isHeavy) square.additionalLabel = "heavy Weapons"
  else if (square.additionalLabel === "heavy Weapons") delete square.additionalLabel
  if (!isHeavy && square.heavyLetter) delete square.heavyLetter
  delete square._inheritedFromGroupLabel
}

async function ensureHeavyDesignations(tab) {
  if (!tab || !tab.doc) return true

  const persistHeavyLetters = async () => {
    if (window.api?.saveShipyardDoc) {
      await window.api.saveShipyardDoc(tab.doc)
    }
  }

  const cleanupModals = () => {
    document.querySelectorAll(".modalOverlay").forEach(el => {
      if (el.parentElement) el.parentElement.removeChild(el)
    })
  }

  cleanupModals()

  if (!tab.heavyLabelSet || tab.heavyLabelSet.size === 0) {
    const heavyRes = await window.api.readHeavyList?.()
    if (heavyRes && heavyRes.ok && Array.isArray(heavyRes.options)) {
      tab.heavyLabelSet = new Set(
        heavyRes.options
          .map(x => normalizeLabel(x).clean.toLowerCase())
          .filter(Boolean)
      )
    }
  }

  const panel = document.getElementById("heavyDesignationPanel")
  const labelEl = document.getElementById("heavyDesignationLabel")
  const inputEl = document.getElementById("heavyDesignationInput")
  const btnCancel = document.getElementById("heavyDesignationCancel")
  const btnApply = document.getElementById("heavyDesignationApply")
  const rawPanel = document.getElementById("shipyardRawPanel")
  const actionsRow = document.getElementById("shipyardDoneActions")

  let isPanelActive = false

  const showPanel = () => {
    if (!panel) return
    isPanelActive = true
    panel.classList.remove("hidden")
    rawPanel?.classList.add("hidden")
    actionsRow?.classList.add("hidden")
  }

  const hidePanel = () => {
    if (!panel) return
    isPanelActive = false
    panel.classList.add("hidden")
    rawPanel?.classList.remove("hidden")
    actionsRow?.classList.remove("hidden")
  }

  const cleanup = () => {
    hidePanel()
    cleanupModals()
    if (btnCancel) btnCancel.onclick = null
    if (btnApply) btnApply.onclick = null
    if (panel) panel.onkeydown = null
    if (inputEl) inputEl.onkeydown = null
  }

  try {
    const squares = tab.doc.squares || []

    for (const s of squares) {
      const label = getSquareDisplayName(s).toLowerCase()
      const isHeavy =
        s.additionalLabel === "heavy Weapons" ||
        tab.heavyLabelSet?.has(label)

      if (!isHeavy) continue
      if (typeof s.heavyLetter === "string" && s.heavyLetter.trim()) continue

      await ensureTabImageLoaded(tab)

      tab.selected = { kind: "square", id: s.id }
      renderCanvas()
      renderShipyardProps(tab.selected)
      showPanel()

      if (labelEl) {
        labelEl.textContent =
          `Box ${s.id}: ${getSquareDisplayName(s) || "Heavy Weapon"}`
      }

      if (inputEl) {
        inputEl.value = s.heavyLetter || ""
        inputEl.focus()
        inputEl.select()
      }

      const designation = await new Promise(resolve => {
        const localCleanup = () => {
          if (btnCancel) btnCancel.onclick = null
          if (btnApply) btnApply.onclick = null
          if (panel) panel.onkeydown = null
          if (inputEl) inputEl.onkeydown = null
        }

        btnCancel && (btnCancel.onclick = () => {
          localCleanup()
          resolve(null)
        })

        btnApply && (btnApply.onclick = () => {
          localCleanup()
          const val = (inputEl?.value || "").trim().toUpperCase()
          resolve(val)
        })

        const onKey = e => {
          if (e.key === "Escape") {
            localCleanup()
            resolve(null)
          }
          if (e.key === "Enter") {
            localCleanup()
            const val = (inputEl?.value || "").trim().toUpperCase()
            resolve(val)
          }
        }

        panel && (panel.onkeydown = onKey)
        inputEl && (inputEl.onkeydown = onKey)
      })

      if (designation === null) {
        cleanup()
        return false
      }

      if (!designation) {
        cleanup()
        alert("Please enter a letter designation for heavy weapons.")
        return false
      }

      if (!/^[A-Z0-9]+$/.test(designation)) {
        cleanup()
        alert("Heavy weapon designation must contain only letters (A-Z) or numbers (0-9).")
        return false
      }

      s.heavyLetter = designation
      await persistHeavyLetters()

      renderShipyardProps({ kind: "square", id: s.id })
    }

    const heavySquares = squares.filter(s => {
      const name = getSquareDisplayName(s).toLowerCase()
      return (
        s.additionalLabel === "heavy Weapons" ||
        tab.heavyLabelSet?.has(name)
      )
    })

    const missing = heavySquares.filter(
      s => !(typeof s.heavyLetter === "string" && s.heavyLetter.trim())
    )

    if (missing.length > 0) {
      const counters = new Map()

      for (const s of missing) {
        const key = getSquareDisplayName(s).toLowerCase() || "heavy"
        const idx = (counters.get(key) || 0) + 1
        counters.set(key, idx)

        const base = String.fromCharCode(65 + ((idx - 1) % 26))
        const repeat = Math.floor((idx - 1) / 26)
        s.heavyLetter = repeat > 0 ? base.repeat(repeat + 1) : base
      }

      await persistHeavyLetters()
    }

    cleanup()
    renderShipyardProps(tab.selected || null)
    return true

  } catch (error) {
    console.error("Error in ensureHeavyDesignations:", error)
    cleanup()
    return false
  }
}

function finalizeHeavyLetters(tab) {
  if (!tab || !tab.doc || !Array.isArray(tab.doc.squares)) return

  // If heavy list is missing, try to hydrate from Data.json
  if (!tab.heavyLabelSet || tab.heavyLabelSet.size === 0) {
    const heavyOpts = window.api?.readHeavyList ? window.api.readHeavyList() : null
    if (heavyOpts && typeof heavyOpts.then === "function") {
      // best-effort sync-ish fallback; ignore if it fails
      heavyOpts.then((res) => {
        if (res && res.ok && Array.isArray(res.options)) {
          tab.heavyLabelSet = new Set(
            res.options
              .map(x => normalizeLabel(x).clean.toLowerCase())
              .filter(Boolean)
          )
        }
      })
    }
  }

  const counters = new Map()
  for (const s of tab.doc.squares) {
    const name = getSquareDisplayName(s).toLowerCase()
    const isHeavy =
      s.additionalLabel === "heavy Weapons" ||
      tab.heavyLabelSet?.has(name)

    if (!isHeavy) {
      delete s.heavyLetter
      continue
    }

    if (!s.additionalLabel) s.additionalLabel = "heavy Weapons"
    if (typeof s.heavyLetter === "string" && s.heavyLetter.trim()) continue

    const idx = (counters.get(name || "heavy") || 0) + 1
    counters.set(name || "heavy", idx)
    const base = String.fromCharCode(65 + ((idx - 1) % 26))
    const repeat = Math.floor((idx - 1) / 26)
    s.heavyLetter = repeat > 0 ? base.repeat(repeat + 1) : base
  }

  // Remove heavy boxes from groups, then prune empty groups
  const heavyIds = new Set(
    tab.doc.squares
      .filter(s => s.additionalLabel === "heavy Weapons" || tab.heavyLabelSet?.has(getSquareDisplayName(s).toLowerCase()))
      .map(s => s.id)
  )

  for (const s of tab.doc.squares) {
    if (heavyIds.has(s.id)) s.groupId = null
  }

  if (Array.isArray(tab.doc.groups)) {
    tab.doc.groups = tab.doc.groups.filter(g => {
      return tab.doc.squares.some(s => s.groupId === g.id)
    })
  }

  // ensure plain objects (no proxies) before stringify
  tab.doc.squares = tab.doc.squares.map(s => JSON.parse(JSON.stringify(s)))
}

// Backfill older JSON files that kept the caret in the label/name.
function fixLegacyCaretLabels(doc) {
  if (!doc) return

  if (Array.isArray(doc.groups)) {
    for (const g of doc.groups) {
      if (!g) continue
      const rawLabel = typeof g.label === "string" ? g.label : ""
      const rawName = typeof g.name === "string" ? g.name : ""
      const source = rawLabel.trim() ? rawLabel : rawName
      if (!source) continue
      const norm = normalizeLabel(source)
      const clean = norm.clean
      if (!clean) continue
      if (norm.hasCaret || rawLabel !== clean || rawName !== clean) {
        g.label = clean
        g.name = clean
      }
    }
  }

  if (!Array.isArray(doc.squares)) return
  for (const s of doc.squares) {
    if (!s) continue
    const rawLabel = typeof s.label === "string" ? s.label : ""
    const rawName = typeof s.name === "string" ? s.name : ""
    const source = rawLabel.trim() ? rawLabel : rawName
    if (!source) continue
    const norm = normalizeLabel(source)
    const clean = norm.clean
    if (!clean) continue
    if (norm.hasCaret || rawLabel !== clean || rawName !== clean) {
      s.label = clean
      s.name = clean
    }
  }
}

// Apply Data.json-based heavy labeling to existing squares.
function applyHeavyLabelsFromFile(tab) {
  if (!tab || !tab.doc || !tab.heavyLabelSet) return
  const squares = tab.doc.squares || []
  for (const s of squares) {
    const raw = typeof s.label === "string" && s.label.trim() ? s.label : (typeof s.name === "string" ? s.name : "")
    const clean = normalizeLabel(raw).clean
    if (!clean) continue
    if (tab.heavyLabelSet.has(clean.toLowerCase())) {
      s.label = clean
      s.name = clean
      s.additionalLabel = "heavy Weapons"
    }
  }
}

function syncGroupLabelsToSquares(tab) {
  if (!tab || !tab.doc) return
  const groups = tab.doc.groups || []
  const squares = tab.doc.squares || []
  for (const s of squares) {
    if (s.groupId === null || s.groupId === undefined) continue
    const g = groups.find(x => x.id === s.groupId)
    if (!g) continue
    const glabel = getGroupDisplayName(g)
    if (!glabel) continue
    applySquareLabel(s, glabel, tab)
    s._inheritedFromGroupLabel = true
  }
}

function applyLoadedDatasetToTab(tab, loaded, doc) {
  if (tab.doc?.squares) {
    cleanupSquareTimers(tab.doc.squares)
  }
  tab.doc = doc
  tab.jsonPath = loaded.jsonPath
  tab.imageDataUrl = loaded.imageDataUrl
  tab.imageObj = null
  tab.selected = null
  fixLegacyCaretLabels(tab.doc)
  syncGroupLabelsToSquares(tab)
}

function parsePosTextToBbox(posText) {
  const parts = String(posText || "")
    .split(",")
    .map((part) => Number(part.trim()))
  if (parts.length !== 4 || parts.some((part) => !Number.isFinite(part))) return null
  let [x1, y1, x2, y2] = parts
  if (x2 < x1) [x1, x2] = [x2, x1]
  if (y2 < y1) [y1, y2] = [y2, y1]
  return { x1, y1, x2, y2 }
}

function pointsFromBbox(bbox) {
  return [
    [bbox.x1, bbox.y1],
    [bbox.x2, bbox.y1],
    [bbox.x2, bbox.y2],
    [bbox.x1, bbox.y2]
  ]
}

function mergeBbox(target, bbox) {
  if (!bbox) return target
  if (!target) return { ...bbox }
  target.x1 = Math.min(target.x1, bbox.x1)
  target.y1 = Math.min(target.y1, bbox.y1)
  target.x2 = Math.max(target.x2, bbox.x2)
  target.y2 = Math.max(target.y2, bbox.y2)
  return target
}

function formatJsonEditGroupName(rawKey) {
  const text = String(rawKey || "").trim()
  if (!text) return "Unknown"
  return text.replace(/[_-]+/g, " ")
}

function buildJsonEditDocFromSuperluminal(rawDoc) {
  const ssd = rawDoc?.ssd
  if (!ssd || typeof ssd !== "object") {
    return { ok: false, error: "JSON is missing both Veil squares and Superluminal ssd data." }
  }

  const squares = []
  const groups = []
  let nextSquareId = 0
  let nextGroupId = 1

  for (const [ssdKey, value] of Object.entries(ssd)) {
    if (!Array.isArray(value) || value.length === 0) continue

    const groupLabel = formatJsonEditGroupName(ssdKey)
    const group = {
      id: nextGroupId++,
      name: groupLabel,
      label: groupLabel,
      count: 0,
      bbox: null,
      center: null,
      __jsonEdit: {
        format: "superluminal",
        ssdKey
      }
    }

    for (let i = 0; i < value.length; i++) {
      const entry = value[i]
      const bbox = parsePosTextToBbox(entry?.pos)
      if (!bbox) continue

      const rawLabelText = String(entry?.type || entry?.name || entry?.label || "").trim()
      const labelText = normalizeLabel(rawLabelText).clean || groupLabel
      const square = {
        id: nextSquareId++,
        groupId: group.id,
        label: labelText,
        name: labelText,
        points: pointsFromBbox(bbox),
        bbox,
        center: {
          x: (bbox.x1 + bbox.x2) / 2,
          y: (bbox.y1 + bbox.y2) / 2
        },
        sideLengthPx: Math.max(1, Math.min(Math.abs(bbox.x2 - bbox.x1), Math.abs(bbox.y2 - bbox.y1))),
        __jsonEdit: {
          format: "superluminal",
          ssdKey,
          groupLabel,
          entryIndex: i,
          pos: String(entry?.pos || "").trim(),
          entry: JSON.parse(JSON.stringify(entry || {}))
        }
      }

      squares.push(square)
      group.count += 1
      group.bbox = mergeBbox(group.bbox, bbox)
    }

    if (group.count > 0 && group.bbox) {
      group.center = {
        x: (group.bbox.x1 + group.bbox.x2) / 2,
        y: (group.bbox.y1 + group.bbox.y2) / 2
      }
      groups.push(group)
    }
  }

  return {
    ok: true,
    doc: {
      squares,
      groups
    },
    format: "superluminal"
  }
}

function buildJsonEditView(rawDoc) {
  const hasVeilSquares = Array.isArray(rawDoc?.squares) && rawDoc.squares.length > 0
  if (hasVeilSquares) {
    return {
      ok: true,
      doc: rawDoc,
      format: "veil"
    }
  }
  return buildJsonEditDocFromSuperluminal(rawDoc)
}

function getJsonEditTabTitle(rawDoc, fallback) {
  const shipName = String(rawDoc?.shipName || "").trim()
  if (shipName) return `${shipName} JSON`
  const empire = String(rawDoc?.shipData?.empire || "").trim()
  const type = String(rawDoc?.shipData?.type || "").trim()
  const joined = [empire, type].filter(Boolean).join(" ")
  if (joined) return `${joined} JSON`
  const base = String(fallback || "").trim()
  return base ? `${base} JSON` : "Edit JSON"
}

async function runJsonEditor() {
  setAddSquareMode(false)
  state.jsonEditAddBoxSsdKey = ""
  state.jsonEditAddBoxSpec = null
  setAddGroupMode(false)
  state.refitMode = false
  const refitBtn = document.getElementById("btnRefitRemove")
  if (refitBtn) refitBtn.classList.remove("refitActive")
  document.querySelectorAll(".modalOverlay").forEach(el => el.remove())
  const canvasSafe = document.getElementById("canvas")
  if (canvasSafe) canvasSafe.style.pointerEvents = "auto"

  setDamageMode(false)
  setWeaponMode(false)
  setSystemMode(false)
  hideShipyardPanel()

  const picker = window.api.pickSuperluminalShip || window.api.pickSuperluminalInput
  if (!picker) {
    alert("Folder picker is not available.")
    return
  }

  const picked = await picker()
  if (!picked || !picked.ok) {
    alert(picked?.error || "Folder selection cancelled.")
    return
  }

  let rawDoc = null
  try {
    rawDoc = JSON.parse(picked.jsonText || "")
  } catch (e) {
    alert("JSON parse failed: " + e.message)
    return
  }

  const normalized = buildJsonEditView(rawDoc)
  if (!normalized.ok || !normalized.doc) {
    alert(normalized.error || "Unsupported JSON format for box editing.")
    return
  }

  closeAllTabsForNewTask()
  const tab = ensureEmptyActiveTab()
  if (!tab) return

  tab.uiState = "jsonEdit"
  tab.doc = normalized.doc
  tab.jsonPath = picked.jsonPath || null
  tab.jsonEditSourceDoc = rawDoc
  tab.imageDataUrl = picked.imageDataUrl
  tab.imageObj = null
  tab.selected = null
  tab.jsonEditFormat = normalized.format
  tab.title = getJsonEditTabTitle(rawDoc, picked.inputBase || picked.jsonFile || "Edit JSON")
  if (normalized.format === "superluminal") {
    const [typeOptions, boxEntries, arcOptions] = await Promise.all([
      readJsonEditBoxTypeOptions(),
      readJsonEditBoxEntryOptions(),
      readArcOptions()
    ])
    tab.jsonEditTypeOptions = Array.isArray(typeOptions) ? typeOptions : []
    tab.jsonEditBoxEntries = Array.isArray(boxEntries) ? boxEntries : []
    tab.jsonEditArcOptions = Array.isArray(arcOptions) ? arcOptions : []
  }

  await ensureTabImageLoaded(tab)
  renderTabs()
  renderCanvas()
  renderCurrentProps()
}

async function saveJsonEditDoc() {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "jsonEdit") {
    alert("No active JSON edit session.")
    return
  }
  if (!tab.jsonPath) {
    alert("Cannot save: JSON path is missing.")
    return
  }
  const sourceDoc =
    tab.jsonEditFormat === "superluminal"
      ? tab.jsonEditSourceDoc
      : tab.doc
  if (!sourceDoc || typeof sourceDoc !== "object") {
    alert("Cannot save: JSON data is missing.")
    return
  }
  const jsonText = JSON.stringify(sourceDoc, null, 2)
  const saveOptions =
    tab.jsonEditFormat === "superluminal"
      ? { autoUpdateConvertFields: true }
      : undefined
  const res = await window.api.saveJson(tab.jsonPath, jsonText, saveOptions)
  if (res?.ok) {
    if (tab.jsonEditFormat === "superluminal" && typeof res.jsonText === "string") {
      try {
        tab.jsonEditSourceDoc = JSON.parse(res.jsonText)
      } catch {}
    }
    showTempMessage("JSON saved.")
    return
  }
  alert(res?.error || "Save failed.")
}

function getAverageSquareSide(tab) {
  const avgParam = Number(tab?.doc?.params?.avgSquareSidePx)
  if (Number.isFinite(avgParam) && avgParam > 0) return avgParam
  const squares = tab?.doc?.squares || []
  const sizes = squares.map(s => Number(s?.sideLengthPx)).filter(n => Number.isFinite(n) && n > 0)
  if (sizes.length === 0) return 10
  const sum = sizes.reduce((a, b) => a + b, 0)
  return sum / sizes.length
}

function addSquareAtPoint(tab, ix, iy) {
  if (!tab || !tab.doc) return
  const squares = tab.doc.squares || []
  const nextId =
    squares.length === 0
      ? 0
      : Math.max(...squares.map(s => Number(s?.id ?? -1))) + 1

  const side = Math.max(4, Math.round(getAverageSquareSide(tab)))
  const half = side / 2
  const x1 = Math.round(ix - half)
  const y1 = Math.round(iy - half)
  const x2 = Math.round(ix + half)
  const y2 = Math.round(iy + half)

  const square = {
    id: nextId,
    groupId: null,
    points: [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2]
    ],
    bbox: { x1, y1, x2, y2 },
    center: { x: (x1 + x2) / 2, y: (y1 + y2) / 2 },
    sideLengthPx: side
  }

  squares.push(square)
  tab.doc.squares = squares
  tab.selected = { kind: "square", id: nextId }
}

function getSelectedJsonEditSsdKey(tab) {
  if (!tab || tab.uiState !== "jsonEdit") return ""
  const sel = tab.selected
  const groups = Array.isArray(tab.doc?.groups) ? tab.doc.groups : []
  const squares = Array.isArray(tab.doc?.squares) ? tab.doc.squares : []

  const fromSquare = (square) => {
    const key = String(square?.__jsonEdit?.ssdKey || "").trim()
    if (key) return key
    const gid = Number(square?.groupId)
    if (!Number.isFinite(gid)) return ""
    const group = groups.find((g) => Number(g?.id) === gid)
    return String(group?.__jsonEdit?.ssdKey || "").trim()
  }

  if (sel?.kind === "square") {
    const square = squares.find((s) => Number(s?.id) === Number(sel.id))
    const key = fromSquare(square)
    if (key) return key
  }

  if (sel?.kind === "group") {
    const group = groups.find((g) => Number(g?.id) === Number(sel.id))
    const key = String(group?.__jsonEdit?.ssdKey || "").trim()
    if (key) return key
  }

  if (groups.length === 1) {
    const key = String(groups[0]?.__jsonEdit?.ssdKey || "").trim()
    if (key) return key
  }

  return ""
}

function getJsonEditArraySsdKeys(tab) {
  const sourceDoc = tab?.jsonEditSourceDoc
  const ssd = sourceDoc?.ssd
  if (!ssd || typeof ssd !== "object" || Array.isArray(ssd)) return []
  const keys = []
  for (const [key, value] of Object.entries(ssd)) {
    if (Array.isArray(value)) keys.push(String(key || "").trim())
  }
  return keys.filter(Boolean)
}

function compactJsonEditKey(raw) {
  return String(raw || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "")
}

function findJsonEditExistingSsdKey(tab, preferredKey) {
  const target = compactJsonEditKey(preferredKey)
  if (!target) return ""
  const keys = getJsonEditArraySsdKeys(tab)
  return keys.find(key => compactJsonEditKey(key) === target) || ""
}

function isJsonEditWeaponSsdKey(ssdKey) {
  const key = String(ssdKey || "").trim().toLowerCase()
  return key === "heavy" || key === "phaser" || key === "drone"
}

function getJsonEditEntryNames(entry) {
  const names = []
  const add = (value) => {
    const text = String(value || "").trim()
    if (text) names.push(text)
  }
  add(entry?.name)
  add(entry?.label)
  if (Array.isArray(entry?.otherNames)) {
    for (const name of entry.otherNames) add(name)
  }
  return names
}

function findJsonEditBoxEntry(tab, rawName) {
  const target = compactJsonEditKey(rawName)
  if (!target) return null
  const entries = Array.isArray(tab?.jsonEditBoxEntries) ? tab.jsonEditBoxEntries : []
  return entries.find(entry => getJsonEditEntryNames(entry).some(name => compactJsonEditKey(name) === target)) || null
}

function getJsonEditBoxEntryTypes(entry) {
  return Array.isArray(entry?.types)
    ? entry.types.map(type => String(type || "").trim().toLowerCase()).filter(Boolean)
    : []
}

function deriveJsonEditSsdKeyForBox(tab, boxLabel, boxEntry) {
  const label = String(boxEntry?.name || boxEntry?.label || boxLabel || "").trim()
  const types = getJsonEditBoxEntryTypes(boxEntry)
  const hasType = (typeName) => types.includes(String(typeName || "").trim().toLowerCase())

  if (hasType("Drone")) {
    return findJsonEditExistingSsdKey(tab, "drone") || "Drone"
  }
  if (hasType("Heavy")) {
    return findJsonEditExistingSsdKey(tab, "heavy") || "heavy"
  }
  if (hasType("Phaser")) {
    return findJsonEditExistingSsdKey(tab, "phaser") || "phaser"
  }

  const existing = findJsonEditExistingSsdKey(tab, label)
  return existing || label
}

function getJsonEditSelectedBoxLabel(tab) {
  if (!tab || tab.uiState !== "jsonEdit") return ""
  const sel = tab.selected
  const squares = Array.isArray(tab.doc?.squares) ? tab.doc.squares : []
  const groups = Array.isArray(tab.doc?.groups) ? tab.doc.groups : []

  if (sel?.kind === "square") {
    const square = squares.find((s) => Number(s?.id) === Number(sel.id))
    const meta = square?.__jsonEdit && typeof square.__jsonEdit === "object" ? square.__jsonEdit : null
    const entry = getJsonEditSourceEntry(tab, meta)
    const typeText = String(entry?.type || meta?.entry?.type || "").trim()
    if (typeText) return typeText
    const labelText = getSquareDisplayName(square)
    if (labelText) return labelText
    const key = String(meta?.ssdKey || "").trim()
    if (key) return formatJsonEditGroupName(key)
  }

  if (sel?.kind === "group") {
    const group = groups.find((g) => Number(g?.id) === Number(sel.id))
    const groupLabel = getGroupDisplayName(group)
    if (groupLabel) return groupLabel
  }

  return ""
}

function resolveJsonEditAddBoxSpecFromText(tab, rawText) {
  const text = String(rawText || "").trim()
  if (!text) return null

  const rawSectionMatch = text.match(/^section\s*:\s*(.+)$/i)
  if (rawSectionMatch) {
    const ssdKey = String(rawSectionMatch[1] || "").trim()
    if (!ssdKey) return null
    return {
      label: formatJsonEditGroupName(ssdKey),
      ssdKey,
      rawSection: true,
      weapon: isJsonEditWeaponSsdKey(ssdKey)
    }
  }

  const boxEntry = findJsonEditBoxEntry(tab, text)
  const label = String(boxEntry?.name || boxEntry?.label || text).trim()
  const ssdKey = deriveJsonEditSsdKeyForBox(tab, label, boxEntry)
  const types = getJsonEditBoxEntryTypes(boxEntry)
  return {
    label,
    ssdKey,
    boxEntry,
    types,
    weapon: isJsonEditWeaponSsdKey(ssdKey)
  }
}

async function resolveJsonEditAddBoxSpec(tab, options = {}) {
  if (!tab || tab.uiState !== "jsonEdit" || tab.jsonEditFormat !== "superluminal") return null
  const {
    allowPrompt = true,
    forcePrompt = false
  } = options

  if (state.jsonEditAddBoxSpec && !forcePrompt) return state.jsonEditAddBoxSpec
  if (!allowPrompt) return state.jsonEditAddBoxSpec || null

  if (!Array.isArray(tab.jsonEditBoxEntries)) {
    tab.jsonEditBoxEntries = await readJsonEditBoxEntryOptions()
  }

  const defaultText =
    getJsonEditSelectedBoxLabel(tab) ||
    String(state.jsonEditAddBoxSpec?.label || "").trim() ||
    String(state.jsonEditAddBoxSsdKey || "").trim() ||
    ""
  const entered = await promptJsonEditAddBoxText(defaultText, tab.jsonEditBoxEntries)
  if (entered === null) return null

  const spec = resolveJsonEditAddBoxSpecFromText(tab, entered)
  if (!spec || !spec.ssdKey) {
    showTempMessage("Box label/type is required.")
    return null
  }

  state.jsonEditAddBoxSpec = spec
  state.jsonEditAddBoxSsdKey = spec.ssdKey
  return spec
}

function refreshJsonEditGroupGeometry(tab, groupId) {
  if (!tab || !tab.doc) return
  const groups = Array.isArray(tab.doc.groups) ? tab.doc.groups : []
  const group = groups.find((g) => Number(g?.id) === Number(groupId))
  if (!group) return
  const members = (tab.doc.squares || []).filter((s) => Number(s?.groupId) === Number(groupId))
  group.count = members.length
  group.bbox = null
  group.center = null
  for (const square of members) {
    if (!square?.bbox || typeof square.bbox !== "object") continue
    group.bbox = mergeBbox(group.bbox, square.bbox)
  }
  if (group.bbox) {
    group.center = {
      x: (group.bbox.x1 + group.bbox.x2) / 2,
      y: (group.bbox.y1 + group.bbox.y2) / 2
    }
  }
}

async function addJsonEditSquareAtPoint(tab, ix, iy) {
  if (!tab || !tab.doc) return
  if (tab.jsonEditFormat !== "superluminal") {
    addSquareAtPoint(tab, ix, iy)
    return
  }

  const sourceDoc = tab.jsonEditSourceDoc
  if (!sourceDoc || typeof sourceDoc !== "object") {
    showTempMessage("Source JSON is unavailable.")
    return
  }
  if (!sourceDoc.ssd || typeof sourceDoc.ssd !== "object" || Array.isArray(sourceDoc.ssd)) {
    sourceDoc.ssd = {}
  }

  const spec = await resolveJsonEditAddBoxSpec(tab, { allowPrompt: true })
  if (!spec || !spec.ssdKey) {
    showTempMessage("Choose a box type first.")
    return
  }
  const ssdKey = spec.ssdKey

  let targetEntries = sourceDoc.ssd[ssdKey]
  if (!Array.isArray(targetEntries)) {
    targetEntries = []
    sourceDoc.ssd[ssdKey] = targetEntries
  }

  const side = Math.max(4, Math.round(getAverageSquareSide(tab)))
  const half = side / 2
  const x1 = Math.round(ix - half)
  const y1 = Math.round(iy - half)
  const x2 = Math.round(ix + half)
  const y2 = Math.round(iy + half)
  const posText = `${x1},${y1},${x2},${y2}`

  const nextEntry = { pos: posText }
  if (spec.weapon) {
    nextEntry.type = String(spec.label || "").trim()
    const key = String(ssdKey || "").trim().toLowerCase()
    if (key === "heavy" || key === "phaser") nextEntry.designation = ""
    nextEntry.arc = ""
  }
  const entryIndex = targetEntries.length
  targetEntries.push(nextEntry)

  ensureGroups(tab)
  let group = (tab.doc.groups || []).find((g) => String(g?.__jsonEdit?.ssdKey || "").trim() === ssdKey)
  if (!group) {
    const groupLabel = formatJsonEditGroupName(ssdKey)
    group = {
      id: nextGroupId(tab.doc.groups || []),
      name: groupLabel,
      label: groupLabel,
      count: 0,
      bbox: null,
      center: null,
      __jsonEdit: {
        format: "superluminal",
        ssdKey
      }
    }
    tab.doc.groups.push(group)
  }

  const squares = tab.doc.squares || []
  const nextId =
    squares.length === 0
      ? 0
      : Math.max(...squares.map((s) => Number(s?.id ?? -1))) + 1
  const labelText =
    String(spec.label || nextEntry?.type || nextEntry?.name || nextEntry?.label || group.label || group.name || formatJsonEditGroupName(ssdKey)).trim() ||
    formatJsonEditGroupName(ssdKey)

  const square = {
    id: nextId,
    groupId: group.id,
    label: labelText,
    name: labelText,
    points: [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2]
    ],
    bbox: { x1, y1, x2, y2 },
    center: { x: (x1 + x2) / 2, y: (y1 + y2) / 2 },
    sideLengthPx: side,
    __jsonEdit: {
      format: "superluminal",
      ssdKey,
      groupLabel: String(group.label || group.name || formatJsonEditGroupName(ssdKey)).trim(),
      entryIndex,
      pos: posText,
      entry: JSON.parse(JSON.stringify(nextEntry))
    }
  }

  squares.push(square)
  tab.doc.squares = squares
  refreshJsonEditGroupGeometry(tab, group.id)
  tab.selected = { kind: "square", id: nextId }
}

function ensureGroups(tab) {
  if (!tab.doc.groups || !Array.isArray(tab.doc.groups)) {
    tab.doc.groups = []
  }
}

function nextGroupId(groups) {
  const ids = groups.map(g => Number(g?.id)).filter(n => Number.isFinite(n))
  return ids.length === 0 ? 1 : Math.max(...ids) + 1
}

function getGroupById(groups, id) {
  return groups.find(g => Number(g?.id) === Number(id)) || null
}

function mergeGroups(tab, primaryId, secondaryId) {
  if (!tab || !tab.doc) return
  if (Number(primaryId) === Number(secondaryId)) return
  ensureGroups(tab)
  const groups = tab.doc.groups
  const primary = getGroupById(groups, primaryId)
  const secondary = getGroupById(groups, secondaryId)
  for (const s of tab.doc.squares || []) {
    if (Number(s.groupId) === Number(secondaryId)) {
      s.groupId = Number(primaryId)
    }
  }
  if (primary && secondary) {
    if ((!primary.name || !primary.label) && (secondary.name || secondary.label)) {
      primary.name = primary.name || secondary.name || ""
      primary.label = primary.label || secondary.label || ""
    }
  }
  tab.doc.groups = groups.filter(g => Number(g?.id) !== Number(secondaryId))
}

function addSquaresToGroup(tab, firstSquare, secondSquare) {
  if (!tab || !tab.doc || !firstSquare || !secondSquare) return
  ensureGroups(tab)
  const groups = tab.doc.groups
  const firstGroup = firstSquare.groupId
  const secondGroup = secondSquare.groupId

  if (firstGroup == null && secondGroup == null) {
    const newId = nextGroupId(groups)
    groups.push({ id: newId, name: "", label: "" })
    firstSquare.groupId = newId
    secondSquare.groupId = newId
    return
  }

  if (firstGroup != null && secondGroup == null) {
    firstSquare.groupId = Number(firstGroup)
    secondSquare.groupId = Number(firstGroup)
    if (!getGroupById(groups, firstGroup)) {
      groups.push({ id: Number(firstGroup), name: "", label: "" })
    }
    return
  }

  if (firstGroup == null && secondGroup != null) {
    firstSquare.groupId = Number(secondGroup)
    secondSquare.groupId = Number(secondGroup)
    if (!getGroupById(groups, secondGroup)) {
      groups.push({ id: Number(secondGroup), name: "", label: "" })
    }
    return
  }

  if (Number(firstGroup) === Number(secondGroup)) {
    return
  }

  mergeGroups(tab, Number(firstGroup), Number(secondGroup))
}

async function enterConvertMode(picked) {
  closeAllTabsForNewTask()
  const tab = ensureEmptyActiveTab()
  if (!tab) return

  tab.uiState = "convert"
  tab.doc = { squares: [], groups: [] }
  tab.imageDataUrl = picked.imageDataUrl
  tab.imageObj = null
  tab.selected = null
  tab.title = picked.inputBase || "Convert"

  state.convertContext = {
    inputFolder: picked.inputFolder,
    jsonFile: picked.jsonFile,
    imageFile: picked.imageFile,
    jsonPath: picked.jsonPath,
    imagePath: picked.imagePath
  }
  state.convertMenuHidden = false

  clearConvertFields()
  prefillConvertFieldsFromJson(picked?.jsonText)
  if (window.api?.shipyardReadExistingSuperluminal && picked?.inputFolder) {
    try {
      const existingRes = await window.api.shipyardReadExistingSuperluminal(picked.inputFolder)
      if (existingRes?.ok && existingRes.jsonText) {
        prefillConvertFieldsFromJson(existingRes.jsonText)
      }
    } catch {}
  }

  renderTabs()
  renderCanvas()
  renderCurrentProps()
}

async function enterAnalysisMode(picked, doc) {
  closeAllTabsForNewTask()
  const tab = ensureEmptyActiveTab()
  if (!tab) return

  tab.uiState = "analysis"
  tab.doc = doc
  tab.jsonPath = picked.jsonPath
  tab.imageDataUrl = picked.imageDataUrl
  tab.imageObj = null
  tab.selected = null

  const weaponDamage = await readWeaponDamageJson()
  const modeSelections = await collectWeaponModeSelections(doc, weaponDamage)
  if (modeSelections === null) {
    closeActiveTab()
    return
  }
  const externalLabelSet = await readExternalLabelSet()
  tab.analysisFilter = "all"
  tab.analysisRadius = ANALYSIS_HEX_RADIUS_DEFAULT
  tab.analysisModeSelections = modeSelections
  tab.analysis = buildWeaponAnalysis(doc, weaponDamage, modeSelections, {
    weaponFilter: tab.analysisFilter,
    radius: tab.analysisRadius
  })
  tab.analysisShipStats = computeShipAnalysisStats(doc, externalLabelSet)
  tab.analysisCompare = null
  tab.analysisCompareDoc = null
  tab.analysisCompareLabel = null
  tab.analysisCompareModeSelections = null
  tab.analysisCompareMode = "normal"
  tab.analysisCompareStats = null
  tab.analysisHighlight = null
  tab.analysisHexLabels = new Set()
  tab.analysisHover = null
  tab.analysisCompareHexLabels = null
  tab.analysisCompareHover = null
  tab.analysisHoverSide = null

  const shipData = doc?.shipData || {}
  const empire = String(shipData.empire || "").trim()
  const type = String(shipData.type || "").trim()
  const titleBase = empire && type ? `${empire} ${type}` : ""
  tab.title = titleBase || picked.inputBase || "Ship Analysis"

  await ensureTabImageLoaded(tab)
  renderTabs()
  renderCanvas()
  renderCurrentProps()
}

async function enterMovementAnalysisMode() {
  const currentTab = getActiveTab()
  const defaultTurnMode = normalizeMovementTurnModeLetter(currentTab?.doc?.shipData?.turnMode)
  const table = await readTurnMovementTable()
  if (!table) {
    throw new Error("Turn&Movement.json is not available.")
  }

  closeAllTabsForNewTask()
  const tab = ensureEmptyActiveTab()
  if (!tab) return

  tab.uiState = "movementAnalysis"
  tab.doc = { squares: [], groups: [] }
  tab.imageDataUrl = null
  tab.imageObj = null
  tab.selected = null
  tab.turnMovementTable = table
  tab.movementAnalysis = {
    turnModeLetter: defaultTurnMode || "D",
    speed: "",
    endingHeading: ""
  }
  tab.movementExpansionCache = null
  tab.movementEndpointAnalysisCache = null
  tab.movementSelectedEndpointKey = null
  tab.movementSelectedPath = null
  tab.title = "Movement Analysis"

  renderTabs()
  renderCanvas()
  renderCurrentProps()
}

function clearConvertFields() {
  const ids = [
    "convertBpv",
    "convertBreakdown",
    "convertShieldCost",
    "convertLifeSupport",
    "convertSizeClass",
    "convertTurnMode",
    "convertMovementCost",
    "convertMinCrew",
    "convertSpareShuttles",
    "convertDockingPoints",
    "convertExplosionStrength",
    "convertCommandRating"
  ]
  for (const id of ids) {
    const el = document.getElementById(id)
    if (el) el.value = ""
  }
}

function setConvertField(id, value) {
  const el = document.getElementById(id)
  if (!el) return
  if (value === undefined || value === null) {
    el.value = ""
  } else {
    el.value = String(value)
  }
}

function prefillConvertFieldsFromJson(jsonText) {
  if (!jsonText) return
  let doc = null
  try {
    doc = JSON.parse(jsonText)
  } catch {
    return
  }
  if (!doc || typeof doc !== "object") return

  const shipData = doc.shipData && typeof doc.shipData === "object" ? doc.shipData : null
  if (!shipData) return

  const shuttles =
    shipData.shuttles && typeof shipData.shuttles === "object" ? shipData.shuttles : null
  const hasKey = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key)

  if (hasKey(shipData, "bpv")) setConvertField("convertBpv", shipData.bpv)
  if (hasKey(shipData, "breakdown")) setConvertField("convertBreakdown", shipData.breakdown)
  if (hasKey(shipData, "shieldCost")) setConvertField("convertShieldCost", shipData.shieldCost)
  if (hasKey(shipData, "lifeSupport")) setConvertField("convertLifeSupport", shipData.lifeSupport)
  if (hasKey(shipData, "sizeClass")) setConvertField("convertSizeClass", shipData.sizeClass)
  if (hasKey(shipData, "turnMode")) setConvertField("convertTurnMode", shipData.turnMode)
  if (hasKey(shipData, "movementCost")) setConvertField("convertMovementCost", shipData.movementCost)
  if (hasKey(shipData, "minCrew")) setConvertField("convertMinCrew", shipData.minCrew)
  if (hasKey(shipData, "dockingPoints")) setConvertField("convertDockingPoints", shipData.dockingPoints)
  if (hasKey(shipData, "explosionStrength")) {
    setConvertField("convertExplosionStrength", shipData.explosionStrength)
  }
  if (hasKey(shipData, "commandRating")) {
    setConvertField("convertCommandRating", shipData.commandRating)
  }

  if (hasKey(shipData, "spareShuttles")) {
    setConvertField("convertSpareShuttles", shipData.spareShuttles)
  } else if (shuttles && hasKey(shuttles, "spareShuttles")) {
    setConvertField("convertSpareShuttles", shuttles.spareShuttles)
  }
}

function exitConvertMode(options = {}) {
  const { clearFields = false, resetTab = false } = options
  const tab = getActiveTab()
  if (tab && tab.uiState === "convert") {
    if (resetTab) {
      tab.uiState = "empty"
      tab.doc = null
      tab.imageDataUrl = null
      tab.imageObj = null
      tab.selected = null
      tab.title = "Untitled"
    } else {
      tab.uiState = null
    }
  }
  if (clearFields) clearConvertFields()
  state.convertContext = null
  state.convertMenuHidden = false
  renderCurrentProps()
  renderCanvas()
}

function collectConvertFields() {
  const getVal = (id) => String(document.getElementById(id)?.value || "").trim()
  return {
    bpv: getVal("convertBpv"),
    breakdown: getVal("convertBreakdown"),
    shieldCost: getVal("convertShieldCost"),
    lifeSupport: getVal("convertLifeSupport"),
    sizeClass: getVal("convertSizeClass"),
    turnMode: getVal("convertTurnMode"),
    movementCost: getVal("convertMovementCost"),
    minCrew: getVal("convertMinCrew"),
    spareShuttles: getVal("convertSpareShuttles"),
    dockingPoints: getVal("convertDockingPoints"),
    explosionStrength: getVal("convertExplosionStrength"),
    commandRating: getVal("convertCommandRating")
  }
}

function formatArcGroupLabel(group) {
  const key = String(group || "").trim().toLowerCase()
  if (key === "heavy") return "Torpedo/Heavy"
  if (key === "phaser") return "Phaser"
  if (key === "drone") return "Drone"
  return key ? key[0].toUpperCase() + key.slice(1) : "Weapon"
}

function formatRankValueGroupLabel(group) {
  const key = String(group || "").trim().toLowerCase()
  if (key === "damcon") return "Damage Control"
  if (key === "sensor") return "Sensor"
  if (key === "scanner") return "Scanner"
  return formatArcGroupLabel(group)
}

function buildArcPromptTitle(item) {
  const label = String(item?.label || "").trim()
  const groupLabel = formatArcGroupLabel(item?.group)
  const squareId = item?.squareId
  const base = label ? `${label} - ${groupLabel}` : groupLabel
  const idText =
    squareId === undefined || squareId === null || squareId === ""
      ? ""
      : ` (Box ${squareId})`
  return `Select arc(s) for ${base}${idText}`
}

function buildDesignationPromptTitle(item) {
  const label = String(item?.label || "").trim()
  const groupLabel = formatArcGroupLabel(item?.group)
  const squareId = item?.squareId
  const base = label ? `${label} - ${groupLabel}` : groupLabel
  const idText =
    squareId === undefined || squareId === null || squareId === ""
      ? ""
      : ` (Box ${squareId})`
  return `Enter designation for ${base}${idText}`
}

function buildRankValuePromptTitle(item) {
  const label = String(item?.label || "").trim()
  const groupLabel = label || formatRankValueGroupLabel(item?.group)
  const squareId = item?.squareId
  const idText =
    squareId === undefined || squareId === null || squareId === ""
      ? ""
      : ` (Box ${squareId})`
  return `Set rank/value for ${groupLabel}${idText}`
}

function getRankAssignmentKey(item) {
  return String(item?.key ?? item?.squareId ?? "").trim()
}

function getRankItemCenter(item) {
  const bbox = item?.bbox && typeof item.bbox === "object" ? item.bbox : null
  if (!bbox) return null
  const x1 = Number(bbox.x1)
  const y1 = Number(bbox.y1)
  const x2 = Number(bbox.x2)
  const y2 = Number(bbox.y2)
  if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) {
    return null
  }
  return {
    x: (x1 + x2) / 2,
    y: (y1 + y2) / 2
  }
}

function compareRankItemsForDisplay(a, b) {
  const centerA = getRankItemCenter(a)
  const centerB = getRankItemCenter(b)
  if (centerA && centerB) {
    if (Math.abs(centerA.y - centerB.y) > 0.001) return centerA.y - centerB.y
    if (Math.abs(centerA.x - centerB.x) > 0.001) return centerA.x - centerB.x
  }

  const squareA = Number(a?.squareId)
  const squareB = Number(b?.squareId)
  const hasSquareA = Number.isFinite(squareA)
  const hasSquareB = Number.isFinite(squareB)
  if (hasSquareA && hasSquareB && squareA !== squareB) return squareA - squareB
  if (hasSquareA && !hasSquareB) return -1
  if (!hasSquareA && hasSquareB) return 1

  const keyA = getRankAssignmentKey(a)
  const keyB = getRankAssignmentKey(b)
  return keyA.localeCompare(keyB)
}

function formatRankRangeOptionLabel(item, index) {
  const label = String(item?.label || "").trim()
  const groupLabel = formatRankValueGroupLabel(item?.group)
  const base = label || groupLabel || "Entry"
  const squareId = item?.squareId
  if (squareId !== undefined && squareId !== null && squareId !== "") {
    return `${base} (Box ${squareId})`
  }
  const key = getRankAssignmentKey(item)
  if (key) return `${base} (${key})`
  return `${base} ${index + 1}`
}

function orientRankOrder(items, firstKey, lastKey) {
  const firstIdx = items.findIndex((item) => getRankAssignmentKey(item) === firstKey)
  const lastIdx = items.findIndex((item) => getRankAssignmentKey(item) === lastKey)
  if (firstIdx !== -1 && lastIdx !== -1 && firstIdx > lastIdx) {
    return items.slice().reverse()
  }
  return items
}

function orderRankItemsByEndpoints(items, firstKey, lastKey) {
  const normalized = (Array.isArray(items) ? items : []).slice()
  if (normalized.length < 2) return normalized

  const byKey = new Map(normalized.map((item) => [getRankAssignmentKey(item), item]))
  const firstItem = byKey.get(firstKey) || normalized[0]
  const lastItem = byKey.get(lastKey) || normalized[normalized.length - 1]

  const firstCenter = getRankItemCenter(firstItem)
  const lastCenter = getRankItemCenter(lastItem)
  if (!firstCenter || !lastCenter) {
    const fallback = normalized.slice().sort(compareRankItemsForDisplay)
    return orientRankOrder(fallback, firstKey, lastKey)
  }

  const vx = lastCenter.x - firstCenter.x
  const vy = lastCenter.y - firstCenter.y
  if (Math.abs(vx) < 0.001 && Math.abs(vy) < 0.001) {
    const fallback = normalized.slice().sort(compareRankItemsForDisplay)
    return orientRankOrder(fallback, firstKey, lastKey)
  }

  const project = (item) => {
    const center = getRankItemCenter(item)
    if (!center) return null
    return (center.x - firstCenter.x) * vx + (center.y - firstCenter.y) * vy
  }

  const ordered = normalized.slice().sort((a, b) => {
    const projA = project(a)
    const projB = project(b)
    const hasProjA = projA !== null
    const hasProjB = projB !== null
    if (hasProjA && hasProjB) {
      if (Math.abs(projA - projB) > 0.001) return projA - projB
      return compareRankItemsForDisplay(a, b)
    }
    if (hasProjA && !hasProjB) return -1
    if (!hasProjA && hasProjB) return 1
    return compareRankItemsForDisplay(a, b)
  })

  return orientRankOrder(ordered, firstKey, lastKey)
}

function setArcHighlight(item) {
  setArcHighlights(item ? [{ item }] : [])
}

function setArcHighlights(items) {
  const normalized = []
  const list = Array.isArray(items) ? items : []
  for (const entry of list) {
    if (!entry) continue
    const source = entry?.item && typeof entry.item === "object" ? entry.item : entry
    const points = Array.isArray(source.points) ? source.points : null
    const bbox = source?.bbox && typeof source.bbox === "object" ? source.bbox : null
    if (!points && !bbox) continue

    const style = entry?.style && typeof entry.style === "object" ? entry.style : entry
    const next = { points, bbox }

    if (typeof style?.strokeStyle === "string" && style.strokeStyle.trim()) {
      next.strokeStyle = style.strokeStyle.trim()
    }
    if (Array.isArray(style?.lineDash)) next.lineDash = style.lineDash
    const lineWidth = Number(style?.lineWidth)
    if (Number.isFinite(lineWidth) && lineWidth > 0) next.lineWidth = lineWidth
    const alpha = Number(style?.alpha)
    if (Number.isFinite(alpha) && alpha >= 0 && alpha <= 1) next.alpha = alpha

    normalized.push(next)
  }

  if (normalized.length === 0) {
    state.arcHighlight = null
  } else if (normalized.length === 1) {
    state.arcHighlight = normalized[0]
  } else {
    state.arcHighlight = { entries: normalized }
  }
}

function clearArcHighlight() {
  state.arcHighlight = null
}

async function readArcOptions() {
  if (!window.api || typeof window.api.shipyardReadSectionEntries !== "function") {
    return null
  }
  const res = await window.api.shipyardReadSectionEntries("Arcs")
  if (!res || !res.ok || !Array.isArray(res.entries)) return null
  return res.entries.map(entry => String(entry || "").trim()).filter(Boolean)
}

async function readJsonEditBoxEntryOptions() {
  if (Array.isArray(jsonEditBoxEntryOptionsCache)) return jsonEditBoxEntryOptionsCache

  try {
    const res = window.api && typeof window.api.shipyardReadSectionEntryObjects === "function"
      ? await window.api.shipyardReadSectionEntryObjects("Boxes")
      : null
    if (res && res.ok && Array.isArray(res.entries)) {
      jsonEditBoxEntryOptionsCache = res.entries
        .map((entry) => {
          const name = String(entry?.name || entry?.label || "").trim()
          const label = String(entry?.label || entry?.name || "").trim()
          const types = Array.isArray(entry?.types)
            ? entry.types.map(type => String(type || "").trim()).filter(Boolean)
            : []
          const otherNames = Array.isArray(entry?.otherNames)
            ? entry.otherNames.map(name => String(name || "").trim()).filter(Boolean)
            : []
          return { name, label, types, otherNames }
        })
        .filter(entry => entry.name.length > 0)
      return jsonEditBoxEntryOptionsCache
    }
  } catch {
    // Fall back to plain label options below.
  }

  const names = await readJsonEditBoxTypeOptions()
  jsonEditBoxEntryOptionsCache = (Array.isArray(names) ? names : [])
    .map(name => ({ name: String(name || "").trim(), label: String(name || "").trim(), types: [], otherNames: [] }))
    .filter(entry => entry.name.length > 0)
  return jsonEditBoxEntryOptionsCache
}

async function readJsonEditBoxTypeOptions() {
  if (Array.isArray(jsonEditBoxTypeOptionsCache)) return jsonEditBoxTypeOptionsCache

  try {
    const res = window.api && typeof window.api.shipyardReadSectionEntries === "function"
      ? await window.api.shipyardReadSectionEntries("Boxes")
      : null
    if (res && res.ok && Array.isArray(res.entries)) {
      jsonEditBoxTypeOptionsCache = res.entries
        .map(option => String(option || "").trim())
        .filter(Boolean)
      return jsonEditBoxTypeOptionsCache
    }
  } catch {
    // Fall back to the legacy label reader below.
  }

  if (!window.api || typeof window.api.shipyardReadBoxes !== "function") {
    jsonEditBoxTypeOptionsCache = []
    return jsonEditBoxTypeOptionsCache
  }

  try {
    const res = await window.api.shipyardReadBoxes()
    if (!res || !res.ok || !Array.isArray(res.options)) {
      jsonEditBoxTypeOptionsCache = []
      return jsonEditBoxTypeOptionsCache
    }
    jsonEditBoxTypeOptionsCache = res.options
      .map(option => String(option || "").trim())
      .filter(Boolean)
    return jsonEditBoxTypeOptionsCache
  } catch {
    jsonEditBoxTypeOptionsCache = []
    return jsonEditBoxTypeOptionsCache
  }
}

async function readWeaponDamageJson() {
  if (weaponDamageCache) return weaponDamageCache
  if (!window.api || typeof window.api.shipyardReadWeaponDamage !== "function") {
    return null
  }
  const res = await window.api.shipyardReadWeaponDamage()
  if (!res || !res.ok || !res.data) return null
  weaponDamageCache = res.data
  return weaponDamageCache
}

async function readTurnMovementTable() {
  if (turnMovementTableCache) return turnMovementTableCache
  if (!window.api || typeof window.api.shipyardReadTurnMovement !== "function") {
    return null
  }
  const res = await window.api.shipyardReadTurnMovement()
  if (!res || !res.ok || !res.data) return null
  turnMovementTableCache = res.data
  return turnMovementTableCache
}

function normalizeMovementTurnModeLetter(raw) {
  const value = String(raw || "").trim().toUpperCase()
  return /^[A-F]{1,2}$/.test(value) ? value : ""
}

function normalizeMovementHeadingIndex(raw) {
  if (raw === "" || raw === null || raw === undefined) return null
  const n = Math.floor(Number(raw))
  if (!Number.isFinite(n)) return null
  if (n < 0 || n > 5) return null
  return n
}

function getMovementEndingHeadingOptions() {
  const opts = [{ value: "", label: "Any Ending Direction" }]
  for (let i = 0; i < 6; i++) {
    opts.push({
      value: String(i),
      label: MOVEMENT_HEADING_OPTION_LABELS[i] || `Heading ${i}`
    })
  }
  return opts
}

function getMovementHeadingLabel(heading) {
  const idx = normalizeMovementHeadingIndex(heading)
  if (idx === null) return "Any"
  return MOVEMENT_HEADING_OPTION_LABELS[idx] || `Heading ${idx}`
}

function getTurnModeLetterOptions(table) {
  const fromTable = Array.isArray(table?.turnModeLetters)
    ? table.turnModeLetters
    : []
  const normalized = fromTable
    .map(item => normalizeMovementTurnModeLetter(item))
    .filter(Boolean)
  if (normalized.length > 0) return normalized

  const fromCategories = Array.isArray(table?.categories)
    ? table.categories
        .map(cat => normalizeMovementTurnModeLetter(cat?.id || cat?.label || ""))
        .filter(Boolean)
    : []
  if (fromCategories.length > 0) return fromCategories

  return ["AA", "A", "B", "C", "D", "E", "F"]
}

function ensureMovementAnalysisState(tab) {
  if (!tab) return null
  if (!tab.movementAnalysis || typeof tab.movementAnalysis !== "object") {
    tab.movementAnalysis = { turnModeLetter: "D", speed: "", endingHeading: "" }
  }
  const stateObj = tab.movementAnalysis
  stateObj.turnModeLetter = normalizeMovementTurnModeLetter(stateObj.turnModeLetter) || "D"
  if (stateObj.speed === undefined || stateObj.speed === null) stateObj.speed = ""
  else stateObj.speed = String(stateObj.speed)
  const speedNum = Number(String(stateObj.speed).trim())
  if (String(stateObj.speed).trim() !== "" && Number.isFinite(speedNum) && speedNum > MOVEMENT_SPEED_MAX) {
    stateObj.speed = String(MOVEMENT_SPEED_MAX)
  }
  const normalizedEndingHeading = normalizeMovementHeadingIndex(stateObj.endingHeading)
  stateObj.endingHeading = normalizedEndingHeading === null ? "" : String(normalizedEndingHeading)
  return stateObj
}

function isSpeedInTurnMovementCell(cell, speed) {
  if (!cell || typeof cell !== "object") return false
  const min = Number(cell.min)
  const max = Number(cell.max)
  const openEnded = cell.openEnded === true
  if (!Number.isFinite(min) || !Number.isFinite(speed)) return false
  if (openEnded) return speed >= min
  if (!Number.isFinite(max)) return false
  return speed >= min && speed <= max
}

function findMovementTurnMatch(table, turnModeLetter, speed) {
  const letter = normalizeMovementTurnModeLetter(turnModeLetter)
  if (!letter || !Number.isFinite(speed)) return null
  const rows = Array.isArray(table?.rows) ? table.rows : []
  for (const row of rows) {
    const values = row && typeof row.values === "object" ? row.values : null
    const cell = values ? values[letter] : null
    if (!isSpeedInTurnMovementCell(cell, speed)) continue
    return { row, cell, letter }
  }
  return null
}

function setMovementAnalysisInput(field, value) {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "movementAnalysis") return
  const movement = ensureMovementAnalysisState(tab)
  if (!movement) return

  let affectsExpansion = false
  if (field === "turnModeLetter") {
    movement.turnModeLetter = normalizeMovementTurnModeLetter(value) || movement.turnModeLetter || "D"
    affectsExpansion = true
  } else if (field === "speed") {
    const nextRaw = value === undefined || value === null ? "" : String(value)
    const nextNum = Number(String(nextRaw).trim())
    if (String(nextRaw).trim() !== "" && Number.isFinite(nextNum) && nextNum > MOVEMENT_SPEED_MAX) {
      movement.speed = String(MOVEMENT_SPEED_MAX)
    } else {
      movement.speed = nextRaw
    }
    affectsExpansion = true
  } else if (field === "endingHeading") {
    const heading = normalizeMovementHeadingIndex(value)
    movement.endingHeading = heading === null ? "" : String(heading)
  }

  if (affectsExpansion) tab.movementExpansionCache = null
  tab.movementEndpointAnalysisCache = null
  tab.movementSelectedEndpointKey = null
  tab.movementSelectedPath = null
  renderMovementAnalysisPanel(tab)
  renderCanvas()
}

function countMovementHeadingBits(mask) {
  let value = Number(mask) || 0
  let count = 0
  while (value) {
    count += value & 1
    value >>= 1
  }
  return count
}

function clampMovementTurnProgress(value, requiredStraight) {
  const raw = Math.floor(Number(value) || 0)
  const cap = Math.max(0, Math.floor(Number(requiredStraight) || 0))
  if (raw < 0) return 0
  if (raw > cap) return cap
  return raw
}

function clampMovementSideslipProgress(value) {
  const raw = Math.floor(Number(value) || 0)
  if (raw < 0) return 0
  if (raw > MOVEMENT_SIDESLIP_REQUIRED_STRAIGHT) return MOVEMENT_SIDESLIP_REQUIRED_STRAIGHT
  return raw
}

function buildMovementStepTransitions(stateItem, requiredStraight) {
  if (!stateItem || typeof stateItem !== "object") return []
  const headingRaw = Math.floor(Number(stateItem.heading))
  if (!Number.isFinite(headingRaw)) return []
  const heading = ((headingRaw % 6) + 6) % 6
  const turnProgress = clampMovementTurnProgress(stateItem.turnProgress, requiredStraight)
  const slipProgress = clampMovementSideslipProgress(stateItem.slipProgress)
  const canTurn = turnProgress >= requiredStraight
  const canSideslip = slipProgress >= MOVEMENT_SIDESLIP_REQUIRED_STRAIGHT
  const q0 = Number(stateItem.q) || 0
  const r0 = Number(stateItem.r) || 0
  const nextStates = []

  const pushMove = (moveHeadingRaw, nextHeadingRaw, moveKind) => {
    const moveHeading = ((Math.floor(moveHeadingRaw) % 6) + 6) % 6
    const nextHeading = ((Math.floor(nextHeadingRaw) % 6) + 6) % 6
    const dir = MOVEMENT_HEX_DIRS[moveHeading]
    if (!dir) return

    let nextTurnProgress = 0
    let nextSlipProgress = 0
    if (moveKind === "turn") {
      nextTurnProgress = 0
      nextSlipProgress = 0
    } else if (moveKind === "sideslip") {
      nextTurnProgress = clampMovementTurnProgress(turnProgress + 1, requiredStraight)
      nextSlipProgress = 0
    } else {
      nextTurnProgress = clampMovementTurnProgress(turnProgress + 1, requiredStraight)
      nextSlipProgress = clampMovementSideslipProgress(slipProgress + 1)
    }

    nextStates.push({
      q: q0 + dir.q,
      r: r0 + dir.r,
      heading: nextHeading,
      turnProgress: nextTurnProgress,
      slipProgress: nextSlipProgress,
      moveKind
    })
  }

  // Straight move
  pushMove(heading, heading, "straight")

  // Regular 60-degree turns (move into the new facing hex, facing changes)
  if (canTurn) {
    pushMove(heading + 1, heading + 1, "turn")
    pushMove(heading - 1, heading - 1, "turn")
  }

  // Sideslip (move into front-side hex, facing unchanged)
  if (canSideslip) {
    pushMove(heading + 1, heading, "sideslip")
    pushMove(heading - 1, heading, "sideslip")
  }

  return nextStates
}

function drawMovementFacingTriangle(ctx, cx, cy, heading, markerSize, fillStyle, strokeStyle, strokeWidth = 1) {
  const headingIdx = ((Math.floor(Number(heading) || 0) % 6) + 6) % 6
  const dir = MOVEMENT_HEX_DIRS[headingIdx] || MOVEMENT_HEX_DIRS[0]
  const dx = 1.5 * (Number(dir.q) || 0)
  const dy = Math.sqrt(3) * ((Number(dir.r) || 0) + (Number(dir.q) || 0) / 2)
  let angle = Math.atan2(dy, dx)
  if (!Number.isFinite(angle)) angle = -Math.PI / 2

  const side = Math.max(4, Number(markerSize) || 4)
  const triHeight = (Math.sqrt(3) / 2) * side
  const centerToTip = (2 * triHeight) / 3
  const centerToBase = triHeight / 3
  const perpX = -Math.sin(angle)
  const perpY = Math.cos(angle)
  const dirX = Math.cos(angle)
  const dirY = Math.sin(angle)

  const tipX = cx + dirX * centerToTip
  const tipY = cy + dirY * centerToTip
  const baseCx = cx - dirX * centerToBase
  const baseCy = cy - dirY * centerToBase
  const leftX = baseCx + perpX * (side / 2)
  const leftY = baseCy + perpY * (side / 2)
  const rightX = baseCx - perpX * (side / 2)
  const rightY = baseCy - perpY * (side / 2)

  ctx.beginPath()
  ctx.moveTo(tipX, tipY)
  ctx.lineTo(leftX, leftY)
  ctx.lineTo(rightX, rightY)
  ctx.closePath()
  if (fillStyle) {
    ctx.fillStyle = fillStyle
    ctx.fill()
  }
  if (strokeStyle) {
    ctx.strokeStyle = strokeStyle
    ctx.lineWidth = strokeWidth
    ctx.stroke()
  }
}

function buildMovementPathFromFinalNode(finalNode) {
  if (!finalNode || typeof finalNode !== "object") return null
  const reversed = []
  let cursor = finalNode
  while (cursor) {
    reversed.push({
      q: cursor.q,
      r: cursor.r,
      heading: cursor.heading,
      moveKind: cursor.moveKind || null
    })
    cursor = cursor.parent || null
  }
  reversed.reverse()
  return reversed
}

function buildMovementExpansion(tab) {
  const movement = ensureMovementAnalysisState(tab)
  const table = tab?.turnMovementTable || null
  const base = {
    ok: false,
    status: "missing",
    turnModeLetter: movement?.turnModeLetter || "D",
    speedRaw: movement ? String(movement.speed || "").trim() : "",
    speedValue: null,
    requestedMoves: 0,
    plottedMoves: 0,
    capped: false,
    requiredStraight: null,
    finalStateCount: 0,
    maxRange: 0,
    finalHeadingMaskByHex: new Map(),
    sampleFinalNodeByHex: new Map(),
    sampleFinalNodeByHexHeading: new Map(),
    match: null,
    note: ""
  }

  if (!movement || !table || !Array.isArray(table.rows)) {
    base.status = "missingTable"
    return base
  }

  if (!base.speedRaw) {
    base.status = "needsSpeed"
    return base
  }

  const speedValue = Number(base.speedRaw)
  if (!Number.isFinite(speedValue) || speedValue < 0) {
    base.status = "invalidSpeed"
    return base
  }
  base.speedValue = speedValue

  const match = findMovementTurnMatch(table, base.turnModeLetter, speedValue)
  if (!match) {
    base.status = "noMatch"
    return base
  }
  base.match = match

  let requiredStraight = Number(match.row?.straightMovesBeforeTurn ?? match.row?.turnMode)
  if (!Number.isFinite(requiredStraight) || requiredStraight < 0) {
    base.status = "invalidRow"
    return base
  }
  requiredStraight = Math.floor(requiredStraight)
  base.requiredStraight = requiredStraight

  let requestedMoves = Math.max(0, Math.round(speedValue))
  let plottedMoves = requestedMoves
  if (plottedMoves > MOVEMENT_ENDPOINT_MAX_MOVES) {
    plottedMoves = MOVEMENT_ENDPOINT_MAX_MOVES
    base.capped = true
    base.note = `Endpoint plot capped at ${MOVEMENT_ENDPOINT_MAX_MOVES} moves for performance.`
  }
  base.requestedMoves = requestedMoves
  base.plottedMoves = plottedMoves

  let current = new Map()
  const startTurnProgress = clampMovementTurnProgress(requiredStraight, requiredStraight)
  const startSlipProgress = MOVEMENT_SIDESLIP_REQUIRED_STRAIGHT
  const startHeading = MOVEMENT_ASSUMED_FORWARD_HEADING
  const startKey = `0,0,${startHeading},${startTurnProgress},${startSlipProgress}`
  current.set(startKey, {
    q: 0,
    r: 0,
    heading: startHeading,
    turnProgress: startTurnProgress,
    slipProgress: startSlipProgress,
    moveKind: null,
    parent: null
  })

  for (let step = 0; step < plottedMoves; step++) {
    const next = new Map()
    for (const stateItem of current.values()) {
      const transitions = buildMovementStepTransitions(stateItem, requiredStraight)
      for (const nextState of transitions) {
        const key = `${nextState.q},${nextState.r},${nextState.heading},${nextState.turnProgress},${nextState.slipProgress}`
        if (next.has(key)) continue
        next.set(key, {
          q: nextState.q,
          r: nextState.r,
          heading: nextState.heading,
          turnProgress: nextState.turnProgress,
          slipProgress: nextState.slipProgress,
          moveKind: nextState.moveKind || null,
          parent: stateItem
        })
      }
    }
    current = next
    if (current.size === 0) break
  }

  const finalNodes = Array.from(current.values())
  const finalHeadingMaskByHex = new Map()
  const sampleFinalNodeByHex = new Map()
  const sampleFinalNodeByHexHeading = new Map()
  let maxRange = 0
  for (const node of finalNodes) {
    const hexKey = `${node.q},${node.r}`
    const headingBit = 1 << (((Math.floor(Number(node.heading) || 0) % 6) + 6) % 6)
    const prevMask = finalHeadingMaskByHex.get(hexKey) || 0
    finalHeadingMaskByHex.set(hexKey, prevMask | headingBit)
    if (!sampleFinalNodeByHex.has(hexKey)) sampleFinalNodeByHex.set(hexKey, node)
    const byHeadingKey = `${hexKey}|${node.heading}`
    if (!sampleFinalNodeByHexHeading.has(byHeadingKey)) sampleFinalNodeByHexHeading.set(byHeadingKey, node)
    const range = getAnalysisHexRange(node.q, node.r)
    if (range > maxRange) maxRange = range
  }

  base.ok = true
  base.status = "ok"
  base.finalHeadingMaskByHex = finalHeadingMaskByHex
  base.sampleFinalNodeByHex = sampleFinalNodeByHex
  base.sampleFinalNodeByHexHeading = sampleFinalNodeByHexHeading
  base.finalStateCount = finalNodes.length
  base.maxRange = maxRange
  base.assumedForwardHeading = MOVEMENT_ASSUMED_FORWARD_HEADING
  base.includesSideslip = true
  return base
}

function getMovementExpansion(tab) {
  if (!tab || tab.uiState !== "movementAnalysis") return null
  const movement = ensureMovementAnalysisState(tab)
  const table = tab?.turnMovementTable || null
  const cacheKey = JSON.stringify({
    letter: movement?.turnModeLetter || "D",
    speed: String(movement?.speed || "").trim(),
    tableVersion: Number(table?.version) || 0,
    rowCount: Array.isArray(table?.rows) ? table.rows.length : 0
  })
  const cached = tab.movementExpansionCache
  if (cached && cached.key === cacheKey && cached.value) return cached.value
  const value = buildMovementExpansion(tab)
  tab.movementExpansionCache = { key: cacheKey, value }
  return value
}

function buildMovementEndpointAnalysis(tab) {
  const movement = ensureMovementAnalysisState(tab)
  const expansion = getMovementExpansion(tab)
  const speedRaw = movement ? String(movement.speed || "").trim() : ""
  const endingHeadingFilter = normalizeMovementHeadingIndex(movement?.endingHeading)
  const base = {
    ok: false,
    status: expansion?.status || "missing",
    turnModeLetter: expansion?.turnModeLetter || movement?.turnModeLetter || "D",
    speedRaw,
    endingHeading: endingHeadingFilter,
    speedValue: expansion?.speedValue ?? null,
    requestedMoves: Number(expansion?.requestedMoves) || 0,
    plottedMoves: Number(expansion?.plottedMoves) || 0,
    capped: Boolean(expansion?.capped),
    requiredStraight: expansion?.requiredStraight ?? null,
    endpointCount: 0,
    finalStateCount: Number(expansion?.finalStateCount) || 0,
    maxRange: 0,
    endpointItems: [],
    endpointKeySet: new Set(),
    sampleFinalNodeByHex: new Map(),
    match: expansion?.match || null,
    note: expansion?.note || ""
  }

  if (!expansion || !expansion.ok) {
    return base
  }

  const endpointMaskByHex = new Map()
  const sampleFinalNodeByHex = new Map()
  const finalHeadingMaskByHex = expansion.finalHeadingMaskByHex instanceof Map
    ? expansion.finalHeadingMaskByHex
    : new Map()
  for (const [hexKey, maskValue] of finalHeadingMaskByHex.entries()) {
    const mask = Number(maskValue) || 0
    let filteredMask = mask
    if (endingHeadingFilter !== null) filteredMask &= (1 << endingHeadingFilter)
    if (!filteredMask) continue
    endpointMaskByHex.set(hexKey, filteredMask)
    if (endingHeadingFilter === null) {
      const sampleNode = expansion.sampleFinalNodeByHex instanceof Map
        ? expansion.sampleFinalNodeByHex.get(hexKey)
        : null
      if (sampleNode) sampleFinalNodeByHex.set(hexKey, sampleNode)
    } else {
      const byHeadingKey = `${hexKey}|${endingHeadingFilter}`
      const sampleNode = expansion.sampleFinalNodeByHexHeading instanceof Map
        ? expansion.sampleFinalNodeByHexHeading.get(byHeadingKey)
        : null
      if (sampleNode) sampleFinalNodeByHex.set(hexKey, sampleNode)
    }
  }

  const endpointItems = []
  const endpointKeySet = new Set()
  let maxRange = 0
  for (const [hexKey, headingMask] of endpointMaskByHex.entries()) {
    const [qText, rText] = hexKey.split(",")
    const q = Number(qText)
    const r = Number(rText)
    if (!Number.isFinite(q) || !Number.isFinite(r)) continue
    const range = getAnalysisHexRange(q, r)
    endpointKeySet.add(hexKey)
    endpointItems.push({
      key: hexKey,
      q,
      r,
      headingMask,
      headingCount: countMovementHeadingBits(headingMask),
      range
    })
    if (range > maxRange) maxRange = range
  }

  endpointItems.sort((a, b) => {
    if (a.range !== b.range) return a.range - b.range
    if (a.r !== b.r) return a.r - b.r
    return a.q - b.q
  })

  base.ok = true
  base.status = "ok"
  base.endpointItems = endpointItems
  base.endpointKeySet = endpointKeySet
  base.sampleFinalNodeByHex = sampleFinalNodeByHex
  base.endpointCount = endpointItems.length
  base.maxRange = maxRange
  base.assumedForwardHeading = expansion.assumedForwardHeading
  base.includesSideslip = expansion.includesSideslip === true
  return base
}

function getMovementEndpointAnalysis(tab) {
  if (!tab || tab.uiState !== "movementAnalysis") return null
  const movement = ensureMovementAnalysisState(tab)
  const table = tab?.turnMovementTable || null
  const cacheKey = JSON.stringify({
    letter: movement?.turnModeLetter || "D",
    speed: String(movement?.speed || "").trim(),
    endingHeading: normalizeMovementHeadingIndex(movement?.endingHeading),
    tableVersion: Number(table?.version) || 0,
    rowCount: Array.isArray(table?.rows) ? table.rows.length : 0
  })
  const cached = tab.movementEndpointAnalysisCache
  if (cached && cached.key === cacheKey && cached.value) return cached.value
  const value = buildMovementEndpointAnalysis(tab)
  tab.movementEndpointAnalysisCache = { key: cacheKey, value }
  return value
}

function getMovementEndpointMapRadius(tab, endpointResult = null) {
  const result = endpointResult || getMovementEndpointAnalysis(tab)
  return Math.max(
    5,
    Math.min(
      80,
      (result?.ok ? Math.max(result.maxRange + 1, result.plottedMoves + 1) : 8)
    )
  )
}

function findMovementEndpointPathToHex(tab, targetQ, targetR) {
  const result = getMovementEndpointAnalysis(tab)
  if (!result?.ok) return null
  const targetKey = `${targetQ},${targetR}`
  if (!(result.endpointKeySet instanceof Set) || !result.endpointKeySet.has(targetKey)) return null
  const finalNode = result.sampleFinalNodeByHex instanceof Map
    ? result.sampleFinalNodeByHex.get(targetKey)
    : null
  if (!finalNode) return null
  return buildMovementPathFromFinalNode(finalNode)
}

function selectMovementEndpointPath(tab, q, r) {
  if (!tab || tab.uiState !== "movementAnalysis") return
  const key = `${q},${r}`
  if (tab.movementSelectedEndpointKey === key) {
    tab.movementSelectedEndpointKey = null
    tab.movementSelectedPath = null
    renderCanvas()
    renderCurrentProps()
    return
  }

  const result = getMovementEndpointAnalysis(tab)
  if (!result?.ok || !(result.endpointKeySet instanceof Set) || !result.endpointKeySet.has(key)) {
    tab.movementSelectedEndpointKey = null
    tab.movementSelectedPath = null
    renderCanvas()
    renderCurrentProps()
    return
  }

  const path = findMovementEndpointPathToHex(tab, q, r)
  if (!Array.isArray(path) || path.length === 0) {
    showTempMessage("No path found for that endpoint.")
    return
  }

  tab.movementSelectedEndpointKey = key
  tab.movementSelectedPath = path
  renderCanvas()
  renderCurrentProps()
}

async function ensureDesignationAssignments(items, designationAssignments) {
  if (!Array.isArray(items) || items.length === 0) return true

  for (const item of items) {
    const assignmentKey = String(item?.key ?? item?.squareId ?? "").trim()
    if (!assignmentKey) continue
    if (designationAssignments[assignmentKey]) continue
    const currentValue = designationAssignments[assignmentKey] || item?.designation || ""
    const title = buildDesignationPromptTitle(item)
    setArcHighlight(item)
    renderCanvas()
    const choice = await promptDesignationInput(title, currentValue)
    clearArcHighlight()
    renderCanvas()
    if (choice === null) return false
    designationAssignments[assignmentKey] = choice
  }
  return true
}

async function ensureArcAssignments(items, arcAssignments) {
  if (!Array.isArray(items) || items.length === 0) return true
  const arcOptions = await readArcOptions()
  if (!arcOptions || arcOptions.length === 0) {
    alert("No arc options found in Data.json under the Arcs section.")
    return false
  }

  for (const item of items) {
    const assignmentKey = String(item?.key ?? item?.squareId ?? "").trim()
    if (!assignmentKey) continue
    if (arcAssignments[assignmentKey]) continue
    const currentValue = arcAssignments[assignmentKey] || item?.arc || ""
    const title = buildArcPromptTitle(item)
    const syncSelectionHighlight = (selectedValues) => {
      const hasSelection = Array.isArray(selectedValues) && selectedValues.length > 0
      if (hasSelection) setArcHighlight(item)
      else clearArcHighlight()
      renderCanvas()
    }
    setArcHighlight(item)
    renderCanvas()
    const choice = await promptArcDropdown(title, currentValue, arcOptions, {
      onSelectionChange: syncSelectionHighlight
    })
    clearArcHighlight()
    renderCanvas()
    if (choice === null) return false
    arcAssignments[assignmentKey] = choice
  }
  return true
}

async function ensureRankValueAssignments(items, rankValueAssignments) {
  if (!Array.isArray(items) || items.length === 0) return true

  const groupNeedsRanks = new Map()
  for (const item of items) {
    const assignmentKey = getRankAssignmentKey(item)
    if (!assignmentKey) continue
    const existing = rankValueAssignments[assignmentKey]
    const hasRank = existing && typeof existing === "object" && existing.rank !== undefined && existing.rank !== null
    if (hasRank) continue
    const groupKey = String(item?.group || "").trim().toLowerCase()
    if (!groupKey) continue
    if (!groupNeedsRanks.has(groupKey)) groupNeedsRanks.set(groupKey, [])
    groupNeedsRanks.get(groupKey).push(item)
  }

  for (const [groupKey, groupItems] of groupNeedsRanks.entries()) {
    if (!Array.isArray(groupItems) || groupItems.length < 2) continue
    const selected = await promptRankRangeSelection(groupKey, groupItems)
    if (selected === null) return false
    const ordered = orderRankItemsByEndpoints(groupItems, selected.firstKey, selected.lastKey)
    for (let i = 0; i < ordered.length; i++) {
      const item = ordered[i]
      const assignmentKey = getRankAssignmentKey(item)
      if (!assignmentKey) continue
      const existing =
        rankValueAssignments[assignmentKey] && typeof rankValueAssignments[assignmentKey] === "object"
          ? rankValueAssignments[assignmentKey]
          : {}
      existing.rank = i + 1
      existing.rankLocked = true
      rankValueAssignments[assignmentKey] = existing
    }
  }

  for (const item of items) {
    const assignmentKey = getRankAssignmentKey(item)
    if (!assignmentKey) continue
    const existing = rankValueAssignments[assignmentKey]
    const hasRankValue =
      existing &&
      typeof existing === "object" &&
      existing.rank !== undefined &&
      existing.rank !== null &&
      existing.value !== undefined &&
      existing.value !== null
    if (hasRankValue) continue
    const currentRank = existing?.rank ?? item?.rank
    const currentValue = existing?.value ?? item?.value
    const lockRank = !!existing?.rankLocked
    const title = buildRankValuePromptTitle(item)
    setArcHighlight(item)
    renderCanvas()
    const choice = await promptRankValueInput(title, currentRank, currentValue, { lockRank })
    clearArcHighlight()
    renderCanvas()
    if (choice === null) return false
    rankValueAssignments[assignmentKey] =
      existing && typeof existing === "object"
        ? { ...existing, ...choice }
        : choice
  }
  return true
}

// Sector index order (clockwise from the top arrow boundary):
// 0 = RF, 1 = R, 2 = RR, 3 = LR, 4 = L, 5 = LF.
// FH/RH hemisphere coverage is handled separately in analysis geometry.
const ARC_SECTOR_MAP = {
  "360": [0, 1, 2, 3, 4, 5],
  RF: [0],
  R: [1],
  RR: [2],
  LR: [3],
  L: [4],
  LF: [5],
  FA: [0, 5],
  FX: [0, 1, 4, 5],
  RA: [2, 3],
  RX: [1, 2, 3, 4],
  RS: [0, 1, 2],
  LS: [3, 4, 5],
  FH: [0, 1, 4, 5],
  RH: [1, 2, 3, 4],
  LP: [4, 5],
  RP: [0, 1],
  AP: [2, 3],
  LPA: [2, 3, 4],
  RPA: [1, 2, 3]
}

const PLASMA_ARC_HALF_WIDTH = Math.PI / 2
const PLASMA_ARC_RANGES = {
  FP: { center: 0 },
  LP: { center: (Math.PI * 5) / 3 },
  RP: { center: Math.PI / 3 }
}

function angleDelta(a, b) {
  let diff = Math.abs(a - b) % (Math.PI * 2)
  if (diff > Math.PI) diff = Math.PI * 2 - diff
  return diff
}

function isAngleInPlasmaArc(angle, arcKey) {
  const info = PLASMA_ARC_RANGES[arcKey]
  if (!info) return false
  return angleDelta(angle, info.center) <= PLASMA_ARC_HALF_WIDTH + ANALYSIS_HEX_ARC_EPS
}


function buildWeaponDamageIndex(weaponDamage) {
  const index = new Map()
  const weapons = weaponDamage && Array.isArray(weaponDamage.weapons)
    ? weaponDamage.weapons
    : []
  for (const weapon of weapons) {
    const id = String(weapon?.id || "").trim()
    if (!id) continue
    index.set(id, weapon)
  }
  return index
}

function mapWeaponTypeToId(rawType) {
  const text = String(rawType || "").toLowerCase()
  if (!text) return null

  if (text.includes("photon")) return "photon_torpedo"
  if (text.includes("particle")) return "particle_cannon"
  if (text.includes("plasmatic") || text.includes("pulsar") || /\bppd\b/.test(text)) {
    return "plasmatic_pulsar_device"
  }
  if (text.includes("plasma")) {
    if (/\bplasma\s*-?\s*(torpedo\s*-?\s*)?r\b/.test(text)) return "plasma_torpedo_r"
    if (/\bplasma\s*-?\s*(torpedo\s*-?\s*)?s\b/.test(text)) return "plasma_torpedo_s"
    if (/\bplasma\s*-?\s*(torpedo\s*-?\s*)?g\b/.test(text)) return "plasma_torpedo_g"
    if (/\bplasma\s*-?\s*(torpedo\s*-?\s*)?f\b/.test(text)) return "plasma_torpedo_f"
    if (/\bplasma\s*-?\s*(torpedo\s*-?\s*)?d\b/.test(text)) return "plasma_torpedo_d"
  }
  if (text.includes("disruptor")) return "disruptor"
  if (text.includes("fusion")) return "fusion_beam"
  if (text.includes("hellbore")) return "hellbore"

  if (text.includes("phaser")) {
    if (/\bphaser\s*-?\s*g\b/.test(text)) return "phaser_3"
    const numMatch = text.match(/phaser[^0-9]*([1-4])/)
    if (numMatch) return `phaser_${numMatch[1]}`
    const romanMatch = text.match(/phaser[^iv]*(iv|iii|ii|i)\b/)
    if (romanMatch) {
      const roman = romanMatch[1]
      const romanMap = { i: 1, ii: 2, iii: 3, iv: 4 }
      const value = romanMap[roman]
      if (value) return `phaser_${value}`
    }
  }

  return null
}

function isModeAllowedAtRange(mode, range) {
  const bands = mode?.toHit?.rangeBands
  if (!Array.isArray(bands) || bands.length === 0) return true
  const band = bands.find(b => range >= b.min && range <= b.max)
  if (!band) return false
  if (band.allowed === false) return false
  return true
}

function getWeaponModes(weaponDef) {
  if (Array.isArray(weaponDef?.firingProfiles)) return weaponDef.firingProfiles
  if (Array.isArray(weaponDef?.modes)) return weaponDef.modes
  return []
}

function formatModeLabel(modeId, variantKey) {
  const base = String(modeId || "mode").replace(/_/g, " ")
  return variantKey ? `${base} (${variantKey})` : base
}

function isDamageSpecObject(spec) {
  if (!spec || typeof spec !== "object") return false
  if (Array.isArray(spec.rangeBands)) return true
  if (Array.isArray(spec.damageByDie)) return true
  if (spec.type) return true
  if (typeof spec.damage === "number") return true
  if (spec.damage && typeof spec.damage === "object") return true
  return false
}

function buildWeaponFiringOptions(weaponDef) {
  const options = []
  const modes = getWeaponModes(weaponDef)
  for (const mode of modes) {
    const modeId = String(mode?.id || "").trim() || "mode"
    if (mode?.damage && !isDamageSpecObject(mode.damage)) {
      for (const [variantKey, variantSpec] of Object.entries(mode.damage)) {
        if (!variantSpec || typeof variantSpec !== "object") continue
        options.push({
          key: `${modeId}:${variantKey}`,
          modeId,
          variantKey,
          mode,
          label: formatModeLabel(modeId, variantKey)
        })
      }
      continue
    }
    options.push({
      key: modeId,
      modeId,
      variantKey: null,
      mode,
      label: formatModeLabel(modeId, null)
    })
  }
  return options
}

function getModeDamageSpec(mode, variantKey) {
  if (!mode) return null
  if (mode.damageTable) return mode.damageTable
  if (!mode.damage) return null
  if (variantKey && mode.damage && typeof mode.damage === "object") {
    return mode.damage[variantKey] || null
  }
  return mode.damage
}

function getAverageDamageFromRangeBands(rangeBands, range) {
  if (!Array.isArray(rangeBands)) return 0
  const band = rangeBands.find(b => range >= b.min && range <= b.max)
  if (!band) return 0
  if (typeof band.damage === "number") return band.damage
  if (Array.isArray(band.damageByDie)) {
    let sum = 0
    let count = 0
    for (const value of band.damageByDie) {
      const num = Number(value)
      if (Number.isFinite(num)) {
        sum += num
        count += 1
      } else {
        count += 1
      }
    }
    return count > 0 ? sum / count : 0
  }
  return 0
}

function getAverageDamageFromSpec(spec, range) {
  if (!spec) return 0
  if (spec.type === "lookupByEnergy") {
    const rows = Array.isArray(spec.overloadEnergyToWarheadStrength)
      ? spec.overloadEnergyToWarheadStrength
      : []
    let max = 0
    for (const row of rows) {
      const value = Number(row?.warheadStrength)
      if (Number.isFinite(value) && value > max) max = value
    }
    return max
  }
  if (Array.isArray(spec.rangeBands)) {
    return getAverageDamageFromRangeBands(spec.rangeBands, range)
  }
  if (typeof spec.damage === "number") return spec.damage
  if (Array.isArray(spec.damageByDie)) {
    let sum = 0
    let count = 0
    for (const value of spec.damageByDie) {
      const num = Number(value)
      if (Number.isFinite(num)) {
        sum += num
        count += 1
      } else {
        count += 1
      }
    }
    return count > 0 ? sum / count : 0
  }
  if (spec.damage && typeof spec.damage === "object") {
    return getAverageDamageFromSpec(spec.damage, range)
  }
  if (spec && typeof spec === "object") {
    let max = 0
    for (const value of Object.values(spec)) {
      const candidate = getAverageDamageFromSpec(value, range)
      if (candidate > max) max = candidate
    }
    return max
  }
  return 0
}

function getModeAverageDamageAtRange(mode, variantKey, range) {
  if (!isModeAllowedAtRange(mode, range)) return 0
  const spec = getModeDamageSpec(mode, variantKey)
  if (!spec) return 0
  return getAverageDamageFromSpec(spec, range)
}

function getWeaponAverageDamageAtRange(weaponDef, selection, range) {
  const modes = getWeaponModes(weaponDef)
  if (!modes.length) return 0
  let picked = null
  if (selection?.modeId) {
    picked = modes.find(mode => String(mode?.id || "") === selection.modeId) || null
  }
  if (!picked) picked = modes[0]
  return getModeAverageDamageAtRange(picked, selection?.variantKey || null, range)
}

function buildWeaponRangeDamage(weaponDef, selection, radius) {
  const maxRange = normalizeAnalysisHexRadius(radius)
  const damageByRange = []
  for (let range = 0; range <= maxRange; range++) {
    damageByRange[range] = getWeaponAverageDamageAtRange(weaponDef, selection, range)
  }
  return damageByRange
}

function getSsdEntriesByKey(doc, key) {
  if (!doc || !doc.ssd || typeof doc.ssd !== "object") return []
  const wanted = String(key || "").trim().toLowerCase()
  if (!wanted) return []
  const matches = []
  for (const [sectionKey, sectionEntries] of Object.entries(doc.ssd)) {
    if (String(sectionKey || "").trim().toLowerCase() !== wanted) continue
    if (!Array.isArray(sectionEntries)) continue
    matches.push(...sectionEntries)
  }
  return matches
}

async function collectWeaponModeSelections(doc, weaponDamage) {
  if (!doc || !doc.ssd || typeof doc.ssd !== "object") return new Map()
  const weaponDamageIndex = buildWeaponDamageIndex(weaponDamage)
  const weaponIds = new Set()
  const weaponKeys = ["heavy", "phaser", "drone"]
  for (const key of weaponKeys) {
    const entries = getSsdEntriesByKey(doc, key)
    for (const entry of entries) {
      const weaponId = mapWeaponTypeToId(entry?.type || entry?.name)
      if (weaponId) weaponIds.add(weaponId)
    }
  }

  const selections = new Map()
  for (const weaponId of weaponIds) {
    const weaponDef = weaponDamageIndex.get(weaponId)
    if (!weaponDef) continue
    const options = buildWeaponFiringOptions(weaponDef)
    if (options.length === 0) continue
    if (options.length === 1) {
      selections.set(weaponId, options[0])
      continue
    }
    const title = `Select firing mode for ${weaponDef.name || weaponId}`
    const optionLabels = options.map(opt => opt.label)
    const choice = await promptLabelDropdown(title, optionLabels[0], optionLabels)
    if (choice === null) return null
    const picked = options.find(opt => opt.label === choice) || options[0]
    selections.set(weaponId, picked)
  }
  return selections
}

function shouldIncludeWeaponEntry(entry, key, filter) {
  if (filter !== "phaser") return true
  if (key === "phaser") return true
  const label = String(entry?.type || entry?.name || "").toLowerCase()
  return label.includes("phaser")
}

function buildWeaponAnalysis(doc, weaponDamage, modeSelections, options = {}) {
  const analysisRadius = normalizeAnalysisHexRadius(options.radius)
  const analysis = {
    weaponCount: 0,
    arcEntryCount: 0,
    emptyArcCount: 0,
    partCounts: {},
    unknownParts: new Set(),
    maxHeat: 0,
    avgHeat: 0,
    weaponEntries: [],
    weaponHighlights: [],
    filter: options.weaponFilter || "all",
    radius: analysisRadius,
    weaponTableDefs: {}
  }

  if (!doc || !doc.ssd || typeof doc.ssd !== "object") return analysis

  const weaponDamageIndex = buildWeaponDamageIndex(weaponDamage)
  const damageById = new Map()
  const modeMap = modeSelections instanceof Map ? modeSelections : new Map()
  const filter = analysis.filter

  const weaponKeys = ["heavy", "phaser", "drone"]
  for (const key of weaponKeys) {
    const entries = getSsdEntriesByKey(doc, key)
    for (const entry of entries) {
      if (!shouldIncludeWeaponEntry(entry, key, filter)) continue
      analysis.weaponCount += 1
      const arcRaw = String(entry?.arc || "").trim()
      if (!arcRaw) {
        analysis.emptyArcCount += 1
        continue
      }
      analysis.arcEntryCount += 1
      const parts = arcRaw
        .split("+")
        .map(part => part.trim().toUpperCase())
        .filter(Boolean)
      const uniqueParts = Array.from(new Set(parts))
      if (uniqueParts.length === 0) continue

      const typeLabel = String(entry?.type || entry?.name || "Weapon").trim()
      const designation = String(entry?.designation || "").trim()
      const weaponLabel = designation ? `${typeLabel} (${designation})` : typeLabel
      const arcText = uniqueParts.join("+")
      const isPhaserG = /\bphaser\s*-?\s*g\b/i.test(typeLabel)
      const damageMultiplier = isPhaserG ? 4 : 1
      const weaponId = mapWeaponTypeToId(entry?.type || entry?.name)
      const bbox = parsePosTextToBbox(entry?.pos)
      const points = bbox ? pointsFromBbox(bbox) : null

      const sectors = new Set()
      const plasmaParts = new Set()
      const hasAll = uniqueParts.includes("360")
      const hasFrontHemisphere = uniqueParts.includes("FH")
      const hasRearHemisphere = uniqueParts.includes("RH")
      for (const part of uniqueParts) {
        analysis.partCounts[part] = (analysis.partCounts[part] || 0) + 1
        if (part === "FH" || part === "RH" || part === "360") continue
        if (PLASMA_ARC_RANGES[part]) {
          plasmaParts.add(part)
          continue
        }
        const mapped = ARC_SECTOR_MAP[part]
        if (!mapped && part !== "FH" && part !== "RH" && part !== "360") {
          analysis.unknownParts.add(part)
          continue
        }
        if (!mapped) continue
        for (const idx of mapped) sectors.add(idx)
      }
      analysis.weaponHighlights.push({
        key: `weapon-${analysis.weaponHighlights.length}`,
        name: typeLabel,
        designation,
        label: weaponLabel,
        arcText,
        partsCount: uniqueParts.length,
        sectors,
        plasmaParts,
        hasAll,
        hasFrontHemisphere,
        hasRearHemisphere,
        points,
        bbox
      })

      if (!weaponId) continue
      const weaponDef = weaponDamageIndex.get(weaponId)
      if (!weaponDef) continue
      const tableDef = weaponDef?.analysisTable
      if (
        tableDef &&
        typeof tableDef === "object" &&
        !analysis.weaponTableDefs[weaponId]
      ) {
        analysis.weaponTableDefs[weaponId] = {
          id: weaponId,
          name: String(weaponDef?.name || weaponId),
          title: String(tableDef?.title || weaponDef?.name || weaponId),
          columns: Array.isArray(tableDef?.columns) ? tableDef.columns : [],
          rows: Array.isArray(tableDef?.rows) ? tableDef.rows : []
        }
      }
      const selection = modeMap.get(weaponId) || null
      const damageKey = selection?.key ? `${weaponId}:${selection.key}` : weaponId
      if (!damageById.has(damageKey)) {
        damageById.set(damageKey, buildWeaponRangeDamage(weaponDef, selection, analysisRadius))
      }

      analysis.weaponEntries.push({
        id: weaponId,
        label: weaponLabel,
        name: typeLabel,
        designation,
        sectors,
        plasmaParts,
        hasAll,
        hasFrontHemisphere,
        hasRearHemisphere,
        damageMultiplier,
        damageByRange: damageById.get(damageKey)
      })
    }
  }

  let maxTotalDamage = 0
  let totalDamageSum = 0
  let totalHexes = 0
  for (let q = -analysisRadius; q <= analysisRadius; q++) {
    for (let r = -analysisRadius; r <= analysisRadius; r++) {
      const s = -q - r
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) > analysisRadius) continue
      if (q === 0 && r === 0) continue
      const sectors = getAnalysisHexSectors(q, r)
      const hexAngle = getAnalysisHexAngle(q, r)
      if (!sectors.length) continue
      const range = getAnalysisHexRange(q, r)
      const isFrontHemisphere = isAnalysisFrontHemisphere(q, r)
      const isRearHemisphere = isAnalysisRearHemisphere(q, r)
      let total = 0
      for (const weapon of analysis.weaponEntries) {
        if (!weaponCoversHex(weapon, sectors, isFrontHemisphere, isRearHemisphere, hexAngle)) continue
        const perWeapon = weapon.damageByRange?.[range] || 0
        total += perWeapon * (weapon.damageMultiplier || 1)
      }
      totalDamageSum += total
      totalHexes += 1
      if (total > maxTotalDamage) maxTotalDamage = total
    }
  }

  analysis.maxHeat = maxTotalDamage
  analysis.avgHeat = totalHexes > 0 ? totalDamageSum / totalHexes : 0
  return analysis
}

function formatAnalysisCount(value) {
  if (!Number.isFinite(value)) return "0"
  if (Number.isInteger(value)) return String(value)
  const rounded = Math.round(value * 100) / 100
  if (Number.isInteger(rounded)) return String(rounded)
  return rounded.toFixed(2).replace(/\.?0+$/, "")
}

function formatSignedAnalysisCount(value) {
  if (!Number.isFinite(value) || value === 0) return "0"
  const absVal = Math.abs(value)
  const base = formatAnalysisCount(absVal)
  return value > 0 ? `+${base}` : `-${base}`
}

function getAnalysisHexDamage(analysis, sectors, hexAngle, range, isFrontHemisphere, isRearHemisphere) {
  const weaponList = Array.isArray(analysis?.weaponEntries)
    ? analysis.weaponEntries
    : []
  if (!weaponList.length) return 0
  let total = 0
  for (const weapon of weaponList) {
    if (!weaponCoversHex(weapon, sectors, isFrontHemisphere, isRearHemisphere, hexAngle)) continue
    const perWeapon = weapon.damageByRange?.[range] || 0
    total += perWeapon * (weapon.damageMultiplier || 1)
  }
  return total
}

function computeAnalysisDiffScale(primary, secondary, radius) {
  if (!primary || !secondary) return 0
  const safeRadius = normalizeAnalysisHexRadius(radius)
  let maxAbs = 0
  for (let q = -safeRadius; q <= safeRadius; q++) {
    for (let r = -safeRadius; r <= safeRadius; r++) {
      const s = -q - r
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) > safeRadius) continue
      if (q === 0 && r === 0) continue
      const sectors = getAnalysisHexSectors(q, r)
      if (!sectors.length) continue
      const hexAngle = getAnalysisHexAngle(q, r)
      const range = getAnalysisHexRange(q, r)
      const isFrontHemisphere = isAnalysisFrontHemisphere(q, r)
      const isRearHemisphere = isAnalysisRearHemisphere(q, r)
      const a = getAnalysisHexDamage(primary, sectors, hexAngle, range, isFrontHemisphere, isRearHemisphere)
      const b = getAnalysisHexDamage(secondary, sectors, hexAngle, range, isFrontHemisphere, isRearHemisphere)
      const diff = Math.abs(a - b)
      if (diff > maxAbs) maxAbs = diff
    }
  }
  return maxAbs
}

function getAnalysisDamageBreakdown(analysis, q, r) {
  if (!analysis) return null
  const weaponList = Array.isArray(analysis.weaponEntries)
    ? analysis.weaponEntries
    : []
  if (!weaponList.length) return null
  if (q === 0 && r === 0) return { range: 0, total: 0, rows: [] }
  const sectors = getAnalysisHexSectors(q, r)
  if (!sectors.length) return null
  const hexAngle = getAnalysisHexAngle(q, r)
  const range = getAnalysisHexRange(q, r)
  const isFrontHemisphere = isAnalysisFrontHemisphere(q, r)
  const isRearHemisphere = isAnalysisRearHemisphere(q, r)
  const rows = []
  let total = 0
  for (const weapon of weaponList) {
    if (!weaponCoversHex(weapon, sectors, isFrontHemisphere, isRearHemisphere, hexAngle)) continue
    const perWeapon = weapon.damageByRange?.[range] || 0
    const dmg = perWeapon * (weapon.damageMultiplier || 1)
    total += dmg
    if (dmg > 0) {
      const label = String(weapon?.label || weapon?.name || weapon?.id || "Weapon")
      rows.push({ label, damage: dmg })
    }
  }
  rows.sort((a, b) => b.damage - a.damage || a.label.localeCompare(b.label))
  return { range, total, rows }
}

function hideAnalysisTooltip() {
  const el = document.getElementById("analysisTooltip")
  if (!el) return
  el.style.display = "none"
}

function updateAnalysisTooltip(tab, geom, hover, labelKeys, options = {}) {
  const tooltip = document.getElementById("analysisTooltip")
  const canvas = document.getElementById("canvas")
  const wrap = document.getElementById("canvasWrap")
  if (!tooltip || !canvas || !wrap || !geom) return

  if (!hover || !labelKeys || !labelKeys.has(hover.key)) {
    tooltip.style.display = "none"
    return
  }

  const analysis = options.analysis || tab?.analysis
  const compareAnalysis = options.compareAnalysis || null
  const compareMode = options.compareMode || "normal"
  const breakdown = getAnalysisDamageBreakdown(analysis, hover.q, hover.r)
  if (!breakdown) {
    tooltip.style.display = "none"
    return
  }

  tooltip.innerHTML = ""
  const header = document.createElement("div")
  header.className = "analysisTooltipHeader"
  header.textContent = `Range: ${breakdown.range}`
  tooltip.appendChild(header)

  const total = document.createElement("div")
  total.className = "analysisTooltipLine"
  total.textContent = `Total: ${formatAnalysisCount(breakdown.total)}`
  tooltip.appendChild(total)

  if (compareAnalysis && compareMode === "difference") {
    const compareBreakdown = getAnalysisDamageBreakdown(compareAnalysis, hover.q, hover.r)
    if (compareBreakdown) {
      const other = document.createElement("div")
      other.className = "analysisTooltipLine"
      other.textContent = `Other: ${formatAnalysisCount(compareBreakdown.total)}`
      tooltip.appendChild(other)

      const delta = breakdown.total - compareBreakdown.total
      const deltaLine = document.createElement("div")
      deltaLine.className = "analysisTooltipLine"
      deltaLine.textContent = `Delta: ${formatSignedAnalysisCount(delta)}`
      tooltip.appendChild(deltaLine)
    }
  }

  if (!breakdown.rows.length) {
    const none = document.createElement("div")
    none.className = "analysisTooltipLine analysisTooltipMuted"
    none.textContent = "No damage at range"
    tooltip.appendChild(none)
  } else {
    const label = document.createElement("div")
    label.className = "analysisTooltipLine analysisTooltipMuted"
    label.textContent = "Breakdown:"
    tooltip.appendChild(label)

    for (const row of breakdown.rows) {
      const line = document.createElement("div")
      line.className = "analysisTooltipLine"
      line.textContent = `${row.label}: ${formatAnalysisCount(row.damage)}`
      tooltip.appendChild(line)
    }
  }

  const { size, axialToPixel } = geom
  const pos = axialToPixel(hover.q, hover.r)
  const canvasRect = canvas.getBoundingClientRect()
  const wrapRect = wrap.getBoundingClientRect()
  const offsetX = canvasRect.left - wrapRect.left
  const offsetY = canvasRect.top - wrapRect.top
  const paneOffsetX = Number(options.offsetX) || 0
  const paneOffsetY = Number(options.offsetY) || 0
  const paneWidth = Number(options.width) || geom.width
  const paneHeight = Number(options.height) || geom.height

  tooltip.style.display = "block"
  const tipRect = tooltip.getBoundingClientRect()
  const maxX = offsetX + paneOffsetX + paneWidth
  const maxY = offsetY + paneOffsetY + paneHeight

  let boxX = offsetX + paneOffsetX + pos.x + size * 0.9
  let boxY = offsetY + paneOffsetY + pos.y + size * 0.6
  if (boxX + tipRect.width > maxX - 6) boxX = offsetX + paneOffsetX + pos.x - tipRect.width - size * 0.6
  if (boxX < offsetX + paneOffsetX + 6) boxX = offsetX + paneOffsetX + 6
  if (boxY + tipRect.height > maxY - 6) boxY = offsetY + paneOffsetY + pos.y - tipRect.height - size * 0.6
  if (boxY < offsetY + paneOffsetY + 6) boxY = offsetY + paneOffsetY + 6

  tooltip.style.left = `${Math.round(boxX)}px`
  tooltip.style.top = `${Math.round(boxY)}px`
}

function renderAnalysisWeaponTables(container, analysis) {
  if (!container) return
  container.innerHTML = ""

  const defs = Object.values(analysis?.weaponTableDefs || {})
    .filter(def => def && Array.isArray(def.columns) && Array.isArray(def.rows))
    .filter(def => def.columns.length > 0 && def.rows.length > 0)
    .sort((a, b) => String(a.name || "").localeCompare(String(b.name || "")))

  if (defs.length === 0) return

  for (const def of defs) {
    const card = document.createElement("div")
    card.className = "analysisWeaponTable"

    const title = document.createElement("div")
    title.className = "analysisWeaponTableTitle"
    title.textContent = String(def.title || def.name || "Weapon Table")
    card.appendChild(title)

    const table = document.createElement("table")
    table.className = "analysisWeaponTableGrid"

    const thead = document.createElement("thead")
    const headRow = document.createElement("tr")
    for (const col of def.columns) {
      const th = document.createElement("th")
      th.textContent = String(col ?? "")
      headRow.appendChild(th)
    }
    thead.appendChild(headRow)
    table.appendChild(thead)

    const tbody = document.createElement("tbody")
    for (const rowData of def.rows) {
      const row = Array.isArray(rowData) ? rowData : []
      const tr = document.createElement("tr")
      const cellCount = Math.max(def.columns.length, row.length)
      for (let i = 0; i < cellCount; i++) {
        const text = String(row[i] ?? "")
        if (i === 0) {
          const th = document.createElement("th")
          th.scope = "row"
          th.textContent = text
          tr.appendChild(th)
        } else {
          const td = document.createElement("td")
          td.textContent = text
          tr.appendChild(td)
        }
      }
      tbody.appendChild(tr)
    }
    table.appendChild(tbody)

    card.appendChild(table)
    container.appendChild(card)
  }
}

function renderAnalysisPanel(tab) {
  const summary = document.getElementById("analysisSummary")
  const statsEl = document.getElementById("analysisShipStats")
  const list = document.getElementById("analysisArcList")
  const tablesEl = document.getElementById("analysisWeaponTables")
  const filterAll = document.getElementById("analysisFilterAll")
  const filterPhaser = document.getElementById("analysisFilterPhaser")
  const radiusInput = document.getElementById("analysisHexRadius")
  const compareClear = document.getElementById("analysisCompareClear")
  const compareModeNormal = document.getElementById("analysisCompareModeNormal")
  const compareModeDiff = document.getElementById("analysisCompareModeDiff")
  if (!summary || !list) return

  const analysis = tab?.analysis
  if (!analysis) {
    summary.textContent = "Load a ship to view analysis."
    if (statsEl) statsEl.textContent = ""
    list.innerHTML = ""
    if (tablesEl) tablesEl.innerHTML = ""
    return
  }

  const filter = analysis.filter || tab?.analysisFilter || "all"
  if (filterAll) filterAll.classList.toggle("damageActive", filter === "all")
  if (filterPhaser) filterPhaser.classList.toggle("damageActive", filter === "phaser")
  if (radiusInput) radiusInput.value = String(getAnalysisHexRadius(tab))
  if (compareClear) compareClear.disabled = !tab.analysisCompare
  const compareActive = Boolean(tab.analysisCompare)
  const compareMode = tab.analysisCompareMode || "normal"
  if (compareModeNormal) {
    compareModeNormal.disabled = !compareActive
    compareModeNormal.classList.toggle("damageActive", compareActive && compareMode === "normal")
  }
  if (compareModeDiff) {
    compareModeDiff.disabled = !compareActive
    compareModeDiff.classList.toggle("damageActive", compareActive && compareMode === "difference")
  }

  const compareSummary = tab.analysisCompare
    ? ` | Compare: ${tab.analysisCompareLabel || "Ship"} (W:${tab.analysisCompare.weaponCount}, A:${tab.analysisCompare.arcEntryCount}, M:${tab.analysisCompare.emptyArcCount})` +
      ` | Mode: ${compareMode === "difference" ? "Difference" : "Normal"}`
    : ""

  summary.textContent =
    `Weapons: ${analysis.weaponCount} | ` +
    `Arcs: ${analysis.arcEntryCount} | ` +
    `Missing arcs: ${analysis.emptyArcCount}` +
    ` | Radius: ${getAnalysisHexRadius(tab)} | ` +
    `Filter: ${filter === "phaser" ? "Phasers only" : "All weapons"}` +
    compareSummary

  if (statsEl) {
    const primaryStats = tab.analysisShipStats || null
    const compareStats = tab.analysisCompareStats || null
    const lines = []
    if (primaryStats) {
      const shieldLine = formatShieldCountsLine(primaryStats.shieldCounts)
      const boxLine = `Internal boxes: ${primaryStats.nonExternalBoxes}`
      if (tab.analysisCompare) {
        lines.push(`Primary - ${shieldLine} | ${boxLine}`)
      } else {
        lines.push(`${shieldLine} | ${boxLine}`)
      }
    }
    if (tab.analysisCompare && compareStats) {
      const compareLabel = tab.analysisCompareLabel || "Compare"
      const shieldLine = formatShieldCountsLine(compareStats.shieldCounts)
      const boxLine = `Internal boxes: ${compareStats.nonExternalBoxes}`
      lines.push(`${compareLabel} - ${shieldLine} | ${boxLine}`)
    }
    statsEl.textContent = lines.join("\n")
  }

  const weapons = Array.isArray(analysis.weaponHighlights)
    ? analysis.weaponHighlights
    : []
  weapons.sort((a, b) => {
    const nameA = String(a?.name || "").toLowerCase()
    const nameB = String(b?.name || "").toLowerCase()
    if (nameA !== nameB) return nameA.localeCompare(nameB)
    const desA = String(a?.designation || "").toLowerCase()
    const desB = String(b?.designation || "").toLowerCase()
    return desA.localeCompare(desB, undefined, { numeric: true })
  })
  const activeKey = tab.analysisHighlight && typeof tab.analysisHighlight === "object"
    ? tab.analysisHighlight.key
    : ""
  list.innerHTML = ""
  for (const weapon of weapons) {
    const row = document.createElement("div")
    row.className = "analysisRow"
    const canMapHighlight = analysisWeaponHasMapHighlight(weapon)
    const canSsdHighlight = analysisWeaponHasSsdGeometry(weapon)
    const canHighlight = canMapHighlight || canSsdHighlight
    if (activeKey && weapon?.key === activeKey) row.classList.add("active")
    if (canHighlight) {
      row.title = canMapHighlight
        ? "Click to highlight weapon arcs on map"
        : "No bearing on map; click to highlight this weapon on the SSD"
      row.onclick = () => {
        const next = weapon?.key === activeKey ? null : weapon
        tab.analysisHighlight = next
        renderAnalysisPanel(tab)
        renderCanvas()
      }
    } else {
      row.classList.add("disabled")
      row.title = "No known arcs to highlight"
    }

    const label = document.createElement("div")
    label.className = "analysisLabel"
    label.textContent = weapon?.label || "Weapon"

    const value = document.createElement("div")
    value.textContent = weapon?.arcText || ""

    row.appendChild(label)
    row.appendChild(value)
    list.appendChild(row)
  }

  if (analysis.unknownParts.size > 0) {
    const warn = document.createElement("div")
    warn.className = "hint"
    warn.textContent = `Unknown arcs: ${Array.from(analysis.unknownParts).join(", ")}`
    list.appendChild(warn)
  }

  renderAnalysisWeaponTables(tablesEl, analysis)
}

function renderMovementAnalysisPanel(tab) {
  const modeSelect = document.getElementById("movementAnalysisTurnMode")
  const speedInput = document.getElementById("movementAnalysisSpeed")
  const endingHeadingSelect = document.getElementById("movementAnalysisEndingHeading")
  const summary = document.getElementById("movementAnalysisSummary")
  const rangesEl = document.getElementById("movementAnalysisRanges")
  if (!modeSelect || !speedInput || !endingHeadingSelect || !summary || !rangesEl) return

  const table = tab?.turnMovementTable || null
  const movement = ensureMovementAnalysisState(tab)
  if (!movement) {
    summary.textContent = "Movement analysis is not available."
    rangesEl.textContent = ""
    return
  }

  const modeOptions = getTurnModeLetterOptions(table)
  if (!modeOptions.includes(movement.turnModeLetter)) {
    movement.turnModeLetter = modeOptions[0] || "D"
  }

  const optionKey = modeOptions.join("|")
  if (modeSelect.dataset.optionKey !== optionKey) {
    modeSelect.innerHTML = ""
    for (const mode of modeOptions) {
      const opt = document.createElement("option")
      opt.value = mode
      opt.textContent = mode
      modeSelect.appendChild(opt)
    }
    modeSelect.dataset.optionKey = optionKey
  }
  modeSelect.value = movement.turnModeLetter

  const desiredSpeed = movement.speed === "" ? "" : String(movement.speed)
  if (speedInput.value !== desiredSpeed) speedInput.value = desiredSpeed

  const endingOptions = getMovementEndingHeadingOptions()
  const endingOptionKey = endingOptions.map(opt => `${opt.value}:${opt.label}`).join("|")
  if (endingHeadingSelect.dataset.optionKey !== endingOptionKey) {
    endingHeadingSelect.innerHTML = ""
    for (const optInfo of endingOptions) {
      const opt = document.createElement("option")
      opt.value = optInfo.value
      opt.textContent = optInfo.label
      endingHeadingSelect.appendChild(opt)
    }
    endingHeadingSelect.dataset.optionKey = endingOptionKey
  }
  endingHeadingSelect.value = String(movement.endingHeading || "")

  if (!table || !Array.isArray(table.rows)) {
    summary.textContent = "Turn & Movement table could not be loaded."
    rangesEl.textContent = ""
    return
  }

  const rangeLines = []
  for (const row of table.rows) {
    const values = row && typeof row.values === "object" ? row.values : null
    const cell = values ? values[movement.turnModeLetter] : null
    if (!cell) continue
    const straightMoves = Number(row?.straightMovesBeforeTurn ?? row?.turnMode)
    const straightLabel = Number.isFinite(straightMoves)
      ? String(Math.trunc(straightMoves))
      : String(row?.straightMovesBeforeTurn ?? row?.turnMode ?? "?")
    rangeLines.push(`${straightLabel}: ${cell.raw || `${cell.min}-${cell.max}`}`)
  }
  rangesEl.textContent = rangeLines.length > 0
    ? `Turn Mode ${movement.turnModeLetter} ranges (straight moves -> speed)\n${rangeLines.join("\n")}`
    : `No ranges found for turn mode ${movement.turnModeLetter}.`

  const speedRaw = String(movement.speed || "").trim()
  if (!speedRaw) {
    summary.textContent = `Select a speed to calculate straight moves before turning for turn mode ${movement.turnModeLetter}.`
    return
  }

  const speed = Number(speedRaw)
  if (!Number.isFinite(speed) || speed < 0) {
    summary.textContent = "Enter a valid speed (0 or greater)."
    return
  }

  const match = findMovementTurnMatch(table, movement.turnModeLetter, speed)
  if (!match) {
    summary.textContent = `No turn entry for turn mode ${movement.turnModeLetter} at speed ${speedRaw}.`
    return
  }

  const straightMoves = Number(match.row?.straightMovesBeforeTurn ?? match.row?.turnMode)
  const straightLabel = Number.isFinite(straightMoves)
    ? String(Math.trunc(straightMoves))
    : String(match.row?.straightMovesBeforeTurn ?? match.row?.turnMode ?? "?")
  const moveWord = straightLabel === "1" ? "move" : "moves"
  const rangeLabel = match.cell?.raw || ""
  const lines = [
    `Turn mode ${match.letter} at speed ${speedRaw}: turn available after ${straightLabel} straight ${moveWord}.` +
      (rangeLabel ? ` (Matched speed range ${rangeLabel})` : "")
  ]
  const selectedEndingHeading = normalizeMovementHeadingIndex(movement.endingHeading)
  if (selectedEndingHeading !== null) {
    lines.push(`Ending direction filter: ${getMovementHeadingLabel(selectedEndingHeading)}.`)
  }

  const endpointResult = getMovementEndpointAnalysis(tab)
  if (endpointResult?.ok) {
    lines.push(
      `Endpoints-only plot: ${endpointResult.endpointCount} unique endpoints after ${endpointResult.plottedMoves} moves ` +
      `(assumed forward-facing start; sideslips included${selectedEndingHeading !== null ? `; ending facing = ${getMovementHeadingLabel(selectedEndingHeading)}` : ""}; see yellow triangle on map).`
    )
    if (endpointResult.capped && endpointResult.note) lines.push(endpointResult.note)
  }

  summary.textContent = lines.join("\n")
}

async function setAnalysisFilter(filter) {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "analysis" || !tab.doc) return
  const next = filter === "phaser" ? "phaser" : "all"
  if (tab.analysisFilter === next && tab.analysis) return

  tab.analysisFilter = next
  const weaponDamage = await readWeaponDamageJson()
  const modeSelections = tab.analysisModeSelections instanceof Map
    ? tab.analysisModeSelections
    : new Map()
  tab.analysis = buildWeaponAnalysis(tab.doc, weaponDamage, modeSelections, {
    weaponFilter: next,
    radius: tab.analysisRadius
  })
  if (tab.analysisCompare && tab.analysisCompareDoc) {
    const compareSelections = tab.analysisCompareModeSelections instanceof Map
      ? tab.analysisCompareModeSelections
      : new Map()
    tab.analysisCompare = buildWeaponAnalysis(tab.analysisCompareDoc, weaponDamage, compareSelections, {
      weaponFilter: next,
      radius: tab.analysisRadius
    })
  }
  tab.analysisHighlight = null
  tab.analysisHover = null
  tab.analysisCompareHover = null
  tab.analysisHoverSide = null
  tab.analysisHexLabels = new Set()
  tab.analysisCompareHexLabels = tab.analysisCompare ? new Set() : null

  renderAnalysisPanel(tab)
  renderCanvas()
}

async function setAnalysisRadius(value) {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "analysis" || !tab.doc) return
  const next = normalizeAnalysisHexRadius(value)
  if (tab.analysisRadius === next && tab.analysis) return

  tab.analysisRadius = next
  const weaponDamage = await readWeaponDamageJson()
  const modeSelections = tab.analysisModeSelections instanceof Map
    ? tab.analysisModeSelections
    : new Map()
  tab.analysis = buildWeaponAnalysis(tab.doc, weaponDamage, modeSelections, {
    weaponFilter: tab.analysisFilter,
    radius: tab.analysisRadius
  })
  if (tab.analysisCompare && tab.analysisCompareDoc) {
    const compareSelections = tab.analysisCompareModeSelections instanceof Map
      ? tab.analysisCompareModeSelections
      : new Map()
    tab.analysisCompare = buildWeaponAnalysis(tab.analysisCompareDoc, weaponDamage, compareSelections, {
      weaponFilter: tab.analysisFilter,
      radius: tab.analysisRadius
    })
  }
  tab.analysisHighlight = null
  tab.analysisHover = null
  tab.analysisCompareHover = null
  tab.analysisHoverSide = null
  tab.analysisHexLabels = new Set()
  tab.analysisCompareHexLabels = tab.analysisCompare ? new Set() : null

  renderAnalysisPanel(tab)
  renderCanvas()
}

async function loadAnalysisComparison() {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "analysis" || !tab.doc) return
  const picker = window.api.pickSuperluminalShip || window.api.pickSuperluminalInput
  if (!picker) {
    alert("Folder picker is not available.")
    return
  }
  const picked = await picker()
  if (!picked || !picked.ok) {
    alert(picked?.error || "Folder selection cancelled.")
    return
  }

  let doc = null
  try {
    doc = JSON.parse(picked.jsonText || "")
  } catch (e) {
    alert("JSON parse failed: " + e.message)
    return
  }

  if (!doc || typeof doc !== "object" || !doc.ssd || typeof doc.ssd !== "object") {
    alert("Weapon analysis comparison requires a SuperLuminal JSON with an ssd section.")
    return
  }

  const weaponDamage = await readWeaponDamageJson()
  const modeSelections = await collectWeaponModeSelections(doc, weaponDamage)
  if (modeSelections === null) return
  const externalLabelSet = await readExternalLabelSet()

  if (!tab.analysisCompareMode) tab.analysisCompareMode = "normal"
  tab.analysisCompareModeSelections = modeSelections
  tab.analysisCompareDoc = doc
  tab.analysisCompareLabel = getShipLabelFromDoc(doc, picked.inputBase || picked.jsonFile || "Comparison")
  tab.analysisCompare = buildWeaponAnalysis(doc, weaponDamage, modeSelections, {
    weaponFilter: tab.analysisFilter,
    radius: tab.analysisRadius
  })
  tab.analysisCompareStats = computeShipAnalysisStats(doc, externalLabelSet)
  tab.analysisCompareHexLabels = new Set()
  tab.analysisCompareHover = null
  tab.analysisHoverSide = null

  renderAnalysisPanel(tab)
  renderCanvas()
}

function setAnalysisCompareMode(mode) {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "analysis") return
  if (!tab.analysisCompare) return
  const next = mode === "difference" ? "difference" : "normal"
  if (tab.analysisCompareMode === next) return
  tab.analysisCompareMode = next
  tab.analysisHover = null
  tab.analysisCompareHover = null
  tab.analysisHoverSide = null
  hideAnalysisTooltip()
  renderAnalysisPanel(tab)
  renderCanvas()
}

function clearAnalysisComparison() {
  const tab = getActiveTab()
  if (!tab || tab.uiState !== "analysis") return
  if (!tab.analysisCompare) return
  tab.analysisCompare = null
  tab.analysisCompareDoc = null
  tab.analysisCompareLabel = null
  tab.analysisCompareModeSelections = null
  tab.analysisCompareStats = null
  tab.analysisCompareHexLabels = null
  tab.analysisCompareHover = null
  tab.analysisHoverSide = null
  tab.analysisCompareMode = "normal"
  hideAnalysisTooltip()
  renderAnalysisPanel(tab)
  renderCanvas()
}

function normalizeMissingShipPath(pathText) {
  return String(pathText || "")
    .trim()
    .replace(/[\\/]+/g, "\\")
}

function toCount(value, fallback = 0) {
  const n = Number(value)
  if (!Number.isFinite(n) || n < 0) return Math.max(0, Number(fallback) || 0)
  return Math.floor(n)
}

function buildMissingSuperluminalModel(res) {
  const normalizedMissing = Array.isArray(res?.missing)
    ? res.missing
      .map((item) => normalizeMissingShipPath(item))
      .filter(Boolean)
    : []

  const uniqueMissing = Array.from(new Set(normalizedMissing))
    .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }))

  const missingCount = uniqueMissing.length
  const veilCount = toCount(res?.veilCount, missingCount)
  const superluminalCount = toCount(res?.superluminalCount, 0)
  const matchedCount = Math.max(0, veilCount - missingCount)
  const coverageRatio = veilCount > 0 ? matchedCount / veilCount : 1
  const coveragePercent = Math.round(Math.min(1, Math.max(0, coverageRatio)) * 100)

  const byGroup = new Map()
  for (const relPath of uniqueMissing) {
    const parts = relPath.split("\\").filter(Boolean)
    const groupKey = parts.length > 1 ? parts[0] : "(Root)"
    byGroup.set(groupKey, (byGroup.get(groupKey) || 0) + 1)
  }

  const groups = Array.from(byGroup.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name, undefined, { sensitivity: "base" }))

  return {
    veilShipsDir: String(res?.veilShipsDir || "").trim() || "(unknown)",
    superluminalShipsDir: String(res?.superluminalShipsDir || "").trim() || "(unknown)",
    veilCount,
    superluminalCount,
    missingCount,
    matchedCount,
    coveragePercent,
    groups,
    missingPaths: uniqueMissing
  }
}

function buildMissingSuperluminalReportText(model) {
  const lines = [
    `Missing from SuperLuminal: ${model.missingCount}`,
    `Matched: ${model.matchedCount}`,
    `Coverage: ${model.coveragePercent}%`,
    `Veil ships: ${model.veilCount}`,
    `SuperLuminal ships: ${model.superluminalCount}`,
    "",
    `Veil folder: ${model.veilShipsDir}`,
    `SuperLuminal folder: ${model.superluminalShipsDir}`,
    "",
    "Missing ship folders:",
    ...(model.missingPaths.length > 0 ? model.missingPaths : ["(none)"])
  ]
  return lines.join("\n")
}

async function copyTextToClipboardSafe(text) {
  const value = String(text || "")
  if (!value) return false

  if (navigator?.clipboard && typeof navigator.clipboard.writeText === "function") {
    try {
      await navigator.clipboard.writeText(value)
      return true
    } catch {}
  }

  try {
    const temp = document.createElement("textarea")
    temp.value = value
    temp.setAttribute("readonly", "readonly")
    temp.style.position = "fixed"
    temp.style.opacity = "0"
    temp.style.pointerEvents = "none"
    document.body.appendChild(temp)
    temp.focus()
    temp.select()
    const copied = document.execCommand("copy")
    document.body.removeChild(temp)
    return copied
  } catch {
    return false
  }
}

function showMissingSuperluminalResultsModal(model) {
  const overlay = document.createElement("div")
  overlay.className = "modalOverlay missingShipsOverlay"

  const card = document.createElement("div")
  card.className = "modalCard missingShipsCard"

  const header = document.createElement("div")
  header.className = "missingShipsHeader"

  const headerLeft = document.createElement("div")
  const title = document.createElement("div")
  title.className = "modalTitle"
  title.textContent = "Veil vs SuperLuminal Comparison"
  const desc = document.createElement("div")
  desc.className = "modalDesc missingShipsDesc"
  desc.textContent = "Missing ship folders from Veil that are not yet in SuperLuminal."
  const pathLineA = document.createElement("div")
  pathLineA.className = "missingShipsPath"
  pathLineA.textContent = `Veil: ${model.veilShipsDir}`
  const pathLineB = document.createElement("div")
  pathLineB.className = "missingShipsPath"
  pathLineB.textContent = `SuperLuminal: ${model.superluminalShipsDir}`
  headerLeft.appendChild(title)
  headerLeft.appendChild(desc)
  headerLeft.appendChild(pathLineA)
  headerLeft.appendChild(pathLineB)

  const closeTop = document.createElement("button")
  closeTop.className = "missingShipsCloseBtn"
  closeTop.textContent = "Close"

  header.appendChild(headerLeft)
  header.appendChild(closeTop)

  const stats = document.createElement("div")
  stats.className = "missingShipsStats"
  const makeStat = (label, value, tone = "") => {
    const box = document.createElement("div")
    box.className = `missingShipsStat ${tone}`.trim()
    const v = document.createElement("div")
    v.className = "missingShipsStatValue"
    v.textContent = String(value)
    const l = document.createElement("div")
    l.className = "missingShipsStatLabel"
    l.textContent = label
    box.appendChild(v)
    box.appendChild(l)
    return box
  }
  stats.appendChild(makeStat("Veil Ship Folders", model.veilCount))
  stats.appendChild(makeStat("SuperLuminal Folders", model.superluminalCount))
  stats.appendChild(makeStat("Missing", model.missingCount, model.missingCount > 0 ? "isAlert" : "isGood"))

  const coverage = document.createElement("div")
  coverage.className = "missingShipsCoverage"
  const coverageLabel = document.createElement("div")
  coverageLabel.className = "missingShipsCoverageLabel"
  coverageLabel.textContent = `Coverage: ${model.coveragePercent}% (${model.matchedCount}/${model.veilCount} matched)`
  const coverageMeter = document.createElement("div")
  coverageMeter.className = "missingShipsCoverageMeter"
  const coverageFill = document.createElement("div")
  coverageFill.className = "missingShipsCoverageFill"
  if (model.coveragePercent >= 80) coverageFill.classList.add("isGood")
  else if (model.coveragePercent <= 40) coverageFill.classList.add("isAlert")
  coverageFill.style.width = `${Math.max(0, Math.min(100, model.coveragePercent))}%`
  coverageMeter.appendChild(coverageFill)
  coverage.appendChild(coverageLabel)
  coverage.appendChild(coverageMeter)

  const body = document.createElement("div")
  body.className = "missingShipsBody"

  const left = document.createElement("div")
  left.className = "missingShipsLeft"
  const groupTitle = document.createElement("div")
  groupTitle.className = "missingShipsSectionTitle"
  groupTitle.textContent = "By Faction/Folder"
  const groupWrap = document.createElement("div")
  groupWrap.className = "missingShipsGroups"
  if (model.groups.length === 0) {
    const none = document.createElement("div")
    none.className = "missingShipsHint"
    none.textContent = "No missing folders."
    groupWrap.appendChild(none)
  } else {
    for (const group of model.groups) {
      const chip = document.createElement("div")
      chip.className = "missingShipsGroupChip"
      chip.textContent = `${group.name}: ${group.count}`
      groupWrap.appendChild(chip)
    }
  }

  const rawTitle = document.createElement("div")
  rawTitle.className = "missingShipsSectionTitle"
  rawTitle.textContent = "Raw List"
  const rawList = document.createElement("textarea")
  rawList.className = "missingShipsRaw"
  rawList.readOnly = true
  rawList.value = model.missingPaths.length > 0 ? model.missingPaths.join("\n") : "(none)"
  left.appendChild(groupTitle)
  left.appendChild(groupWrap)
  left.appendChild(rawTitle)
  left.appendChild(rawList)

  const right = document.createElement("div")
  right.className = "missingShipsRight"
  const listTitle = document.createElement("div")
  listTitle.className = "missingShipsSectionTitle"
  listTitle.textContent = "Missing Ship Folder List"
  const filterRow = document.createElement("div")
  filterRow.className = "missingShipsFilterRow"
  const filterInput = document.createElement("input")
  filterInput.type = "text"
  filterInput.placeholder = "Filter missing folder names..."
  filterInput.className = "missingShipsFilterInput"
  const listCount = document.createElement("div")
  listCount.className = "missingShipsListCount"
  filterRow.appendChild(filterInput)
  filterRow.appendChild(listCount)
  const list = document.createElement("ul")
  list.className = "missingShipsList"

  const renderList = () => {
    const query = String(filterInput.value || "").trim().toLowerCase()
    const items = query
      ? model.missingPaths.filter((item) => item.toLowerCase().includes(query))
      : model.missingPaths

    list.innerHTML = ""
    listCount.textContent = `${items.length} shown`

    if (items.length === 0) {
      const li = document.createElement("li")
      li.className = "missingShipsListEmpty"
      li.textContent = "No matching folders."
      list.appendChild(li)
      return
    }

    for (let i = 0; i < items.length; i++) {
      const li = document.createElement("li")
      li.className = "missingShipsItem"
      const index = document.createElement("span")
      index.className = "missingShipsItemIndex"
      index.textContent = `${i + 1}.`
      const name = document.createElement("span")
      name.className = "missingShipsItemName"
      name.textContent = items[i]
      li.appendChild(index)
      li.appendChild(name)
      list.appendChild(li)
    }
  }
  filterInput.oninput = () => renderList()
  renderList()

  right.appendChild(listTitle)
  right.appendChild(filterRow)
  right.appendChild(list)

  body.appendChild(left)
  body.appendChild(right)

  const actions = document.createElement("div")
  actions.className = "modalActions missingShipsActions"
  const copyListBtn = document.createElement("button")
  copyListBtn.textContent = "Copy List"
  const copyReportBtn = document.createElement("button")
  copyReportBtn.textContent = "Copy Report"
  const closeBtn = document.createElement("button")
  closeBtn.textContent = "Close"
  actions.appendChild(copyListBtn)
  actions.appendChild(copyReportBtn)
  actions.appendChild(closeBtn)

  card.appendChild(header)
  card.appendChild(stats)
  card.appendChild(coverage)
  card.appendChild(body)
  card.appendChild(actions)
  overlay.appendChild(card)
  document.body.appendChild(overlay)

  const cleanup = () => {
    window.removeEventListener("keydown", onKey)
    if (overlay.parentElement) overlay.parentElement.removeChild(overlay)
  }

  const close = () => cleanup()
  const onKey = (e) => {
    if (e.key === "Escape") close()
  }

  window.addEventListener("keydown", onKey)
  overlay.onclick = (e) => {
    if (e.target === overlay) close()
  }
  closeTop.onclick = () => close()
  closeBtn.onclick = () => close()

  copyListBtn.onclick = async () => {
    const ok = await copyTextToClipboardSafe(rawList.value)
    if (ok) showTempMessage("Missing list copied.")
    else alert("Could not copy list to clipboard.")
  }

  copyReportBtn.onclick = async () => {
    const ok = await copyTextToClipboardSafe(buildMissingSuperluminalReportText(model))
    if (ok) showTempMessage("Comparison report copied.")
    else alert("Could not copy report to clipboard.")
  }

  if (model.missingCount > 0) filterInput.focus()
  else closeBtn.focus()
}

async function showMissingSuperluminalShips() {
  if (!window.api || typeof window.api.shipyardListMissingSuperluminalShips !== "function") {
    alert("Missing-ship scan is not available.")
    return
  }

  const res = await window.api.shipyardListMissingSuperluminalShips()
  if (!res || !res.ok) {
    alert(res?.error || "Failed to compare Veil and SuperLuminal ship folders.")
    return
  }

  const model = buildMissingSuperluminalModel(res)
  showMissingSuperluminalResultsModal(model)
  if (model.missingCount === 0) showTempMessage("No missing ships found.")
}

/* -----------------------------
   Shipyard flow
----------------------------- */
async function runShipyard() {
  setAddSquareMode(false)
  setAddGroupMode(false)
  // Make sure no lingering overlays or disabled canvas states block interaction.
  document.querySelectorAll(".modalOverlay").forEach(el => el.remove())
  const canvasSafe = document.getElementById("canvas")
  if (canvasSafe) canvasSafe.style.pointerEvents = "auto"

  const folderPick = await window.api.shipyardPickFolderNamed()
  if (!folderPick) return
  if (!folderPick.ok) {
    alert(folderPick.error || "Folder pick failed.")
    return
  }

  const loaded = await window.api.shipyardLoadFolderDataset(folderPick.folderPath)
  if (!loaded || !loaded.ok) {
    alert(loaded?.error || "Failed to load dataset from folder.")
    return
  }

  let doc = null
  try {
    doc = JSON.parse(loaded.jsonText)
  } catch (e) {
    alert("JSON parse failed: " + e.message)
    return
  }

  closeAllTabsForNewTask()
  const tab = ensureEmptyActiveTab()
  if (!tab) return

  applyLoadedDatasetToTab(tab, loaded, doc)

  const folderName = baseNameFromPath(folderPick.folderPath) || "Ship"
  const defaultShipName = stripVeilMarkerName(folderName) || folderName

  // Mark this tab as Shipyard so the left Ship Controls area stays hidden.
  tab.uiState = "shipyard"
  tab.shipName = ensureVeilMarkerName(defaultShipName || folderName)
  tab.title = tab.shipName

  // Ensure normal selection mode during shipyard
  setDamageMode(false)
  setWeaponMode(false)
  setSystemMode(false)

  await ensureTabImageLoaded(tab)
  renderTabs()
  renderCanvas()
  renderCurrentProps()

  const shipNameChoice = await shipyardAskShipName(defaultShipName)
  if (shipNameChoice.action === "cancel") {
    closeTabFromShipyard()
    return
  }
  const chosenShipName =
    shipNameChoice.action === "rename"
      ? String(shipNameChoice.name || "").trim()
      : defaultShipName
  const targetFolderName = ensureVeilMarkerName(chosenShipName || folderName)
  tab.shipName = targetFolderName
  tab.title = targetFolderName
  renderTabs()
  renderCurrentProps()

  const jsonStem = fileStemFromPath(loaded.jsonPath)
  const imageStem = loaded.imagePath ? fileStemFromPath(loaded.imagePath) : ""
  const currentShipName = String(doc?.shipName || "").trim()
    const needsRename =
      (jsonStem && jsonStem !== targetFolderName) ||
      (imageStem && imageStem !== targetFolderName) ||
      (currentShipName && currentShipName !== targetFolderName) ||
      (!currentShipName && targetFolderName)
    const needsMarkerRename = targetFolderName !== folderName

    if (needsRename || needsMarkerRename) {
      const prepared = await window.api.shipyardRenameFolderAndFiles(
        targetFolderName,
        folderPick.folderPath
      )
      if (!prepared || !prepared.ok) {
        hideShipyardPanel()
        alert(prepared?.error || "Failed to rename folder/files.")
      return
    }

    try {
      doc = JSON.parse(prepared.jsonText)
    } catch (e) {
      hideShipyardPanel()
      alert("JSON parse failed: " + e.message)
      return
    }

    tab.shipName = prepared.shipName
    tab.title = prepared.shipName
    tab.imageDataUrl = prepared.imageDataUrl
    tab.imageObj = null
    tab.jsonPath = prepared.jsonPath
    tab.doc = doc
    tab.selected = null
    fixLegacyCaretLabels(tab.doc)
    } else {
      if (tab.doc && typeof tab.doc === "object") {
        tab.doc.shipName = targetFolderName
      }
      tab.shipName = targetFolderName
      tab.title = targetFolderName
    }

  await ensureTabImageLoaded(tab)
  renderTabs()
  renderCanvas()
  renderCurrentProps()

  const labelsRes = await window.api.shipyardReadBoxes()
  if (!labelsRes || !labelsRes.ok) {
    hideShipyardPanel()
    alert(labelsRes?.error || "Failed to read Data.json")
    return
  }

  const labelOptions = labelsRes.options
  tab.heavyLabelSet = new Set()
  tab.boxLabelOptions = labelOptions.slice()
  if (labelOptions.length === 0) {
    hideShipyardPanel()
    alert("Data.json has no label options.")
    return
  }

  // Load Data.json to mark heavy weapons.
  const heavyRes = await window.api.readHeavyList?.()
  if (heavyRes && heavyRes.ok && Array.isArray(heavyRes.options)) {
    for (const opt of heavyRes.options) {
      const clean = normalizeLabel(opt).clean.toLowerCase()
      if (clean) tab.heavyLabelSet.add(clean)
    }
  }
  applyHeavyLabelsFromFile(tab)


  // Ensure normal click selection (no special modes) during shipyard review.
  setDamageMode(false)
  setWeaponMode(false)
  setSystemMode(false)

  const squaresNeedingLabels = () => {
    return (tab.doc.squares || []).filter(
      s => s.groupId === null && (!isSquareLabeled(s) || s._inheritedFromGroupLabel)
    )
  }

  const unlabeledGroups = (tab.doc.groups || []).filter(g => !isGroupLabeled(g))
  const initialSquares = squaresNeedingLabels()
  const total = unlabeledGroups.length + initialSquares.length

  if (total === 0) {
    showShipyardDoneStep("Nothing to label. You can still review properties and save.")
  } else {
    showShipyardLabelStep()

    let done = 0

    for (const g of unlabeledGroups) {
      const stillExists = (tab.doc.groups || []).some(x => x.id === g.id)
      if (!stillExists) continue

      done += 1
      tab.selected = { kind: "group", id: g.id }
      renderCanvas()
      renderCurrentProps()

      const res = await shipyardPickLabel(`Label Group ${g.id} (${done}/${total})`, labelOptions, "group")

      if (res.action === "cancel") {
        closeTabFromShipyard()
        return
      }

      if (res.action === "remove") {
        const ok = confirm(`Remove Group ${g.id}?\n\nThis will also remove ALL squares inside it.`)
        if (ok) removeGroupById(tab, g.id)
        continue
      }

      if (res.action === "apply") {
        applyGroupLabel(g, res.value)
        for (const s of tab.doc.squares || []) {
          if (s.groupId === g.id) {
            applySquareLabel(s, g.label || g.name || res.value, tab)
            s._inheritedFromGroupLabel = true
          }
        }
      }
    }

    for (const s of squaresNeedingLabels()) {
      const stillExists = (tab.doc.squares || []).some(x => x.id === s.id)
      if (!stillExists) continue

      done += 1
      tab.selected = { kind: "square", id: s.id }
      renderCanvas()
      renderCurrentProps()

      const res = await shipyardPickLabel(`Label Square ${s.id} (${done}/${total})`, labelOptions, "square")

      if (res.action === "cancel") {
        closeTabFromShipyard()
        return
      }

      if (res.action === "remove") {
        const ok = confirm(`Remove Square ${s.id}?`)
        if (ok) removeSquareById(tab, s.id)
        continue
      }

      if (res.action === "apply") applySquareLabel(s, res.value, tab)
    }

    // Let the user freely select any box before saving.
    tab.selected = null
    const canvas = document.getElementById("canvas")
    if (canvas) canvas.style.pointerEvents = "auto"
    // Ensure no modal overlays remain
    document.querySelectorAll(".modalOverlay").forEach(el => el.parentElement?.removeChild(el))
    renderCanvas()
    renderCurrentProps()
    showShipyardDoneStep("All unlabeled items processed. Click the image to select, review properties, then Save JSON.")
  }

  while (true) {
    const action = await shipyardDoneAskSave()

    if (action.action === "close") {
      hideShipyardPanel()
      renderCanvas()
      renderCurrentProps()
      return
    }

    if (action.action === "save") {
      const okHeavy = await ensureHeavyDesignations(tab)
      if (!okHeavy) {
        alert("Save cancelled until heavy weapon designations are provided.")
        continue
      }
      finalizeHeavyLetters(tab)
      if (!tab.jsonPath) {
        alert("Cannot save: JSON path is missing.")
        continue
      }

      const jsonText = JSON.stringify(tab.doc, null, 2)
      const res = await window.api.saveJson(tab.jsonPath, jsonText)

      if (res && res.ok) {
        closeTabFromShipyard()
        return
      }

      alert(res?.error || "Save failed.")
    }
  }
}

/* -----------------------------
   Events
----------------------------- */
function wireEvents() {
  const bindClick = (id, fn) => {
    const el = document.getElementById(id)
    if (el) el.onclick = fn
  }


  bindClick("btnShipyard", async () => {
    try {
      await runShipyard()
    } catch (e) {
      hideShipyardPanel()
      alert("Shipyard error: " + e.message)
    }
  })

  bindClick("btnEditJson", async () => {
    try {
      await runJsonEditor()
    } catch (e) {
      alert(e?.message || "JSON editor failed.")
    }
  })

  bindClick("btnShipyardCancel", () => {
    const tab = getActiveTab()
    if (!tab || tab.uiState !== "shipyard") {
      alert("No active Shipyard session to cancel.")
      return
    }
    closeTabFromShipyard()
  })

  bindClick("shipyardAddSquare", () => {
    setAddSquareMode(!state.addSquareMode)
    if (state.addSquareMode) {
      setAddGroupMode(false)
      showTempMessage("Click the image to place a new square.")
    }
  })

  bindClick("shipyardAddGroup", () => {
    setAddGroupMode(!state.addGroupMode)
    if (state.addGroupMode) {
      setAddSquareMode(false)
      showTempMessage("Select two squares to group.")
    }
  })

  bindClick("jsonEditAddBox", async () => {
    const tab = getActiveTab()
    if (!tab || tab.uiState !== "jsonEdit") {
      alert("No active JSON edit session.")
      return
    }
    const next = !state.addSquareMode
    setAddSquareMode(next)
    if (state.addSquareMode) {
      setAddGroupMode(false)
      if (tab.jsonEditFormat === "superluminal") {
        let spec = null
        try {
          spec = await resolveJsonEditAddBoxSpec(tab, { allowPrompt: true, forcePrompt: true })
        } catch (e) {
          setAddSquareMode(false)
          alert(e?.message || "Failed to start Add Box.")
          return
        }
        if (!spec || !spec.ssdKey) {
          setAddSquareMode(false)
          return
        }
        showTempMessage(`Click the image to add ${spec.label || "a new box"} to ${formatJsonEditGroupName(spec.ssdKey)}.`)
      } else {
        showTempMessage("Click the image to place a brand-new box.")
      }
    } else if (tab.jsonEditFormat === "superluminal") {
      const selectedKey = getSelectedJsonEditSsdKey(tab)
      if (selectedKey) state.jsonEditAddBoxSsdKey = selectedKey
      state.jsonEditAddBoxSpec = null
    }
  })

  bindClick("jsonEditSave", async () => {
    try {
      await saveJsonEditDoc()
    } catch (e) {
      alert(e?.message || "Save failed.")
    }
  })

  bindClick("btnConvertSSD", async () => {
    if (!window.api || typeof window.api.convertSSDJson !== "function") {
      alert("SSD conversion is not available.")
      return
    }

    try {
      if (!window.api.pickSuperluminalInput) {
        alert("Folder picker is not available.")
        return
      }
      const picked = await window.api.pickSuperluminalInput()
      if (!picked || !picked.ok) {
        alert(picked?.error || "Folder selection cancelled.")
        return
      }
      await enterConvertMode(picked)
    } catch (e) {
      alert(e?.message || "SSD conversion failed.")
    }
  })

  bindClick("btnMissingSuperluminal", async () => {
    try {
      await showMissingSuperluminalShips()
    } catch (e) {
      alert(e?.message || "Missing-ship scan failed.")
    }
  })

  bindClick("btnWeaponAnalysis", async () => {
    try {
      const picker = window.api.pickSuperluminalShip || window.api.pickSuperluminalInput
      if (!picker) {
        alert("Folder picker is not available.")
        return
      }
      const picked = await picker()
      if (!picked || !picked.ok) {
        alert(picked?.error || "Folder selection cancelled.")
        return
      }

      let doc = null
      try {
        doc = JSON.parse(picked.jsonText || "")
      } catch (e) {
        alert("JSON parse failed: " + e.message)
        return
      }

      if (!doc || typeof doc !== "object" || !doc.ssd || typeof doc.ssd !== "object") {
        alert("Weapon analysis requires a SuperLuminal JSON with an ssd section.")
        return
      }

      await enterAnalysisMode(picked, doc)
    } catch (e) {
      alert(e?.message || "Weapon analysis failed.")
    }
  })

  bindClick("btnMovementAnalysis", async () => {
    try {
      await enterMovementAnalysisMode()
    } catch (e) {
      alert(e?.message || "Movement analysis failed.")
    }
  })

  bindClick("convertCancel", () => {
    exitConvertMode()
  })

  bindClick("convertConfirm", async () => {
    if (!state.convertContext) return
    const shipData = collectConvertFields()
    const designationAssignments = {}
    const arcAssignments = {}
    const rankValueAssignments = {}
    state.convertMenuHidden = true
    renderCurrentProps()
    try {
      while (true) {
        const res = await window.api.convertSSDJson({
          ...state.convertContext,
          shipData,
          designationAssignments,
          arcAssignments,
          rankValueAssignments
        })

        if (res?.needsDesignations?.items?.length) {
          const ok = await ensureDesignationAssignments(res.needsDesignations.items, designationAssignments)
          if (!ok) {
            state.convertMenuHidden = false
            renderCurrentProps()
            return
          }
          continue
        }

        if (res?.needsArcs?.items?.length) {
          const ok = await ensureArcAssignments(res.needsArcs.items, arcAssignments)
          if (!ok) {
            state.convertMenuHidden = false
            renderCurrentProps()
            return
          }
          continue
        }

        if (res?.needsRankValues?.items?.length) {
          const ok = await ensureRankValueAssignments(res.needsRankValues.items, rankValueAssignments)
          if (!ok) {
            state.convertMenuHidden = false
            renderCurrentProps()
            return
          }
          continue
        }

        if (!res || !res.ok) {
          state.convertMenuHidden = false
          renderCurrentProps()
          alert(res?.error || "SSD conversion failed.")
          return
        }

          showTempMessage("Converted SSD saved.")
          const warningText = res?.warning ? `\n\n${res.warning}` : ""
          alert(`Converted SSD saved:\n${res.path}${warningText}`)
          exitConvertMode({ clearFields: true, resetTab: true })
          return
        }
    } catch (e) {
      state.convertMenuHidden = false
      renderCurrentProps()
      alert(e?.message || "SSD conversion failed.")
    }
  })

  bindClick("analysisClose", () => {
    closeActiveTab()
  })
  bindClick("movementAnalysisClose", () => {
    closeActiveTab()
  })
  bindClick("jsonEditClose", () => {
    setAddSquareMode(false)
    state.jsonEditAddBoxSsdKey = ""
    state.jsonEditAddBoxSpec = null
    closeActiveTab()
  })
  bindClick("analysisFilterAll", () => {
    setAnalysisFilter("all")
  })
  bindClick("analysisFilterPhaser", () => {
    setAnalysisFilter("phaser")
  })
  const analysisRadiusInput = document.getElementById("analysisHexRadius")
  if (analysisRadiusInput) {
    analysisRadiusInput.onchange = () => {
      setAnalysisRadius(analysisRadiusInput.value)
    }
  }
  const movementModeInput = document.getElementById("movementAnalysisTurnMode")
  if (movementModeInput) {
    movementModeInput.onchange = () => {
      setMovementAnalysisInput("turnModeLetter", movementModeInput.value)
    }
  }
  const movementSpeedInput = document.getElementById("movementAnalysisSpeed")
  if (movementSpeedInput) {
    const syncMovementSpeed = () => {
      setMovementAnalysisInput("speed", movementSpeedInput.value)
    }
    movementSpeedInput.oninput = syncMovementSpeed
    movementSpeedInput.onchange = syncMovementSpeed
  }
  const movementEndingHeadingInput = document.getElementById("movementAnalysisEndingHeading")
  if (movementEndingHeadingInput) {
    movementEndingHeadingInput.onchange = () => {
      setMovementAnalysisInput("endingHeading", movementEndingHeadingInput.value)
    }
  }
  bindClick("analysisCompareLoad", async () => {
    try {
      await loadAnalysisComparison()
    } catch (e) {
      alert(e?.message || "Weapon comparison failed.")
    }
  })
  bindClick("analysisCompareClear", () => {
    clearAnalysisComparison()
  })
  bindClick("analysisCompareModeNormal", () => {
    setAnalysisCompareMode("normal")
  })
  bindClick("analysisCompareModeDiff", () => {
    setAnalysisCompareMode("difference")
  })

  const btnEditSetup = document.getElementById("btnEditSetup")
  if (btnEditSetup) {
    btnEditSetup.onclick = async () => {
      try {
        const res = await window.api.editSetupPaths()
        if (!res) return
        if (!res.ok) {
          alert(res.error || "Failed to update setup paths.")
          return
        }
        alert("Setup paths saved.")
      } catch (e) {
        alert(e?.message || "Failed to update setup paths.")
      }
    }
  }

  const btnRecomputeGroups = document.getElementById("btnRecomputeGroups")
  if (btnRecomputeGroups) {
    btnRecomputeGroups.onclick = () => {
      renderCanvas()
      renderCurrentProps()
    }
  }

  const btnShipyardEditLabel = document.getElementById("shipyardEditLabel")
  if (btnShipyardEditLabel) {
    btnShipyardEditLabel.onclick = () => editSelectedLabelShipyard()
  }

  const canvas = document.getElementById("canvas")
  canvas.addEventListener("click", async (e) => {
    const tab = getActiveTab()
    if (!tab || !tab.doc) return

    if (tab.uiState === "analysis") {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const pane = getAnalysisPaneAtPoint(tab, rect, x, y)
      if (!pane.side || !pane.rect) return
      if (pane.side === "compare" && !tab.analysisCompare) return
      if (analysisPaneUsesSsdPreview(tab, pane.side)) return
      const localX = x - pane.rect.x
      const localY = y - pane.rect.y
      const radius = pane.side === "compare"
        ? normalizeAnalysisHexRadius(tab.analysisCompare?.radius ?? tab.analysisRadius)
        : getAnalysisHexRadius(tab)
      const hex = analysisPointToHex(pane.rect, localX, localY, radius)
      if (hex) {
        if (pane.side === "compare") {
          if (!tab.analysisCompareHexLabels) tab.analysisCompareHexLabels = new Set()
          const key = `${hex.q},${hex.r}`
          if (tab.analysisCompareHexLabels.has(key)) tab.analysisCompareHexLabels.delete(key)
          else tab.analysisCompareHexLabels.add(key)
        } else {
          if (!tab.analysisHexLabels) tab.analysisHexLabels = new Set()
          const key = `${hex.q},${hex.r}`
          if (tab.analysisHexLabels.has(key)) tab.analysisHexLabels.delete(key)
          else tab.analysisHexLabels.add(key)
        }
        renderCanvas()
      }
      return
    }

    if (tab.uiState === "movementAnalysis") {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const radius = getMovementEndpointMapRadius(tab)
      const hex = analysisPointToHex(rect, x, y, radius)
      if (!hex) return
      selectMovementEndpointPath(tab, hex.q, hex.r)
      return
    }

    if (isShipyardLabelVisible()) {
      return
    }

    // Energy locked? Still allow phaser/system recalcs
    const { ix, iy } = canvasToImagePoint(tab, e.clientX, e.clientY)

    if (state.addGroupMode) {
      const s = hitTestSquare(tab, ix, iy)
      if (!s) {
        showTempMessage("Select a square.")
        return
      }
      if (state.addGroupSeedId === null) {
        state.addGroupSeedId = s.id
        tab.selected = { kind: "square", id: s.id }
        showTempMessage("Select a second square.")
        renderCanvas()
        renderCurrentProps()
        return
      }
      const first = (tab.doc.squares || []).find(x => Number(x.id) === Number(state.addGroupSeedId))
      if (!first) {
        state.addGroupSeedId = null
        return
      }
      addSquaresToGroup(tab, first, s)
      state.addGroupSeedId = null
      tab.selected = { kind: "square", id: s.id }
      renderCanvas()
      renderCurrentProps()
      return
    }

    if (state.addSquareMode) {
      if (tab.uiState === "jsonEdit") await addJsonEditSquareAtPoint(tab, ix, iy)
      else addSquareAtPoint(tab, ix, iy)
      renderCanvas()
      renderCurrentProps()
      return
    }

    if (state.damageMode) {
      const s = hitTestSquare(tab, ix, iy)
      if (s) {
        s.damaged = true
        tab.selected = { kind: "square", id: s.id }
        refreshEnergyAfterMarkerChange(tab)
        renderCanvas()
        renderCurrentProps()
      }
      return
    }

    if (state.refitMode) {
      const s = hitTestSquare(tab, ix, iy)
      if (s) {
        delete s.damaged
        s.refitRemoved = true
        tab.selected = { kind: "square", id: s.id }
        refreshEnergyAfterMarkerChange(tab)
        renderCanvas()
        renderCurrentProps()
      }
      return
    }

    if (state.systemMode) {
      const s = hitTestSquare(tab, ix, iy)
      if (s) {
        const name = getSquareDisplayName(s).toLowerCase()
        let phaserUsage = s.phaserCapUsage

        if (name.startsWith("phaser")) {
          if (name.includes("phaser 3")) {
            phaserUsage = 0.5
          } else if (name.includes("phaser 1") || name.includes("phaser 2")) {
            if (typeof phaserUsage !== "number") {
              const mode = await promptPhaserPower12()
              if (mode === null) return
              phaserUsage = mode === "low" ? 0.5 : 1
            }
          } else if (name.includes("phaser 4")) {
            if (typeof phaserUsage !== "number") {
              const mode = await promptPhaserPower4()
              if (mode === null) return
              if (mode === "low") phaserUsage = 0.5
              else if (mode === "medium") phaserUsage = 1
              else phaserUsage = 2
            }
          } else if (name.includes("phaser g")) {
            const nextCount = (Number(s.phaserGCount) || 0) + 1
            s.phaserGCount = nextCount
            phaserUsage = 0.25 * nextCount
            if (s.phaserGTimer) {
              clearTimeout(s.phaserGTimer)
              delete s.phaserGTimer
            }
            if (nextCount < 4) {
              s.phaserGTimer = setTimeout(() => {
                if (s.phaserGCount < 4) {
                  delete s.systemDeployed
                  refreshEnergyAfterMarkerChange(tab)
                  renderCanvas()
                  renderCurrentProps()
                }
              }, 1000)
            }
          }
        }

        s.systemDeployed = true
        if (typeof phaserUsage === "number") s.phaserCapUsage = phaserUsage
        refreshEnergyAfterMarkerChange(tab)
        tab.selected = { kind: "square", id: s.id }
        renderCanvas()
        renderCurrentProps()
      }
      return
    }

    const s = hitTestSquare(tab, ix, iy)
    if (s) tab.selected = { kind: "square", id: s.id }
    else {
      const g = hitTestGroup(tab, ix, iy)
      if (!g) tab.selected = null
      else tab.selected = { kind: "group", id: g.id }
    }

    renderCanvas()
    renderCurrentProps()
  })

  canvas.addEventListener("mousemove", (e) => {
    const tab = getActiveTab()
    if (!tab || tab.uiState !== "analysis") return
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const pane = getAnalysisPaneAtPoint(tab, rect, x, y)
    if (!pane.side || !pane.rect) {
      const hadHover = tab.analysisHover || tab.analysisCompareHover || tab.analysisHoverSide
      tab.analysisHover = null
      tab.analysisCompareHover = null
      tab.analysisHoverSide = null
      if (hadHover) renderCanvas()
      return
    }
    if (pane.side === "compare" && !tab.analysisCompare) return
    if (analysisPaneUsesSsdPreview(tab, pane.side)) {
      const hadPrimaryHover = tab.analysisHover || tab.analysisHoverSide === "primary"
      tab.analysisHover = null
      if (tab.analysisHoverSide === "primary") tab.analysisHoverSide = null
      if (hadPrimaryHover) {
        hideAnalysisTooltip()
        renderCanvas()
      }
      return
    }
    const localX = x - pane.rect.x
    const localY = y - pane.rect.y
    const radius = pane.side === "compare"
      ? normalizeAnalysisHexRadius(tab.analysisCompare?.radius ?? tab.analysisRadius)
      : getAnalysisHexRadius(tab)
    const hex = analysisPointToHex(pane.rect, localX, localY, radius)
    const labelKeys = pane.side === "compare"
      ? tab.analysisCompareHexLabels
      : tab.analysisHexLabels
    let next = null
    if (hex && labelKeys instanceof Set) {
      const key = `${hex.q},${hex.r}`
      if (labelKeys.has(key)) {
        next = { q: hex.q, r: hex.r, key }
      }
    }
    const prevKey = pane.side === "compare"
      ? (tab.analysisCompareHover ? tab.analysisCompareHover.key : "")
      : (tab.analysisHover ? tab.analysisHover.key : "")
    const nextKey = next ? next.key : ""
    const nextSide = next ? pane.side : null
    if (prevKey === nextKey && tab.analysisHoverSide === nextSide) return

    if (pane.side === "compare") {
      tab.analysisCompareHover = next
      tab.analysisHover = null
    } else {
      tab.analysisHover = next
      tab.analysisCompareHover = null
    }
    tab.analysisHoverSide = nextSide
    renderCanvas()
  })

  canvas.addEventListener("mouseleave", (e) => {
    const tab = getActiveTab()
    if (!tab || tab.uiState !== "analysis") return
    const tooltip = document.getElementById("analysisTooltip")
    if (tooltip && (tooltip.matches(":hover") || tooltip.contains(e.relatedTarget))) return
    if (!tab.analysisHover && !tab.analysisCompareHover && !tab.analysisHoverSide) return
    tab.analysisHover = null
    tab.analysisCompareHover = null
    tab.analysisHoverSide = null
    renderCanvas()
  })

  const analysisTooltip = document.getElementById("analysisTooltip")
  if (analysisTooltip) {
    analysisTooltip.addEventListener("mouseleave", () => {
      const tab = getActiveTab()
      if (!tab || tab.uiState !== "analysis") return
      if (canvas.matches(":hover")) return
      if (!tab.analysisHover && !tab.analysisCompareHover && !tab.analysisHoverSide) return
      tab.analysisHover = null
      tab.analysisCompareHover = null
      tab.analysisHoverSide = null
      renderCanvas()
    })
  }

  canvas.addEventListener("contextmenu", (e) => {
    if (!state.damageMode && !state.systemMode && !state.refitMode) return
    const tab = getActiveTab()
    if (!tab || !tab.doc) return

    if (isShipyardLabelVisible()) return

    const { ix, iy } = canvasToImagePoint(tab, e.clientX, e.clientY)
    const s = hitTestSquare(tab, ix, iy)
    if (!s) return

    e.preventDefault()

    if (state.damageMode) delete s.damaged
    if (state.refitMode) {
      delete s.refitRemoved
    }
    if (state.systemMode) delete s.systemDeployed
    if (state.systemMode || state.damageMode || state.refitMode) {
      refreshEnergyAfterMarkerChange(tab)
    }

    tab.selected = { kind: "square", id: s.id }
    renderCanvas()
    renderCurrentProps()
  })

  window.addEventListener("resize", () => {
    renderCanvas()
    syncLeftPanelScrollbar()
  })
}

/* -----------------------------
   Damage marking helpers
----------------------------- */
function setDamageMode(on) {
  state.damageMode = !!on
  if (state.damageMode) {
    state.systemMode = false
    state.refitMode = false
    const sysBtn = document.getElementById("btnSystemDeploy")
    if (sysBtn) sysBtn.classList.remove("systemActive")
    const refitBtn = document.getElementById("btnRefitRemove")
    if (refitBtn) refitBtn.classList.remove("refitActive")
  }
  const btn = document.getElementById("btnMarkDamage")
  if (!btn) return
  if (state.damageMode) btn.classList.add("damageActive")
  else btn.classList.remove("damageActive")
}

function setWeaponMode(on) {
  // Weapon mode removed
}

function setSystemMode(on) {
  state.systemMode = !!on
  if (state.systemMode) {
    state.damageMode = false
    state.refitMode = false
    const dmgBtn = document.getElementById("btnMarkDamage")
    if (dmgBtn) dmgBtn.classList.remove("damageActive")
    const refitBtn = document.getElementById("btnRefitRemove")
    if (refitBtn) refitBtn.classList.remove("refitActive")
  }
  const btn = document.getElementById("btnSystemDeploy")
  if (!btn) return
  if (state.systemMode) btn.classList.add("systemActive")
  else btn.classList.remove("systemActive")
}

/* -----------------------------
   Team prompt (friendly/enemy)
----------------------------- */
function promptPhaserPower12() {
  return new Promise((resolve) => {
    const overlay = document.createElement("div")
    overlay.className = "modalOverlay"

    const card = document.createElement("div")
    card.className = "modalCard"

    const title = document.createElement("div")
    title.className = "modalTitle"
    title.textContent = "Phaser Firing Mode"

    const desc = document.createElement("div")
    desc.className = "modalDesc"
    desc.textContent = "Choose whether to fire a low-power (0.5) or full (1.0) shot."

    const actions = document.createElement("div")
    actions.className = "modalActions"

    const btnLow = document.createElement("button")
    btnLow.textContent = "Low Power (0.5)"
    const btnFull = document.createElement("button")
    btnFull.textContent = "Full Power (1.0)"
    const btnCancel = document.createElement("button")
    btnCancel.textContent = "Cancel"

    actions.appendChild(btnLow)
    actions.appendChild(btnFull)
    actions.appendChild(btnCancel)

    card.appendChild(title)
    card.appendChild(desc)
    card.appendChild(actions)
    overlay.appendChild(card)
    document.body.appendChild(overlay)

    const cleanup = (choice) => {
      document.body.removeChild(overlay)
      resolve(choice)
    }

    btnLow.onclick = () => cleanup("low")
    btnFull.onclick = () => cleanup("full")
    btnCancel.onclick = () => cleanup(null)

    const onKey = (e) => {
      if (e.key === "Escape") cleanup(null)
    }
    window.addEventListener("keydown", onKey, { once: true })
  })
}

/* -----------------------------
   Phaser-4 power prompt
----------------------------- */
function promptPhaserPower4() {
  return new Promise((resolve) => {
    const overlay = document.createElement("div")
    overlay.className = "modalOverlay"

    const card = document.createElement("div")
    card.className = "modalCard"

    const title = document.createElement("div")
    title.className = "modalTitle"
    title.textContent = "Phaser-4 Firing Mode"

    const desc = document.createElement("div")
    desc.className = "modalDesc"
    desc.textContent = "Choose the power level for this Phaser-4 shot."

    const actions = document.createElement("div")
    actions.className = "modalActions"

    const btnLow = document.createElement("button")
    btnLow.textContent = "Low (0.5)"
    const btnMed = document.createElement("button")
    btnMed.textContent = "Medium (1.0)"
    const btnFull = document.createElement("button")
    btnFull.textContent = "Normal (2.0)"
    const btnCancel = document.createElement("button")
    btnCancel.textContent = "Cancel"

    actions.appendChild(btnLow)
    actions.appendChild(btnMed)
    actions.appendChild(btnFull)
    actions.appendChild(btnCancel)

    card.appendChild(title)
    card.appendChild(desc)
    card.appendChild(actions)
    overlay.appendChild(card)
    document.body.appendChild(overlay)

    const cleanup = (choice) => {
      document.body.removeChild(overlay)
      resolve(choice)
    }

    btnLow.onclick = () => cleanup("low")
    btnMed.onclick = () => cleanup("medium")
    btnFull.onclick = () => cleanup("full")
    btnCancel.onclick = () => cleanup(null)

    const onKey = (e) => {
      if (e.key === "Escape") cleanup(null)
    }
    window.addEventListener("keydown", onKey, { once: true })
  })
}

/* -----------------------------
   Boot
----------------------------- */
renderAppVersion()
loadAppVersion()
addEmptyTab()
setupXXVIUI()
wireEvents()
renderCanvas()
renderCurrentProps()
syncLeftPanelScrollbar()

