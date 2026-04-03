const { app, BrowserWindow, ipcMain, dialog, Menu, shell } = require("electron")
let autoUpdater = null
try {
  ;({ autoUpdater } = require("electron-updater"))
} catch {
  autoUpdater = null
}
const path = require("path")
const fs = require("fs/promises")
const { spawn } = require("child_process")

const configPath = path.join(app.getPath("userData"), "settings.json")
let mainWindow = null
let shipyardXXVIProcess = null
let userConfig = { shipsDir: null, gamesDir: null, superluminalShipsDir: null }
const IMAGE_EXTENSIONS = Object.freeze([
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff"
])
const IMAGE_MIME_BY_EXTENSION = Object.freeze({
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".tif": "image/tiff",
  ".tiff": "image/tiff"
})

function formatErrorMessage(error, fallback = "Unexpected error.") {
  const msg = error?.message
  if (typeof msg === "string" && msg.trim()) return msg.trim()
  const text = String(error || "").trim()
  return text || fallback
}

function ok(payload = {}) {
  return { ok: true, ...payload }
}

function fail(error, fallback) {
  return { ok: false, error: formatErrorMessage(error, fallback) }
}

function handleObjectIpc(channel, handler) {
  ipcMain.handle(channel, async (event, ...args) => {
    try {
      const payload = await handler(event, ...args)
      return ok(payload || {})
    } catch (error) {
      return fail(error)
    }
  })
}

async function loadSettings() {
  try {
    const txt = await fs.readFile(configPath, "utf-8")
    const parsed = JSON.parse(txt)
    if (parsed && typeof parsed === "object") {
      userConfig = {
        shipsDir: parsed.shipsDir || null,
        gamesDir: parsed.gamesDir || null,
        superluminalShipsDir: parsed.superluminalShipsDir || null
      }
    }
  } catch {
    // ignore
  }
}

async function saveSettings() {
  try {
    const dir = path.dirname(configPath)
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(configPath, JSON.stringify(userConfig, null, 2), "utf-8")
  } catch {
    // ignore
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  mainWindow = win
  win.on("closed", () => {
    if (mainWindow === win) {
      mainWindow = null
    }
  })

  win.loadFile("index.html")
  win.setMenuBarVisibility(false)
}

app.whenReady().then(() => {
  loadSettings().catch(() => {})
  Menu.setApplicationMenu(null)
  createWindow()
  if (autoUpdater) {
    autoUpdater.autoDownload = true
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit()
})

/* ------------------------------------------
   Helpers
------------------------------------------ */
function sanitizeFileName(name) {
  return String(name)
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "_")
    .replace(/\.+$/g, "")
    .slice(0, 80)
}

const VEIL_FOLDER_MARKER = "#"

function stripVeilMarker(name) {
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

function makeVeilUnmarkedName(rawName) {
  return sanitizeFileName(stripVeilMarker(rawName))
}

async function ensureUniquePath(targetPath) {
  try {
    await fs.access(targetPath)
  } catch {
    return targetPath
  }

  const dir = path.dirname(targetPath)
  const base = path.basename(targetPath)

  for (let i = 2; i < 1000; i++) {
    const candidate = path.join(dir, `${base} (${i})`)
    try {
      await fs.access(candidate)
    } catch {
      return candidate
    }
  }
  throw new Error("Could not find an available folder name.")
}

async function renameFolderAndFilesToName(shipNameRaw, folderPathRaw) {
  const shipName = sanitizeFileName(shipNameRaw)
  if (!shipName) throw new Error("Ship name is empty after sanitizing.")

  const folderPath = folderPathRaw
  const parentDir = path.dirname(folderPath)
  const desiredFolder = path.join(parentDir, shipName)

  const newFolderPath =
    path.resolve(desiredFolder) === path.resolve(folderPath)
      ? folderPath
      : await ensureUniquePath(desiredFolder)

  if (path.resolve(newFolderPath) !== path.resolve(folderPath)) {
    await fs.rename(folderPath, newFolderPath)
  }

  const files = await fs.readdir(newFolderPath)
  const { jsonFile, imageFile: imgFile } = requireDatasetFiles(files)

  const oldJsonPath = path.join(newFolderPath, jsonFile)
  const oldImgPath = path.join(newFolderPath, imgFile)

  const imgExt = path.extname(oldImgPath)
  const newJsonPath = path.join(newFolderPath, `${shipName}.json`)
  const newImgPath = path.join(newFolderPath, `${shipName}${imgExt}`)

  if (path.resolve(oldJsonPath) !== path.resolve(newJsonPath)) {
    try { await fs.unlink(newJsonPath) } catch {}
    await fs.rename(oldJsonPath, newJsonPath)
  }

  if (path.resolve(oldImgPath) !== path.resolve(newImgPath)) {
    try { await fs.unlink(newImgPath) } catch {}
    await fs.rename(oldImgPath, newImgPath)
  }

  const jsonTextRaw = await fs.readFile(newJsonPath, "utf-8")
  let doc = null
  try {
    doc = JSON.parse(jsonTextRaw)
  } catch (e) {
    throw new Error("JSON parse failed after rename: " + e.message)
  }

  doc.shipName = shipName
  doc.imageFile = path.basename(newImgPath)

  const jsonText = JSON.stringify(doc, null, 2)
  await fs.writeFile(newJsonPath, jsonText, "utf-8")

  const imageDataUrl = await fileToDataUrl(newImgPath)

  return {
    shipName,
    folderPath: newFolderPath,
    jsonPath: newJsonPath,
    imagePath: newImgPath,
    imageDataUrl,
    jsonText
  }
}

function findFirstByExt(files, exts) {
  const lower = files.map(f => ({ f, lf: f.toLowerCase() }))
  for (const ext of exts) {
    const hit = lower.find(x => x.lf.endsWith(ext))
    if (hit) return hit.f
  }
  return null
}

function findDatasetFiles(files) {
  const jsonFile = findFirstByExt(files, [".json"])
  const imageFile = findFirstByExt(files, IMAGE_EXTENSIONS)
  return { jsonFile, imageFile }
}

function requireDatasetFiles(files) {
  const { jsonFile, imageFile } = findDatasetFiles(files)
  if (!jsonFile) throw new Error("No .json found in folder.")
  if (!imageFile) throw new Error("No image found in folder.")
  return { jsonFile, imageFile }
}

function sendToRenderer(channel, payload) {
  const win = mainWindow || BrowserWindow.getAllWindows()[0]
  if (!win || win.isDestroyed()) return
  win.webContents.send(channel, payload)
}

async function pickSuperluminalFolder(defaultPath) {
  const picked = await dialog.showOpenDialog({
    title: "Select ship folder (image + JSON)",
    properties: ["openDirectory"],
    defaultPath: defaultPath || undefined
  })

  if (picked.canceled || !picked.filePaths || picked.filePaths.length === 0) {
    return { ok: false, error: "No folder selected." }
  }

  const inputFolder = picked.filePaths[0]
  const files = await fs.readdir(inputFolder)
  let jsonFile = null
  let imageFile = null
  try {
    ;({ jsonFile, imageFile } = requireDatasetFiles(files))
  } catch (error) {
    return fail(error)
  }

  const jsonPath = path.join(inputFolder, jsonFile)
  const imagePath = path.join(inputFolder, imageFile)
  const jsonText = await fs.readFile(jsonPath, "utf-8")
  const imageDataUrl = await fileToDataUrl(imagePath)
  const inputBase = path.basename(jsonPath, path.extname(jsonPath))

  return {
    ok: true,
    inputFolder,
    jsonFile,
    imageFile,
    jsonPath,
    imagePath,
    jsonText,
    imageDataUrl,
    inputBase
  }
}

function normalizeLabel(raw) {
  return String(raw || "").trim().toLowerCase()
}

function keyFromLabel(raw) {
  const key = normalizeLabel(raw).replace(/[^a-z0-9]+/g, "")
  return key || "misc"
}

function formatRankValueLabel(key) {
  const normalized = normalizeLabel(key)
  if (!normalized) return ""
  if (normalized === "damcon") return "Damage Control"
  return normalized[0].toUpperCase() + normalized.slice(1)
}

let dataJsonCache = null
let dataJsonSectionsCache = null
let weaponDamageCache = null
let turnMovementCache = null

async function readDataJson() {
  if (dataJsonCache) return dataJsonCache
  const baseDir = app.isPackaged ? process.resourcesPath : app.getAppPath()
  const targetPath = path.join(baseDir, "Data.json")

  try {
    await fs.access(targetPath)
  } catch (err) {
    throw new Error("Data.json not found.")
  }

  const raw = await fs.readFile(targetPath, "utf-8")
  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    throw new Error("Data.json is not valid JSON.")
  }

  let entries = null
  let sectionsMap = null
  if (Array.isArray(parsed)) {
    entries = parsed
  } else if (parsed && typeof parsed === "object") {
    if (Array.isArray(parsed.sections)) {
      entries = []
      sectionsMap = {}
      for (const section of parsed.sections) {
        if (!section || typeof section !== "object") continue
        const label = normalizeLabel(section.label || "")
        const items = Array.isArray(section.entries) ? section.entries : []
        entries.push(...items)
        if (label) sectionsMap[label] = items
      }
    } else if (Array.isArray(parsed.entries)) {
      entries = parsed.entries
    }
  }

  if (!Array.isArray(entries)) {
    throw new Error("Data.json must be an array or contain a sections/entries list.")
  }

  dataJsonCache = entries
    .map((entry) => {
      const types = Array.isArray(entry?.types)
        ? entry.types.map(t => String(t || "").trim()).filter(Boolean)
        : []
      return {
        name: String(entry?.name || "").trim(),
        label: String(entry?.label || entry?.name || "").trim(),
        types
      }
    })
    .filter(entry => entry.name.length > 0)

  dataJsonSectionsCache = sectionsMap

  return dataJsonCache
}

async function readVeilBoxesList() {
  const candidates = [
    path.resolve(__dirname, "..", "Veil", "Data", "Boxes.txt"),
    path.resolve(__dirname, "..", "..", "Veil", "Data", "Boxes.txt")
  ]
  for (const candidate of candidates) {
    try {
      await fs.access(candidate)
      const raw = await fs.readFile(candidate, "utf-8")
      return raw
        .split(/\r?\n/)
        .map(line => String(line || "").trim())
        .filter(Boolean)
    } catch {}
  }
  return []
}

async function buildVeilToDataNameMap(dataNameSet) {
  const map = new Map()
  const veilNames = await readVeilBoxesList()
  if (!veilNames.length) return map
  for (const name of veilNames) {
    if (dataNameSet.has(name)) {
      map.set(name, name)
      continue
    }
    const sheildMatch = name.match(/^Sheild #([1-6])$/)
    if (sheildMatch) {
      const corrected = `Shield #${sheildMatch[1]}`
      if (dataNameSet.has(corrected)) {
        map.set(name, corrected)
      }
    }
  }
  return map
}

function buildOutputKeyMap(dataNames) {
  const map = new Map()
  for (const name of dataNames) {
    const clean = String(name || "").trim()
    if (!clean) continue
    const key = keyFromLabel(clean)
    if (!map.has(key)) map.set(key, clean)
  }
  return map
}

function getShieldIndexFromLabel(raw) {
  const match = String(raw || "").match(/sh(?:ie|ei)ld[^0-9]*(\d+)/i)
  if (!match) return 0
  const idx = Number(match[1])
  if (!Number.isFinite(idx) || idx < 1 || idx > 6) return 0
  return idx
}

function buildShieldOutputLabels(outputKeyMap) {
  const labels = []
  for (let i = 1; i <= 6; i++) {
    const fromData = outputKeyMap.get(`shield${i}`) || outputKeyMap.get(`sheild${i}`)
    labels.push(fromData || `Shield #${i}`)
  }
  return labels
}

function applySuperluminalAutoUpdateFields(doc, options = {}) {
  if (!doc || typeof doc !== "object" || Array.isArray(doc)) return doc

  const ssd = doc.ssd
  if (!ssd || typeof ssd !== "object" || Array.isArray(ssd)) return doc

  const shieldOutputLabels = Array.isArray(options.shieldOutputLabels)
    ? options.shieldOutputLabels
    : []
  const hitAndRunTerms = Array.isArray(options.hitAndRunTerms)
    ? options.hitAndRunTerms
    : []

  const shields = ssd.shields
  if (shieldOutputLabels.length >= 6 && shields && typeof shields === "object" && !Array.isArray(shields)) {
    const normalized = {}
    for (const [key, value] of Object.entries(shields)) {
      const idx = getShieldIndexFromLabel(key)
      if (idx > 0) normalized[shieldOutputLabels[idx - 1]] = value
      else normalized[key] = value
    }
    ssd.shields = normalized
  }

  const sectionCounts = new Map()
  for (const [key, value] of Object.entries(ssd)) {
    if (!Array.isArray(value)) continue
    const k = keyFromLabel(key)
    sectionCounts.set(k, (sectionCounts.get(k) || 0) + value.length)
  }

  if (!doc.shipData || typeof doc.shipData !== "object" || Array.isArray(doc.shipData)) {
    return doc
  }

  const getCount = (...keys) =>
    keys.reduce((sum, key) => sum + (sectionCounts.get(String(key || "").trim()) || 0), 0)

  doc.shipData.crewUnits = getCount("crewunits")
  doc.shipData.boardingParties = getCount("boardingparties")
  doc.shipData.tBombs = getCount("transporterbombs", "transportbombs")
  doc.shipData.dBombs = getCount("transporterdummies", "transportdummies")
  doc.shipData.probes = getCount("probesammo", "probeammo")

  const hitAndRunOut = {}
  for (const term of hitAndRunTerms) {
    const label = String(term || "").trim()
    if (!label) continue
    const key = keyFromLabel(label)
    const count = sectionCounts.get(key) || 0
    if (count > 0) hitAndRunOut[label] = count
  }
  if (Object.keys(hitAndRunOut).length > 0) {
    doc.shipData.hitAndRun = hitAndRunOut
  } else if (doc.shipData.hitAndRun) {
    delete doc.shipData.hitAndRun
  }

  return doc
}

async function readWeaponDamageJson() {
  if (weaponDamageCache) return weaponDamageCache
  const baseDir = app.isPackaged ? process.resourcesPath : app.getAppPath()
  const targetPath = path.join(baseDir, "weaponDamage.json")

  try {
    await fs.access(targetPath)
  } catch (err) {
    throw new Error("weaponDamage.json not found.")
  }

  const raw = await fs.readFile(targetPath, "utf-8")
  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    throw new Error("weaponDamage.json is not valid JSON.")
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("weaponDamage.json must be a JSON object.")
  }
  if (!Array.isArray(parsed.weapons)) {
    throw new Error("weaponDamage.json must contain a weapons array.")
  }

  weaponDamageCache = parsed
  return weaponDamageCache
}

async function readTurnMovementJson() {
  if (turnMovementCache) return turnMovementCache
  const baseDir = app.isPackaged ? process.resourcesPath : app.getAppPath()
  const targetPath = path.join(baseDir, "Turn&Movement.json")

  try {
    await fs.access(targetPath)
  } catch (err) {
    throw new Error("Turn&Movement.json not found.")
  }

  const raw = await fs.readFile(targetPath, "utf-8")
  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    throw new Error("Turn&Movement.json is not valid JSON.")
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Turn&Movement.json must be a JSON object.")
  }
  if (!Array.isArray(parsed.rows)) {
    throw new Error("Turn&Movement.json must contain a rows array.")
  }

  turnMovementCache = parsed
  return turnMovementCache
}

async function readDataNames() {
  const entries = await readDataJson()
  return entries.map(e => e.label || e.name)
}

async function readDataNamesByType(typeName) {
  const entries = await readDataJson()
  const target = normalizeLabel(typeName)
  return entries
    .filter(e => (e.types || []).some(t => normalizeLabel(t) === target))
    .map(e => e.label || e.name)
}

async function readDataSectionEntries(sectionLabel) {
  await readDataJson()
  if (!sectionLabel) return []
  const key = normalizeLabel(sectionLabel)
  const items = dataJsonSectionsCache && dataJsonSectionsCache[key]
  if (!Array.isArray(items)) return []
  return items
    .map(entry => String(entry?.name || "").trim())
    .filter(Boolean)
}

function blankValuesDeep(value) {
  if (Array.isArray(value)) {
    return value.map(item => blankValuesDeep(item))
  }
  if (value && typeof value === "object") {
    const out = {}
    for (const [k, v] of Object.entries(value)) {
      out[k] = blankValuesDeep(v)
    }
    return out
  }
  return ""
}

function formatPosFromSquare(square) {
  const bbox = square?.bbox
  if (bbox && typeof bbox === "object") {
    const x1 = Math.round(Number(bbox.x1 ?? 0))
    const y1 = Math.round(Number(bbox.y1 ?? 0))
    const x2 = Math.round(Number(bbox.x2 ?? 0))
    const y2 = Math.round(Number(bbox.y2 ?? 0))
    return `${x1},${y1},${x2},${y2}`
  }
  const center = square?.center
  const side = Number(square?.sideLengthPx ?? 0)
  if (center && typeof center === "object") {
    const cx = Math.round(Number(center.x ?? 0))
    const cy = Math.round(Number(center.y ?? 0))
    const size = Math.round(side)
    return `${cx},${cy},${size}`
  }
  return ""
}

function bboxFromPosText(posText) {
  if (!posText) return null
  const parts = String(posText)
    .split(",")
    .map(p => Number(p.trim()))
    .filter(n => Number.isFinite(n))

  if (parts.length === 4) {
    const [x1, y1, x2, y2] = parts
    return { x1, y1, x2, y2 }
  }

  if (parts.length === 3) {
    const [cx, cy, size] = parts
    const half = size / 2
    return { x1: cx - half, y1: cy - half, x2: cx + half, y2: cy + half }
  }

  return null
}

function cloneTemplateItem(item) {
  if (!item || typeof item !== "object") return {}
  return JSON.parse(JSON.stringify(item))
}

async function readTemplatePath() {
  const candidates = [
    path.join(__dirname, "SLConfig.json"),
    path.resolve(__dirname, "..", "SLConfig.json")
  ]
  for (const candidate of candidates) {
    try {
      await fs.access(candidate)
      return candidate
    } catch {}
  }
  return null
}

async function readHitAndRunTerms() {
  return readDataNamesByType("Hit and Run")
}

// Default to the shared Ships folder (Warframe Data Matrix) when picking ship data
async function getDefaultShipsPath() {
  const candidates = [
    userConfig.shipsDir ? path.resolve(userConfig.shipsDir) : null,
    // Grandparent/Ships (e.g., Desktop/Ships)
    path.resolve(__dirname, "..", "..", "Ships"),
    // Parent/Ships (app folder sibling)
    path.resolve(__dirname, "..", "Ships"),
    path.resolve(app.getPath("documents"), "Warframe Data Matrix", "Ships")
  ]

  for (const candidate of candidates) {
    try {
      await fs.access(candidate)
      return candidate
    } catch {}
  }

  return null
}

async function getDefaultSuperluminalShipsPath() {
  if (userConfig.superluminalShipsDir) {
    try {
      const resolved = path.resolve(userConfig.superluminalShipsDir)
      await fs.access(resolved)
      return resolved
    } catch {}
  }
  return await getDefaultShipsPath()
}

function getSuperluminalOutputFolder(inputFolder) {
  const baseFolderName = stripVeilMarker(path.basename(inputFolder))
  const lowerName = baseFolderName.toLowerCase()
  const folderName = lowerName.endsWith(" superluminal")
    ? baseFolderName
    : `${baseFolderName} superluminal`
  const root = userConfig.superluminalShipsDir
    ? path.resolve(userConfig.superluminalShipsDir)
    : path.dirname(inputFolder)
  return path.join(root, folderName)
}

async function readExistingSuperluminalDoc(inputFolder) {
  try {
    const targetFolder = getSuperluminalOutputFolder(inputFolder)
    await fs.access(targetFolder)
    const files = await fs.readdir(targetFolder)
    const jsonFile = findFirstByExt(files, [".json"])
    if (!jsonFile) return null
    const jsonPath = path.join(targetFolder, jsonFile)
    const raw = await fs.readFile(jsonPath, "utf-8")
    const doc = JSON.parse(raw)
    if (!doc || typeof doc !== "object") return null
    return { doc, jsonPath }
  } catch {
    return null
  }
}

function seedAssignmentsFromExistingDoc(
  existingDoc,
  matchesByKey,
  designationAssignments,
  arcAssignments,
  rankValueAssignments,
  outputKeyMap
) {
  if (!existingDoc || !existingDoc.ssd || typeof existingDoc.ssd !== "object") return

  const sortedMatches = (key) =>
    (matchesByKey.get(key) || [])
      .slice()
      .sort((a, b) => (a.square?.id ?? 0) - (b.square?.id ?? 0))

  const seedWeaponAssignments = (key) => {
    const altKey = outputKeyMap?.get(key)
    const existingEntries = Array.isArray(existingDoc.ssd?.[key])
      ? existingDoc.ssd[key]
      : Array.isArray(existingDoc.ssd?.[altKey])
        ? existingDoc.ssd[altKey]
        : []
    if (existingEntries.length === 0) return
    const matches = sortedMatches(key)
    for (let i = 0; i < matches.length; i++) {
      const match = matches[i]
      const entry = existingEntries[i]
      if (!match || !entry) continue
      const squareId = match?.square?.id
      if (squareId === undefined || squareId === null) continue
      const keyId = String(squareId)
      if (!designationAssignments[keyId]) {
        const designation = String(entry?.designation || "").trim()
        if (designation) designationAssignments[keyId] = designation
      }
      if (!arcAssignments[keyId]) {
        const arc = String(entry?.arc || "").trim()
        if (arc) arcAssignments[keyId] = arc
      }
    }
  }

  const seedRankAssignments = (key) => {
    const altKey = outputKeyMap?.get(key)
    const existingEntries = Array.isArray(existingDoc.ssd?.[key])
      ? existingDoc.ssd[key]
      : Array.isArray(existingDoc.ssd?.[altKey])
        ? existingDoc.ssd[altKey]
        : []
    if (existingEntries.length === 0) return
    const matches = sortedMatches(key)
    for (let i = 0; i < matches.length; i++) {
      const match = matches[i]
      const entry = existingEntries[i]
      if (!match || !entry) continue
      const squareId = match?.square?.id
      if (squareId === undefined || squareId === null) continue
      const keyId = String(squareId)
      const existingAssign =
        rankValueAssignments[keyId] && typeof rankValueAssignments[keyId] === "object"
          ? rankValueAssignments[keyId]
          : {}
      let changed = false
      if ((existingAssign.value === undefined || existingAssign.value === null) &&
          entry.value !== undefined && entry.value !== null) {
        existingAssign.value = entry.value
        changed = true
      }
      if ((existingAssign.rank === undefined || existingAssign.rank === null) &&
          entry.rank !== undefined && entry.rank !== null) {
        existingAssign.rank = entry.rank
        changed = true
      }
      if (changed) {
        rankValueAssignments[keyId] = existingAssign
      }
    }
  }

  seedWeaponAssignments("heavy")
  seedWeaponAssignments("drone")
  seedWeaponAssignments("phaser")
  seedRankAssignments("sensor")
  seedRankAssignments("scanner")
  seedRankAssignments("damcon")
}

function mimeFromExt(extLower) {
  return IMAGE_MIME_BY_EXTENSION[extLower] || "application/octet-stream"
}

async function fileToDataUrl(filePath) {
  try {
    try {
      await fs.access(filePath)
    } catch (err) {
      throw new Error(`Image file not found: ${filePath}`)
    }

    const buf = await fs.readFile(filePath)

    if (!buf || buf.length === 0) {
      throw new Error(`Image file is empty: ${filePath}`)
    }

    const ext = path.extname(filePath).toLowerCase()
    const mime = mimeFromExt(ext)

    if (!IMAGE_EXTENSIONS.includes(ext)) {
      throw new Error(`Invalid image extension: ${ext}`)
    }

    const b64 = buf.toString("base64")
    return `data:${mime};base64,${b64}`
  } catch (error) {
    console.error("Failed to convert file to data URL:", error)
    throw error
  }
}

/* ------------------------------------------
   IPC: Save JSON (overwrite same path)
------------------------------------------ */
ipcMain.handle("saveJson", async (event, jsonPath, jsonText, saveOptions) => {
  try {
    let outputText = String(jsonText ?? "")
    const autoUpdateConvertFields =
      saveOptions &&
      typeof saveOptions === "object" &&
      saveOptions.autoUpdateConvertFields === true

    if (autoUpdateConvertFields) {
      let doc = null
      try {
        doc = JSON.parse(outputText)
      } catch (e) {
        return { ok: false, error: `Invalid JSON: ${e.message}` }
      }
      const dataNames = await readDataNames()
      const outputKeyMap = buildOutputKeyMap(dataNames)
      const shieldOutputLabels = buildShieldOutputLabels(outputKeyMap)
      const hitAndRunTerms = await readHitAndRunTerms()
      applySuperluminalAutoUpdateFields(doc, {
        hitAndRunTerms,
        shieldOutputLabels
      })
      outputText = JSON.stringify(doc, null, 2)
    }

    await fs.writeFile(jsonPath, outputText, "utf-8")
    return ok({ path: jsonPath, jsonText: outputText })
  } catch (e) {
    return fail(e)
  }
})

/* ------------------------------------------
   IPC: Read Data.json label options
------------------------------------------ */
handleObjectIpc("shipyard:readBoxes", async () => {
  const options = await readDataNames()
  return { options }
})

handleObjectIpc("shipyard:readExistingSuperluminal", async (event, inputFolder) => {
  if (!inputFolder) throw new Error("No input folder provided.")
  const existing = await readExistingSuperluminalDoc(inputFolder)
  if (!existing || !existing.doc) throw new Error("No existing Superluminal data found.")
  return { jsonText: JSON.stringify(existing.doc, null, 2) }
})

handleObjectIpc("shipyard:readSectionEntries", async (event, sectionLabel) => {
  const entries = await readDataSectionEntries(sectionLabel)
  return { entries }
})

/* ------------------------------------------
   IPC: Read heavy weapon options
------------------------------------------ */
handleObjectIpc("shipyard:readHeavyList", async () => {
  const options = await readDataNamesByType("Heavy")
  return { options }
})

handleObjectIpc("shipyard:readExternalList", async () => {
  const options = await readDataNamesByType("External")
  return { options }
})

handleObjectIpc("shipyard:readWeaponDamage", async () => {
  const data = await readWeaponDamageJson()
  return { data }
})

handleObjectIpc("shipyard:readTurnMovement", async () => {
  const data = await readTurnMovementJson()
  return { data }
})

function validateShipJSON(doc) {
  if (!doc || typeof doc !== "object") {
    throw new Error("Invalid JSON: not an object")
  }

  if (!Array.isArray(doc.squares)) {
    throw new Error("Invalid JSON: 'squares' must be an array")
  }

  if (doc.groups && !Array.isArray(doc.groups)) {
    throw new Error("Invalid JSON: 'groups' must be an array")
  }

  for (let i = 0; i < doc.squares.length; i++) {
    const sq = doc.squares[i]
    if (!sq || typeof sq !== "object") {
      throw new Error(`Invalid square at index ${i}`)
    }
    if (typeof sq.id === "undefined") {
      throw new Error(`Square at index ${i} missing 'id'`)
    }
  }

  return true
}

/* ------------------------------------------
   IPC: Shipyard pick folder
------------------------------------------ */
ipcMain.handle("shipyard:pickFolderNamed", async () => {
  const defaultPath = await getDefaultShipsPath()
  const picked = await dialog.showOpenDialog({
    title: "Choose ship folder (contains .json + image)",
    properties: ["openDirectory"],
    defaultPath: defaultPath || undefined
  })

  if (picked.canceled || !picked.filePaths || picked.filePaths.length === 0) return null
  return { ok: true, folderPath: picked.filePaths[0] }
})

/* ------------------------------------------
   NEW IPC: Load dataset from folder WITHOUT renaming
------------------------------------------ */
ipcMain.handle("shipyard:loadFolderDataset", async (event, folderPath) => {
  try {
    const files = await fs.readdir(folderPath)
    let jsonFile = null
    let imgFile = null
    try {
      ;({ jsonFile, imageFile: imgFile } = requireDatasetFiles(files))
    } catch (error) {
      return fail(error)
    }

    const jsonPath = path.join(folderPath, jsonFile)
    const imagePath = path.join(folderPath, imgFile)

    const jsonText = await fs.readFile(jsonPath, "utf-8")
    let doc
    try {
      doc = JSON.parse(jsonText)
      validateShipJSON(doc)
    } catch (parseError) {
      return {
        ok: false,
        error: `Invalid ship data: ${parseError.message}`
      }
    }
    const imageDataUrl = await fileToDataUrl(imagePath)

    return ok({ folderPath, jsonPath, imagePath, jsonText, imageDataUrl })
  } catch (e) {
    return fail(e)
  }
})

/* ------------------------------------------
   IPC: Rename folder + files to ship name
------------------------------------------ */
handleObjectIpc("shipyard:renameFolderAndFiles", async (event, shipNameRaw, folderPathRaw) => {
  return await renameFolderAndFilesToName(shipNameRaw, folderPathRaw)
})

ipcMain.handle("setup:editPaths", async () => {
  try {
    const pickDir = async (title, defaultPath) => {
      const picked = await dialog.showOpenDialog({
        title,
        properties: ["openDirectory"],
        defaultPath: defaultPath || undefined
      })
      if (picked.canceled || !picked.filePaths || picked.filePaths.length === 0) return null
      return picked.filePaths[0]
    }

    const currentGames = userConfig.gamesDir
    const currentShips = userConfig.shipsDir
    const currentSuperluminal = userConfig.superluminalShipsDir

    const gamesDir = await pickDir("Select Games folder", currentGames)
    const shipsDir = await pickDir("Select Ships folder", currentShips)
    const superluminalShipsDir = await pickDir(
      "Select Superluminal Ships folder",
      currentSuperluminal
    )

    if (gamesDir) userConfig.gamesDir = gamesDir
    if (shipsDir) userConfig.shipsDir = shipsDir
    if (superluminalShipsDir) userConfig.superluminalShipsDir = superluminalShipsDir

    await saveSettings()

    return {
      ok: true,
      gamesDir: userConfig.gamesDir,
      shipsDir: userConfig.shipsDir,
      superluminalShipsDir: userConfig.superluminalShipsDir
    }
  } catch (e) {
    return { ok: false, error: e.message }
  }
})

ipcMain.handle("shipyard:runXXVI", async () => {
  if (shipyardXXVIProcess) {
    return { ok: false, error: "Shipyard XXVI is already running." }
  }

  const scriptPath = path.join(__dirname, "Shipyard XXVI.py")
  let resolved = false

  return new Promise((resolve) => {
    const safeResolve = (payload) => {
      if (resolved) return
      resolved = true
      resolve(payload)
    }

    const emitLine = (line, streamName) => {
      const text = String(line || "").trim()
      if (!text) return
      let percent = null
      const match = text.match(/(\d{1,3}(?:\.\d+)?)%/)
      if (match) {
        percent = Math.min(100, Number(match[1]))
      }
      sendToRenderer("shipyard:xxvi:progress", { line: text, percent, stream: streamName })
    }

    const attachStream = (stream, streamName) => {
      if (!stream) return
      let buffer = ""
      stream.on("data", (chunk) => {
        buffer += chunk.toString()
        const parts = buffer.split(/[\r\n]+/)
        buffer = parts.pop() || ""
        for (const part of parts) {
          emitLine(part, streamName)
        }
      })
      stream.on("end", () => {
        const leftover = buffer.trim()
        if (leftover) emitLine(leftover, streamName)
      })
    }

    const startProcess = (cmd, args) => {
      const child = spawn(cmd, args, { cwd: __dirname, windowsHide: true })
      shipyardXXVIProcess = child

      child.once("spawn", () => {
        safeResolve({ ok: true })
      })

      child.on("error", (err) => {
        if (err.code === "ENOENT" && cmd === "python") {
          shipyardXXVIProcess = null
          startProcess("py", ["-3", "-u", scriptPath])
          return
        }
        shipyardXXVIProcess = null
        sendToRenderer("shipyard:xxvi:error", { error: err.message })
        safeResolve({ ok: false, error: err.message })
      })

      attachStream(child.stdout, "stdout")
      attachStream(child.stderr, "stderr")

      child.on("close", (code) => {
        shipyardXXVIProcess = null
        sendToRenderer("shipyard:xxvi:done", { code })
      })
    }

    startProcess("python", ["-u", scriptPath])
  })
})

ipcMain.handle("shipyard:pickSuperluminalInput", async () => {
  try {
    const defaultPath = await getDefaultShipsPath()
    return await pickSuperluminalFolder(defaultPath)
  } catch (e) {
    return { ok: false, error: e?.message || "Failed to pick folder." }
  }
})

ipcMain.handle("shipyard:pickSuperluminalShip", async () => {
  try {
    const defaultPath = await getDefaultSuperluminalShipsPath()
    return await pickSuperluminalFolder(defaultPath)
  } catch (e) {
    return { ok: false, error: e?.message || "Failed to pick folder." }
  }
})

ipcMain.handle("shipyard:convertSSDJson", async (event, payload) => {
  try {
    const templatePath = await readTemplatePath()
    if (!templatePath) {
      return { ok: false, error: "SLConfig.json template not found." }
    }

    const dataNames = await readDataNames()
    const dataNameSet = new Set(
      dataNames.map(name => String(name || "").trim()).filter(Boolean)
    )
    const veilNameMap = await buildVeilToDataNameMap(dataNameSet)
    const outputKeyMap = buildOutputKeyMap(dataNames)
    const shieldOutputLabels = buildShieldOutputLabels(outputKeyMap)
    const heavyTerms = await readDataNamesByType("Heavy")
    const heavySet = new Set(
      heavyTerms.map(name => String(name || "").trim()).filter(Boolean)
    )
    const droneTerms = await readDataNamesByType("Drone")
    const droneSet = new Set(
      droneTerms.map(name => String(name || "").trim()).filter(Boolean)
    )
    const phaserTerms = await readDataNamesByType("Phaser")
    const phaserSet = new Set(
      phaserTerms.map(name => String(name || "").trim()).filter(Boolean)
    )
    const hitAndRunTerms = await readHitAndRunTerms()
    const hitAndRunSet = new Set(
      hitAndRunTerms.map(name => String(name || "").trim()).filter(Boolean)
    )

    let inputFolder = payload?.inputFolder || null
    let jsonFile = payload?.jsonFile || null
    let imageFile = payload?.imageFile || null
    let inputPath = payload?.jsonPath || null
    let imagePath = payload?.imagePath || null
    if (!inputFolder || !jsonFile || !imageFile || !inputPath || !imagePath) {
      const defaultPath = await getDefaultShipsPath()
      const picked = await pickSuperluminalFolder(defaultPath)
      if (!picked.ok) return picked
      inputFolder = picked.inputFolder
      jsonFile = picked.jsonFile
      imageFile = picked.imageFile
      inputPath = picked.jsonPath
      imagePath = picked.imagePath
    }
    const raw = await fs.readFile(inputPath, "utf-8")
    let inputDoc = null
    try {
      inputDoc = JSON.parse(raw)
    } catch (e) {
      return { ok: false, error: "Input JSON parse failed: " + e.message }
    }

    const templateRaw = await fs.readFile(templatePath, "utf-8")
    const template = JSON.parse(templateRaw)
    const output = JSON.parse(JSON.stringify(template))

    const inputBaseRaw = path.basename(inputPath, path.extname(inputPath))
    const inputBase = stripVeilMarker(inputBaseRaw)
    const nameParts = inputBase.split(" ").filter(Boolean)
    const empire = nameParts.length > 0 ? nameParts[0] : ""
    const type = nameParts.length > 1 ? nameParts.slice(1).join(" ") : ""
    const overrides = payload?.shipData || null
    const arcAssignments =
      payload?.arcAssignments && typeof payload.arcAssignments === "object"
        ? payload.arcAssignments
        : {}
    const designationAssignments =
      payload?.designationAssignments && typeof payload.designationAssignments === "object"
        ? payload.designationAssignments
        : {}
    const rankValueAssignments =
      payload?.rankValueAssignments && typeof payload.rankValueAssignments === "object"
        ? payload.rankValueAssignments
        : {}

  const hasSquares = Array.isArray(inputDoc?.squares) && inputDoc.squares.length > 0
  const hasSsd = inputDoc?.ssd && typeof inputDoc.ssd === "object"
  const hasShipData = inputDoc?.shipData && typeof inputDoc.shipData === "object"
  const isSuperluminalInput = !hasSquares && (hasSsd || hasShipData)
  const existingSuperluminal =
    !isSuperluminalInput && inputFolder ? await readExistingSuperluminalDoc(inputFolder) : null
  const existingDoc = existingSuperluminal?.doc || null

  if (isSuperluminalInput) {
      const updated = JSON.parse(JSON.stringify(inputDoc))
      if (!updated.ssd || typeof updated.ssd !== "object") {
        updated.ssd = {}
      }

      const templateShipData =
        template.shipData && typeof template.shipData === "object"
          ? template.shipData
          : {}
      const existingShipData =
        updated.shipData && typeof updated.shipData === "object"
          ? updated.shipData
          : {}

      const mergedShipData = {
        ...blankValuesDeep(templateShipData),
        ...existingShipData
      }

      if (!mergedShipData.empire && empire) mergedShipData.empire = empire
      if (!mergedShipData.type && type) mergedShipData.type = type

      if (overrides && typeof overrides === "object") {
        for (const [k, v] of Object.entries(overrides)) {
          const text = String(v ?? "").trim()
          if (!text) continue
          if (k in mergedShipData) {
            mergedShipData[k] = v
          }
        }
        if ("spareShuttles" in overrides) {
          const spareText = String(overrides.spareShuttles ?? "").trim()
          if (spareText) {
            if (!mergedShipData.shuttles || typeof mergedShipData.shuttles !== "object") {
              mergedShipData.shuttles = {}
            }
            mergedShipData.shuttles.spareShuttles = overrides.spareShuttles
          }
        }
      }

      updated.shipData = mergedShipData

      const arcKeys = ["drone", "phaser", "heavy"]
      const designationTargets = []
      const arcTargets = []
      for (const key of arcKeys) {
        const entries = Array.isArray(updated.ssd?.[key]) ? updated.ssd[key] : []
        for (let i = 0; i < entries.length; i++) {
          const entry = entries[i]
          const keyId = `ssd:${key}:${i}`
          const posText = String(entry?.pos || "").trim()
          const bbox = bboxFromPosText(posText)
          const designation = String(entry?.designation || "").trim()
          designationTargets.push({
            key: keyId,
            squareId: designation || null,
            label: String(entry?.type || "").trim(),
            group: key,
            designation,
            bbox
          })
          arcTargets.push({
            key: keyId,
            squareId: entry?.designation ? String(entry.designation).trim() : null,
            label: String(entry?.type || "").trim(),
            group: key,
            arc: String(entry?.arc || "").trim(),
            bbox
          })
        }
      }

      const missingDesignations = designationTargets.filter((target) => {
        const assignmentKey = String(target.key || "").trim()
        const assigned = assignmentKey
          ? String(designationAssignments[assignmentKey] || "").trim()
          : ""
        return !assigned
      })
      if (missingDesignations.length > 0) {
        return { ok: false, needsDesignations: { items: missingDesignations } }
      }

      const missingArcs = arcTargets.filter((target) => {
        const assignmentKey = String(target.key || "").trim()
        const assigned = assignmentKey ? String(arcAssignments[assignmentKey] || "").trim() : ""
        if (assigned) return false
        const existingArc = String(target.arc || "").trim()
        return !existingArc
      })
      if (missingArcs.length > 0) {
        return { ok: false, needsArcs: { items: missingArcs } }
      }

      const rankValueKeys = ["sensor", "scanner", "damcon"]
      const templateSsd =
        template.ssd && typeof template.ssd === "object" ? template.ssd : {}
      const rankValueTargets = []
      for (const key of rankValueKeys) {
        const entries = Array.isArray(updated.ssd?.[key]) ? updated.ssd[key] : []
        if (entries.length === 0) continue
        const templateEntries = Array.isArray(templateSsd?.[key]) ? templateSsd[key] : []
        for (let i = 0; i < entries.length; i++) {
          const entry = entries[i] || {}
          const keyId = `ssd:${key}:${i}`
          const posText = String(entry?.pos || "").trim()
          const bbox = bboxFromPosText(posText)
          const templateItem = templateEntries[i] || templateEntries[0] || {}
          const value =
            entry?.value !== undefined && entry?.value !== null
              ? entry.value
              : templateItem?.value
          const rank =
            entry?.rank !== undefined && entry?.rank !== null
              ? entry.rank
              : templateItem?.rank
          rankValueTargets.push({
            key: keyId,
            label: formatRankValueLabel(key),
            group: key,
            value,
            rank,
            bbox
          })
        }
      }
      const missingRankValues = rankValueTargets.filter((target) => {
        const assignmentKey = String(target.key || "").trim()
        const assigned = assignmentKey ? rankValueAssignments[assignmentKey] : null
        if (!assigned || typeof assigned !== "object") return true
        return (
          assigned.rank === undefined ||
          assigned.rank === null ||
          assigned.value === undefined ||
          assigned.value === null
        )
      })
      if (missingRankValues.length > 0) {
        return { ok: false, needsRankValues: { items: missingRankValues } }
      }

      for (const key of arcKeys) {
        const entries = Array.isArray(updated.ssd?.[key]) ? updated.ssd[key] : []
        for (let i = 0; i < entries.length; i++) {
          const entryKey = `ssd:${key}:${i}`
          const designationAssigned = String(designationAssignments[entryKey] || "").trim()
          if (designationAssigned) entries[i].designation = designationAssigned
          const assigned = String(arcAssignments[entryKey] || "").trim()
          if (assigned) entries[i].arc = assigned
        }
      }

      for (const key of rankValueKeys) {
        const entries = Array.isArray(updated.ssd?.[key]) ? updated.ssd[key] : []
        for (let i = 0; i < entries.length; i++) {
          const entryKey = `ssd:${key}:${i}`
          const assigned = rankValueAssignments[entryKey]
          if (!assigned || typeof assigned !== "object") continue
          if (assigned.value !== undefined) entries[i].value = assigned.value
          if (assigned.rank !== undefined) entries[i].rank = assigned.rank
        }
      }

      const shields = updated.ssd?.shields
      if (shields && typeof shields === "object" && !Array.isArray(shields)) {
        const normalized = {}
        for (const [key, value] of Object.entries(shields)) {
          const shieldIdx = getShieldIndexFromLabel(key)
          if (shieldIdx > 0) normalized[shieldOutputLabels[shieldIdx - 1]] = value
          else normalized[key] = value
        }
        updated.ssd.shields = normalized
      }

      await fs.writeFile(inputPath, JSON.stringify(updated, null, 2), "utf-8")
      return { ok: true, path: inputPath }
    }

    const squares = Array.isArray(inputDoc?.squares) ? inputDoc.squares : []
    const groups = Array.isArray(inputDoc?.groups) ? inputDoc.groups : []
    const matchesByKey = new Map()
    let hasShields = false
    let crewUnitsCount = 0
    let boardingPartiesCount = 0
    let tBombsCount = 0
    let dBombsCount = 0
    let probesCount = 0
    const hitAndRunCounts = {}
    const shieldCounts = [0, 0, 0, 0, 0, 0]

    const addMatch = (key, square, label) => {
      if (!matchesByKey.has(key)) matchesByKey.set(key, [])
      matchesByKey.get(key).push({ square, label })
    }

    const labelKeyMap = new Map([
      ["Boarding Parties", "boarding"],
      ["Crew Units", "crew"],
      ["Transporter Bombs", "tbomb"],
      ["Transporter Dummies", "dbomb"],
      ["Flag Bridge", "flagbridge"],
      ["Bridge", "bridge"],
      ["Emergency Bridge", "emer"],
      ["Cloak", "cloak"],
      ["Transporter", "tran"],
      ["Tractor", "trac"],
      ["Forward Hull", "fhull"],
      ["Aft Hull", "ahull"],
      ["Left Warp", "lwarp"],
      ["Right Warp", "rwarp"],
      ["Center Warp", "cwarp"],
      ["Impulse", "imp"],
      ["Battery", "btty"],
      ["Lab", "lab"],
      ["Shuttle", "shuttle"],
      ["Auxiliary Control", "aux"],
      ["APR", "apr"],
      ["Probe", "probe"],
      ["Probes (Ammo)", "probeammo"],
      ["Damage Control", "damcon"],
      ["Excess Damage", "excessdamage"],
      ["Sensor", "sensor"],
      ["Scanner", "scanner"]
    ])
    for (const [name, key] of labelKeyMap.entries()) {
      outputKeyMap.set(key, name)
    }

    for (const s of squares) {
      const labelRaw = String(s?.label || s?.name || "").trim()
      if (!labelRaw) continue
      const labelName = veilNameMap.get(labelRaw) || labelRaw
      if (!dataNameSet.has(labelName)) continue

      const shieldIdx = getShieldIndexFromLabel(labelName)
      if (shieldIdx > 0) {
        hasShields = true
        shieldCounts[shieldIdx - 1] += 1
      }

      if (labelName === "Crew Units") {
        crewUnitsCount += 1
      }
      if (labelName === "Boarding Parties") {
        boardingPartiesCount += 1
      }
      if (labelName === "Transporter Bombs") {
        tBombsCount += 1
      }
      if (labelName === "Transporter Dummies") {
        dBombsCount += 1
      }
      if (labelName === "Probes (Ammo)") {
        probesCount += 1
      }

      const isDrone = droneSet.has(labelName)
      if (isDrone) {
        addMatch("drone", s, labelName)
        continue
      }

      const isHeavy = heavySet.has(labelName)

      if (isHeavy) {
        addMatch("heavy", s, labelName)
        continue
      }

      const isPhaser = phaserSet.has(labelName)
      if (isPhaser) {
        addMatch("phaser", s, labelName)
        continue
      }

      let matched = false
      if (hitAndRunSet.has(labelName)) {
        hitAndRunCounts[labelName] = (hitAndRunCounts[labelName] || 0) + 1
        addMatch("hitandrun", s, labelName)
        matched = true
      }

      if (!matched) {
        const mappedKey = labelKeyMap.get(labelName)
        if (mappedKey) {
          addMatch(mappedKey, s, labelName)
          matched = true
        }
      }
      if (!matched) {
        addMatch(keyFromLabel(labelName), s, labelName)
      }

    }

    if (existingDoc) {
      seedAssignmentsFromExistingDoc(
        existingDoc,
        matchesByKey,
        designationAssignments,
        arcAssignments,
        rankValueAssignments,
        outputKeyMap
      )
    }

    const arcKeys = ["drone", "phaser", "heavy"]
    const arcTargets = []
    const designationTargets = []
    for (const key of arcKeys) {
      const matches = matchesByKey.get(key) || []
      for (const match of matches) {
        const squareId = match?.square?.id
        if (squareId === undefined || squareId === null) continue
        const points = Array.isArray(match.square?.points) ? match.square.points : null
        const bbox =
          match.square?.bbox && typeof match.square.bbox === "object"
            ? match.square.bbox
            : null
        const designation =
          key === "heavy" || key === "drone"
            ? String(match.square?.heavyLetter || "").trim()
            : ""
        designationTargets.push({
          key: String(squareId),
          squareId: Number(squareId),
          label: String(match?.label || "").trim(),
          group: key,
          designation,
          points,
          bbox
        })
        arcTargets.push({
          key: String(squareId),
          squareId: Number(squareId),
          label: String(match?.label || "").trim(),
          group: key,
          points,
          bbox
        })
      }
    }
    const missingDesignations = designationTargets.filter((target) => {
      const assignmentKey = String(target.key || target.squareId || "").trim()
      const assigned = assignmentKey
        ? String(designationAssignments[assignmentKey] || "").trim()
        : ""
      return !assigned
    })
    if (missingDesignations.length > 0) {
      return { ok: false, needsDesignations: { items: missingDesignations } }
    }
    const missingArcs = arcTargets.filter((target) => {
      const assignmentKey = String(target.key || target.squareId || "").trim()
      const assigned = assignmentKey ? String(arcAssignments[assignmentKey] || "").trim() : ""
      if (assigned) return false
      const existingArc = String(target.arc || "").trim()
      return !existingArc
    })
    if (missingArcs.length > 0) {
      return { ok: false, needsArcs: { items: missingArcs } }
    }

    const templateSsd =
      output.ssd && typeof output.ssd === "object" ? output.ssd : {}
    const rankValueKeys = ["sensor", "scanner", "damcon"]
    const rankValueTargets = []
    for (const key of rankValueKeys) {
      const matches = (matchesByKey.get(key) || [])
        .slice()
        .sort((a, b) => (a.square?.id ?? 0) - (b.square?.id ?? 0))
      if (matches.length === 0) continue
      const templateEntries = Array.isArray(templateSsd?.[key]) ? templateSsd[key] : []
      for (let i = 0; i < matches.length; i++) {
        const match = matches[i]
        const squareId = match?.square?.id
        if (squareId === undefined || squareId === null) continue
        const templateItem = templateEntries[i] || templateEntries[0] || {}
        const posText = formatPosFromSquare(match.square)
        const bbox =
          match.square?.bbox && typeof match.square.bbox === "object"
            ? match.square.bbox
            : bboxFromPosText(posText)
        rankValueTargets.push({
          key: String(squareId),
          squareId: Number(squareId),
          label: String(match?.label || formatRankValueLabel(key)).trim(),
          group: key,
          value: templateItem?.value,
          rank: templateItem?.rank,
          bbox
        })
      }
    }
    const missingRankValues = rankValueTargets.filter((target) => {
      const assignmentKey = String(target.key || target.squareId || "").trim()
      const assigned = assignmentKey ? rankValueAssignments[assignmentKey] : null
      if (!assigned || typeof assigned !== "object") return true
      return (
        assigned.rank === undefined ||
        assigned.rank === null ||
        assigned.value === undefined ||
        assigned.value === null
      )
    })
    if (missingRankValues.length > 0) {
      return { ok: false, needsRankValues: { items: missingRankValues } }
    }

    const shuttleBays = []
    for (const g of groups) {
      const glabel = normalizeLabel(g?.label || g?.name || "")
      if (!glabel.includes("shuttle")) continue
      let count = Array.isArray(g?.members) ? g.members.length : 0
      if (!count && g?.id !== undefined && g?.id !== null) {
        count = squares.filter(s => Number(s.groupId) === Number(g.id)).length
      }
      if (count > 0) {
        shuttleBays.push({ size: count, admin: count })
      }
    }

  if (output.shipData && typeof output.shipData === "object") {
    output.shipData = blankValuesDeep(output.shipData)
    const existingShipData =
      existingDoc && existingDoc.shipData && typeof existingDoc.shipData === "object"
        ? existingDoc.shipData
        : null
    if (existingShipData) {
      for (const [k, v] of Object.entries(existingShipData)) {
        if (k in output.shipData) {
          output.shipData[k] = v
        }
      }
    }
    output.shipData.empire = empire
    output.shipData.type = type
      if (overrides && typeof overrides === "object") {
        for (const [k, v] of Object.entries(overrides)) {
          if (k in output.shipData) {
            output.shipData[k] = v
          }
        }
      }
      if (overrides && "spareShuttles" in overrides) {
        if (!output.shipData.shuttles || typeof output.shipData.shuttles !== "object") {
          output.shipData.shuttles = {}
        }
        output.shipData.shuttles.spareShuttles = overrides.spareShuttles
      }
      if (shuttleBays.length > 0) {
        if (!output.shipData.shuttles || typeof output.shipData.shuttles !== "object") {
          output.shipData.shuttles = {}
        }
        output.shipData.shuttles.bay = shuttleBays
      }
    }

    if (!output.ssd || typeof output.ssd !== "object") {
      output.ssd = {}
    }

    const newSsd = {}

    const buildEntries = (key, templateArr, useLabelTypeAlways) => {
      const matches = (matchesByKey.get(key) || [])
        .sort((a, b) => (a.square?.id ?? 0) - (b.square?.id ?? 0))
      if (matches.length === 0) return null

      const keepType = key === "heavy" || key === "phaser" || key === "drone"
      const entries = []
      for (let i = 0; i < matches.length; i++) {
        const match = matches[i]
        const templateItem = templateArr[i] || templateArr[0] || {}
        const entry = cloneTemplateItem(templateItem)
        const labelText = String(match.label || "").trim()
        const pos = formatPosFromSquare(match.square)
        const heavyLetter =
          key === "heavy" || key === "drone"
            ? String(match.square?.heavyLetter || "").trim()
            : ""
        const designationAssignment =
          key === "heavy" || key === "drone" || key === "phaser"
            ? String(designationAssignments[String(match.square?.id)] || "").trim()
            : ""
        const arcAssignment =
          key === "heavy" || key === "drone" || key === "phaser"
            ? String(arcAssignments[String(match.square?.id)] || "").trim()
            : ""
        const rankValueAssignment =
          key === "sensor" || key === "scanner" || key === "damcon"
            ? rankValueAssignments[String(match.square?.id)]
            : null

        if (useLabelTypeAlways || i >= templateArr.length) {
          if ("type" in entry || labelText) entry.type = labelText
          if ("designation" in entry) entry.designation = ""
          if ("arc" in entry) entry.arc = ""
        }

        if (!keepType && "type" in entry) {
          delete entry.type
        }

        const designationValue = designationAssignment || heavyLetter
        if (designationValue && "designation" in entry) {
          entry.designation = designationValue
        }

        if (arcAssignment) {
          entry.arc = arcAssignment
        }

        if (rankValueAssignment && typeof rankValueAssignment === "object") {
          if (rankValueAssignment.value !== undefined) entry.value = rankValueAssignment.value
          if (rankValueAssignment.rank !== undefined) entry.rank = rankValueAssignment.rank
        }

        entry.pos = pos
        entries.push(entry)
      }
      return entries
    }

    const preferredOrder = [
      "heavy",
      "drone",
      "phaser",
      "flagbridge",
      "bridge",
      "cloak",
      "tran",
      "trac",
      "fhull",
      "imp",
      "cwarp",
      "ahull",
      "btty",
      "lab",
      "lwarp",
      "rwarp",
      "emer",
      "shuttle",
      "fighter",
      "aux",
      "apr",
      "probe",
      "probeammo",
      "boarding",
      "crew",
      "hitandrun",
      "tbomb",
      "dbomb",
      "sensor",
      "scanner",
      "damcon",
      "excessdamage",
      "shields"
    ]

    const handled = new Set()
    const addKey = (key) => {
      if (handled.has(key)) return
      const outputKey = outputKeyMap.get(key) || key
      const value = templateSsd[key]
      if (Array.isArray(value)) {
        const useLabelTypeAlways = key === "heavy" || key === "phaser" || key === "drone"
        const entries = buildEntries(key, value, useLabelTypeAlways)
        if (entries) newSsd[outputKey] = entries
        handled.add(key)
        return
      }
      if (key === "shields") {
        if (hasShields) {
          const next = {}
          const templateShields =
            templateSsd.shields && typeof templateSsd.shields === "object" && !Array.isArray(templateSsd.shields)
              ? templateSsd.shields
              : {}
          for (const [shieldKey, shieldValue] of Object.entries(templateShields)) {
            if (getShieldIndexFromLabel(shieldKey) === 0) {
              next[shieldKey] = shieldValue
            }
          }
          for (let i = 1; i <= 6; i++) {
            next[shieldOutputLabels[i - 1]] = shieldCounts[i - 1]
          }
          newSsd[outputKey] = next
        }
        handled.add(key)
      }
    }

    for (const key of preferredOrder) {
      if (key in templateSsd || key === "shields") {
        addKey(key)
      } else if (matchesByKey.has(key)) {
        const entries = buildEntries(key, [], true)
        if (entries) newSsd[outputKeyMap.get(key) || key] = entries
        handled.add(key)
      }
    }

    for (const [key] of matchesByKey.entries()) {
      if (handled.has(key)) continue
      const entries = buildEntries(key, [], true)
      if (entries) newSsd[outputKeyMap.get(key) || key] = entries
    }

    output.ssd = newSsd

    if (output.shipData && typeof output.shipData === "object") {
      output.shipData.crewUnits = crewUnitsCount
      output.shipData.boardingParties = boardingPartiesCount
      output.shipData.tBombs = tBombsCount
      output.shipData.dBombs = dBombsCount
      output.shipData.probes = probesCount

      const hitAndRunOut = {}
      for (const term of hitAndRunTerms) {
        const key = String(term || "").trim()
        if (!key) continue
        const count = hitAndRunCounts[key] || 0
        if (count > 0) hitAndRunOut[key] = count
      }
      if (Object.keys(hitAndRunOut).length > 0) {
        output.shipData.hitAndRun = hitAndRunOut
      } else if (output.shipData.hitAndRun) {
        delete output.shipData.hitAndRun
      }
    }

    const targetFolder = getSuperluminalOutputFolder(inputFolder)
    await fs.mkdir(targetFolder, { recursive: true })

    const outputBaseName =
      stripVeilMarker(path.basename(inputFolder)) ||
      stripVeilMarker(path.basename(jsonFile, path.extname(jsonFile))) ||
      "Ship"
    const outputJsonPath = path.join(
      targetFolder,
      isSuperluminalInput ? jsonFile : `${outputBaseName}.json`
    )
    const outputImagePath = path.join(
      targetFolder,
      isSuperluminalInput ? imageFile : `${outputBaseName}${path.extname(imageFile)}`
    )

    await fs.writeFile(outputJsonPath, JSON.stringify(output, null, 2), "utf-8")
    await fs.copyFile(imagePath, outputImagePath)

    let warning = null
    const inputFolderName = path.basename(inputFolder)
    const strippedInputFolderName = stripVeilMarker(inputFolderName)
    const inputWasMarked = strippedInputFolderName !== inputFolderName
    if (!isSuperluminalInput && inputWasMarked) {
      const unmarkedName = makeVeilUnmarkedName(inputFolderName)
      if (unmarkedName && unmarkedName !== inputFolderName) {
        try {
          await renameFolderAndFilesToName(unmarkedName, inputFolder)
        } catch (err) {
          warning = `Converted, but failed to remove marker from folder: ${err?.message || err}`
        }
      }
    }

    const result = { ok: true, path: outputJsonPath }
    if (warning) result.warning = warning
    return result
  } catch (e) {
    return { ok: false, error: e?.message || "Conversion failed." }
  }
})

ipcMain.handle("app:getVersion", () => {
  return app.getVersion()
})

ipcMain.handle("app:openUpdates", async () => {
  await shell.openExternal("https://github.com/Bennettthedog/Shipyard/releases")
  return { ok: true }
})

ipcMain.handle("app:checkForUpdates", async () => {
  if (!autoUpdater) {
    return { ok: false, error: "Updater not installed." }
  }
  if (!app.isPackaged) {
    return { ok: false, error: "Updates are only available in packaged builds." }
  }
  try {
    autoUpdater.checkForUpdates()
    return { ok: true }
  } catch (e) {
    return { ok: false, error: e?.message || "Failed to check for updates." }
  }
})

if (autoUpdater) {
  autoUpdater.on("update-not-available", () => {
    dialog.showMessageBox({
      type: "info",
      title: "No Updates",
      message: "You are already on the latest version."
    })
  })

  autoUpdater.on("error", (e) => {
    dialog.showMessageBox({
      type: "error",
      title: "Update Error",
      message: e?.message || "An error occurred while checking for updates."
    })
  })

  autoUpdater.on("update-downloaded", async () => {
    const res = await dialog.showMessageBox({
      type: "question",
      title: "Update Ready",
      message: "An update has been downloaded. Restart now to install?",
      buttons: ["Restart", "Later"],
      defaultId: 0,
      cancelId: 1
    })
    if (res.response === 0) {
      autoUpdater.quitAndInstall()
    }
  })
}
