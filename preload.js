const { contextBridge, ipcRenderer } = require("electron")

const invoke = (channel, ...args) => ipcRenderer.invoke(channel, ...args)
const on = (channel, cb) => {
  if (typeof cb !== "function") return () => {}
  const wrapped = (_event, payload) => cb(payload)
  ipcRenderer.on(channel, wrapped)
  return () => ipcRenderer.removeListener(channel, wrapped)
}

contextBridge.exposeInMainWorld("api", {
  saveJson: (jsonPath, jsonText, options) => invoke("saveJson", jsonPath, jsonText, options),

  shipyardReadBoxes: () => invoke("shipyard:readBoxes"),
  shipyardReadExistingSuperluminal: (inputFolder) =>
    invoke("shipyard:readExistingSuperluminal", inputFolder),
  shipyardReadSectionEntries: (sectionLabel) =>
    invoke("shipyard:readSectionEntries", sectionLabel),
  readHeavyList: () => invoke("shipyard:readHeavyList"),
  readExternalList: () => invoke("shipyard:readExternalList"),
  shipyardReadWeaponDamage: () => invoke("shipyard:readWeaponDamage"),
  shipyardReadTurnMovement: () => invoke("shipyard:readTurnMovement"),
  shipyardListMissingSuperluminalShips: () =>
    invoke("shipyard:listMissingSuperluminalShips"),
  shipyardPickFolderNamed: () => invoke("shipyard:pickFolderNamed"),

  // NEW: load without rename
  shipyardLoadFolderDataset: (folderPath) => invoke("shipyard:loadFolderDataset", folderPath),

  // rename (optional)
  shipyardRenameFolderAndFiles: (shipName, folderPath) =>
    invoke("shipyard:renameFolderAndFiles", shipName, folderPath),

  runShipyardXXVI: () => invoke("shipyard:runXXVI"),
  onShipyardXXVIProgress: (cb) => on("shipyard:xxvi:progress", cb),
  onShipyardXXVIDone: (cb) => on("shipyard:xxvi:done", cb),
  onShipyardXXVIError: (cb) => on("shipyard:xxvi:error", cb),
  pickSuperluminalInput: () => invoke("shipyard:pickSuperluminalInput"),
  pickSuperluminalShip: () => invoke("shipyard:pickSuperluminalShip"),
  convertSSDJson: (payload) => invoke("shipyard:convertSSDJson", payload),

  editSetupPaths: () => invoke("setup:editPaths"),
  getAppVersion: () => invoke("app:getVersion"),
  openUpdates: () => invoke("app:openUpdates"),
  checkForUpdates: () => invoke("app:checkForUpdates")
})
