import cv2
import numpy as np
import argparse
import json
import os
import ctypes
import uuid
from ctypes import wintypes
from tkinter import Tk, filedialog, messagebox

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
HRESULT = ctypes.c_long
COINIT_APARTMENTTHREADED = 0x2
CLSCTX_INPROC_SERVER = 0x1
FOS_PICKFOLDERS = 0x20
FOS_FORCEFILESYSTEM = 0x40
FOS_ALLOWMULTISELECT = 0x200
FOS_PATHMUSTEXIST = 0x800
SIGDN_FILESYSPATH = 0x80058000
HRESULT_CANCELLED = 0x800704C7
HRESULT_CHANGED_MODE = 0x80010106


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, text):
        value = uuid.UUID(text)
        tail = (ctypes.c_ubyte * 8).from_buffer_copy(value.bytes[8:])
        return cls(value.time_low, value.time_mid, value.time_hi_version, tail)


def hresultValue(hr):
    return ctypes.c_ulong(hr).value


def hresultFailed(hr):
    return hr < 0


def getComMethod(interfacePtr, index, restype, *argtypes):
    vtable = ctypes.cast(
        interfacePtr,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def releaseComObject(interfacePtr):
    if not interfacePtr:
        return
    release = getComMethod(interfacePtr, 2, wintypes.ULONG)
    release(interfacePtr)


def listImagePathsInFolder(folderPath):
    if not folderPath or not os.path.isdir(folderPath):
        return []

    imagePaths = []
    for name in sorted(os.listdir(folderPath), key = str.lower):
        fullPath = os.path.join(folderPath, name)
        if not os.path.isfile(fullPath):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            imagePaths.append(fullPath)
    return imagePaths


def listShipImagePathsInFolders(folderPaths):
    imagePaths = []
    seen = set()
    for folderPath in folderPaths:
        folderImages = listImagePathsInFolder(folderPath)
        if not folderImages:
            continue

        selectedImage = folderImages[0]
        key = os.path.normcase(os.path.abspath(selectedImage))
        if key in seen:
            continue
        seen.add(key)
        imagePaths.append(selectedImage)

        if len(folderImages) > 1:
            print(f"Multiple images found in folder; using first image only: {folderPath}")
    return imagePaths


def chooseFolderPathsOneByOne(root):
    folderPaths = []
    while True:
        folderPath = filedialog.askdirectory(
            title = "Choose a ship folder",
            parent = root
        )
        if not folderPath:
            break
        folderPaths.append(folderPath)
        addAnother = messagebox.askyesno(
            "Add Another Folder",
            "Queue another ship folder?",
            parent = root
        )
        if not addAnother:
            break
    return folderPaths


def chooseFolderPathsAtOnceWindows():
    if os.name != "nt":
        return None

    try:
        ole32 = ctypes.OleDLL("ole32")
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        ole32.CoInitializeEx.restype = HRESULT
        ole32.CoUninitialize.argtypes = []
        ole32.CoUninitialize.restype = None
        ole32.CoCreateInstance.argtypes = [
            ctypes.POINTER(GUID),
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ole32.CoCreateInstance.restype = HRESULT
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole32.CoTaskMemFree.restype = None

        hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        hrValue = hresultValue(hr)
        if hresultFailed(hr) and hrValue != HRESULT_CHANGED_MODE:
            return None

        shouldUninit = hrValue in (0, 1)
        dialog = ctypes.c_void_p()
        results = ctypes.c_void_p()
        folderPaths = []

        clsidFileOpenDialog = GUID.from_string("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")
        iidFileOpenDialog = GUID.from_string("{D57C7288-D4AD-4768-BE02-9D969532D960}")

        try:
            hr = ole32.CoCreateInstance(
                ctypes.byref(clsidFileOpenDialog),
                None,
                CLSCTX_INPROC_SERVER,
                ctypes.byref(iidFileOpenDialog),
                ctypes.byref(dialog)
            )
            if hresultFailed(hr):
                return None

            getOptions = getComMethod(dialog, 10, HRESULT, ctypes.POINTER(wintypes.DWORD))
            setOptions = getComMethod(dialog, 9, HRESULT, wintypes.DWORD)
            setTitle = getComMethod(dialog, 17, HRESULT, ctypes.c_wchar_p)
            show = getComMethod(dialog, 3, HRESULT, wintypes.HWND)
            getResults = getComMethod(dialog, 27, HRESULT, ctypes.POINTER(ctypes.c_void_p))

            options = wintypes.DWORD()
            hr = getOptions(dialog, ctypes.byref(options))
            if hresultFailed(hr):
                return None

            options.value |= (
                FOS_PICKFOLDERS |
                FOS_FORCEFILESYSTEM |
                FOS_ALLOWMULTISELECT |
                FOS_PATHMUSTEXIST
            )
            hr = setOptions(dialog, options.value)
            if hresultFailed(hr):
                return None

            setTitle(dialog, "Choose one or more ship folders")
            hr = show(dialog, 0)
            if hresultValue(hr) == HRESULT_CANCELLED:
                return []
            if hresultFailed(hr):
                return None

            hr = getResults(dialog, ctypes.byref(results))
            if hresultFailed(hr):
                return None

            getCount = getComMethod(results, 7, HRESULT, ctypes.POINTER(wintypes.DWORD))
            getItemAt = getComMethod(results, 8, HRESULT, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))

            count = wintypes.DWORD()
            hr = getCount(results, ctypes.byref(count))
            if hresultFailed(hr):
                return None

            for index in range(count.value):
                item = ctypes.c_void_p()
                hr = getItemAt(results, index, ctypes.byref(item))
                if hresultFailed(hr):
                    continue

                try:
                    getDisplayName = getComMethod(
                        item,
                        5,
                        HRESULT,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_void_p)
                    )
                    namePtr = ctypes.c_void_p()
                    hr = getDisplayName(item, SIGDN_FILESYSPATH, ctypes.byref(namePtr))
                    if hresultFailed(hr) or not namePtr.value:
                        continue
                    try:
                        folderPaths.append(ctypes.wstring_at(namePtr.value))
                    finally:
                        ole32.CoTaskMemFree(namePtr)
                finally:
                    releaseComObject(item)

            return folderPaths
        finally:
            releaseComObject(results)
            releaseComObject(dialog)
            if shouldUninit:
                ole32.CoUninitialize()
    except Exception:
        return None

def chooseImagePaths():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    pickFolder = messagebox.askyesnocancel(
        "Choose SSD Input",
        "Yes: choose one or more ship folders and queue the image from each.\n"
        "No: choose one or more image files.\n"
        "Cancel: stop.",
        parent = root
    )

    if pickFolder is None:
        root.destroy()
        return []

    if pickFolder:
        folderPaths = chooseFolderPathsAtOnceWindows()
        if folderPaths is None:
            folderPaths = chooseFolderPathsOneByOne(root)

        if not folderPaths:
            root.destroy()
            return []

        imagePaths = listShipImagePathsInFolders(folderPaths)
        if not imagePaths:
            messagebox.showinfo(
                "No Images Found",
                "The selected folders do not contain any supported ship images.",
                parent = root
            )
        root.destroy()
        return imagePaths

    paths = filedialog.askopenfilenames(
        title = "Choose one or more images",
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
        parent = root
    )
    root.destroy()
    return [str(p) for p in paths if p]


def rotatePortraitImageInPlace(imagePath, imageBgr):
    h, w = imageBgr.shape[:2]
    if h <= w:
        return imageBgr, False, True

    rotated = cv2.rotate(imageBgr, cv2.ROTATE_90_CLOCKWISE)
    saved = cv2.imwrite(imagePath, rotated)
    return rotated, True, bool(saved)


def orderPoints(pts):
    rect = np.zeros((4, 2), dtype = np.float32)
    s = pts.sum(axis = 1)
    diff = np.diff(pts, axis = 1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def polygonMask(shapeHW, polyPts):
    mask = np.zeros(shapeHW, dtype = np.uint8)
    cv2.fillPoly(mask, [polyPts.astype(np.int32)], 255)
    return mask


def borderIsDarkEnough(gray, quadPts, borderThickness = 2, borderMaxMean = 170):
    h, w = gray.shape[:2]
    poly = quadPts.astype(np.float32)
    mask = polygonMask((h, w), poly)

    k = max(1, int(borderThickness))
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations = 1)
    borderRing = cv2.subtract(mask, eroded)

    vals = gray[borderRing > 0]
    if vals.size < 50:
        return True

    meanVal = float(np.mean(vals))
    return meanVal <= float(borderMaxMean)


def isSquareApprox(approx, minArea, maxArea, aspectTol, rightAngleTolDeg):
    if len(approx) != 4:
        return False
    if not cv2.isContourConvex(approx):
        return False

    area = cv2.contourArea(approx)
    if area < minArea:
        return False
    if maxArea is not None and area > maxArea:
        return False

    pts = approx.reshape(4, 2).astype(np.float32)
    rect = orderPoints(pts)

    x, y, bw, bh = cv2.boundingRect(rect.astype(np.int32))
    if bh == 0:
        return False
    aspect = bw / float(bh)
    if abs(aspect - 1.0) > aspectTol:
        return False

    def angleDeg(a, b, c):
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom == 0:
            return 0.0
        cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    angles = [
        angleDeg(rect[3], rect[0], rect[1]),
        angleDeg(rect[0], rect[1], rect[2]),
        angleDeg(rect[1], rect[2], rect[3]),
        angleDeg(rect[2], rect[3], rect[0]),
    ]
    for ang in angles:
        if abs(ang - 90.0) > rightAngleTolDeg:
            return False

    return True


def preprocessForBoxes(
    imageBgr,
    adaptiveBlockSize = 31,
    adaptiveC = 7,
    openKernel = 3,
    openIters = 1,
    closeKernel = 3,
    closeIters = 1
):
    gray = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    grayEq = clahe.apply(gray)

    blockSize = int(adaptiveBlockSize)
    if blockSize % 2 == 0:
        blockSize += 1
    blockSize = max(3, blockSize)

    binImg = cv2.adaptiveThreshold(
        grayEq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize, int(adaptiveC)
    )

    ok = max(1, int(openKernel))
    openKernelMat = np.ones((ok, ok), np.uint8)
    opened = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, openKernelMat, iterations = int(openIters))

    k = max(1, int(closeKernel))
    closeKernelMat = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closeKernelMat, iterations = int(closeIters))

    return grayEq, closed


def findSquaresOnce(
    imageBgr,
    minArea = 50,
    maxArea = None,
    aspectTol = 0.22,
    rightAngleTolDeg = 20,
    approxEpsFactor = 0.02,
    borderCheck = True,
    borderThickness = 2,
    borderMaxMean = 170,
    adaptiveBlockSize = 31,
    adaptiveC = 7,
    openKernel = 3,
    openIters = 1,
    closeKernel = 3,
    closeIters = 1,
    progressTracker = None,
    progressMsg = "",
):
    grayEq, binClosed = preprocessForBoxes(
        imageBgr,
        adaptiveBlockSize = adaptiveBlockSize,
        adaptiveC = adaptiveC,
        openKernel = openKernel,
        openIters = openIters,
        closeKernel = closeKernel,
        closeIters = closeIters
    )

    contours, _ = cv2.findContours(binClosed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if progressTracker:
        progressTracker.add_total(len(contours))
    if not progressMsg:
        progressMsg = "Detecting squares (contours)"

    def tick():
        if progressTracker:
            progressTracker.advance(1, progressMsg)

    squares = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            tick()
            continue

        eps = float(approxEpsFactor) * peri
        approx = cv2.approxPolyDP(cnt, eps, True)

        if not isSquareApprox(approx, minArea, maxArea, aspectTol, rightAngleTolDeg):
            tick()
            continue

        rect = orderPoints(approx.reshape(4, 2).astype(np.float32))

        if borderCheck:
            if not borderIsDarkEnough(grayEq, rect, borderThickness = borderThickness, borderMaxMean = borderMaxMean):
                tick()
                continue

        squares.append(rect)
        tick()

    debug = {"gray": grayEq, "binary": binClosed}
    return squares, debug


def estimateCellSize(h, w, cellDiv = 66.0):
    return max(h, w) / float(cellDiv)


def extractGridLines(binImg, cellSize, lineFrac = 0.60, minKernel = 9, lineIters = 1, dilateAfter = 1):
    k = int(max(minKernel, cellSize * float(lineFrac)))
    k = max(3, k)

    horizKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    vertKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))

    horiz = cv2.erode(binImg, horizKernel, iterations = int(lineIters))
    horiz = cv2.dilate(horiz, horizKernel, iterations = int(lineIters))

    vert = cv2.erode(binImg, vertKernel, iterations = int(lineIters))
    vert = cv2.dilate(vert, vertKernel, iterations = int(lineIters))

    lines = cv2.bitwise_or(horiz, vert)

    if dilateAfter > 0:
        dk = np.ones((3, 3), np.uint8)
        lines = cv2.dilate(lines, dk, iterations = int(dilateAfter))

    return lines


def findSquaresSharedBordersOnce(
    imageBgr,
    minArea = 50,
    maxArea = None,
    aspectTol = 0.22,
    rightAngleTolDeg = 20,
    approxEpsFactor = 0.02,
    adaptiveBlockSize = 31,
    adaptiveC = 7,
    openKernel = 3,
    openIters = 1,
    closeKernel = 3,
    closeIters = 1,
    cellDiv = 66.0,
    lineFrac = 0.60,
    minLineKernel = 9,
    lineIters = 1,
    lineDilate = 1,
    progressTracker = None,
    progressMsg = "",
):
    grayEq, binClosed = preprocessForBoxes(
        imageBgr,
        adaptiveBlockSize = adaptiveBlockSize,
        adaptiveC = adaptiveC,
        openKernel = openKernel,
        openIters = openIters,
        closeKernel = closeKernel,
        closeIters = closeIters
    )

    h, w = grayEq.shape[:2]
    cellSize = estimateCellSize(h, w, cellDiv = cellDiv)

    lineMask = extractGridLines(
        binClosed,
        cellSize = cellSize,
        lineFrac = lineFrac,
        minKernel = minLineKernel,
        lineIters = lineIters,
        dilateAfter = lineDilate
    )

    inv = cv2.bitwise_not(lineMask)

    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity = 4)

    if progressTracker:
        progressTracker.add_total(max(0, numLabels - 1))
    if not progressMsg:
        progressMsg = "Detecting squares (components)"

    def tick():
        if progressTracker:
            progressTracker.advance(1, progressMsg)

    squares = []
    for i in range(1, numLabels):
        x, y, bw, bh, area = stats[i]

        if x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h:
            tick()
            continue

        if area < minArea:
            tick()
            continue
        if maxArea is not None and area > maxArea:
            tick()
            continue

        comp = (labels == i).astype(np.uint8) * 255

        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            tick()
            continue

        cnt = max(contours, key = cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            tick()
            continue

        eps = float(approxEpsFactor) * peri
        approx = cv2.approxPolyDP(cnt, eps, True)

        if not isSquareApprox(approx, minArea, maxArea, aspectTol, rightAngleTolDeg):
            tick()
            continue

        rect = orderPoints(approx.reshape(4, 2).astype(np.float32))
        squares.append(rect)
        tick()

    debug = {"gray": grayEq, "binary": binClosed, "lines": lineMask, "cells": inv}
    return squares, debug


def rectToBBox(rect):
    x, y, w, h = cv2.boundingRect(rect.astype(np.int32))
    return (x, y, x + w, y + h)


def bboxArea(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def bboxIntersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return iw * ih


def iou(a, b):
    inter = bboxIntersection(a, b)
    areaA = bboxArea(a)
    areaB = bboxArea(b)
    denom = areaA + areaB - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def rectSideLength(rect):
    r = rect.astype(np.float32)
    d01 = np.linalg.norm(r[1] - r[0])
    d12 = np.linalg.norm(r[2] - r[1])
    d23 = np.linalg.norm(r[3] - r[2])
    d30 = np.linalg.norm(r[0] - r[3])
    return float((d01 + d12 + d23 + d30) / 4.0)


def removeLargeNearDuplicateSquares(
    squares,
    iouThresh = 0.85,
    containThresh = 0.95,
    areaRatioMin = 1.02,
):
    if not squares:
        return [], []

    items = []
    for r in squares:
        b = rectToBBox(r)
        a = bboxArea(b)
        items.append((a, r, b))

    items.sort(key = lambda x: x[0])

    kept = []
    keptBoxes = []
    keptAreas = []
    removed = []

    for areaC, rectC, boxC in items:
        drop = False

        for areaK, boxK in zip(keptAreas, keptBoxes):
            inter = bboxIntersection(boxC, boxK)
            if inter <= 0:
                continue

            fracSmallCovered = inter / float(max(1, areaK))
            iouVal = iou(boxC, boxK)

            if (fracSmallCovered >= float(containThresh) or iouVal >= float(iouThresh)) and areaC >= areaK * float(areaRatioMin):
                drop = True
                break

        if drop:
            removed.append(rectC)
        else:
            kept.append(rectC)
            keptBoxes.append(boxC)
            keptAreas.append(areaC)

    return kept, removed


def filterExtremeSizeSquaresFromAvg(
    squares,
    minScaleFromAvg = 0.10,
    maxScaleFromAvg = 2.00,
    minCount = 12,
):
    if not squares:
        return [], [], 0.0, (0.0, 0.0)

    if len(squares) < int(minCount):
        return squares, [], 0.0, (0.0, 0.0)

    sizes = np.array([rectSideLength(r) for r in squares], dtype = np.float32)
    avg = float(np.mean(sizes))
    if avg <= 1e-6:
        return squares, [], avg, (0.0, 0.0)

    lo = avg * float(minScaleFromAvg)
    hi = avg * float(maxScaleFromAvg)

    kept = []
    removed = []
    for r, s in zip(squares, sizes.tolist()):
        if s <= lo or s >= hi:
            removed.append(r)
        else:
            kept.append(r)

    if len(kept) < max(1, int(0.20 * len(squares))):
        return squares, [], avg, (lo, hi)

    return kept, removed, avg, (lo, hi)


class ProgressTracker:
    def __init__(self, cb, emitEvery = 25):
        self.cb = cb
        self.emitEvery = max(1, int(emitEvery))
        self.done = 0
        self.total = 0
        self.lastEmit = 0

    def add_total(self, n):
        if n > 0:
            self.total += int(n)

    def advance(self, n = 1, msg = ""):
        if n <= 0:
            return
        self.done += int(n)
        if self.cb:
            if (self.done - self.lastEmit) >= self.emitEvery or self.done >= self.total:
                self.lastEmit = self.done
                self.cb(self.done, max(1, self.total), msg)

    def finalize(self, msg = ""):
        if self.cb:
            self.cb(self.done, max(1, self.total), msg)


def detectSquaresMultiScale(
    imageBgr,
    scales = (1.0, 1.6, 2.2),
    minArea = 50,
    maxArea = None,
    aspectTol = 0.22,
    rightAngleTolDeg = 20,
    approxEpsFactor = 0.02,
    borderCheck = True,
    borderThickness = 2,
    borderMaxMean = 170,
    adaptiveBlockSize = 31,
    adaptiveC = 7,
    openKernel = 3,
    openIters = 1,
    closeKernel = 3,
    closeIters = 1,
    sharedBorders = True,
    cellDiv = 66.0,
    lineFrac = 0.60,
    minLineKernel = 9,
    lineIters = 1,
    lineDilate = 1,
    progressCb = None,
):
    allSquares = []
    debugBest = None
    bestCount = -1

    progressTracker = ProgressTracker(progressCb, emitEvery = 25) if progressCb else None

    for s in scales:
        if s == 1.0:
            resized = imageBgr
        else:
            resized = cv2.resize(imageBgr, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)

        squaresA, debugA = findSquaresOnce(
            resized,
            minArea = minArea,
            maxArea = maxArea,
            aspectTol = aspectTol,
            rightAngleTolDeg = rightAngleTolDeg,
            approxEpsFactor = approxEpsFactor,
            borderCheck = borderCheck,
            borderThickness = max(1, int(borderThickness)),
            borderMaxMean = borderMaxMean,
            adaptiveBlockSize = adaptiveBlockSize,
            adaptiveC = adaptiveC,
            openKernel = openKernel,
            openIters = openIters,
            closeKernel = max(1, int(closeKernel)),
            closeIters = max(1, int(closeIters)),
            progressTracker = progressTracker,
            progressMsg = f"Detecting squares (scale {s:.2f}, contours)",
        )

        squaresB = []
        debugB = None
        if sharedBorders:
            squaresB, debugB = findSquaresSharedBordersOnce(
                resized,
                minArea = minArea,
                maxArea = maxArea,
                aspectTol = aspectTol,
                rightAngleTolDeg = rightAngleTolDeg,
                approxEpsFactor = approxEpsFactor,
                adaptiveBlockSize = adaptiveBlockSize,
                adaptiveC = adaptiveC,
                openKernel = openKernel,
                openIters = openIters,
                closeKernel = max(1, int(closeKernel)),
                closeIters = max(1, int(closeIters)),
                cellDiv = cellDiv,
                lineFrac = lineFrac,
                minLineKernel = minLineKernel,
                lineIters = lineIters,
                lineDilate = lineDilate,
                progressTracker = progressTracker,
                progressMsg = f"Detecting squares (scale {s:.2f}, components)",
            )

        for r in squaresA:
            allSquares.append((r / float(s)).astype(np.float32))
        for r in squaresB:
            allSquares.append((r / float(s)).astype(np.float32))

        totalCount = len(squaresA) + len(squaresB)
        if totalCount > bestCount:
            bestCount = totalCount
            debugBest = {
                "gray": debugA.get("gray", None),
                "binary": debugA.get("binary", None),
                "lines": None if debugB is None else debugB.get("lines", None),
                "cells": None if debugB is None else debugB.get("cells", None),
            }

    if progressTracker:
        progressTracker.finalize("Detecting squares (done)")

    return allSquares, debugBest


def drawSquaresOn(outBgr, squares, colorBgr, thickness = 2):
    for rect in squares:
        pts = rect.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(outBgr, [pts], True, colorBgr, int(thickness))
    return outBgr


# -------------------------------
# GROUPING (distance <= 1/2 avg side; groups >= 2)
# -------------------------------
def bboxRectDistance(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    dx = 0
    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2

    dy = 0
    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2

    return float(np.hypot(dx, dy))


def groupSquaresByProximity(squares, maxDistPx, minGroupSize = 2):
    n = len(squares)
    if n == 0:
        return []

    boxes = [rectToBBox(r) for r in squares]

    adj = [[] for _ in range(n)]
    for i in range(n):
        bi = boxes[i]
        for j in range(i + 1, n):
            bj = boxes[j]
            if bboxRectDistance(bi, bj) <= float(maxDistPx):
                adj[i].append(j)
                adj[j].append(i)

    groups = []
    seen = [False] * n

    for i in range(n):
        if seen[i]:
            continue

        stack = [i]
        seen[i] = True
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        if len(comp) >= int(minGroupSize):
            groups.append(comp)

    groups.sort(key = lambda g: -len(g))
    return groups


def drawGroupsOverlay(imageBgr, squares, groups, alpha = 0.35, thickness = 2, drawIds = True):
    out = imageBgr.copy()

    # draw all squares faintly so ungrouped ones still appear
    out = drawSquaresOn(out, squares, (0, 255, 255), thickness = 1)

    overlay = out.copy()

    rng = np.random.default_rng(1337)
    colors = []
    for _ in range(len(groups)):
        c = rng.integers(40, 256, size = 3, dtype = np.int32).tolist()
        colors.append((int(c[0]), int(c[1]), int(c[2])))

    for gi, idxs in enumerate(groups):
        color = colors[gi]

        for idx in idxs:
            rect = squares[idx]
            pts = rect.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], True, color, int(thickness))

        if drawIds:
            centers = []
            for idx in idxs:
                x1, y1, x2, y2 = rectToBBox(squares[idx])
                centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
            cx = int(sum(c[0] for c in centers) / max(1, len(centers)))
            cy = int(sum(c[1] for c in centers) / max(1, len(centers)))
            cv2.putText(out, f"G{gi + 1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)
    return out


# -------------------------------
# JSON EXPORT
# -------------------------------
def rectToPointsList(rect):
    r = rect.astype(np.float32).tolist()
    return [[float(x), float(y)] for x, y in r]


def bboxUnion(boxes):
    if not boxes:
        return (0, 0, 0, 0)
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (int(x1), int(y1), int(x2), int(y2))


def bboxCenter(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def exportGroupsJson(
    jsonPath,
    imageShapeHW,
    squares,
    groups,
    avgSide,
    groupDistThreshPx
):
    h, w = imageShapeHW

    # Map square index -> group id (only for grouped squares)
    squareToGroup = {}
    for gi, idxs in enumerate(groups, start = 1):
        for idx in idxs:
            squareToGroup[int(idx)] = int(gi)

    squaresJson = []
    for i, rect in enumerate(squares):
        b = rectToBBox(rect)
        cx, cy = bboxCenter(b)
        s = rectSideLength(rect)
        squareEntry = {
            "id": int(i),
            "groupId": squareToGroup.get(int(i), None),
            "points": rectToPointsList(rect),
            "bbox": {"x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])},
            "center": {"x": float(cx), "y": float(cy)},
            "sideLengthPx": float(s),
        }
        squaresJson.append(squareEntry)

    groupsJson = []
    for gi, idxs in enumerate(groups, start = 1):
        memberBoxes = [rectToBBox(squares[idx]) for idx in idxs]
        gb = bboxUnion(memberBoxes)
        gcx, gcy = bboxCenter(gb)
        groupsJson.append({
            "id": int(gi),
            "members": [int(x) for x in idxs],
            "bbox": {"x1": int(gb[0]), "y1": int(gb[1]), "x2": int(gb[2]), "y2": int(gb[3])},
            "center": {"x": float(gcx), "y": float(gcy)},
            "count": int(len(idxs)),
        })

    payload = {
        "image": {"width": int(w), "height": int(h)},
        "params": {
            "avgSquareSidePx": float(avgSide),
            "groupDistanceThresholdPx": float(groupDistThreshPx),
            "minGroupSize": 2
        },
        "groups": groupsJson,
        "squares": squaresJson,
    }

    with open(jsonPath, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2)

    return jsonPath


def removeSingleBoxGroups(groups):
    """
    Removes groups that contain only one box.
    """
    return [group for group in groups if len(group) > 1]


def removeOverlappingBoxes(squares, iouThreshold=0.5):
    """
    Removes all but one of the boxes that overlap more than the specified IoU threshold with another box.
    Keeps the smallest box in terms of area.
    """
    def calculateIoU(boxA, boxB):
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])

        interArea = max(0, x2 - x1) * max(0, y2 - y1)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea - interArea
        return interArea / unionArea if unionArea > 0 else 0

    def bboxArea(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    groups = []
    for i, rectA in enumerate(squares):
        bboxA = rectToBBox(rectA)
        added = False
        for group in groups:
            if any(calculateIoU(bboxA, rectToBBox(squares[j])) > iouThreshold for j in group):
                group.append(i)
                added = True
                break
        if not added:
            groups.append([i])

    toKeep = set()
    for group in groups:
        smallest = min(group, key=lambda idx: bboxArea(rectToBBox(squares[idx])))
        toKeep.add(smallest)

    kept = [rect for idx, rect in enumerate(squares) if idx in toKeep]
    removed = [rect for idx, rect in enumerate(squares) if idx not in toKeep]
    return kept, removed


def fitToScreen(imageBgr, maxW, maxH, padding = 80):
    h, w = imageBgr.shape[:2]
    maxW = max(1, int(maxW) - int(padding))
    maxH = max(1, int(maxH) - int(padding))
    if maxW <= 0 or maxH <= 0:
        return imageBgr, 1.0

    scale = min(maxW / float(w), maxH / float(h), 1.0)
    if scale >= 1.0:
        return imageBgr, 1.0

    newW = max(1, int(w * scale))
    newH = max(1, int(h * scale))
    return cv2.resize(imageBgr, (newW, newH), interpolation = cv2.INTER_AREA), float(scale)


def rectFromDrag(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    size = int(max(1, min(abs(dx), abs(dy))))
    if size <= 1:
        return None

    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    x2 = x0 + sx * size
    y2 = y0 + sy * size

    x1i, x2i = sorted([int(x0), int(x2)])
    y1i, y2i = sorted([int(y0), int(y2)])

    rect = np.array([[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]], dtype = np.float32)
    return rect


def processImage(imagePath, args, itemIndex, itemTotal):
    if itemTotal > 1:
        print(f"Processing SSD {itemIndex}/{itemTotal}: {imagePath}")

    image = cv2.imread(imagePath)
    if image is None:
        print("Failed to read image:", imagePath)
        return False

    image, rotatedForLandscape, rotateSaved = rotatePortraitImageInPlace(imagePath, image)
    if rotatedForLandscape:
        if rotateSaved:
            print("Rotated portrait image to landscape and saved:", imagePath)
        else:
            print("Rotated portrait image for this session, but failed to save:", imagePath)

    scales = tuple(float(x.strip()) for x in args.scales.split(",") if x.strip())

    # Simple console progress bar for long-running detection.
    def progressBar(step, total, msg = ""):
        total = max(1, int(total))
        pct = min(1.0, max(0.0, float(step) / total))
        barLen = 30
        filled = int(barLen * pct)
        bar = "#" * filled + "." * (barLen - filled)
        print(f"\r[{bar}] {pct * 100:5.1f}% {msg}", end = "", flush = True)
        if step >= total:
            print()

    rawSquares, debugBest = detectSquaresMultiScale(
        image,
        scales = scales,
        minArea = args.minArea,
        aspectTol = args.aspectTol,
        rightAngleTolDeg = args.angleTol,
        approxEpsFactor = args.eps,
        borderCheck = not args.noBorderCheck,
        borderThickness = args.borderThickness,
        borderMaxMean = args.borderMaxMean,
        adaptiveBlockSize = args.block,
        adaptiveC = args.C,
        openKernel = args.openK,
        openIters = args.openIters,
        closeKernel = args.closeK,
        closeIters = args.closeIters,
        sharedBorders = not args.noSharedBorders,
        cellDiv = args.cellDiv,
        lineFrac = args.lineFrac,
        minLineKernel = args.minLineKernel,
        lineIters = args.lineIters,
        lineDilate = args.lineDilate,
        progressCb = progressBar,
    )

    pre = len(rawSquares)

    overlapRemoved = []
    squaresAfterOverlap = rawSquares
    if not args.noOverlapRemove:
        squaresAfterOverlap, overlapRemoved = removeLargeNearDuplicateSquares(
            rawSquares,
            iouThresh = args.overlapIou,
            containThresh = args.containFrac,
            areaRatioMin = args.areaRatioMin,
        )

    sizeRemoved = []
    squaresFinal = squaresAfterOverlap
    avgSide = 0.0
    loHi = (0.0, 0.0)

    if not args.noSizeFilter:
        squaresFinal, sizeRemoved, avgSide, loHi = filterExtremeSizeSquaresFromAvg(
            squaresAfterOverlap,
            minScaleFromAvg = args.minScaleFromAvg,
            maxScaleFromAvg = args.maxScaleFromAvg,
            minCount = args.minSizeFilterCount,
        )

    if avgSide <= 1e-6 and squaresFinal:
        avgSide = float(np.mean([rectSideLength(r) for r in squaresFinal]))

    # Remove overlapping boxes
    squaresFinal, overlapRemoved2 = removeOverlappingBoxes(squaresFinal, iouThreshold=0.5)

    removedSquares = overlapRemoved + sizeRemoved + overlapRemoved2

    def computeGroupsAndImage(squares):
        if squares:
            avg = float(np.mean([rectSideLength(r) for r in squares]))
        else:
            avg = 0.0

        distThresh = 0.5 * float(avg) if avg > 0 else 0.0
        groupsLocal = groupSquaresByProximity(
            squares,
            maxDistPx=distThresh,
            minGroupSize=2
        )
        groupsLocal = removeSingleBoxGroups(groupsLocal)

        img = drawGroupsOverlay(
            image,
            squares,
            groupsLocal,
            alpha=args.groupAlpha,
            thickness=2,
            drawIds=(not args.noGroupIds)
        )

        if removedSquares:
            img = drawSquaresOn(img, removedSquares, (0, 0, 255), thickness = 2)

        return img, groupsLocal, avg, distThresh

    print(f"Detected squares (raw): {pre}")
    if not args.noOverlapRemove:
        print(f"Removed by overlap (larger near-duplicates): {len(overlapRemoved)}")
    if not args.noSizeFilter:
        if avgSide > 0:
            lo, hi = loHi
            print(f"Avg side length: {avgSide:.2f} px  |  keep range: ({lo:.2f}, {hi:.2f}) px")
        print(f"Removed by size: {len(sizeRemoved)}")
    root = Tk()
    root.withdraw()
    screenW = root.winfo_screenwidth()
    screenH = root.winfo_screenheight()
    root.destroy()

    windowName = "Groups (proximity within 0.5 avg side; groups >= 2)"
    addedSquares = []
    pendingStart = {"pt": None}
    latest = {"groups": [], "avg": 0.0, "dist": 0.0}

    def refreshDisplay():
        img, groupsLocal, avgLocal, distLocal = computeGroupsAndImage(squaresFinal)
        latest["groups"] = groupsLocal
        latest["avg"] = avgLocal
        latest["dist"] = distLocal
        displayImg, displayScale = fitToScreen(img, screenW, screenH)
        cv2.imshow(windowName, displayImg)
        return displayScale

    displayScale = refreshDisplay()

    def onMouse(event, x, y, flags, param):
        nonlocal displayScale
        if event == cv2.EVENT_LBUTTONDOWN:
            pendingStart["pt"] = (int(x / displayScale), int(y / displayScale))
        elif event == cv2.EVENT_LBUTTONUP and pendingStart["pt"] is not None:
            x0, y0 = pendingStart["pt"]
            x1, y1 = int(x / displayScale), int(y / displayScale)
            pendingStart["pt"] = None
            rect = rectFromDrag(x0, y0, x1, y1)
            if rect is None:
                return
            squaresFinal.append(rect)
            addedSquares.append(rect)
            displayScale = refreshDisplay()

    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(windowName, onMouse)

    print("Interactive add: drag to add a square, press 'u' to undo last added, press 'q' or Esc to finish.")
    while True:
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("u"), 8):
            if addedSquares:
                last = addedSquares.pop()
                if last in squaresFinal:
                    squaresFinal.remove(last)
                displayScale = refreshDisplay()

    cv2.destroyAllWindows()

    print(f"Final kept squares: {len(squaresFinal)}")
    print(f"Grouping distance threshold: {latest['dist']:.2f} px (0.5 * avgSide)")
    print(f"Groups (size >= 2): {len(latest['groups'])}")

    # JSON output path default: alongside image, with suffix
    jsonPath = args.jsonOut
    if not jsonPath:
        base, _ = os.path.splitext(imagePath)
        jsonPath = base + "_groups.json"

    try:
        savedJson = exportGroupsJson(
            jsonPath = jsonPath,
            imageShapeHW = image.shape[:2],
            squares = squaresFinal,
            groups = latest["groups"],
            avgSide = latest["avg"],
            groupDistThreshPx = latest["dist"]
        )
        print("Saved JSON:", savedJson)
    except Exception as e:
        print("Failed to write JSON:", e)

    if args.save:
        finalImg, _, _, _ = computeGroupsAndImage(squaresFinal)
        cv2.imwrite(args.save, finalImg)
        print("Saved image:", args.save)

    if itemTotal > 1:
        print(f"Finished SSD {itemIndex}/{itemTotal}: {imagePath}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        nargs = "+",
        default = None,
        help = "One or more image paths. If omitted, a picker lets you choose files or one or more folders."
    )
    parser.add_argument(
        "--folder",
        nargs = "+",
        default = None,
        help = "One or more ship folders. The first supported image in each selected folder will be queued."
    )
    parser.add_argument(
        "--save",
        type = str,
        default = None,
        help = "Optional output path to save an annotated image. Only valid when processing one image."
    )
    parser.add_argument(
        "--jsonOut",
        type = str,
        default = None,
        help = "Optional output path to save groups/squares JSON. Only valid when processing one image."
    )

    parser.add_argument("--minArea", type = int, default = 50)
    parser.add_argument("--aspectTol", type = float, default = 0.22)
    parser.add_argument("--angleTol", type = float, default = 20)
    parser.add_argument("--eps", type = float, default = 0.02)

    parser.add_argument("--block", type = int, default = 31)
    parser.add_argument("--C", type = int, default = 7)
    parser.add_argument("--openK", type = int, default = 3)
    parser.add_argument("--openIters", type = int, default = 1)
    parser.add_argument("--closeK", type = int, default = 3)
    parser.add_argument("--closeIters", type = int, default = 1)

    parser.add_argument("--scales", type = str, default = "1.0,1.6,2.2")

    parser.add_argument("--noBorderCheck", action = "store_true")
    parser.add_argument("--borderThickness", type = int, default = 2)
    parser.add_argument("--borderMaxMean", type = int, default = 170)

    parser.add_argument("--noSharedBorders", action = "store_true")
    parser.add_argument("--cellDiv", type = float, default = 66.0)
    parser.add_argument("--lineFrac", type = float, default = 0.60)
    parser.add_argument("--minLineKernel", type = int, default = 9)
    parser.add_argument("--lineIters", type = int, default = 1)
    parser.add_argument("--lineDilate", type = int, default = 1)

    # Size filter
    parser.add_argument("--noSizeFilter", action = "store_true")
    parser.add_argument("--minScaleFromAvg", type = float, default = 0.70)
    parser.add_argument("--maxScaleFromAvg", type = float, default = 2.00)
    parser.add_argument("--minSizeFilterCount", type = int, default = 12)

    # Overlap removal
    parser.add_argument("--noOverlapRemove", action = "store_true")
    parser.add_argument("--overlapIou", type = float, default = 0.85)
    parser.add_argument("--containFrac", type = float, default = 0.95)
    parser.add_argument("--areaRatioMin", type = float, default = 1.02)

    # Group display
    parser.add_argument("--groupAlpha", type = float, default = 0.35, help = "Overlay alpha for group fill.")
    parser.add_argument("--noGroupIds", action = "store_true", help = "Disable drawing group id labels.")

    args = parser.parse_args()

    if args.image and args.folder:
        print("Use either --image or --folder, not both.")
        return 1

    if args.folder:
        missingFolders = [folderPath for folderPath in args.folder if not os.path.isdir(folderPath)]
        if missingFolders:
            print("Folder not found:", missingFolders[0])
            return 1
        imagePaths = listShipImagePathsInFolders(args.folder)
        if not imagePaths:
            print("No supported ship images found in the selected folders.")
            return 1
    else:
        imagePaths = list(args.image) if args.image else chooseImagePaths()

    if not imagePaths:
        print("No images selected.")
        return 1

    if len(imagePaths) > 1 and (args.save or args.jsonOut):
        print("The --save and --jsonOut options only support one image at a time.")
        return 1

    total = len(imagePaths)
    completed = 0
    for index, imagePath in enumerate(imagePaths, start = 1):
        if processImage(imagePath, args, index, total):
            completed += 1

    if total > 1:
        print(f"Queued SSD run complete: {completed}/{total} processed.")
    return 0 if completed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
