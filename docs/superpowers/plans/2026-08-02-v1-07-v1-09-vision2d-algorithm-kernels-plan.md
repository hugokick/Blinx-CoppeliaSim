# V1-07～V1-09 Vision2D Algorithm Kernels Implementation Plan

> **Execution requirement:** Follow this plan with `executing-plans`,
> `test-driven-development`, `systematic-debugging`, and
> `verification-before-completion`. Every behavioral task must show a real RED
> before production code changes.

**Goal:** Starting from `origin/main` at
`9af91087d29cf3186ef84f3528c5f4c495b1ff71`, deliver isolated, deterministic,
pure二维 V1-07 QR/EAN-13/RS1D code recognition, V1-08 OCR, and V1-09 surface-defect
detection kernels with original synthetic tests. Do not perform formal course
integration.

**Worktree:**
`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-v1-09-algorithm-kernels`

**Branch:** `codex/v2-2-v1-07-v1-09-algorithm-kernels`

**Interpreter:** `.\.venv-vision\Scripts\python.exe` (OpenCV 4.14, NumPy 2.4,
pytest 9.1). The worktree-local `.venv-vision` is an ignored junction to the
already validated local environment; never add it to Git.

## 0. Non-negotiable rules

- Use only this worktree; do not edit the main worktree or any H: drive source.
- Do not modify `vision_platform/vision2d/__init__.py`, existing V1-02～V1-05
  runtime, student runner/SDK, CLI, PyQt, gateway, catalog, templates, formal
  experiment files, `.ttt`, URDF, STL, robot assets, or V1-06 code.
- Use `apply_patch` for source, tests, Markdown and manifest edits.
- For every behavior: RED test first, run the focused command and record the
  failure, implement the minimum code, rerun focused tests, then run the listed
  regression before committing.
- Keep the public API in the three new submodules; tests import those modules
  directly. No network, downloaded model/font/data, binary fixture, or temp
  artifact is allowed.
- A skipped online/hardware test is not PASS. This work package has no online
  test and leaves hardware/course/human acceptance pending.
- Each algorithm receives an independent commit. Do not squash the three
  algorithm commits together.

## Task 1: Record design and plan

**Files:**

- Create `docs/superpowers/specs/2026-08-02-v1-07-v1-09-vision2d-algorithm-kernels-design.md`.
- Create this plan file.

Steps:

1. Confirm branch, HEAD, clean status, worktree path and OpenCV capabilities
   (`QRCodeDetector`, `QRCodeEncoder_create`, `cv2.barcode.BarcodeDetector`,
   `cv2.ml.KNearest_create`, `cv2.ml.SVM_create`).
2. Read the roadmap, existing V1-02～V1-05 design/plan, current integration
   design/plan, `RETAINED_FILES.txt`, and `requirements-vision.txt`.
3. Review the design for exact input/output/error/performance contracts,
   RS1D limitations, OCR train/test isolation, and defect threshold rules.
4. Run `git diff --check`; commit both documents:

```powershell
git add -- docs/superpowers/specs/2026-08-02-v1-07-v1-09-vision2d-algorithm-kernels-design.md `
  docs/superpowers/plans/2026-08-02-v1-07-v1-09-vision2d-algorithm-kernels-plan.md
git commit -m "docs(vision2d): design v1-07 to v1-09 kernels"
```

## Task 2: V1-07 RED tests and original code fixtures

**Files:**

- Create `tests/test_vision2d/synthetic_v107_v109.py`.
- Create `tests/test_vision2d/test_code_recognition.py`.

The test fixture must generate QR images with OpenCV's in-process encoder,
standards-shaped EAN-13 images with the project-owned encoding tables, and RS1D
images with a documented original bit-bar encoder. It must support scale,
rotation, brightness, checksum failure and seeded noise without writing files.

RED steps:

1. Add direct imports of `vision_platform.vision2d.code_recognition` and tests
   for QR round-trip payload and four-corner location.
2. Add multi-QR, rotated/scaled/noisy QR, blank, damaged and invalid-input tests.
   A damaged code must produce `PARTIAL`/`NO_TARGETS` with a non-success failure
   signal; it may not be silently skipped.
3. Add standard EAN-13 round-trip, rotated/scaled/noisy, checksum validation,
   RS1D round-trip, rotated/scaled/noisy, multiple-code, unsupported-format
   boundary, blank and invalid-input tests. RS1D must not stand in for EAN-13.
4. Add deterministic duplicate suppression, finite confidence/bbox assertions,
   maximum-code bound and a performance smoke assertion that only checks finite
   elapsed time.
5. Run the focused RED command and retain the observed import/API failures:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_code_recognition.py
```

Do not create `code_recognition.py` before this RED run.

## Task 3: V1-07 implementation and isolated commit

**Files:**

- Create `vision_platform/vision2d/code_recognition.py`.
- Modify `RETAINED_FILES.txt` by appending the new module and V1-07 tests only.

Implementation steps:

1. Add frozen `CodeReading` and `CodeRecognitionResult`, stable status/error
   validation, BGR uint8 input validation, and `schema_version=1`.
2. Implement QR single/multi detection using OpenCV 4.14, preserving detector
   polygons in original coordinates and returning `PARTIAL` for located-but-
   undecoded points.
3. Implement standard `encode_ean13_payload`/EAN-13 decoding with 95-module
   guard/parity/checksum validation, then bounded `encode_rs1d_payload` decoding
   with magic/length/checksum validation, fixed angle search, row run-length
   parsing, inverse coordinate mapping and duplicate suppression.
4. Ensure all outputs are JSON-like/finitely bounded except immutable tuples,
   and no code path reads a path, imports a network dependency, or silently
   falls back to a different barcode standard.
5. Run focused GREEN, then the existing vision2d suite:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_code_recognition.py
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d
```

6. Run `git diff --check`, inspect the diff for forbidden paths, and commit:

```powershell
git add -- vision_platform/vision2d/code_recognition.py `
  tests/test_vision2d/synthetic_v107_v109.py tests/test_vision2d/test_code_recognition.py `
  RETAINED_FILES.txt
git commit -m "feat(vision2d): add v1-07 code recognition kernel"
```

## Task 4: V1-08 RED tests and original glyph fixtures

**Files:**

- Modify `tests/test_vision2d/synthetic_v107_v109.py` with an original 5×7
  bitmap glyph table and text renderer.
- Create `tests/test_vision2d/test_ocr.py`.

RED steps:

1. Add direct imports of `vision_platform.vision2d.ocr`.
2. Add deterministic KNN training tests for held-out train/test counts,
   fixed seed repeatability, label isolation, and no test sample passed to
   OpenCV training. Add an SVM smoke test using the same in-memory glyphs.
3. Add recognition tests for clean text, scale/rotation/noise/brightness,
   sorted bboxes and finite confidence; expected text mismatch must be visible.
4. Add empty, pure-noise, broken-stroke, touching/stuck glyph, too-small sample,
   invalid image and invalid model tests. No test may use a system font, OCR
   executable, downloaded model or network.
5. Run the focused RED command and record the missing-module/API failure before
   adding production OCR code:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_ocr.py
```

## Task 5: V1-08 implementation and isolated commit

**Files:**

- Create `vision_platform/vision2d/ocr.py`.
- Modify `RETAINED_FILES.txt` by appending the OCR module and test.

Implementation steps:

1. Add immutable `GlyphModel`, `TrainingReport`, `OCRCharacter` and `OCRResult`
   contracts plus `TrainingError`/input validation.
2. Normalize glyph arrays to 20×20 float32. Split per-label samples with a
   local seeded generator. Train OpenCV KNN (`k=3`) or linear SVM only on the
   train matrix; compute held-out accuracy on a separate matrix and preserve
   counts/seed/method in the model.
3. Implement grayscale/adaptive threshold, morphology, PCA deskew, connected
   components, projection-based sorting/splitting, and explicit empty/stuck/
   broken failure codes.
4. Infer with the stored ML model, convert KNN distances or SVM margins into
   clipped confidence, map deskew bboxes back to original image coordinates,
   and mark `expected_text` mismatch/low-confidence as non-success.
5. Keep all randomness local and all runtime limits bounded; never use external
   fonts or data.
6. Run focused GREEN, then all vision2d tests:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_ocr.py
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d
```

7. Review for train/test leakage and forbidden integration changes; run
   `git diff --check`; commit:

```powershell
git add -- vision_platform/vision2d/ocr.py `
  tests/test_vision2d/synthetic_v107_v109.py tests/test_vision2d/test_ocr.py `
  RETAINED_FILES.txt
git commit -m "feat(vision2d): add v1-08 ocr kernel"
```

## Task 6: V1-09 RED tests and original defect fixtures

**Files:**

- Extend `tests/test_vision2d/synthetic_v107_v109.py` with reference/candidate
  masks and deterministic brightness/noise variants.
- Create `tests/test_vision2d/test_defect_detection.py`.

RED steps:

1. Add direct imports of `vision_platform.vision2d.defect_detection`.
2. Add pass case and stable-light/noise cases.
3. Add independent missing, hole, foreign, broken and dimension cases; assert
   defect type, finite region/metric, explicit threshold and status.
4. Add blank reference, size mismatch, invalid dtype/channel and unusable
   polarity tests. Confirm no defect is silently dropped.
5. Run focused RED before creating the production module:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_defect_detection.py
```

## Task 7: V1-09 implementation and isolated commit

**Files:**

- Create `vision_platform/vision2d/defect_detection.py`.
- Modify `RETAINED_FILES.txt` by appending the defect module and test.

Implementation steps:

1. Add frozen `DefectConfig`, `DefectFinding` and `DefectResult` contracts with
   explicit relative thresholds, morphology size, maximum centroid shift and
   stable failure codes.
2. Validate same-size uint8 gray/BGR inputs. Segment polarity from border
   statistics, Otsu/adaptive threshold, normalize brightness and clean small
   noise by the configured odd morphology kernel.
3. Compute bounded centroid alignment, reference/candidate metrics, symmetric
   difference masks, connected components, contour hierarchy and bbox/area/
   perimeter measurements.
4. Classify missing, hole, foreign, broken and dimension using configured
   relative thresholds derived from the reference area/image area; include the
   actual pixel threshold and confidence in every finding.
5. Return `PASS` only with no findings, `PARTIAL` with findings, and `REJECTED`
   for invalid/unusable inputs. Preserve failure code and metrics in all
   non-success results.
6. Run focused GREEN and the full vision2d suite:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d/test_defect_detection.py
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d
```

7. Run `git diff --check`, audit no formal integration paths changed, and commit:

```powershell
git add -- vision_platform/vision2d/defect_detection.py `
  tests/test_vision2d/synthetic_v107_v109.py tests/test_vision2d/test_defect_detection.py `
  RETAINED_FILES.txt
git commit -m "feat(vision2d): add v1-09 defect detection kernel"
```

## Task 8: Cross-module contract, determinism and retained-file audit

**Files:**

- Modify only the three new test files if a cross-module assertion is missing.
- Modify `RETAINED_FILES.txt` only by appending missing new paths.

Run:

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q tests/test_vision2d
.\.venv-vision\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
```

Audit that the final diff contains only the three runtime modules, their
synthetic/test sources, this design/plan and retained entries. Confirm no
`.png`, `.jpg`, `.onnx`, `.pkl`, model cache, downloaded data, `.ttt`, student,
CLI, PyQt, SDK, gateway or public export changes are present. Confirm each
algorithm has its own commit and the worktree-local environment remains ignored.

## Task 9: Independent review, push and handoff

Perform a fresh P0/P1/P2 review of the final diff and tests. Re-run the complete
verification after any review fix, then:

```powershell
git fetch origin
git push -u origin codex/v2-2-v1-07-v1-09-algorithm-kernels
git rev-parse HEAD
git rev-parse origin/codex/v2-2-v1-07-v1-09-algorithm-kernels
git status --short --branch
```

The handoff must report the exact branch/remote commit, per-algorithm commit
hashes, input/output/error contracts, original synthetic generation methods,
focused and full test counts, `git diff --check`, retained paths, independent
review findings, and explicit boundaries: no formal course integration,
CoppeliaSim, PyQt, SDK/gateway, hardware, commercial barcode compatibility,
real OCR/defect accuracy, `PENDING_HARDWARE`, and
`PENDING_HUMAN_ACCEPTANCE`. Recommend a future adapter/experiment integration
only; do not implement it in this branch.
