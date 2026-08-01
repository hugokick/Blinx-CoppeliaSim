from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vision_platform.student.protocol import RunState
from vision_platform.vision_quality.evidence import load_recorded_bundle
from vision_platform.vision_quality.models import VisionResultEvidenceError


_TERMINAL_STATES = frozenset(
    {RunState.PASSED, RunState.FAILED, RunState.CANCELLED}
)
_MAX_LAYER_BYTES = 64 * 1024 * 1024
_SAFE_ERROR_MESSAGES = {
    "VISION_RESULT_EVIDENCE_INVALID": "无法载入视觉结果证据",
    "VISION_RESULT_PANEL_ERROR": "视觉结果面板不可用",
    "VISION_RESULT_PANEL_CALLBACK_FAILED": "无法更新视觉结果",
}


class _SnapshotBridge(QObject):
    updated = pyqtSignal(object)


class VisionResultPanel(QWidget):
    """Read-only viewer for verified vision-result evidence bundles."""

    def __init__(self, *, controller=None, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._unsubscribe = None
        self._bundle: dict[str, Any] | None = None
        self._layer_images: dict[str, np.ndarray] = {}
        self._build_ui()
        self.clear_result()

        self._bridge = _SnapshotBridge(self)
        self._bridge.updated.connect(self._render_snapshot)
        if controller is not None:
            try:
                self._unsubscribe = controller.subscribe(
                    self._bridge.updated.emit
                )
            except BaseException as error:
                self._render_error(error, fallback_code="VISION_RESULT_PANEL_ERROR")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        heading_row = QHBoxLayout()
        heading = QLabel("视觉结果证据")
        heading.setObjectName("visionResultHeading")
        self.status_label = QLabel("尚无视觉结果")
        self.status_label.setObjectName("visionResultStatus")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        heading_row.addWidget(self.status_label, 2)
        root.addLayout(heading_row)

        metadata = QFrame()
        metadata.setObjectName("visionResultMetadata")
        metadata_layout = QGridLayout(metadata)
        metadata_layout.setContentsMargins(12, 8, 12, 8)
        self.profile_label = QLabel("配置档：—")
        self.metrics_label = QLabel("结果状态：—　图层：0")
        self.profile_label.setWordWrap(True)
        self.profile_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        metadata_layout.addWidget(self.profile_label, 0, 0, 1, 2)
        metadata_layout.addWidget(self.metrics_label, 1, 0, 1, 2)
        root.addWidget(metadata)

        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("显示图层"))
        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(180)
        self.layer_combo.currentIndexChanged.connect(
            self._render_selected_layer
        )
        layer_row.addWidget(self.layer_combo)
        layer_row.addStretch(1)
        root.addLayout(layer_row)

        splitter = QSplitter(Qt.Horizontal)
        preview_frame = QFrame()
        preview_frame.setObjectName("visionPreviewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self.preview_label = QLabel("等待视觉结果")
        self.preview_label.setObjectName("visionPreview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 280)
        preview_layout.addWidget(self.preview_label, 1)

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("结构化视觉结果将在此显示")
        self.result_text.setMinimumSize(300, 280)
        splitter.addWidget(preview_frame)
        splitter.addWidget(self.result_text)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.boundary_label = QLabel(
            "PENDING_HARDWARE｜仿真证据不是课程成绩，也不替代海康相机、"
            "真实机械臂或真实光学精度验收。"
        )
        self.boundary_label.setObjectName("visionResultBoundary")
        self.boundary_label.setWordWrap(True)
        root.addWidget(self.boundary_label)

        self.setStyleSheet(
            """
            QLabel#visionResultHeading {
                font-size: 18px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#visionResultStatus {
                color: #315a78;
                font-weight: 600;
            }
            QFrame#visionResultMetadata {
                background: #edf3f7;
                border-left: 4px solid #2f789c;
                border-radius: 3px;
            }
            QFrame#visionPreviewFrame {
                background: #15222c;
                border: 1px solid #49616f;
                border-radius: 4px;
            }
            QLabel#visionPreview {
                color: #b9c9d4;
            }
            QLabel#visionResultBoundary {
                color: #6f4e18;
                background: #fff6d9;
                border: 1px solid #dfc574;
                border-radius: 3px;
                padding: 7px 9px;
            }
            """
        )

    def load_run(self, run_directory: str | Path) -> None:
        try:
            directory = Path(run_directory).expanduser().resolve(strict=True)
            if not directory.is_dir():
                raise VisionResultEvidenceError(
                    "evidence run must be a directory"
                )
            names = sorted(
                entry.name
                for entry in directory.iterdir()
                if entry.name.startswith("vision-bundle-")
                and entry.name.endswith(".json")
            )
            if not names:
                self.clear_result("本次运行没有视觉结果包")
                return

            artifact_name = names[-1]
            bundle = load_recorded_bundle(directory, artifact_name)
            images = self._load_verified_images(bundle)
            self._show_bundle(bundle, images, artifact_name)
        except BaseException as error:
            self._render_error(
                error,
                fallback_code="VISION_RESULT_EVIDENCE_INVALID",
            )

    def clear_result(self, message: str = "尚无视觉结果") -> None:
        self._bundle = None
        self._layer_images.clear()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.blockSignals(False)
        self.status_label.setText(message)
        self.profile_label.setText("配置档：—")
        self.metrics_label.setText("结果状态：—　图层：0")
        self.preview_label.clear()
        self.preview_label.setText("等待视觉结果")
        self.result_text.clear()

    def release_subscription(self) -> None:
        callback = self._unsubscribe
        if callback is None:
            return
        self._unsubscribe = None
        try:
            callback()
        except Exception:
            self._unsubscribe = callback
            raise

    @pyqtSlot(object)
    def _render_snapshot(self, snapshot: Any) -> None:
        try:
            state = snapshot.state
            if state not in _TERMINAL_STATES:
                return
            evidence_dir = snapshot.evidence_dir
            if evidence_dir is None:
                self.clear_result("本次运行没有视觉结果包")
                return
            self.load_run(evidence_dir)
        except BaseException as error:
            self._render_error(
                error,
                fallback_code="VISION_RESULT_PANEL_CALLBACK_FAILED",
            )

    def _load_verified_images(
        self,
        bundle: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        resolved = bundle.get("_resolved_layer_paths")
        layers = bundle.get("layers")
        if not isinstance(resolved, dict) or not isinstance(layers, list):
            raise VisionResultEvidenceError("bundle layer paths are missing")
        images: dict[str, np.ndarray] = {}
        for layer in layers:
            layer_id = layer["layer_id"]
            path = resolved.get(layer_id)
            if not isinstance(path, Path):
                raise VisionResultEvidenceError(
                    "bundle layer path is invalid"
                )
            images[layer_id] = self._read_verified_bgr(path, layer)
        return images

    @staticmethod
    def _read_verified_bgr(path: Path, layer: dict[str, Any]) -> np.ndarray:
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise VisionResultEvidenceError("bundle layer path changed")
        with path.open("rb") as handle:
            encoded = handle.read(_MAX_LAYER_BYTES + 1)
        if len(encoded) > _MAX_LAYER_BYTES:
            raise VisionResultEvidenceError("bundle layer is too large")
        if hashlib.sha256(encoded).hexdigest() != layer["sha256"]:
            raise VisionResultEvidenceError("bundle layer hash changed")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if (
            image is None
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[1] != layer["width"]
            or image.shape[0] != layer["height"]
        ):
            raise VisionResultEvidenceError("bundle layer pixels are invalid")
        return image

    def _show_bundle(
        self,
        bundle: dict[str, Any],
        images: dict[str, np.ndarray],
        artifact_name: str,
    ) -> None:
        self._bundle = bundle
        self._layer_images = images
        layers = bundle["layers"]
        profile = bundle["profile"]
        profile_id = profile.get("profile_id", "—")
        resolution = profile.get("resolution")
        resolution_text = "—"
        if isinstance(resolution, list) and len(resolution) == 2:
            resolution_text = f"{resolution[0]} × {resolution[1]}"
        angle_text = self._format_number(
            profile.get("perspective_angle_deg"),
            digits=1,
            suffix="°",
        )
        height_text = self._format_number(
            profile.get("camera_rig_z_m"),
            digits=2,
            suffix=" m",
        )
        key_light_text = self._format_rgb(profile.get("key_diffuse_rgb"))
        fill_light_text = self._format_rgb(profile.get("fill_diffuse_rgb"))

        self.status_label.setText(f"已验证：{artifact_name}")
        self.profile_label.setText(
            f"配置档：{profile_id}　分辨率：{resolution_text}\n"
            f"视场角：{angle_text}　相机高度：{height_text}　"
            f"主光 RGB：{key_light_text}　补光 RGB：{fill_light_text}"
        )
        self.metrics_label.setText(
            f"结果状态：{bundle['status']}　图层：{len(layers)}"
        )
        public_bundle = {
            key: value
            for key, value in bundle.items()
            if not key.startswith("_")
        }
        self.result_text.setPlainText(
            json.dumps(
                public_bundle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
        )

        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in layers:
            self.layer_combo.addItem(layer["title"], layer["layer_id"])
        self.layer_combo.setCurrentIndex(0)
        self.layer_combo.blockSignals(False)
        self._render_selected_layer()

    @staticmethod
    def _format_number(value: Any, *, digits: int, suffix: str) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "—"
        numeric = float(value)
        if not math.isfinite(numeric):
            return "—"
        return f"{numeric:.{digits}f}{suffix}"

    @classmethod
    def _format_rgb(cls, value: Any) -> str:
        if not isinstance(value, list) or len(value) != 3:
            return "—"
        parts = [cls._format_number(item, digits=2, suffix="") for item in value]
        if "—" in parts:
            return "—"
        return ", ".join(parts)

    @pyqtSlot()
    def _render_selected_layer(self) -> None:
        layer_id = self.layer_combo.currentData()
        image = self._layer_images.get(layer_id)
        if image is None:
            self.preview_label.clear()
            self.preview_label.setText("等待视觉结果")
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        target = self.preview_label.size()
        self.preview_label.setPixmap(
            pixmap.scaled(
                target,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _render_error(
        self,
        error: BaseException,
        *,
        fallback_code: str,
    ) -> None:
        try:
            candidate = getattr(error, "code", None)
        except BaseException:
            candidate = None
        code = (
            candidate
            if candidate in _SAFE_ERROR_MESSAGES
            else fallback_code
        )
        message = _SAFE_ERROR_MESSAGES.get(
            code,
            "视觉结果暂时不可用",
        )
        self.clear_result(f"{code}：{message}")
