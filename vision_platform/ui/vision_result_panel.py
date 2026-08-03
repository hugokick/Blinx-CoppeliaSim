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
        self.metrics_label.setWordWrap(True)
        self.profile_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        metadata_layout.addWidget(self.profile_label, 0, 0, 1, 2)
        metadata_layout.addWidget(self.metrics_label, 1, 0, 1, 2)
        root.addWidget(metadata)

        self.evidence_paths_label = QLabel("证据图层：—")
        self.evidence_paths_label.setWordWrap(True)
        self.evidence_paths_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.evidence_paths_label)

        self.route_summary_label = QLabel("代码路由：—")
        self.route_summary_label.setObjectName("visionRouteSummary")
        self.route_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.route_summary_label)
        self.route_text = QPlainTextEdit()
        self.route_text.setObjectName("visionRouteDetails")
        self.route_text.setReadOnly(True)
        self.route_text.setMaximumBlockCount(400)
        self.route_text.setMinimumHeight(150)
        root.addWidget(self.route_text)

        self.ocr_summary_label = QLabel("OCR 分拣：—")
        self.ocr_summary_label.setObjectName("visionOcrSummary")
        self.ocr_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.ocr_summary_label)
        self.ocr_text = QPlainTextEdit()
        self.ocr_text.setObjectName("visionOcrDetails")
        self.ocr_text.setReadOnly(True)
        self.ocr_text.setMaximumBlockCount(240)
        self.ocr_text.setMinimumHeight(150)
        root.addWidget(self.ocr_text)

        self.defect_summary_label = QLabel("表面缺陷分拣：—")
        self.defect_summary_label.setObjectName("visionDefectSummary")
        self.defect_summary_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        root.addWidget(self.defect_summary_label)
        self.defect_text = QPlainTextEdit()
        self.defect_text.setObjectName("visionDefectDetails")
        self.defect_text.setReadOnly(True)
        self.defect_text.setMaximumBlockCount(360)
        self.defect_text.setMinimumHeight(180)
        root.addWidget(self.defect_text)

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
            QLabel#visionOcrSummary {
                color: #315a78;
                font-weight: 600;
            }
            QLabel#visionDefectSummary {
                color: #315a78;
                font-weight: 600;
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
        self.evidence_paths_label.setText("证据图层：—")
        self.route_summary_label.setText("代码路由：—")
        self.route_text.clear()
        self.ocr_summary_label.setText("OCR 分拣：—")
        self.ocr_text.clear()
        self.defect_summary_label.setText("表面缺陷分拣：—")
        self.defect_text.clear()
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
        metrics = f"结果状态：{bundle['status']}　图层：{len(layers)}"
        result = bundle.get("result")
        if isinstance(result, dict) and result.get("template_id"):
            template_id = result["template_id"]
            score = self._format_number(result.get("score"), digits=3, suffix="")
            bbox = result.get("bbox_px")
            center = result.get("center_px")
            bbox_text = self._format_sequence(bbox, digits=0)
            center_text = self._format_sequence(center, digits=1)
            metrics += (
                f"　模板：{template_id}　分数：{score}"
                f"　框：{bbox_text}　中心：{center_text}"
            )
        training = result.get("training") if isinstance(result, dict) else None
        if isinstance(training, dict):
            accuracy = self._format_number(
                training.get("held_out_accuracy"),
                digits=3,
                suffix="",
            )
            if accuracy != "—":
                metrics += f"　OCR训练准确率：{accuracy}"
        self.metrics_label.setText(metrics)
        self.evidence_paths_label.setText(
            "证据图层："
            + "　".join(
                f"{layer.get('layer_id', '—')}: {layer.get('path', '—')}"
                for layer in layers
                if isinstance(layer, dict)
            )
        )
        self._show_code_routes(result)
        self._show_ocr_sorting(result)
        self._show_defect_sorting(result)
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

    def _show_ocr_sorting(self, result: Any) -> None:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("training"), dict)
            or not isinstance(result.get("results"), list)
        ):
            self.ocr_summary_label.setText("OCR 分拣：—")
            self.ocr_text.clear()
            return

        plan_id = result.get("plan_id", "—")
        plan_text = str(plan_id)
        if len(plan_text) > 24:
            plan_text = f"{plan_text[:12]}…{plan_text[-8:]}"
        status = result.get("status", "—")
        self.ocr_summary_label.setText(
            f"OCR 分拣：{status}　计划：{plan_text}"
        )
        training = result["training"]
        accuracy = self._format_number(
            training.get("held_out_accuracy"),
            digits=3,
            suffix="",
        )
        lines = [
            f"training_accuracy={accuracy}",
            f"train_count={training.get('train_count', '—')} "
            f"test_count={training.get('test_count', '—')}",
        ]
        for item in result["results"]:
            if not isinstance(item, dict):
                continue
            confidence = self._format_number(
                item.get("confidence"),
                digits=3,
                suffix="",
            )
            lines.append(
                f"{item.get('identifier', '—')} | confidence={confidence} | "
                f"route={item.get('route_id', '—')} | "
                f"entry={item.get('entry_id', '—')} | "
                f"status={item.get('status', '—')}"
            )
        lines.extend(
            (
                f"motion_status={result.get('motion_status', '—')}",
                f"final_occupancy={result.get('final_occupancy', '—')}",
                f"same_run_evidence={result.get('same_run_evidence', '—')}",
                f"robot_home={result.get('robot_home', '—')} "
                f"tool_off={result.get('tool_off', '—')}",
                f"error={str(result.get('error', '—'))[:1800]}",
                f"human_acceptance={result.get('human_acceptance', '—')}",
                f"hardware_status={result.get('hardware_status', '—')}",
            )
        )
        self.ocr_text.setPlainText("\n".join(lines))

    def _show_code_routes(self, result: Any) -> None:
        required = {
            "plan_id", "status", "entries", "motion_events",
            "completed_entry_ids", "final_occupancy", "error",
            "human_acceptance", "hardware_status",
        }
        if not isinstance(result, dict) or not required <= set(result):
            self.route_summary_label.setText("代码路由：—")
            self.route_text.clear()
            return
        entries = result["entries"]
        if not isinstance(entries, list):
            self.route_summary_label.setText("代码路由：证据格式无效")
            self.route_text.clear()
            return
        lines = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"{entry.get('entry_id', '—')} | {entry.get('code_type', '—')} | "
                f"{entry.get('payload', '—')} | {entry.get('part_id', '—')} -> "
                f"{entry.get('route_id', '—')} | pick={entry.get('pick_xyz_mm', '—')} | "
                f"drop={entry.get('drop_xyz_mm', '—')}"
            )
        lines.append(f"motion_events={result['motion_events']}")
        lines.append(f"completed={result['completed_entry_ids']}")
        lines.append(f"final_occupancy={result['final_occupancy']}")
        lines.append(f"error={result['error']}")
        lines.append(f"human={result['human_acceptance']}")
        lines.append(f"hardware={result['hardware_status']}")
        self.route_summary_label.setText(
            f"代码路由：{result['status']}　计划：{result['plan_id']}"
        )
        self.route_text.setPlainText("\n".join(lines))

    def _show_defect_sorting(self, result: Any) -> None:
        """Render V1-09 as bounded, read-only teaching evidence."""

        if (
            not isinstance(result, dict)
            or not isinstance(result.get("entries"), list)
            or not result.get("plan_id")
        ):
            self.defect_summary_label.setText("表面缺陷分拣：—")
            self.defect_text.clear()
            return

        plan_id = str(result.get("plan_id", "—"))
        display_plan = (
            plan_id
            if len(plan_id) <= 28
            else f"{plan_id[:12]}…{plan_id[-10:]}"
        )
        status = result.get("status", "—")
        entries = result["entries"]
        self.defect_summary_label.setText(
            f"表面缺陷分拣：{status}　计划：{display_plan}　"
            f"条目：{len(entries)}/6"
        )
        lines = [
            f"plan_id={plan_id}",
            f"config_sha256={result.get('config_sha256', '—')}",
            f"asset_manifest_sha256={result.get('asset_manifest_sha256', '—')}",
            f"thresholds={result.get('thresholds', '—')}",
        ]
        for item in entries:
            if not isinstance(item, dict):
                lines.append("entry=<invalid>")
                continue
            lines.extend(
                (
                    f"{item.get('entry_id', '—')} | decision={item.get('decision', '—')} | "
                    f"defect_type={item.get('defect_type', '—')} | "
                    f"slot={item.get('slot_id', '—')} | status={item.get('status', '—')}",
                    f"  reference_crop_sha256={item.get('reference_crop_sha256', '—')}",
                    f"  candidate_crop_sha256={item.get('candidate_crop_sha256', '—')}",
                    f"  findings_sha256={item.get('findings_sha256', '—')}",
                )
            )
            findings = item.get("findings")
            if not isinstance(findings, list):
                lines.append("  findings=<invalid>")
                continue
            if not findings:
                lines.append("  findings=none")
            for finding in findings:
                if not isinstance(finding, dict):
                    lines.append("  finding=<invalid>")
                    continue
                lines.append(
                    "  finding="
                    f"{finding.get('defect_type', '—')} "
                    f"bbox_px={finding.get('bbox_px', '—')} "
                    f"area_px2={self._format_number(finding.get('area_px2'), digits=3, suffix='')} "
                    f"relative_area={self._format_number(finding.get('relative_area'), digits=4, suffix='')} "
                    f"metric={self._format_number(finding.get('metric'), digits=3, suffix='')} "
                    f"threshold={self._format_number(finding.get('threshold'), digits=3, suffix='')} "
                    f"confidence={self._format_number(finding.get('confidence'), digits=3, suffix='')}"
                )
        lines.extend(
            (
                f"motion_status={result.get('motion_status', '—')}",
                f"final_slots={result.get('final_slots', '—')}",
                f"same_run_evidence={result.get('same_run_evidence', '—')}",
                f"robot_home={result.get('robot_home', '—')} tool_off={result.get('tool_off', '—')}",
                f"error={str(result.get('error', '—'))[:1800]}",
                f"human_acceptance={result.get('human_acceptance', '—')}",
                f"hardware_status={result.get('hardware_status', '—')}",
            )
        )
        self.defect_text.setPlainText("\n".join(lines))

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

    @classmethod
    def _format_sequence(cls, value: Any, *, digits: int) -> str:
        if not isinstance(value, list) or not value:
            return "—"
        parts = [cls._format_number(item, digits=digits, suffix="") for item in value]
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
