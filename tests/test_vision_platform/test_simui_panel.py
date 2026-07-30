from __future__ import annotations

import numpy as np

from vision_platform.models import Frame
from vision_platform.ui.simui_panel import SimUIPanel, build_simui_xml
from vision_platform.ui.view_model import VisionLabViewModel


class RecordingSimUI:
    def __init__(self):
        self.created = []
        self.destroyed = []
        self.labels = []
        self.images = []
        self.enabled = []

    def create(self, xml):
        self.created.append(xml)
        return f"ui-{len(self.created)}"

    def destroy(self, handle):
        self.destroyed.append(handle)

    def setLabelText(self, handle, widget_id, text, suppress_events=True):
        self.labels.append((handle, widget_id, text, suppress_events))

    def setImageData(self, handle, widget_id, data, width, height):
        self.images.append((handle, widget_id, data, width, height))

    def setEnabled(
        self,
        handle,
        widget_id,
        enabled,
        suppress_events=True,
    ):
        self.enabled.append(
            (handle, widget_id, enabled, suppress_events)
        )


def test_simui_xml_contains_required_teaching_controls():
    xml = build_simui_xml()

    for text in (
        "启动",
        "暂停",
        "继续",
        "复位",
        "标定",
        "识别结果",
        "像素坐标",
        "世界坐标",
        "急停",
    ):
        assert text in xml
    assert '<image id="1"' in xml


def test_panel_creates_updates_and_destroys_one_ui():
    simui = RecordingSimUI()
    panel = SimUIPanel(
        simui=simui,
        view_model=VisionLabViewModel(),
    )

    panel.open()
    panel.refresh()
    panel.close()
    panel.close()

    assert len(simui.created) == 1
    assert simui.destroyed == ["ui-1"]
    assert panel.update_count == 1


def test_panel_projects_status_coordinates_and_thumbnail():
    simui = RecordingSimUI()
    view_model = VisionLabViewModel()
    panel = SimUIPanel(simui=simui, view_model=view_model)
    panel.open()
    view_model.set_connection(
        connected=True,
        camera_backend="sim",
        robot_backend="sim",
    )
    view_model.pixel_coordinate = (320.0, 240.0)
    view_model.world_coordinate_mm = (75.0, -20.0, 20.0)
    view_model.set_frame(
        Frame(
            image_bgr=np.full((480, 640, 3), 50, dtype=np.uint8),
            width=640,
            height=480,
            timestamp_s=1.0,
            source="fake",
            sequence_id=0,
        )
    )

    panel.refresh()

    label_texts = [item[2] for item in simui.labels]
    assert any("实验平台已连接" in text for text in label_texts)
    assert any("320.0" in text for text in label_texts)
    assert any("75.0" in text for text in label_texts)
    assert simui.images[-1][3:] == (320, 240)
    assert len(simui.images[-1][2]) == 320 * 240 * 3


def test_button_callbacks_only_enqueue_commands():
    simui = RecordingSimUI()
    commands = []
    panel = SimUIPanel(
        simui=simui,
        view_model=VisionLabViewModel(),
        command_sink=commands.append,
    )
    panel.open()

    panel.dispatch_command("start")
    panel.dispatch_command("pause")
    panel.dispatch_command("emergency_stop")

    assert commands == ["start", "pause", "emergency_stop"]
