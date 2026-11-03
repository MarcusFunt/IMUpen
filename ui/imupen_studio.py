"""IMUpen Studio desktop UI concept. Hardware and data actions are intentionally disabled."""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)


INK = "#1c302f"
MUTED = "#71817f"
TEAL = "#167d72"
AMBER = "#9a6818"
LINE = "#e3e9e5"


def label(text: str, style: str = "body", parent: QWidget | None = None) -> QLabel:
    item = QLabel(text, parent)
    item.setObjectName(style)
    item.setWordWrap(True)
    return item


def card(title: str | None = None, subtitle: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)
    if title:
        layout.addWidget(label(title, "cardTitle"))
    if subtitle:
        layout.addWidget(label(subtitle, "muted"))
    return frame, layout


def badge(text: str, flavor: str = "amber") -> QLabel:
    item = label(text, f"badge{flavor.title()}")
    item.setWordWrap(False)
    return item


def button(text: str, primary: bool = False, enabled: bool = True) -> QPushButton:
    item = QPushButton(text)
    item.setObjectName("primaryButton" if primary else "secondaryButton")
    item.setEnabled(enabled)
    item.setMinimumHeight(40)
    return item


def line_edit(placeholder: str) -> QLineEdit:
    item = QLineEdit()
    item.setPlaceholderText(placeholder)
    return item


class Imu3DView(QWidget):
    """Interactive 3D-style board preview with generated motion, never device data."""

    anglesChanged = Signal(float, float, float)
    motionChanged = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Interactive 3D view of the IMUpen motion sensor board")
        self.setToolTip("Drag to orbit the camera. The synthetic demo pose pauses while you inspect it.")
        self.yaw, self.pitch, self.roll = -22.0, 18.0, -12.0
        self.camera_yaw, self.camera_pitch = 0.0, 0.0
        self.phase = 0.0
        self.playing = True
        self._drag_pos: QPointF | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._advance)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        super().showEvent(event)
        if self.playing:
            self.timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        self.timer.stop()
        super().hideEvent(event)

    def _advance(self) -> None:
        self.phase += 0.024
        self.yaw = 32.0 * math.sin(self.phase * 0.54)
        self.pitch = 22.0 * math.sin(self.phase * 0.91 + 0.4)
        self.roll = 27.0 * math.cos(self.phase * 0.73 + 0.6)
        self._publish_angles()
        self.update()

    def _publish_angles(self) -> None:
        self.anglesChanged.emit(self.yaw, self.pitch, self.roll)

    def toggle_motion(self) -> bool:
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.motionChanged.emit(self.playing)
        return self.playing

    def reset_view(self) -> None:
        self.phase = 0.0
        self.yaw, self.pitch, self.roll = -22.0, 18.0, -12.0
        self.camera_yaw, self.camera_pitch = 0.0, 0.0
        self.playing = True
        self.timer.start()
        self.motionChanged.emit(True)
        self._publish_angles()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self.playing:
                self.playing = False
                self.timer.stop()
                self.motionChanged.emit(False)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._drag_pos
            self._drag_pos = event.position()
            # Drag orbits the view camera; it does not change the sensor pose.
            self.camera_yaw = (self.camera_yaw + delta.x() * 0.65 + 180.0) % 360.0 - 180.0
            self.camera_pitch = max(-62.0, min(62.0, self.camera_pitch + delta.y() * 0.5))
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def _project(self, point: tuple[float, float, float], scale: float, center: QPointF) -> QPointF:
        x, y, z = point
        yaw, pitch, roll = map(math.radians, (self.yaw, self.pitch, self.roll))

        # Rotate around local X, Y, then Z. The fixed isometric camera keeps all three
        # board axes visible while the generated pose changes.
        y1 = y * math.cos(roll) - z * math.sin(roll)
        z1 = y * math.sin(roll) + z * math.cos(roll)
        x2 = x * math.cos(pitch) + z1 * math.sin(pitch)
        z2 = -x * math.sin(pitch) + z1 * math.cos(pitch)
        x3 = x2 * math.cos(yaw) - y1 * math.sin(yaw)
        y3 = x2 * math.sin(yaw) + y1 * math.cos(yaw)
        return self._project_world((x3, y3, z2), scale, center)

    def _project_world(self, point: tuple[float, float, float], scale: float,
                       center: QPointF) -> QPointF:
        """Project world-space geometry through the independently controlled camera."""
        x, y, z = point
        camera_yaw, camera_pitch = map(math.radians, (self.camera_yaw, self.camera_pitch))
        x1 = x * math.cos(camera_yaw) - y * math.sin(camera_yaw)
        y1 = x * math.sin(camera_yaw) + y * math.cos(camera_yaw)
        y2 = y1 * math.cos(camera_pitch) - z * math.sin(camera_pitch)
        z2 = y1 * math.sin(camera_pitch) + z * math.cos(camera_pitch)
        return QPointF(center.x() + scale * (x1 - y2) * 0.72,
                       center.y() + scale * ((x1 + y2) * 0.36 - z2 * 0.85))

    def _polygon(self, painter: QPainter, points: list[tuple[float, float, float]],
                 scale: float, center: QPointF, color: str, edge: str | None = None) -> None:
        painter.setBrush(QColor(color))
        painter.setPen(QPen(QColor(edge or color), 1))
        painter.drawPolygon(QPolygonF([self._project(p, scale, center) for p in points]))

    def _box(self, painter: QPainter, x: float, y: float, width: float, depth: float,
             bottom: float, top: float, color: str, side: str, scale: float,
             center: QPointF) -> None:
        a = (x - width / 2, y - depth / 2)
        b = (x + width / 2, y - depth / 2)
        c = (x + width / 2, y + depth / 2)
        d = (x - width / 2, y + depth / 2)
        low = [(a[0], a[1], bottom), (b[0], b[1], bottom),
               (c[0], c[1], bottom), (d[0], d[1], bottom)]
        high = [(a[0], a[1], top), (b[0], b[1], top),
                (c[0], c[1], top), (d[0], d[1], top)]
        self._polygon(painter, [high[0], high[1], low[1], low[0]], scale, center, side)
        self._polygon(painter, [high[1], high[2], low[2], low[1]], scale, center, color)
        self._polygon(painter, [high[2], high[3], low[3], low[2]], scale, center, side)
        self._polygon(painter, [high[3], high[0], low[0], low[3]], scale, center, color)
        self._polygon(painter, high, scale, center, color, "#55776c")

    def _arrow(self, painter: QPainter, origin: tuple[float, float, float],
               end: tuple[float, float, float], name: str, color: str,
               scale: float, center: QPointF) -> None:
        start_pt, end_pt = self._project(origin, scale, center), self._project(end, scale, center)
        painter.setPen(QPen(QColor(color), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start_pt, end_pt)
        angle = math.atan2(end_pt.y() - start_pt.y(), end_pt.x() - start_pt.x())
        size = 5.2
        left = QPointF(end_pt.x() - size * math.cos(angle - 0.48), end_pt.y() - size * math.sin(angle - 0.48))
        right = QPointF(end_pt.x() - size * math.cos(angle + 0.48), end_pt.y() - size * math.sin(angle + 0.48))
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygonF([end_pt, left, right]))
        painter.setPen(QColor(color))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(QPointF(end_pt.x() + 4, end_pt.y() - 3), name)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(LINE), 1))
        painter.setBrush(QColor("#fbfcfa"))
        painter.drawRoundedRect(area, 9, 9)

        scale = min((area.width() - 72) / 43.5, (area.height() - 50) / 30.0)
        center = QPointF(area.center().x(), area.center().y() + 9)

        # A ground grid and compass are fixed in world coordinates. The board pose
        # changes against them; camera drag changes the view of both together.
        painter.setPen(QPen(QColor("#e6ece8"), 1, Qt.PenStyle.DotLine))
        grid_z = -3.0
        for offset in range(-30, 31, 10):
            painter.drawLine(self._project_world((-30, offset, grid_z), scale, center),
                             self._project_world((30, offset, grid_z), scale, center))
            painter.drawLine(self._project_world((offset, -30, grid_z), scale, center),
                             self._project_world((offset, 30, grid_z), scale, center))

        x0, x1, y0, y1 = -14.0, 14.0, -9.0, 9.0
        bottom, top = -0.9, 0.9
        top_corners = [(x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)]
        low_corners = [(x, y, bottom) for x, y, _ in top_corners]
        self._polygon(painter, [top_corners[0], top_corners[1], low_corners[1], low_corners[0]], scale, center, "#315a50")
        self._polygon(painter, [top_corners[1], top_corners[2], low_corners[2], low_corners[1]], scale, center, "#284b43")
        self._polygon(painter, [top_corners[2], top_corners[3], low_corners[3], low_corners[2]], scale, center, "#315a50")
        self._polygon(painter, [top_corners[3], top_corners[0], low_corners[0], low_corners[3]], scale, center, "#284b43")
        self._polygon(painter, top_corners, scale, center, "#397262", "#2c594e")

        # Eight gold castellated pads on each long edge.
        for i in range(8):
            pad_y = -6.5 + i * (13 / 7)
            for pad_x in (-13.15, 13.15):
                self._box(painter, pad_x, pad_y, 1.25, 1.0, 0.91, 1.06,
                          "#d6b76a", "#a88743", scale, center)

        # USB-C connector and representative components on the board.
        self._box(painter, 0, -9.8, 7.2, 3.4, 0.95, 2.2, "#aeb9b5", "#778580", scale, center)
        self._box(painter, 0, -9.45, 4.5, 0.85, 2.2, 2.24, "#66736f", "#66736f", scale, center)
        self._box(painter, -8.2, -4.5, 2.3, 2.0, 0.95, 1.7, "#202d2e", "#162123", scale, center)
        self._box(painter, -7.0, 4.0, 3.2, 2.6, 0.95, 1.5, "#d0d5d1", "#aab3ae", scale, center)
        self._box(painter, 8.2, 4.0, 2.5, 2.4, 0.95, 1.55, "#202d2e", "#162123", scale, center)
        chip_x, chip_y = 2.0, 1.2
        self._box(painter, chip_x, chip_y, 5.5, 5.5, 0.95, 2.45, "#222b2d", "#141d20", scale, center)

        # Inertial axes originate at the LSM6DS3 package.
        origin = (chip_x, chip_y, 3.0)
        self._arrow(painter, origin, (chip_x + 8.0, chip_y, 3.0), "X", "#d75c50", scale, center)
        self._arrow(painter, origin, (chip_x, chip_y + 8.0, 3.0), "Y", "#268d70", scale, center)
        self._arrow(painter, origin, (chip_x, chip_y, 10.0), "Z", "#4778c8", scale, center)

        # Small world-frame compass remains distinct from the colored device axes.
        compass = QPointF(area.right() - 44, area.top() + 53)
        painter.setPen(QPen(QColor("#dfe7e2"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(compass, 23, 23)
        compass_origin = self._project_world((0, 0, 0), 6.2, compass)
        for end, name, color in [((3, 0, 0), "X", "#d75c50"),
                                 ((0, 3, 0), "Y", "#268d70"),
                                 ((0, 0, 3), "Z", "#4778c8")]:
            compass_end = self._project_world(end, 6.2, compass)
            painter.setPen(QPen(QColor(color), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(compass_origin, compass_end)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(QPointF(compass_end.x() + 2, compass_end.y() - 1), name)
        painter.setPen(QColor("#71817a"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        painter.drawText(QPointF(compass.x() - 19, compass.y() + 35), "WORLD FRAME")

        painter.setPen(QColor("#63726e"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(QPointF(area.left() + 14, area.top() + 21), "XIAO ESP32-S3  ·  LSM6DS3 orientation")
        painter.setPen(QColor("#87938f"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QPointF(area.left() + 14, area.bottom() - 11), "Device axes move with the board · world frame stays separate")


class StudioWindow(QMainWindow):
    PAGES = ["Overview", "Record", "Sessions", "Dataset", "Train & evaluate"]
    ICONS = ["◫", "●", "≡", "▦", "⌁"]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IMUpen Studio")
        self.resize(1390, 880)
        self.setMinimumSize(1020, 700)
        self.setStyleSheet(STYLESHEET)
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._sidebar())

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._topbar())
        self.stack = QStackedWidget()
        self.stack.setObjectName("pageStack")
        for page in (self._overview(), self._record(), self._sessions(), self._dataset(), self._training()):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        main_layout.addWidget(self.stack, 1)
        root_layout.addWidget(main, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Prototype view · device, capture, export, and training services are not connected")

    def _sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(238)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(17, 22, 17, 18)
        layout.setSpacing(8)
        brand = QHBoxLayout()
        mark = label("IM", "brandMark")
        brand.addWidget(mark)
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(label("IMUpen Studio", "brandName"))
        names.addWidget(label("IMU research workspace", "brandSub"))
        brand.addLayout(names, 1)
        layout.addLayout(brand)
        layout.addSpacing(28)
        layout.addWidget(label("WORKSPACE", "railSection"))
        self.nav_buttons: list[QPushButton] = []
        for i, (name, icon) in enumerate(zip(self.PAGES, self.ICONS)):
            nav_name = name.replace("&", "&&")
            nav = QPushButton(f"{icon}    {nav_name}")
            nav.setAccessibleName(name)
            nav.setObjectName("navActive" if i == 0 else "navButton")
            nav.setCursor(Qt.CursorShape.PointingHandCursor)
            nav.clicked.connect(lambda checked=False, index=i: self._navigate(index))
            layout.addWidget(nav)
            self.nav_buttons.append(nav)
        layout.addStretch(1)
        local = QFrame()
        local.setObjectName("railInfo")
        local_layout = QVBoxLayout(local)
        local_layout.setContentsMargins(12, 12, 12, 12)
        local_layout.setSpacing(5)
        local_layout.addWidget(label("LOCAL WORKSPACE", "railSection"))
        local_layout.addWidget(label("Files remain on this computer. Raw captures are treated as immutable.", "railCopy"))
        layout.addWidget(local)
        layout.addWidget(label("IMUpen · software prototype", "railFoot"))
        return side

    def _topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(27, 0, 30, 0)
        crumb = label("Workspace   /   Overview", "muted")
        crumb.setObjectName("breadcrumb")
        layout.addWidget(crumb)
        layout.addStretch(1)
        layout.addWidget(badge("●  DEVICE NOT CONNECTED", "amber"))
        return bar

    def _navigate(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, item in enumerate(self.nav_buttons):
            item.setObjectName("navActive" if i == index else "navButton")
            item.style().unpolish(item)
            item.style().polish(item)
        crumb = self.findChild(QLabel, "breadcrumb")
        if crumb:
            crumb.setText(f"Workspace   /   {self.PAGES[index]}")

    def _page(self, eyebrow: str, title: str, description: str, status: str | None = None) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 26, 30, 35)
        layout.setSpacing(18)
        head = QHBoxLayout()
        copy = QVBoxLayout()
        copy.setSpacing(5)
        copy.addWidget(label(eyebrow.upper(), "eyebrow"))
        copy.addWidget(label(title, "pageTitle"))
        copy.addWidget(label(description, "pageDescription"))
        head.addLayout(copy, 1)
        if status:
            head.addWidget(badge(status, "amber"), 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(head)
        return page, layout

    def _notice(self, title: str, body: str) -> QFrame:
        frame, box = card()
        frame.setObjectName("notice")
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(label("!", "noticeIcon"), 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(3)
        text.addWidget(label(title, "noticeTitle"))
        text.addWidget(label(body, "noticeCopy"))
        row.addLayout(text, 1)
        box.addLayout(row)
        return frame

    def _overview(self) -> QWidget:
        page, layout = self._page("Workspace", "Project overview", "Review your setup and continue with a local capture workflow.", "SETUP INCOMPLETE")
        layout.addWidget(self._notice("Hardware verification is pending", "The interface cannot confirm electrical or sensor performance. Complete plan milestone M6 on the actual XIAO and LSM6DS3 before making hardware claims."))
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        ready, ready_box = card("Recording readiness")
        ready_box.addWidget(label("Connect the board, verify the firmware identity, and prepare a labeled session before recording.", "body"))
        ready_box.addSpacing(4)
        for number, text, state in [("01", "Board connection", "Not connected"), ("02", "Sensor identity", "Not verified"), ("03", "Session metadata", "Ready to enter")]:
            row = QHBoxLayout()
            row.addWidget(label(number, "stepNumber"))
            row.addWidget(label(text, "body"), 1)
            row.addWidget(badge(state.upper(), "slate" if number == "03" else "amber"))
            ready_box.addLayout(row)
        ready_box.addSpacing(6)
        action_row = QHBoxLayout()
        prepare = button("Prepare a recording", True)
        prepare.clicked.connect(lambda: self._navigate(1))
        action_row.addWidget(prepare)
        review = button("Review hardware plan")
        plan_path = Path(__file__).resolve().parents[1] / "COMPLETE_IMPLEMENTATION_PLAN.md"
        review.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(plan_path))))
        action_row.addWidget(review)
        action_row.addStretch(1)
        ready_box.addLayout(action_row)
        grid.addWidget(ready, 0, 0, 2, 1)

        status, status_box = card("Project status", "Implementation milestones")
        for name, state, flavor in [("Capture & reports", "AVAILABLE", "green"), ("Dataset export", "AVAILABLE", "green"), ("Hardware validation · M6", "PENDING", "amber"), ("Recognition training · M7", "PLANNED", "slate")]:
            row = QHBoxLayout()
            row.addWidget(label(name, "body"), 1)
            row.addWidget(badge(state, flavor))
            status_box.addLayout(row)
        grid.addWidget(status, 0, 1)

        recent, recent_box = card("Recent sessions", "Capture folders detected in this workspace")
        recent_box.addStretch(1)
        recent_box.addWidget(label("No captures found", "emptyTitle"))
        recent_box.addWidget(label("When you record data, session reports and quality verdicts will appear here.", "muted"))
        recent_box.addStretch(1)
        browse = button("Browse sessions")
        browse.clicked.connect(lambda: self._navigate(2))
        recent_box.addWidget(browse)
        grid.addWidget(recent, 1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)
        return page

    def _record(self) -> QWidget:
        page, layout = self._page("Capture workflow", "Prepare a recording", "Add session labels, then confirm the device and capture settings.", "DEVICE OFFLINE")
        layout.addWidget(self._notice("No capture service is connected", "This design shows the intended session flow. It will not open a serial port or write capture files."))
        columns = QHBoxLayout()
        columns.setSpacing(16)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)
        form, form_box = card("Session details", "Use a pseudonymous writer code. These fields belong in metadata, not raw telemetry.")
        form_box.addLayout(self._field("Writer code", line_edit("e.g. writer-014")))
        form_box.addLayout(self._field("Prompt / character", line_edit("e.g. loop, spiral, or letter")))
        fields = QHBoxLayout()
        fields.addLayout(self._field("Repetition", line_edit("01")))
        duration = QComboBox()
        duration.addItems(["10 seconds", "30 seconds", "60 seconds"])
        fields.addLayout(self._field("Duration", duration))
        form_box.addLayout(fields)
        form_box.addLayout(self._field("Notes (optional)", line_edit("Session context")))
        left_layout.addWidget(form)
        device, device_box = card("Device connection", "The board should identify as XIAO ESP32-S3 with LSM6DS3 over SPI.")
        row = QHBoxLayout()
        port = QComboBox()
        port.addItem("No serial devices detected")
        port.setEnabled(False)
        row.addWidget(port, 1)
        row.addWidget(button("Refresh", enabled=False))
        device_box.addLayout(row)
        device_box.addWidget(label("Sensor ID and firmware header: unavailable", "muted"))
        left_layout.addWidget(device)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)
        motion, motion_box = card("3D motion view", "Animated demo pose · synthetic · no sensor data")
        preview = Imu3DView()
        motion_box.addWidget(preview, 1)
        angles_row = QHBoxLayout()
        angle_labels = [label("YAW  −22°", "angleReadout"),
                        label("PITCH  +18°", "angleReadout"),
                        label("ROLL  −12°", "angleReadout")]
        for item in angle_labels:
            angles_row.addWidget(item, 1)
        motion_box.addLayout(angles_row)

        def update_angles(yaw: float, pitch: float, roll: float) -> None:
            angle_labels[0].setText(f"YAW  {yaw:+.0f}°")
            angle_labels[1].setText(f"PITCH  {pitch:+.0f}°")
            angle_labels[2].setText(f"ROLL  {roll:+.0f}°")

        preview.anglesChanged.connect(update_angles)
        motion_actions = QHBoxLayout()
        motion_actions.addWidget(label("Drag to orbit the camera; demo pauses while you inspect.", "muted"), 1)
        play = button("Pause demo")
        play.clicked.connect(lambda: preview.toggle_motion())
        preview.motionChanged.connect(lambda active: play.setText("Pause demo" if active else "Play demo"))
        motion_actions.addWidget(play)
        reset = button("Reset view")
        reset.clicked.connect(preview.reset_view)
        motion_actions.addWidget(reset)
        motion_box.addLayout(motion_actions)
        right_layout.addWidget(motion, 1)
        policy, policy_box = card("Capture policy")
        for value in ["Nominal output rate", "208 Hz", "Raw source", "CSV v1 · field order preserved", "Artifacts", "capture.csv · events.jsonl · report.json"]:
            policy_box.addWidget(label(value, "body" if value in ("208 Hz", "CSV v1 · field order preserved", "capture.csv · events.jsonl · report.json") else "muted"))
        right_layout.addWidget(policy)
        right_layout.addStretch(1)
        columns.addWidget(left, 1)
        columns.addWidget(right, 1)
        layout.addLayout(columns, 1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(button("Start capture", True, enabled=False))
        layout.addLayout(actions)
        return page

    def _field(self, name: str, widget: QWidget) -> QVBoxLayout:
        field = QVBoxLayout()
        field.setSpacing(6)
        field_label = label(name, "fieldLabel")
        field_label.setBuddy(widget)
        field.addWidget(field_label)
        widget.setMinimumHeight(40)
        widget.setObjectName("input")
        field.addWidget(widget)
        return field

    def _sessions(self) -> QWidget:
        page, layout = self._page("Local data", "Sessions", "Inspect raw captures and quality reports without changing source files.", "0 SESSIONS")
        layout.addWidget(self._notice("No capture folders found", "Choose a workspace folder containing IMUpen capture directories to review sessions. Raw capture files remain unchanged."))
        listing, box = card("Capture library", "Expected artifacts: capture.csv, events.jsonl, report.json")
        box.addStretch(1)
        box.addWidget(label("Nothing to show yet", "emptyTitle"))
        box.addWidget(label("Recorded sessions will appear here with status, duration, sample count, and source type.", "muted"))
        box.addStretch(1)
        layout.addWidget(listing, 1)
        return page

    def _dataset(self) -> QWidget:
        page, layout = self._page("Derived data", "Dataset export", "Create traceable canonical datasets and optional 100 Hz products.", "NO SOURCE SELECTED")
        layout.addWidget(self._notice("Raw captures stay immutable", "Exports are derived files with their own manifest. Resampling must respect gaps, epochs, and segment boundaries."))
        grid = QGridLayout()
        grid.setSpacing(16)
        settings, settings_box = card("Export settings")
        for name, control in [("Source capture", QComboBox()), ("Source label", QComboBox()), ("Nominal rate", QComboBox()), ("Optional derivative", QComboBox())]:
            if name == "Source capture": control.addItem("Choose a session…")
            elif name == "Source label": control.addItems(["Select real or synthetic"])
            elif name == "Nominal rate": control.addItems(["208 Hz"])
            else: control.addItems(["No resampled output", "100 Hz"])
            control.setEnabled(False)
            settings_box.addLayout(self._field(name, control))
        settings_box.addWidget(button("Export canonical dataset", True, enabled=False))
        manifest, manifest_box = card("Provenance manifest", "Recorded with every dataset product")
        for text in ["Source capture hash", "Source type: real / synthetic", "Configuration and calibration versions", "Dropped or invalid samples", "Processing parameters"]:
            manifest_box.addWidget(label("✓   " + text, "body"))
        grid.addWidget(settings, 0, 0)
        grid.addWidget(manifest, 0, 1)
        layout.addLayout(grid)
        layout.addStretch(1)
        return page

    def _training(self) -> QWidget:
        page, layout = self._page("Recognition experiments · M7", "Train & evaluate", "Plan handwriting experiments with reproducible session and writer splits.", "NOT IMPLEMENTED")
        layout.addWidget(self._notice("There is no recognition model or trainer in this repository", "This section is a future workflow proposal. It cannot train a model, score handwriting, or report accuracy."))
        proposal, box = card("Proposed experiment setup", "Available after real labeled sessions and a recognition backend exist")
        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(16)
        for i, (head, body) in enumerate([
            ("Dataset", "Select a versioned dataset; show its source and manifest."),
            ("Split strategy", "Group by writer or session to reduce leakage."),
            ("Model", "Choose a documented architecture and fixed parameters."),
            ("Evaluation", "Report held-out metrics and per-class errors."),
        ]):
            item = QVBoxLayout()
            item.setSpacing(5)
            item.addWidget(label(head, "cardTitle"))
            item.addWidget(label(body, "muted"))
            grid.addLayout(item, i // 2, i % 2)
        box.addLayout(grid)
        box.addSpacing(8)
        box.addWidget(badge("SYNTHETIC DATA CANNOT VERIFY HANDWRITING ACCURACY", "amber"), 0, Qt.AlignmentFlag.AlignLeft)
        box.addWidget(button("Configure experiment", True, enabled=False))
        layout.addWidget(proposal)
        layout.addStretch(1)
        return page


STYLESHEET = f"""
QMainWindow {{ background: #f4f6f3; color: {INK}; }}
QWidget {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: {INK}; }}
QWidget#sidebar {{ background: #17383a; }}
QLabel#brandMark {{ background: #285958; border: 1px solid #55827a; border-radius: 9px; color: #dff4eb; font-size: 13px; font-weight: 700; padding: 9px; }}
QLabel#brandName {{ color: #f5faf8; font-size: 14px; font-weight: 700; }}
QLabel#brandSub {{ color: #a9c3bd; font-size: 11px; }}
QLabel#railSection {{ color: #87aaa4; font-size: 10px; font-weight: 700; letter-spacing: 1px; }}
QLabel#railCopy, QLabel#railFoot {{ color: #b4c9c3; font-size: 11px; }}
QLabel#railFoot {{ padding: 11px 2px 0; }}
QFrame#railInfo {{ background: #204446; border: 1px solid #315858; border-radius: 9px; }}
QPushButton#navButton, QPushButton#navActive {{ color: #c2d4cf; text-align: left; background: transparent; border: 0; border-radius: 7px; padding: 11px 12px; font-weight: 550; }}
QPushButton#navButton:hover {{ background: #244749; color: white; }}
QPushButton#navActive {{ background: #2b5552; color: white; border-left: 3px solid #67c9ad; padding-left: 9px; }}
QWidget#topbar {{ background: #ffffff; border-bottom: 1px solid {LINE}; min-height: 57px; max-height: 57px; }}
QWidget#pageStack QWidget {{ background: transparent; }}
QLabel#breadcrumb {{ font-size: 12px; }}
QLabel#eyebrow {{ color: #7b8a85; font-size: 10px; font-weight: 700; letter-spacing: 1.4px; }}
QLabel#pageTitle {{ color: {INK}; font-size: 27px; font-weight: 700; }}
QLabel#pageDescription {{ color: {MUTED}; font-size: 13px; }}
QFrame#card {{ background: #ffffff; border: 1px solid {LINE}; border-radius: 10px; }}
QFrame#notice {{ background: #fff9ed; border: 1px solid #eddfc3; border-left: 3px solid #c28a31; border-radius: 8px; }}
QLabel#noticeIcon {{ color: #9a6818; font-weight: 800; font-size: 15px; }}
QLabel#noticeTitle {{ color: #59431e; font-weight: 700; }}
QLabel#noticeCopy {{ color: #786747; }}
QLabel#cardTitle {{ color: {INK}; font-size: 14px; font-weight: 700; }}
QLabel#body {{ color: #425552; }}
QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#emptyTitle {{ color: #405450; font-size: 16px; font-weight: 650; }}
QLabel#fieldLabel {{ color: #485b57; font-size: 12px; font-weight: 650; }}
QLineEdit#input, QComboBox#input {{ background: #fff; border: 1px solid #d7e0db; border-radius: 6px; padding: 8px 10px; color: {INK}; }}
QLineEdit#input:focus, QComboBox#input:focus {{ border: 1px solid #43978a; }}
QLineEdit#input:disabled, QComboBox#input:disabled {{ background: #f3f5f3; color: #84918d; }}
QLabel#stepNumber {{ color: {TEAL}; font-weight: 700; min-width: 25px; }}
QLabel#link {{ color: {TEAL}; font-weight: 650; }}
QLabel#angleReadout {{ background: #f0f5f1; border: 1px solid #e0e8e2; border-radius: 6px; color: #405650; font-size: 11px; font-weight: 700; padding: 8px 10px; }}
QLabel#badgeAmber, QLabel#badgeSlate, QLabel#badgeGreen {{ border-radius: 10px; padding: 5px 9px; font-size: 10px; font-weight: 750; letter-spacing: .4px; }}
QLabel#badgeAmber {{ background: #fff1d5; color: {AMBER}; }}
QLabel#badgeSlate {{ background: #edf1ee; color: #65736e; }}
QLabel#badgeGreen {{ background: #e0f2e9; color: #286847; }}
QPushButton#primaryButton {{ color: white; background: {TEAL}; border: 1px solid {TEAL}; border-radius: 6px; padding: 0 15px; font-weight: 650; }}
QPushButton#primaryButton:hover {{ background: #106b62; }}
QPushButton#primaryButton:enabled {{ color: white; background-color: {TEAL}; border: 1px solid {TEAL}; }}
QPushButton#primaryButton:disabled {{ color: #778681; background-color: #e5ebe7; border: 1px solid #d6dfda; }}
QPushButton#secondaryButton {{ color: #354b47; background: #fff; border: 1px solid #d7e0db; border-radius: 6px; padding: 0 13px; font-weight: 600; }}
QPushButton#secondaryButton:hover {{ background: #f7f9f7; }}
QPushButton:disabled {{ color: #929d99; background: #ecefec; border-color: #e4e9e5; }}
QStatusBar {{ background: #fff; color: #72817c; border-top: 1px solid {LINE}; font-size: 11px; }}
QScrollArea {{ background: #f4f6f3; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #cbd5d0; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("IMUpen Studio")
    app.setFont(QFont("Segoe UI", 10))
    window = StudioWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
