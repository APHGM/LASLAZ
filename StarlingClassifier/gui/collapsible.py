"""
CollapsibleSection — a click-to-expand panel that takes minimal space when closed.
Drop-in replacement for QGroupBox when you want compact UI.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QToolButton, QFrame, QSizePolicy
)


class CollapsibleSection(QWidget):
    """
    Click the header to expand/collapse. Content lives inside .content_layout.
    Usage:
        sec = CollapsibleSection("Bird Detection", start_open=False)
        sec.content_layout.addWidget(my_grid_widget)
    """
    def __init__(self, title: str, start_open: bool = False, parent=None):
        super().__init__(parent)

        self.toggle_btn = QToolButton(text=title, checkable=True)
        self.toggle_btn.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                font-weight: bold;
                padding: 6px 4px;
                text-align: left;
            }
            QToolButton:hover { background: rgba(255,255,255,0.05); }
        """)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if start_open else Qt.ArrowType.RightArrow
        )
        self.toggle_btn.setChecked(start_open)
        self.toggle_btn.toggled.connect(self._on_toggle)

        # Visual divider line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: rgba(255,255,255,0.1);")

        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(20, 4, 8, 8)
        self.content_layout.setSpacing(6)
        self.content.setVisible(start_open)

        # Outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.toggle_btn)
        outer.addWidget(line)
        outer.addWidget(self.content)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def _on_toggle(self, checked: bool):
        self.content.setVisible(checked)
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def set_open(self, open_state: bool):
        self.toggle_btn.setChecked(open_state)
