"""
CollapsibleSection — a click-to-expand panel that takes minimal space when closed.
Drop-in replacement for QGroupBox when you want compact UI.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QFrame, QSizePolicy, QLabel
)


class CollapsibleSection(QWidget):
    """
    Click the header to expand/collapse. Content lives inside .content_layout.
    Set a live status string with set_status(text, color) — shown on the right
    of the header so the user can see current settings without expanding.
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
        # Allow the button to shrink horizontally so long titles don't force
        # the window wider than the available screen width.
        self.toggle_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )

        # Live status badge — shows current config summary
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            "color: #8aa; padding-right: 10px; font-size: 11px;"
        )
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        # Cap the status badge width so it doesn't push the layout wider
        # on small screens — it elides gracefully if too long.
        self.status_label.setMaximumWidth(260)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )

        # Header row: toggle | stretch | status
        header = QWidget()
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(0, 0, 0, 0)
        h_lay.setSpacing(0)
        h_lay.addWidget(self.toggle_btn, 0)
        h_lay.addStretch(1)
        h_lay.addWidget(self.status_label, 0)

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
        outer.addWidget(header)
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

    def set_status(self, text: str, color: str = "#8aa"):
        """Update the small status line on the right side of the header."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {color}; padding-right: 10px; font-size: 11px;"
        )
