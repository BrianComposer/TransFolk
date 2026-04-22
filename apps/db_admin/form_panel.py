from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import TABLE_CONFIG


class FormPanel(QWidget):
    def __init__(self, table_name: str, db_manager, on_saved_callback):
        super().__init__()
        self.table_name = table_name
        self.db_manager = db_manager
        self.on_saved_callback = on_saved_callback
        self.current_id = None
        self.inputs: Dict[str, Any] = {}

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        title = QLabel(f"Editar / crear: {TABLE_CONFIG[self.table_name]['title']}")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        self.form_layout = QFormLayout(container)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        config = TABLE_CONFIG[self.table_name]
        fk_map = config.get("foreign_keys", {})

        for col in config["columns"]:
            if col == "id":
                continue
            widget = self._make_input(col, fk_map)
            self.inputs[col] = widget
            self.form_layout.addRow(col, widget)

        box = QGroupBox("Formulario")
        box_layout = QVBoxLayout(box)
        box_layout.addWidget(container)
        scroll.setWidget(box)
        outer.addWidget(scroll)

        self.save_btn = QPushButton("Guardar")
        self.new_btn = QPushButton("Nuevo")
        self.save_btn.clicked.connect(self.save)
        self.new_btn.clicked.connect(self.clear_form)

        outer.addWidget(self.save_btn)
        outer.addWidget(self.new_btn)
        outer.addStretch()

    def _make_input(self, col: str, fk_map: Dict[str, Any]):
        if col in fk_map:
            combo = QComboBox()
            combo.setEditable(False)
            self._fill_combo(combo, fk_map[col]["table"])
            return combo

        lower = col.lower()
        if "description" in lower or "descripcion" in lower or col == "durations":
            text = QTextEdit()
            text.setFixedHeight(90)
            return text

        if lower.startswith("train_") or "datetime" in lower:
            return QLineEdit()

        if lower in {"bias", "weight_tying", "greedy", "mixed_precision", "deterministic", "early_stopping"}:
            return QCheckBox()

        return QLineEdit()

    def _fill_combo(self, combo: QComboBox, ref_table: str):
        combo.clear()
        combo.addItem("-- seleccionar --", None)
        for row in self.db_manager.fetch_fk_options(ref_table):
            combo.addItem(str(row[1]), row[0])

    def refresh_fk_combos(self):
        fk_map = TABLE_CONFIG[self.table_name].get("foreign_keys", {})
        for col, info in fk_map.items():
            combo = self.inputs[col]
            current = combo.currentData()
            self._fill_combo(combo, info["table"])
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)

    def clear_form(self):
        self.current_id = None
        for col, widget in self.inputs.items():
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(0)
            elif isinstance(widget, QTextEdit):
                widget.clear()
            elif isinstance(widget, QCheckBox):
                widget.setChecked(False)
            else:
                widget.clear()

    def load_data(self, row_data: Dict[str, Any]):
        self.current_id = row_data.get("id")
        for col, widget in self.inputs.items():
            value = row_data.get(col)
            if isinstance(widget, QComboBox):
                idx = widget.findData(value)
                widget.setCurrentIndex(idx if idx >= 0 else 0)
            elif isinstance(widget, QTextEdit):
                widget.setPlainText("" if value is None else str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setText("" if value is None else str(value))

    def _collect_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for col, widget in self.inputs.items():
            if isinstance(widget, QComboBox):
                data[col] = widget.currentData()
            elif isinstance(widget, QTextEdit):
                text = widget.toPlainText().strip()
                data[col] = text if text != "" else None
            elif isinstance(widget, QCheckBox):
                data[col] = 1 if widget.isChecked() else 0
            else:
                text = widget.text().strip()
                data[col] = text if text != "" else None
        return data

    def save(self):
        data = self._collect_data()
        required_name = "name" in self.inputs
        if required_name and not data.get("name"):
            QMessageBox.warning(self, "Validación", "El campo 'name' es obligatorio.")
            return

        fk_map = TABLE_CONFIG[self.table_name].get("foreign_keys", {})
        for fk_col in fk_map:
            if data.get(fk_col) is None:
                QMessageBox.warning(self, "Validación", f"Debes seleccionar un valor para '{fk_col}'.")
                return

        try:
            if self.current_id is None:
                self.db_manager.insert(self.table_name, data)
            else:
                self.db_manager.update(self.table_name, self.current_id, data)
            self.on_saved_callback()
            self.clear_form()
        except Exception as exc:
            QMessageBox.critical(self, "Error al guardar", str(exc))
