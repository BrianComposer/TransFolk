from typing import Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import TABLE_CONFIG


class TablePanel(QWidget):
    def __init__(self, table_name: str, db_manager, form_panel):
        super().__init__()
        self.table_name = table_name
        self.db_manager = db_manager
        self.form_panel = form_panel
        self.rows: List[Dict[str, Any]] = []

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel(TABLE_CONFIG[self.table_name]["title"])
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        top = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Buscar...")
        self.search_edit.textChanged.connect(self.refresh)

        self.refresh_btn = QPushButton("Refrescar")
        self.new_btn = QPushButton("Nuevo")
        self.edit_btn = QPushButton("Cargar en formulario")
        self.delete_btn = QPushButton("Eliminar")

        self.refresh_btn.clicked.connect(self.refresh)
        self.new_btn.clicked.connect(self.form_panel.clear_form)
        self.edit_btn.clicked.connect(self.load_selected)
        self.delete_btn.clicked.connect(self.delete_selected)

        top.addWidget(self.search_edit)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.new_btn)
        top.addWidget(self.edit_btn)
        top.addWidget(self.delete_btn)
        layout.addLayout(top)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self.load_selected())
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.table)

    def refresh(self):
        search = self.search_edit.text()
        self.form_panel.refresh_fk_combos()
        self.rows = self.db_manager.fetch_all(self.table_name, search)
        self._populate()

    def _populate(self):
        if not self.rows:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        columns = list(self.rows[0].keys())
        visible_columns = [c for c in columns if not c.endswith("__display")]

        headers = []
        for col in visible_columns:
            if col + "__display" in columns:
                headers.append(col + " (nombre)")
            else:
                headers.append(col)

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.rows))

        for r, row in enumerate(self.rows):
            for c, col in enumerate(visible_columns):
                display_col = col + "__display"
                value = row.get(display_col) if display_col in row else row.get(col)
                item = QTableWidgetItem("" if value is None else str(value))
                item.setData(Qt.ItemDataRole.UserRole, row.get("id"))
                self.table.setItem(r, c, item)

        self.table.resizeRowsToContents()

    def _selected_row_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def load_selected(self):
        row_id = self._selected_row_id()
        if row_id is None:
            QMessageBox.information(self, "Selección", "Selecciona una fila primero.")
            return
        row_data = self.db_manager.fetch_by_id(self.table_name, row_id)
        if row_data:
            self.form_panel.load_data(row_data)

    def delete_selected(self):
        row_id = self._selected_row_id()
        if row_id is None:
            QMessageBox.information(self, "Selección", "Selecciona una fila primero.")
            return

        result = QMessageBox.question(
            self,
            "Confirmar borrado",
            "¿Seguro que quieres eliminar este registro?",
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        try:
            self.db_manager.delete(self.table_name, row_id)
            self.refresh()
            self.form_panel.clear_form()
        except Exception as exc:
            QMessageBox.critical(self, "Error al eliminar", str(exc))
