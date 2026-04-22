from PySide6.QtWidgets import QHBoxLayout, QWidget

from .form_panel import FormPanel
from .table_panel import TablePanel


class EntityPage(QWidget):
    def __init__(self, table_name: str, db_manager):
        super().__init__()
        self.table_name = table_name
        self.db_manager = db_manager

        layout = QHBoxLayout(self)
        self.form_panel = FormPanel(table_name, db_manager, self._refresh_both)
        self.table_panel = TablePanel(table_name, db_manager, self.form_panel)

        layout.addWidget(self.table_panel, 3)
        layout.addWidget(self.form_panel, 2)

    def _refresh_both(self):
        self.table_panel.refresh()
        self.form_panel.refresh_fk_combos()
