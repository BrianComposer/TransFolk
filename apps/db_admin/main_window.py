from PySide6.QtWidgets import QMainWindow, QTabWidget

from .config import TABLE_CONFIG
from .db_manager import DBManager
from .entity_page import EntityPage


class MainWindow(QMainWindow):
    def __init__(self, db_path: str):
        super().__init__()
        self.setWindowTitle("TransFolk SQLite Admin")
        self.resize(1500, 850)

        self.db_manager = DBManager(db_path)

        tabs = QTabWidget()
        for table_name in TABLE_CONFIG.keys():
            page = EntityPage(table_name, self.db_manager)
            tabs.addTab(page, TABLE_CONFIG[table_name]["title"])

        self.setCentralWidget(tabs)
