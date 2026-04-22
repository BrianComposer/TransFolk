import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from apps.db_admin.main_window import MainWindow


def resolve_db_path() -> Path:
    """
    Resuelve la ruta de la base de datos con esta prioridad:
    1) argumento por línea de comandos
    2) raíz global obtenida desde Settings -> <root>/apps/db/transfolk_config.db
    3) fallback relativo al propio archivo main.py
    """
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()

    # Opción preferente: usar Settings del proyecto
    try:
        from transfolk_config.settings import Settings  # cambia este import si tu clase Settings está en otro módulo

        settings = Settings()
        candidate = Path(settings.root) / "apps" / "db" / "transfolk_config.db"
        if candidate.exists():
            return candidate.resolve()
    except Exception:
        pass

    # Fallback por si Settings falla o cambia en el futuro
    current_file = Path(__file__).resolve()
    candidate = current_file.parent.parent / "db" / "transfolk_config.db"
    return candidate.resolve()


def main():
    app = QApplication(sys.argv)

    db_path = resolve_db_path()

    if not db_path.exists():
        QMessageBox.critical(
            None,
            "Base de datos no encontrada",
            f"No se ha encontrado la base de datos SQLite en:\n\n{db_path}"
        )
        sys.exit(1)

    window = MainWindow(str(db_path))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()