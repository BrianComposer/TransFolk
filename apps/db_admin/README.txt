TransFolk SQLite Admin (versión simple)

Ubicación recomendada dentro del proyecto:
apps/db_admin/

Estructura:
apps/
├── db/
│   └── transfolk_config.db
└── db_admin/
    ├── __init__.py
    ├── config.py
    ├── db_manager.py
    ├── entity_page.py
    ├── form_panel.py
    ├── main.py
    ├── main_window.py
    └── table_panel.py

Instalación:
pip install PySide6

Ejecución desde la raíz del proyecto:
python -m apps.db_admin.main

O indicando ruta explícita:
python -m apps.db_admin.main apps/db/transfolk_config.db

Notas:
- La app usa sqlite3 estándar.
- Muestra nombres relacionales en columnas con sufijo "(nombre)".
- Los combos permiten elegir relaciones por nombre.
- Los booleanos se muestran como checkboxes.
- Las columnas de texto largo usan QTextEdit.
