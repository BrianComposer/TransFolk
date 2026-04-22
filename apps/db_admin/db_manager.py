import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .config import TABLE_CONFIG


class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def fetch_fk_options(self, table: str) -> List[sqlite3.Row]:
        config = TABLE_CONFIG[table]
        display = config["display_column"]
        pk = config["pk"]
        query = f"SELECT {pk}, {display} FROM {table} ORDER BY {display}"
        with self.get_connection() as conn:
            return conn.execute(query).fetchall()

    def fetch_all(self, table: str, search_text: str = "") -> List[Dict[str, Any]]:
        config = TABLE_CONFIG[table]
        columns = config["columns"]
        pk = config["pk"]
        fk_map = config.get("foreign_keys", {})

        select_parts = [f"main.{col}" for col in columns]
        joins = []
        display_aliases = []

        for fk_col, fk_info in fk_map.items():
            ref_table = fk_info["table"]
            ref_display = fk_info["display"]
            alias = f"{fk_col}_ref"
            joins.append(
                f"LEFT JOIN {ref_table} {alias} ON main.{fk_col} = {alias}.id"
            )
            display_aliases.append(
                f"{alias}.{ref_display} AS {fk_col}__display"
            )

        sql = f"SELECT {', '.join(select_parts + display_aliases)} FROM {table} main "
        if joins:
            sql += " " + " ".join(joins)

        params: List[Any] = []
        searchable = [
            col for col in columns
            if col != pk and not col.startswith("id_")
        ]

        if search_text.strip():
            like_parts = []
            for col in searchable:
                like_parts.append(f"CAST(main.{col} AS TEXT) LIKE ?")
                params.append(f"%{search_text.strip()}%")
            for fk_col in fk_map:
                like_parts.append(f"CAST({fk_col}_ref.{fk_map[fk_col]['display']} AS TEXT) LIKE ?")
                params.append(f"%{search_text.strip()}%")
            if like_parts:
                sql += " WHERE " + " OR ".join(like_parts)

        order_by = config.get("display_column", pk)
        sql += f" ORDER BY main.{order_by}"

        with self.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def insert(self, table: str, data: Dict[str, Any]) -> None:
        filtered = {k: v for k, v in data.items() if k != "id"}
        columns = list(filtered.keys())
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        with self.get_connection() as conn:
            conn.execute(sql, [filtered[c] for c in columns])

    def update(self, table: str, row_id: Any, data: Dict[str, Any]) -> None:
        filtered = {k: v for k, v in data.items() if k != "id"}
        assignments = ", ".join([f"{k} = ?" for k in filtered.keys()])
        sql = f"UPDATE {table} SET {assignments} WHERE id = ?"
        with self.get_connection() as conn:
            conn.execute(sql, [*filtered.values(), row_id])

    def delete(self, table: str, row_id: Any) -> None:
        sql = f"DELETE FROM {table} WHERE id = ?"
        with self.get_connection() as conn:
            conn.execute(sql, [row_id])

    def fetch_by_id(self, table: str, row_id: Any) -> Optional[Dict[str, Any]]:
        config = TABLE_CONFIG[table]
        sql = f"SELECT {', '.join(config['columns'])} FROM {table} WHERE id = ?"
        with self.get_connection() as conn:
            row = conn.execute(sql, [row_id]).fetchone()
        return dict(row) if row else None
