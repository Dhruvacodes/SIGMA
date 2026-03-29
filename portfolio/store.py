"""
SIGMA Portfolio Store.
In-memory and SQLite-backed portfolio storage.
"""

from datetime import datetime
from typing import Any

from models.portfolio import UserPortfolio

# In-memory store
_portfolio_store: dict[str, UserPortfolio] = {}


class PortfolioStore:
    """
    Portfolio store for SIGMA.
    Uses in-memory storage with optional SQLite persistence.
    """

    def __init__(self, use_sqlite: bool = False, db_path: str = "./portfolios.db"):
        self.use_sqlite = use_sqlite
        self.db_path = db_path

        if use_sqlite:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                user_id TEXT PRIMARY KEY,
                portfolio_json TEXT,
                last_updated TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def save(self, portfolio: UserPortfolio) -> None:
        """
        Save a portfolio.

        Args:
            portfolio: UserPortfolio to save.
        """
        _portfolio_store[portfolio.user_id] = portfolio

        if self.use_sqlite:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO portfolios (user_id, portfolio_json, last_updated)
                VALUES (?, ?, ?)
            """,
                (
                    portfolio.user_id,
                    portfolio.model_dump_json(),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            conn.close()

    def get(self, user_id: str) -> UserPortfolio | None:
        """
        Get a portfolio by user ID.

        Args:
            user_id: User ID to look up.

        Returns:
            UserPortfolio if found, None otherwise.
        """
        # Check in-memory first
        if user_id in _portfolio_store:
            return _portfolio_store[user_id]

        # Try SQLite if enabled
        if self.use_sqlite:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT portfolio_json FROM portfolios WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                portfolio = UserPortfolio.model_validate_json(row[0])
                _portfolio_store[user_id] = portfolio
                return portfolio

        return None

    def delete(self, user_id: str) -> bool:
        """
        Delete a portfolio.

        Args:
            user_id: User ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        deleted = user_id in _portfolio_store
        if deleted:
            del _portfolio_store[user_id]

        if self.use_sqlite:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolios WHERE user_id = ?", (user_id,))
            deleted = deleted or cursor.rowcount > 0
            conn.commit()
            conn.close()

        return deleted

    def list_all(self) -> list[str]:
        """
        List all user IDs with portfolios.

        Returns:
            List of user IDs.
        """
        user_ids = set(_portfolio_store.keys())

        if self.use_sqlite:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM portfolios")
            for row in cursor.fetchall():
                user_ids.add(row[0])
            conn.close()

        return list(user_ids)


# Module-level singleton
portfolio_store = PortfolioStore()
