from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "app_store.db"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_store() -> None:
    with _conn() as conn:
        conn.execute(
            """
            create table if not exists users (
                user_id text primary key,
                display_name text not null,
                wechat_id text not null default '',
                balance real not null default 0,
                created_at text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists assets (
                id integer primary key autoincrement,
                owner_id text not null,
                asset_type text not null,
                title text not null,
                content_json text not null,
                is_paid integer not null default 0,
                price real not null default 0,
                visibility text not null default 'private',
                created_at text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists purchases (
                id integer primary key autoincrement,
                user_id text not null,
                asset_id integer not null,
                price real not null,
                created_at text not null,
                unique(user_id, asset_id)
            )
            """
        )
        conn.execute(
            """
            insert or ignore into users (user_id, display_name, wechat_id, balance, created_at)
            values ('u_demo', 'DemoUser', '', 1000, ?)
            """,
            (_now(),),
        )
        conn.commit()


def _asset_code(asset_id: int) -> str:
    return f"A{asset_id:06d}"


def _asset_from_row(row: sqlite3.Row) -> dict:
    return {
        "asset_id": _asset_code(int(row["id"])),
        "asset_id_num": int(row["id"]),
        "owner_id": row["owner_id"],
        "asset_type": row["asset_type"],
        "title": row["title"],
        "content": json.loads(row["content_json"]),
        "is_paid": bool(row["is_paid"]),
        "price": float(row["price"]),
        "visibility": row["visibility"],
        "created_at": row["created_at"],
    }


def list_users() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """
            select user_id, display_name, wechat_id, balance, created_at
            from users
            order by created_at asc
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_user(user_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            """
            select user_id, display_name, wechat_id, balance, created_at
            from users
            where user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def create_user(user_id: str, display_name: str, init_balance: float) -> tuple[bool, str]:
    user_id = user_id.strip()
    if not user_id:
        return False, "用户ID不能为空。"
    with _conn() as conn:
        exists = conn.execute("select 1 from users where user_id = ?", (user_id,)).fetchone()
        if exists:
            return False, "用户ID已存在。"
        conn.execute(
            """
            insert into users (user_id, display_name, wechat_id, balance, created_at)
            values (?, ?, '', ?, ?)
            """,
            (user_id, display_name.strip() or user_id, float(init_balance), _now()),
        )
        conn.commit()
    return True, "用户创建成功。"


def update_wechat(user_id: str, wechat_id: str) -> None:
    with _conn() as conn:
        conn.execute("update users set wechat_id = ? where user_id = ?", (wechat_id.strip(), user_id))
        conn.commit()


def create_asset(
    owner_id: str,
    asset_type: str,
    title: str,
    content: dict,
    is_paid: bool,
    price: float,
    visibility: str,
) -> str:
    with _conn() as conn:
        cur = conn.execute(
            """
            insert into assets (owner_id, asset_type, title, content_json, is_paid, price, visibility, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_id,
                asset_type,
                title,
                json.dumps(content, ensure_ascii=False),
                1 if is_paid else 0,
                float(price),
                visibility,
                _now(),
            ),
        )
        conn.commit()
        aid = int(cur.lastrowid)
    return _asset_code(aid)


def list_assets(asset_type: str | None = None) -> list[dict]:
    sql = """
        select id, owner_id, asset_type, title, content_json, is_paid, price, visibility, created_at
        from assets
    """
    params: tuple = ()
    if asset_type:
        sql += " where asset_type = ? "
        params = (asset_type,)
    sql += " order by id desc "
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_asset_from_row(row) for row in rows]


def purchased_asset_ids(user_id: str) -> set[int]:
    with _conn() as conn:
        rows = conn.execute("select asset_id from purchases where user_id = ?", (user_id,)).fetchall()
    return {int(row["asset_id"]) for row in rows}


def buy_asset(user_id: str, asset_id_num: int) -> tuple[bool, str]:
    with _conn() as conn:
        row = conn.execute(
            """
            select id, owner_id, is_paid, price, visibility
            from assets
            where id = ?
            """,
            (asset_id_num,),
        ).fetchone()
        if not row:
            return False, "资产不存在。"

        owner_id = row["owner_id"]
        is_paid = bool(row["is_paid"])
        price = float(row["price"])

        if user_id == owner_id:
            return True, "资产所有者无需购买。"
        if not is_paid:
            return True, "该资产免费可见。"

        purchased = conn.execute(
            "select 1 from purchases where user_id = ? and asset_id = ?",
            (user_id, asset_id_num),
        ).fetchone()
        if purchased:
            return True, "你已购买该资产。"

        buyer = conn.execute("select balance from users where user_id = ?", (user_id,)).fetchone()
        seller = conn.execute("select balance from users where user_id = ?", (owner_id,)).fetchone()
        if not buyer or not seller:
            return False, "用户不存在，无法完成购买。"
        if float(buyer["balance"]) < price:
            return False, "余额不足，请先充值（当前示例仅本地模拟余额）。"

        conn.execute("update users set balance = balance - ? where user_id = ?", (price, user_id))
        conn.execute("update users set balance = balance + ? where user_id = ?", (price, owner_id))
        conn.execute(
            """
            insert into purchases (user_id, asset_id, price, created_at)
            values (?, ?, ?, ?)
            """,
            (user_id, asset_id_num, price, _now()),
        )
        conn.commit()
    return True, f"购买成功，已支付 {price:.2f}。"
