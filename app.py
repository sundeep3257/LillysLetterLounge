import json
import random
import re
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, session, url_for

app = Flask(__name__)
# Render/production: set SECRET_KEY in environment variables.
# (Keeping a dev fallback so local runs still work.)
app.secret_key = os.getenv("SECRET_KEY", "lillys-letter-lounge-dev-secret")


BASE_DIR = Path(__file__).resolve().parent
# Default SQLite DB location; override via env if you mount a persistent disk on Render.
# Example Render env var: DB_PATH=/var/data/wordsearch.db
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "wordsearch.db")))

# Optional: allow configuring Lilly's password without editing code.
LILLY_PASSWORD = os.getenv("LILLY_PASSWORD", "07072009")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Ensure ON DELETE CASCADE works for lilly_progress, etc.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS puzzles (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              creator_name TEXT NOT NULL,
              theme TEXT NOT NULL,
              grid_size INTEGER NOT NULL,
              words_json TEXT NOT NULL,
              grid_json TEXT NOT NULL,
              placements_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lilly_progress (
              puzzle_id INTEGER PRIMARY KEY,
              found_words_json TEXT NOT NULL DEFAULT '[]',
              updated_at TEXT NOT NULL,
              FOREIGN KEY(puzzle_id) REFERENCES puzzles(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


# Ensure tables exist when running under Gunicorn (import-time init is safe for SQLite).
init_db()


def build_puzzle_list() -> list[dict]:
    """Build the puzzle list with status + metadata (used by Lilly + Admin dashboards)."""
    with get_db() as conn:
        puzzles = conn.execute(
            """
            SELECT id, created_at, creator_name, theme, grid_size, words_json
            FROM puzzles
            ORDER BY id DESC
            """
        ).fetchall()

    items: list[dict] = []
    for p in puzzles:
        words = json.loads(p["words_json"])
        total = len(words)
        found = get_progress(int(p["id"]))
        found_count = len(set(found))
        if found_count <= 0:
            status = "not completed"
        elif found_count >= total and total > 0:
            status = "completed"
        else:
            status = "partially completed"

        items.append(
            {
                "id": int(p["id"]),
                "theme": p["theme"],
                "creator_name": p["creator_name"],
                "grid_size": int(p["grid_size"]),
                "word_count": total,
                "found_count": found_count,
                "status": status,
            }
        )

    return items


_WORD_RE = re.compile(r"[A-Za-z]+")


def normalize_words(raw: str) -> list[str]:
    # Accept commas / newlines / any text; keep letter-runs only.
    words = [w.upper() for w in _WORD_RE.findall(raw or "")]
    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def generate_word_search(words: list[str], size: int) -> tuple[list[list[str]], list[dict]]:
    if size < 5 or size > 40:
        raise ValueError("Grid size must be between 5 and 40.")
    if not words:
        raise ValueError("Please provide at least one word.")
    if any(len(w) > size for w in words):
        too_long = max((w for w in words if len(w) > size), key=len)
        raise ValueError(f'Word "{too_long}" is longer than the grid size ({size}).')

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]

    words_sorted = sorted(words, key=len, reverse=True)

    def empty_grid() -> list[list[str]]:
        return [["" for _ in range(size)] for _ in range(size)]

    def can_place(grid: list[list[str]], word: str, r: int, c: int, dr: int, dc: int) -> bool:
        rr, cc = r, c
        for ch in word:
            if rr < 0 or rr >= size or cc < 0 or cc >= size:
                return False
            existing = grid[rr][cc]
            if existing and existing != ch:
                return False
            rr += dr
            cc += dc
        return True

    def do_place(grid: list[list[str]], word: str, r: int, c: int, dr: int, dc: int) -> list[tuple[int, int]]:
        coords: list[tuple[int, int]] = []
        rr, cc = r, c
        for ch in word:
            grid[rr][cc] = ch
            coords.append((rr, cc))
            rr += dr
            cc += dc
        return coords

    # Try multiple full-grid attempts (randomized placements).
    for _attempt in range(80):
        grid = empty_grid()
        placements: list[dict] = []

        ok = True
        for word in words_sorted:
            placed = False
            for _try in range(600):
                dr, dc = random.choice(directions)
                # Choose a start that can possibly fit (bias to valid ranges).
                start_r = random.randrange(size)
                start_c = random.randrange(size)

                end_r = start_r + dr * (len(word) - 1)
                end_c = start_c + dc * (len(word) - 1)
                if not (0 <= end_r < size and 0 <= end_c < size):
                    continue

                if not can_place(grid, word, start_r, start_c, dr, dc):
                    continue

                coords = do_place(grid, word, start_r, start_c, dr, dc)
                placements.append(
                    {
                        "word": word,
                        "start": [start_r, start_c],
                        "dir": [dr, dc],
                        "cells": [[r, c] for (r, c) in coords],
                    }
                )
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        # Fill blanks
        for r in range(size):
            for c in range(size):
                if not grid[r][c]:
                    grid[r][c] = chr(ord("A") + random.randrange(26))

        return grid, placements

    raise ValueError("Could not generate a puzzle with those settings. Try a larger grid or fewer words.")


def lilly_is_authed() -> bool:
    return bool(session.get("lilly_authed"))


def require_lilly() -> None:
    if not lilly_is_authed():
        abort(401)


def get_progress(puzzle_id: int) -> list[str]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT found_words_json FROM lilly_progress WHERE puzzle_id = ?",
            (puzzle_id,),
        ).fetchone()

    if row is None:
        return []
    try:
        data = json.loads(row["found_words_json"])
        return [str(w).upper() for w in data if str(w).strip()]
    except json.JSONDecodeError:
        return []


def set_progress(puzzle_id: int, found_words: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    found_words = [w.upper() for w in found_words]
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO lilly_progress (puzzle_id, found_words_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(puzzle_id) DO UPDATE SET
              found_words_json = excluded.found_words_json,
              updated_at = excluded.updated_at
            """,
            (puzzle_id, json.dumps(found_words), now),
        )
        conn.commit()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        BASE_DIR / "templates",
        "Favicon_LLL.png",
        mimetype="image/png",
    )


@app.route("/lilly/login", methods=["GET", "POST"])
def lilly_login():
    error = None
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if password == LILLY_PASSWORD:
            session["lilly_authed"] = True
            return redirect(url_for("lilly_dashboard"))
        error = "Incorrect password. Try again."
    return render_template("lilly_login.html", error=error)


@app.route("/lilly/logout")
def lilly_logout():
    session.pop("lilly_authed", None)
    return redirect(url_for("home"))


@app.route("/lilly")
def lilly_dashboard():
    if not lilly_is_authed():
        return redirect(url_for("lilly_login"))
    return render_template("lilly_dashboard.html", puzzles=build_puzzle_list())


@app.route("/admin")
def admin_dashboard():
    # Hidden entry via triple-click; no password requested by user.
    return render_template("admin_dashboard.html", puzzles=build_puzzle_list())


@app.route("/admin/puzzles/<int:puzzle_id>/delete", methods=["POST"])
def admin_delete_puzzle(puzzle_id: int):
    with get_db() as conn:
        # Extra safety: delete progress explicitly (even though FK cascade is enabled)
        conn.execute("DELETE FROM lilly_progress WHERE puzzle_id = ?", (puzzle_id,))
        conn.execute("DELETE FROM puzzles WHERE id = ?", (puzzle_id,))
        conn.commit()

    return redirect(url_for("admin_dashboard"))


@app.route("/lilly/puzzles/<int:puzzle_id>")
def lilly_solve(puzzle_id: int):
    if not lilly_is_authed():
        return redirect(url_for("lilly_login"))

    with get_db() as conn:
        row = conn.execute("SELECT * FROM puzzles WHERE id = ?", (puzzle_id,)).fetchone()

    if row is None:
        return "Puzzle not found", 404

    words = json.loads(row["words_json"])
    grid = json.loads(row["grid_json"])
    found_words = set(get_progress(puzzle_id))

    # Only reveal cell locations for words Lilly has already found.
    found_cells: dict[str, list[list[int]]] = {}
    try:
        placements = json.loads(row["placements_json"])
    except json.JSONDecodeError:
        placements = []

    for pl in placements:
        w = str(pl.get("word") or "").upper()
        cells = pl.get("cells") or []
        if w and w in found_words and isinstance(cells, list):
            found_cells[w] = cells

    return render_template(
        "lilly_solve.html",
        puzzle={
            "id": row["id"],
            "theme": row["theme"],
            "creator_name": row["creator_name"],
            "grid_size": row["grid_size"],
            "words": words,
            "grid": grid,
            "found_words": sorted(found_words),
            "found_cells": found_cells,
        },
    )


@app.route("/lilly/puzzles/<int:puzzle_id>/check", methods=["POST"])
def lilly_check_selection(puzzle_id: int):
    require_lilly()

    data = request.get_json(silent=True) or {}
    start = data.get("start")
    end = data.get("end")
    if (
        not isinstance(start, list)
        or not isinstance(end, list)
        or len(start) != 2
        or len(end) != 2
    ):
        return jsonify({"ok": False, "reason": "bad_request"}), 400

    sr, sc = int(start[0]), int(start[1])
    er, ec = int(end[0]), int(end[1])

    with get_db() as conn:
        row = conn.execute("SELECT grid_size, placements_json FROM puzzles WHERE id = ?", (puzzle_id,)).fetchone()

    if row is None:
        return jsonify({"ok": False, "reason": "not_found"}), 404

    size = int(row["grid_size"])
    if not (0 <= sr < size and 0 <= sc < size and 0 <= er < size and 0 <= ec < size):
        return jsonify({"ok": False, "reason": "out_of_bounds"}), 200

    dr = er - sr
    dc = ec - sc
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)

    # Must be straight (horizontal/vertical/diagonal)
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return jsonify({"ok": False, "reason": "not_straight"}), 200

    length = max(abs(dr), abs(dc)) + 1
    cells = [[sr + i * step_r, sc + i * step_c] for i in range(length)]

    try:
        placements = json.loads(row["placements_json"])
    except json.JSONDecodeError:
        placements = []

    # Map placements by exact cell sequence (both directions)
    placement_map: dict[str, str] = {}
    for pl in placements:
        word = str(pl.get("word") or "").upper()
        pl_cells = pl.get("cells") or []
        if not word or not isinstance(pl_cells, list):
            continue
        key_fwd = ";".join([f"{r},{c}" for r, c in pl_cells])
        key_rev = ";".join([f"{r},{c}" for r, c in reversed(pl_cells)])
        placement_map[key_fwd] = word
        placement_map[key_rev] = word

    sel_key = ";".join([f"{r},{c}" for r, c in cells])
    word = placement_map.get(sel_key)
    if not word:
        return jsonify({"ok": False, "reason": "no_match"}), 200

    found_words = set(get_progress(puzzle_id))
    if word in found_words:
        return jsonify({"ok": True, "word": word, "cells": cells, "already_found": True}), 200

    found_words.add(word)
    set_progress(puzzle_id, sorted(found_words))

    return jsonify({"ok": True, "word": word, "cells": cells, "already_found": False}), 200


@app.route("/make", methods=["GET", "POST"])
def make_word_search():
    error = None

    form_defaults = {
        "creator_name": "",
        "theme": "",
        "grid_size": "12",
        "words": "",
    }

    if request.method == "GET":
        return render_template("make.html", error=None, form=form_defaults)

    if request.method == "POST":
        # If user is returning to the editor from the preview page, do NOT regenerate.
        if (request.form.get("mode") or "").strip().lower() == "edit":
            form = {
                "creator_name": (request.form.get("creator_name") or "").strip(),
                "theme": (request.form.get("theme") or "").strip(),
                "grid_size": (request.form.get("grid_size") or "12").strip(),
                "words": request.form.get("words") or "",
            }
            return render_template("make.html", error=None, form=form)

        creator_name = (request.form.get("creator_name") or "").strip()
        theme = (request.form.get("theme") or "").strip()
        raw_words = request.form.get("words") or ""
        grid_size_raw = (request.form.get("grid_size") or "").strip()

        form = {
            "creator_name": creator_name,
            "theme": theme,
            "grid_size": grid_size_raw or "12",
            "words": raw_words,
        }

        if not creator_name:
            error = "Please enter your name."
        elif not theme:
            error = "Please enter a theme."
        else:
            try:
                grid_size = int(grid_size_raw)
            except ValueError:
                grid_size = 0

            words = normalize_words(raw_words)
            try:
                grid, placements = generate_word_search(words, grid_size)

                # Preview first (no DB write yet)
                payload = {
                    "creator_name": creator_name,
                    "theme": theme,
                    "grid_size": grid_size,
                    "words": words,
                    "grid": grid,
                    "placements": placements,
                }

                return render_template("preview.html", payload=payload)
            except ValueError as e:
                error = str(e)

    return render_template("make.html", error=error, form=form if "form" in locals() else form_defaults)


@app.route("/make/save", methods=["POST"])
def save_word_search():
    payload_raw = request.form.get("payload") or ""
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return "Invalid puzzle payload.", 400

    creator_name = (payload.get("creator_name") or "").strip()
    theme = (payload.get("theme") or "").strip()
    grid_size = int(payload.get("grid_size") or 0)
    words = payload.get("words") or []
    grid = payload.get("grid") or []
    placements = payload.get("placements") or []

    # Minimal validation
    if not creator_name or not theme or not isinstance(words, list) or not isinstance(grid, list):
        return "Invalid puzzle payload.", 400

    created_at = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO puzzles (created_at, creator_name, theme, grid_size, words_json, grid_json, placements_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                creator_name,
                theme,
                grid_size,
                json.dumps(words),
                json.dumps(grid),
                json.dumps(placements),
            ),
        )
        puzzle_id = cur.lastrowid
        conn.commit()

    return redirect(url_for("view_puzzle", puzzle_id=puzzle_id))


@app.route("/puzzles/<int:puzzle_id>")
def view_puzzle(puzzle_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM puzzles WHERE id = ?", (puzzle_id,)).fetchone()

    if row is None:
        return "Puzzle not found", 404

    words = json.loads(row["words_json"])
    grid = json.loads(row["grid_json"])

    return render_template(
        "puzzle.html",
        puzzle={
            "id": row["id"],
            "created_at": row["created_at"],
            "creator_name": row["creator_name"],
            "theme": row["theme"],
            "grid_size": row["grid_size"],
            "words": words,
            "grid": grid,
        },
    )


if __name__ == "__main__":
    # Local dev server (Render/Gunicorn will not execute this block)
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "").strip() in {"1", "true", "True"}
    app.run(host="0.0.0.0", port=port, debug=debug)
