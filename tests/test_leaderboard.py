import json

from stretch_demo.tasks.leaderboard import Leaderboard, clock_text, normalize_initials


def test_initials_are_cleaned_up():
    assert normalize_initials(" a.b c d ") == "ABC"
    assert normalize_initials("   ") is None


def test_ranked_by_score_then_time_and_kept_per_controller(tmp_path):
    path = tmp_path / "board.json"
    head, hands = Leaderboard(path, "head"), Leaderboard(path, "hands_vel")
    assert head.add("AAA", 300, 90.0) == 1
    assert head.add("BBB", 400, 120.0) == 1   # higher score wins
    assert head.add("CCC", 400, 100.0) == 1   # same score, faster
    assert hands.add("DDD", 500, 60.0) == 1   # its own board
    assert [e["initials"] for e in head.entries()] == ["CCC", "BBB", "AAA"]
    assert len(json.loads(path.read_text())) == 4  # everything kept in one file


def test_rows_show_top_five_and_mark_the_new_entry(tmp_path):
    board = Leaderboard(tmp_path / "board.json", "head")
    assert board.rows()[1] == ("", "no scores yet")
    for i in range(7):
        board.add(f"P{i}", 100 * i, 60.0)
    rows = board.rows()
    assert len(rows) == 6 and rows[1][0] == "1. P6" and "<-- new" in rows[1][1]


def test_unreadable_file_is_moved_aside_not_lost(tmp_path):
    path = tmp_path / "board.json"
    path.write_text("{not json")
    board = Leaderboard(path, "head")
    assert board.add("AAA", 100, 10.0) == 1
    assert len(list(tmp_path.glob("board.unreadable-*.json"))) == 1


def test_clock_text():
    assert clock_text(83.4) == "1:23"
