from __future__ import annotations

from datetime import datetime


def make_exit_state(status: str, code: str, msg: str, algo: str) -> dict:
    return {
        "status": status,
        "code": code,
        "msg": msg,
        "algo": algo,
        "ts": datetime.now().astimezone(),
    }


def is_fail(exit_state: dict | None) -> bool:
    return exit_state is not None and exit_state.get("status") == "fail"


def is_warn(exit_state: dict | None) -> bool:
    return exit_state is not None and exit_state.get("status") == "warn"
