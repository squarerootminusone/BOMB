"""Move selection strategies for Go self-play.

Provides three strategies behind a common ABC interface:
- RandomMoveSelector: uniformly random legal move (excluding pass)
- MCTSMoveSelector: OpenSpiel MCTS with random rollout evaluation
- KataGoMoveSelector: KataGo engine via GTP subprocess
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class MoveSelector(ABC):
    """Abstract base for move selection strategies.

    Why: A common interface lets callers swap strategies without changing
    the game-playing loop.

    How: Subclasses implement ``select_move`` which inspects game state
    through an ``_OpenSpielGoLogic`` instance and returns a board coordinate.
    """

    @abstractmethod
    def select_move(self, logic: object, board_size: int) -> Tuple[int, int]:
        """Select next move.

        Args:
            logic: An ``_OpenSpielGoLogic`` instance exposing ``.legal_actions()``
                   and ``._state``.
            board_size: Side length of the board (e.g. 5).

        Returns:
            ``(row, col)`` tuple for the chosen intersection.
        """

    def close(self) -> None:
        """Release any held resources (subprocesses, file handles, etc.)."""


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


class RandomMoveSelector(MoveSelector):
    """Pick a uniformly random legal move, excluding pass.

    Why: Baseline strategy useful for data collection and testing.
    How: Filters the pass action from legal actions, then samples uniformly.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    def select_move(self, logic: object, board_size: int) -> Tuple[int, int]:
        pass_id = board_size * board_size
        legal: List[int] = [a for a in logic.legal_actions() if a != pass_id]  # type: ignore[attr-defined]
        if not legal:
            # No non-pass moves available; fall back to center.
            return (board_size // 2, board_size // 2)
        action = int(self._rng.choice(np.asarray(legal, dtype=np.int32)))
        row, col = divmod(action, board_size)
        return (row, col)


# ---------------------------------------------------------------------------
# MCTS (OpenSpiel)
# ---------------------------------------------------------------------------


class MCTSMoveSelector(MoveSelector):
    """Monte-Carlo Tree Search via OpenSpiel's ``MCTSBot``.

    Why: Provides a stronger-than-random policy for generating realistic
    game trajectories without an external engine.

    How: Wraps ``open_spiel.python.algorithms.mcts.MCTSBot`` with a random
    rollout evaluator.  On each call the current game state is cloned so
    the search does not mutate the caller's state.
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        uct_c: float = 2.0,
        seed: int = 42,
        board_size: int = 5,
    ) -> None:
        try:
            import pyspiel
            from open_spiel.python.algorithms import mcts as _mcts
        except ImportError as exc:
            raise ImportError(
                "pyspiel and open_spiel are required for MCTSMoveSelector. "
                "Install with: pip install open_spiel"
            ) from exc

        self._mcts = _mcts
        game = pyspiel.load_game("go", {"board_size": board_size})
        evaluator = _mcts.RandomRolloutEvaluator(
            n_rollouts=1,
            random_state=np.random.RandomState(seed),
        )
        self._bot = _mcts.MCTSBot(
            game,
            uct_c=uct_c,
            max_simulations=num_simulations,
            evaluator=evaluator,
        )

    def select_move(self, logic: object, board_size: int) -> Tuple[int, int]:
        state = logic._state.clone()  # type: ignore[attr-defined]
        action = self._bot.step(state)

        pass_id = board_size * board_size
        if action == pass_id:
            legal = [a for a in state.legal_actions() if a != pass_id]
            if legal:
                action = legal[0]

        row, col = divmod(action, board_size)
        return (row, col)


# ---------------------------------------------------------------------------
# KataGo (GTP subprocess)
# ---------------------------------------------------------------------------


class KataGoMoveSelector(MoveSelector):
    """Communicate with a KataGo engine over the GTP protocol.

    Why: KataGo provides superhuman-strength moves for high-quality
    demonstration trajectories.

    How: Spawns KataGo as a subprocess and exchanges GTP commands over
    stdin/stdout.  Board state is kept in sync by sending ``play`` commands
    for every move made externally; ``genmove`` retrieves the engine's
    chosen move.
    """

    def __init__(
        self,
        katago_path: str = "katago",
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        visits: int = 100,
        board_size: int = 5,
        komi: float = 6.5,
    ) -> None:
        cmd: List[str] = [katago_path, "gtp"]
        if model_path is not None:
            cmd += ["-model", model_path]
        if config_path is not None:
            cmd += ["-config", config_path]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._board_size = int(board_size)

        self._send(f"boardsize {self._board_size}")
        self._send(f"komi {komi}")
        self._send("clear_board")
        if visits > 0:
            self._send(f"kata-set-param maxVisits {visits}")

        self._moves: List[Tuple[str, int, int]] = []

    # -- GTP I/O helpers ----------------------------------------------------

    def _send(self, cmd: str) -> str:
        """Send a GTP command and return the response string.

        Why: All KataGo interaction funnels through this single method so
        error handling and flushing stay consistent.
        """
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()
        return self._read_response()

    def _read_response(self) -> str:
        """Read one GTP response (terminated by a blank line)."""
        assert self._proc.stdout is not None

        lines: List[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break
            stripped = line.strip()
            if not lines:
                # First non-empty line starts with '=' or '?'.
                if not stripped:
                    continue
                lines.append(stripped)
            else:
                if not stripped:
                    break
                lines.append(stripped)
        return "\n".join(lines)

    # -- Coordinate conversion ----------------------------------------------

    @staticmethod
    def _rc_to_gtp(row: int, col: int, board_size: int) -> str:
        """Convert ``(row, col)`` to a GTP vertex like ``D3``.

        GTP columns run A..T (skipping I); rows count 1..N from the bottom.
        """
        col_letter = chr(ord("A") + col + (1 if col >= 8 else 0))
        row_number = board_size - row
        return f"{col_letter}{row_number}"

    @staticmethod
    def _gtp_to_rc(vertex: str, board_size: int) -> Tuple[int, int]:
        """Convert a GTP vertex like ``D3`` to ``(row, col)``."""
        col_letter = vertex[0].upper()
        col = ord(col_letter) - ord("A")
        if col > 8:  # account for skipped 'I'
            col -= 1
        row = board_size - int(vertex[1:])
        return (row, col)

    # -- Public API ---------------------------------------------------------

    def notify_move(self, row: int, col: int, color: str, board_size: int) -> None:
        """Inform KataGo that a move was played externally.

        Args:
            row: Board row (0-indexed from top).
            col: Board column (0-indexed from left).
            color: ``"black"`` or ``"white"``.
            board_size: Side length of the board.
        """
        gtp_color = "B" if color == "black" else "W"
        gtp_coord = self._rc_to_gtp(row, col, board_size)
        self._send(f"play {gtp_color} {gtp_coord}")
        self._moves.append((color, row, col))

    def select_move(self, logic: object, board_size: int) -> Tuple[int, int]:
        current_player = logic._state.current_player()  # type: ignore[attr-defined]
        gtp_color = "B" if current_player == 0 else "W"
        response = self._send(f"genmove {gtp_color}")

        # Parse "= D3" -> vertex string
        vertex = ""
        if "=" in response:
            vertex = response.split("=", 1)[1].strip().split()[0]

        if vertex.upper() in ("PASS", "RESIGN", ""):
            # Engine passed or resigned; fall back to first legal non-pass move.
            pass_id = board_size * board_size
            legal = [a for a in logic.legal_actions() if a != pass_id]  # type: ignore[attr-defined]
            if legal:
                action = legal[0]
                return divmod(action, board_size)
            return (board_size // 2, board_size // 2)

        return self._gtp_to_rc(vertex, board_size)

    def close(self) -> None:
        """Shut down the KataGo subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send("quit")
            except (BrokenPipeError, OSError):
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_selector(strategy: str, seed: int = 42, **kwargs: object) -> MoveSelector:
    """Create a ``MoveSelector`` by name.

    Args:
        strategy: One of ``"random"``, ``"mcts"``, or ``"katago"``.
        seed: RNG seed forwarded to strategies that use one.
        **kwargs: Extra keyword arguments forwarded to the chosen class.

    Returns:
        A ready-to-use ``MoveSelector`` instance.

    Raises:
        ValueError: If *strategy* is not recognised.
    """
    if strategy == "random":
        return RandomMoveSelector(seed=seed)
    elif strategy == "mcts":
        return MCTSMoveSelector(seed=seed, **kwargs)  # type: ignore[arg-type]
    elif strategy == "katago":
        return KataGoMoveSelector(**kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
