from __future__ import annotations

from pathlib import Path
import io
import pstats
from typing import Tuple, Any

try:
    import cProfile  # type: ignore
except Exception:  # pragma: no cover
    cProfile = None  # type: ignore


def write_cprofile_outputs(prof: Any, out_dir: str | Path, *, basename: str = "profile", sort: str = "cumulative") -> Tuple[Path, Path]:
    """Persist a cProfile profile to .prof and a sorted text report.

    Parameters
    ----------
    prof : cProfile.Profile
        A cProfile profile object with .disable() already called.
    out_dir : str | Path
        Directory to write outputs into.
    basename : str
        Base name for the files (default: "profile").
    sort : str
        pstats sort key (e.g., "cumulative", "tottime").

    Returns
    -------
    prof_path : Path
        Path to the written .prof file.
    txt_path : Path
        Path to the written text report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prof_path = out_dir / f"{basename}.prof"
    prof.dump_stats(str(prof_path))

    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).strip_dirs().sort_stats(sort).print_stats()
    txt_path = out_dir / f"{basename}_cprofile_{sort}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return prof_path, txt_path
