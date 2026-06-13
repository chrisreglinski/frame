import importlib.util
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_match_info = _load("match_info", "02_src/02_features/match_info.py")
_match_target = _load("match_target", "02_src/02_features/match_target.py")
_match_team_stats = _load("match_team_stats", "02_src/02_features/match_team_stats.py")
_match_matchup_stats = _load("match_matchup_stats", "02_src/02_features/match_matchup_stats.py")
_abt = _load("abt", "02_src/02_features/abt.py")

build_match_info = _match_info.build_match_info
build_match_target = _match_target.build_match_target
build_match_team_stats = _match_team_stats.build_match_team_stats
build_match_matchup_stats = _match_matchup_stats.build_match_matchup_stats
build_abt = _abt.build_abt
