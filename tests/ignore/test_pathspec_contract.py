"""ChunkHound reads pathspec's compiled patterns to decide directories as git does.

pathspec is bounded below 2 (pathspec 1.x deprecates GitWildMatchPattern, the
class the engine compiles with), but any release inside the bound may still
change its pattern objects. If they change shape, the rules degrade to
pathspec's own matching (no pruning, no git-exact re-includes) and log a
warning; these tests fail first so the change is seen. They compile with the
installed pathspec, so they guard whichever version the lock resolves (0.12
spells a regex's tail "(?:(?P<ps_d>/).*)?$", 1.x "(?:(?P<ps_d>/)|$)" and
matches with re.search).
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement

import chunkhound.utils.ignore_engine as ignore_engine
from chunkhound.utils.ignore_engine import (
    _compile_gitignore_lines,
    _compile_gitwildmatch,
    _spec_rules,
    _transform_gitignore_line,
    build_repo_aware_ignore_engine,
)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Every form a .gitignore line takes: plain file, glob, anchored, directory,
# negation, and the whole-tree wildcards.
_GITIGNORE_FORMS = [
    "Thumbs.db",
    "*.log",
    "/build",
    "build/",
    "/build/",
    "node_modules/",
    "foo/**",
    "**/node_modules/**",
    "!keep.py",
    "!/src/",
    "src/**/x.py",
    "**/foo",
    "a/b/",
    "*",
    "/*",
    "foo/*",
    "**",
    "**/",
    "/**/",
    "a/**/b",
    "dir/**/",
    "[abc].py",
    "?x.py",
    ".env*",
    "!.impeccable/live/config.json",
]


def test_ignore_rules_are_read_from_compiled_pathspec_patterns() -> None:
    lines = [
        *_transform_gitignore_line(".", "foo/"),
        *_transform_gitignore_line(".", "!foo/bar.py"),
        *_transform_gitignore_line("src", "!/keep/"),
    ]
    _spec, rules = _compile_gitignore_lines(lines)

    assert not rules.degraded
    assert [rule.dir_only for rule in rules.rules] == [d for _p, d in lines]
    assert [rule.include for rule in rules.rules] == [
        not p.startswith("!") for p, _d in lines
    ]
    assert all(rule.exact is not None for rule in rules.rules)

    config_rules = _spec_rules(_compile_gitwildmatch(["**/node_modules/**", "dist/"]))
    assert not config_rules.degraded
    assert [rule.dir_only for rule in config_rules.rules] == [False, True]


def test_every_gitignore_pattern_form_is_read_from_the_installed_pathspec() -> None:
    lines = [
        line
        for form in _GITIGNORE_FORMS
        for line in _transform_gitignore_line(".", form)
    ]
    _spec, rules = _compile_gitignore_lines(lines)
    assert not rules.degraded

    excludes = [form for form in _GITIGNORE_FORMS if not form.startswith("!")]
    assert not _spec_rules(_compile_gitwildmatch(excludes)).degraded


def test_a_plain_file_pattern_keeps_ignored_directories_pruned(tmp_path: Path) -> None:
    """A .gitignore that names a file (Thumbs.db) beside node_modules/ (TFS 65863).

    With pathspec 1.1 such a line once stopped all directory pruning: the walks
    entered node_modules, while each file in it was still excluded.
    """
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / ".gitignore").write_text(
        "Thumbs.db\n*.log\nnode_modules/\n", encoding="utf-8"
    )
    engine = build_repo_aware_ignore_engine(
        root, ["gitignore"], ".chignore", ["**/dist/**"]
    )

    assert engine.matches(root / "node_modules", is_dir=True)
    assert engine.matches(root / "pkg" / "node_modules", is_dir=True)
    assert engine.matches(root / "web" / "dist", is_dir=True)
    assert not engine.matches(root / "src", is_dir=True)
    assert engine.matches(root / "src" / "Thumbs.db", is_dir=False)
    assert engine.matches(root / "src" / "debug.log", is_dir=False)
    assert not engine.matches(root / "src" / "app.py", is_dir=False)


def test_pathspec_bound_excludes_releases_without_gitwildmatch() -> None:
    """An install cannot resolve a pathspec without GitWildMatchPattern (TFS 65865).

    pathspec 1.x deprecates the class; a 2.x may remove it, and an unbounded
    requirement would let the next tool install take it and stop every index.
    """
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    (requirement,) = [
        req
        for req in map(Requirement, project["dependencies"])
        if req.name == "pathspec"
    ]

    assert requirement.specifier.contains("0.12.1")
    assert requirement.specifier.contains("1.1.1")
    assert not requirement.specifier.contains("2.0.0")


def test_a_pathspec_without_gitwildmatch_is_named_in_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error names the missing class and the installed pathspec (TFS 65865).

    Loads a separate copy of the engine module against a pathspec whose
    gitwildmatch module has no GitWildMatchPattern, as a release that removed
    it would look.
    """
    monkeypatch.setitem(
        sys.modules,
        "pathspec.patterns.gitwildmatch",
        ModuleType("pathspec.patterns.gitwildmatch"),
    )
    spec = importlib.util.spec_from_file_location(
        "_ignore_engine_without_gitwildmatch", ignore_engine.__file__
    )
    assert spec is not None and spec.loader is not None
    engine_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, engine_module)
    spec.loader.exec_module(engine_module)

    with pytest.raises(RuntimeError) as raised:
        engine_module.build_ignore_engine(
            tmp_path, ["config"], config_exclude=["*.log"]
        )

    message = str(raised.value)
    assert "GitWildMatchPattern" in message
    assert f"pathspec {version('pathspec')}" in message
