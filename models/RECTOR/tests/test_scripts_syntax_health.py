import py_compile
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"


def test_all_scripts_python_files_compile():
    py_files = [
        path for path in SCRIPTS_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    ]

    # Broad safety net over all modules in scripts/.
    assert len(py_files) > 30

    failures = []
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # pragma: no cover
            failures.append((path, str(exc)))

    assert not failures, "Script syntax failures:\n" + "\n".join(
        f"- {path}: {err}" for path, err in failures
    )
