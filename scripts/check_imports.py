import importlib.util
import sys
import traceback
from pathlib import Path


def _module_name(file: str) -> str:
    """Name each file by its position in the tree so relative imports resolve."""
    path = Path(file).resolve()
    try:
        return ".".join(path.relative_to(Path.cwd()).with_suffix("").parts)
    except ValueError:
        return path.stem


if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            spec = importlib.util.spec_from_file_location(_module_name(file), file)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not determine a loader for {file}")
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        except Exception:  # noqa: BLE001 - report every file that fails to import
            has_failure = True
            print(file)
            traceback.print_exc()
            print()

    sys.exit(1 if has_failure else 0)
