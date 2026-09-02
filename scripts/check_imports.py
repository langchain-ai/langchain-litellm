import importlib.util
import sys
import traceback
from pathlib import Path

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        module_name = ".".join(Path(file).with_suffix("").parts)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not determine a loader for {file}")
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
