from pathlib import Path
from typing import List


def _variant_key(item: str) -> tuple[int, tuple[int, ...]]:
    """Return a comparable key where suffixed variants are larger than base."""
    parts = item.split("_")
    if len(parts) == 1:
        return (0, ())

    numeric_suffix: tuple[int, ...] = tuple(int(part) for part in parts[1:])
    return (1, numeric_suffix)


def remove_duplicates_from_list(input_list: List[Path]):
    # --- input list contains strings like 01, 01_01, 01_02, 03, 04, 04_01, and so on
    # ---- only keep the bigger one, so 01 and 01_01 will be removed, but 01_02 will be kept
    best_by_base = {}

    for path in input_list:
        item = path.stem
        base = item.split("_", 1)[0]
        existing_path = best_by_base.get(base)
        if existing_path:
            existing = existing_path.stem
        else:
            existing = None
        if existing is None or _variant_key(item) > _variant_key(existing):
            best_by_base[base] = path

    return sorted(best_by_base.values())


def test_remove_duplicates_from_list():
    dir = r"D:\DPythonProjects\image-generator\archive\major-system-v5"
    input_list = Path(dir).rglob("*.png")
    expected_output = ["01_02", "03", "04_01"]
    actual_output = remove_duplicates_from_list(input_list)

    print("Actual Output:", actual_output)
    print("Expected Output:", expected_output)
    assert actual_output == expected_output
