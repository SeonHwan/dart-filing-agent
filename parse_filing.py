import json
from pathlib import Path
import xml.etree.ElementTree as ET

from dart_sections import Section, parse_sections
from dart_xml_sanitize import sanitize_dart_xml

DATA_DIR = Path(__file__).parent / "data"


def _flatten(sections: list[Section]) -> list[dict]:
    records = []
    for s1 in sections:
        for leaf in s1.iter_leaves():
            if not leaf.blocks:
                continue
            records.append(
                {
                    "atocid": leaf.atocid,
                    "level": leaf.level,
                    "path": leaf.path,
                    "blocks": leaf.blocks,
                }
            )
    return records


def parse_one(xml_path: Path) -> list[dict]:
    raw = xml_path.read_text(encoding="utf-8")
    root = ET.fromstring(sanitize_dart_xml(raw))
    sections = parse_sections(root)
    return _flatten(sections)


def main() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))

    for entry in manifest:
        filing_dir = DATA_DIR / entry["company"] / entry["period"].replace(".", "-")
        xml_path = filing_dir / f"{entry['rcept_no']}.xml"
        records = parse_one(xml_path)

        out_path = filing_dir / "sections.json"
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        n_tables = sum(1 for r in records for b in r["blocks"] if b["type"] == "table")
        n_paras = sum(1 for r in records for b in r["blocks"] if b["type"] == "p")
        print(
            f"{entry['company']} {entry['period']}: 섹션 {len(records)}개,"
            f" 문단 {n_paras:,}개 / 표 {n_tables:,}개 -> {out_path}"
        )


if __name__ == "__main__":
    main()
