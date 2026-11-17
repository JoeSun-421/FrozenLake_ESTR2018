"""Register verified experiments without mixing planned controls into baselines."""
import json
from pathlib import Path


def publish(output, catalog):
    output, catalog = Path(output).resolve(), Path(catalog).resolve()
    project = catalog.parent.parent
    verification = json.loads((output / "verification.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if verification["status"] != "passed" or manifest["backend"] != "gpu":
        raise ValueError("Only verified results from the retained training backend can be published")
    cfg = json.loads((output / "config.json").read_text())
    aggregate = json.loads((output / "aggregate.json").read_text())
    doc = json.loads(catalog.read_text()) if catalog.exists() else dict(schema_version=1, experiments=[])
    record = dict(id=cfg["name"], label=f'{aggregate["size"]}×{aggregate["size"]} — {cfg["variant"]}',
                  size=aggregate["size"], variant=cfg["variant"], path=str(output.relative_to(project)))
    doc["experiments"] = sorted([r for r in doc["experiments"] if r["id"] != record["id"]] + [record],
                                key=lambda r: (r["size"], r["variant"]))
    doc["future_work"] = dict(optimistic_16=dict(config="config/optimistic_16.json",
        status="published" if any(r["id"] == "optimistic_16" for r in doc["experiments"]) else "planned"))
    catalog.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return record
