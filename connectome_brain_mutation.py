#helpers for mutating the c302 connectome on a per-neuron basis
#loads the canonical CSV/JSON data that ship w/ OpenWorm Docker image and
#materialises modified copies that scale the outgoing chemical synapses of selected brain neurons

import csv
import json


BRAIN_SCALE_PREFIX = "brain_scale__"
CONNECTOME_FILENAME = "herm_full_edgelist_MODIFIED.csv"
CONNECTOME_HEADER = ("Source", "Target", "Weight", "Type")
NEURON_METADATA_FILENAME = "owmeta_cache.json"


#indicates a mutation error
class ConnectomeMutationError(RuntimeError):
    pass


#return a mapping of neuron name to its set of declared neuron classes
def load_neuron_metadata(data_dir):
    meta_path = data_dir / NEURON_METADATA_FILENAME
    if not meta_path.exists():
        raise ConnectomeMutationError(
            f"couldn't find the neuron metadata file at: {meta_path}"
        )

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    neuron_info = payload.get("neuron_info")
    if not isinstance(neuron_info, dict):
        raise ConnectomeMutationError(
            "neuron metadata JSON looks weird again"
        )

    metadata = {}
    for name, info in neuron_info.items():

        classes = info[1]
        metadata[name] = set(classes or [])
    return metadata


#read the canonical connectome CSV into memory
def connectome_rows(base_path):
    if not base_path.exists():
        raise ConnectomeMutationError(
            f"couldn't find the connectome CSV at: {base_path}"
        )

    with base_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


#write a mutated connectome CSV applying per-neuron scaling factors
def write_mutated_connectome(
    *,
    rows,
    output_path,
    scale_by_neuron,
    brain_targets=None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONNECTOME_HEADER)
        writer.writeheader()
        for entry in rows:
            src = entry["Source"].strip()
            tgt = entry["Target"].strip()
            syn_type = entry["Type"].strip().lower()
            weight_val = float(entry["Weight"])

            scale = float(scale_by_neuron.get(src, 1.0))
            mutate_targets = brain_targets is None or tgt in brain_targets
            if syn_type == "chemical" and mutate_targets:
                if scale <= 0.0:
                    #drop the connection entirely for ablations (scale <= 0)
                    continue
                if scale != 1.0:
                    weight_val = round(max(0.0, weight_val * scale))
                    if weight_val <= 0.0:
                        continue

            writer.writerow(
                {
                    "Source": src,
                    "Target": tgt,
                    "Weight": f"{int(weight_val)}",
                    "Type": entry["Type"].strip(),
                }
            )
