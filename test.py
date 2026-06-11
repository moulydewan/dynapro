import json

for fname in [
    "output/simulations/medium_baseline.json",
    "output/simulations/medium_proact.json",
    "output/simulations/medium_dynapro.json",
]:
    with open(fname) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"FILE: {fname}  ({len(data)} convos)")
    print(f"{'='*60}")

    for item in data[:2]:  # first 2 conversations only
        print(f"\n--- Conv {item['conv_id']} ---")
        for t in item["conversation"]:
            preview = t["content"][:150].replace("\n", " ")
            print(f"  [{t['role']:>9}]: {preview}...")
        print(f"  → total turns: {len(item['conversation'])}")