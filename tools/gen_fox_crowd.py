#!/usr/bin/env python3
"""Generates Models/Fox/glTF/FoxCrowd.gltf: the stock Fox with EXT_mesh_gpu_instancing
placing N instances (default 32x32 = 1024) on a grid. The skinned mesh node gets the
instancing extension; instance TRANSLATION lives in an embedded data-URI buffer, so the
file still references the stock Fox.bin/textures and needs no extra binaries.

Instancing a skinned mesh is undefined in the EXT spec; the NeuralOIT renderer defines it
as: instance TRS = per-instance placement applied after skinning (each instance gets its
own skin-matrix block / animation phase).

Usage: python3 tools/gen_fox_crowd.py [--side 32] [--step 250]
"""

import argparse
import base64
import json
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=32, help="grid side (side^2 instances)")
    parser.add_argument("--step", type=float, default=250.0, help="grid step, model units")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = root / "Models" / "Fox" / "glTF" / "Fox.gltf"
    dst = root / "Models" / "Fox" / "glTF" / "FoxCrowd.gltf"
    gltf = json.loads(src.read_text())

    side, step = args.side, args.step
    count = side * side

    # Grid translations, centered at the origin (y = 0; the fox stands on its local ground).
    data = bytearray()
    half = (side - 1) * 0.5
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for k in range(count):
        x = (k % side - half) * step
        z = (k // side - half) * step
        t = (x, 0.0, z)
        data += struct.pack("<3f", *t)
        for c in range(3):
            mn[c] = min(mn[c], t[c])
            mx[c] = max(mx[c], t[c])

    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(data)).decode()
    gltf.setdefault("buffers", []).append({"uri": uri, "byteLength": len(data)})
    gltf.setdefault("bufferViews", []).append(
        {"buffer": len(gltf["buffers"]) - 1, "byteOffset": 0, "byteLength": len(data)}
    )
    gltf.setdefault("accessors", []).append(
        {
            "bufferView": len(gltf["bufferViews"]) - 1,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": count,
            "type": "VEC3",
            "min": mn,
            "max": mx,
        }
    )

    # Attach the instancing extension to the skinned mesh node.
    fox = next(n for n in gltf["nodes"] if "mesh" in n and "skin" in n)
    fox.setdefault("extensions", {})["EXT_mesh_gpu_instancing"] = {
        "attributes": {"TRANSLATION": len(gltf["accessors"]) - 1}
    }
    used = set(gltf.get("extensionsUsed", []))
    used.add("EXT_mesh_gpu_instancing")
    gltf["extensionsUsed"] = sorted(used)

    dst.write_text(json.dumps(gltf, separators=(",", ":")))
    print(f"{dst}: {count} instances ({side}x{side}, step {step})")


if __name__ == "__main__":
    main()
