#!/usr/bin/env python3
"""Generates Models/GlassVaseFlowers/glTF/GlassVaseFlowersCrowd.gltf: the stock
GlassVaseFlowers replicated with EXT_mesh_gpu_instancing into an NxN grid (default 8x8 = 64
clusters) — a dense field of transparent glass vases for stress-testing order-independent
transparency (lots of overlapping alpha-blended surfaces in depth). The file still references
the stock GlassVaseFlowers.bin/textures and only adds an embedded data-URI buffer of instance
translations, so no extra binaries are needed.

The model's four scene nodes (flowers x2, the alpha-blend glass vase, the transmission vase) are
flat top-level nodes, several carrying a 90-degree X rotation. EXT_mesh_gpu_instancing applies
the instance transform in the node's LOCAL frame (worldMatrix = nodeMatrix * instanceMatrix), so a
world-space floor grid offset G must be pre-rotated by the node's inverse rotation:
instanceTranslation = R_node^-1 * G. Then every node shifts by the SAME world offset per grid cell,
so each cluster keeps the original two-vase arrangement.

Usage: python3 tools/gen_glass_vase_crowd.py [--side 8] [--step 0.25]
"""

import argparse
import base64
import json
import struct
from pathlib import Path


def quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q = [x, y, z, w] (glTF order)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=8, help="grid side (side^2 clusters)")
    parser.add_argument("--step", type=float, default=0.25, help="grid step, model units (metres)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = root / "Models" / "GlassVaseFlowers" / "glTF" / "GlassVaseFlowers.gltf"
    dst = root / "Models" / "GlassVaseFlowers" / "glTF" / "GlassVaseFlowersCrowd.gltf"
    gltf = json.loads(src.read_text())

    side, step = args.side, args.step
    count = side * side

    # World-space floor grid offsets, centered at the origin (y unchanged: vases keep their height).
    half = (side - 1) * 0.5
    grid = []
    for k in range(count):
        gx = (k % side - half) * step
        gz = (k // side - half) * step
        grid.append((gx, 0.0, gz))

    # This renderer batches multi-instance draws per mesh, so a mesh shared by several instanced
    # nodes desyncs the run length ("instance count N != node run length k*N"). Give each instanced
    # node a unique mesh: the second+ user of a mesh gets a shallow copy (same primitives/accessors,
    # no vertex data duplicated).
    mesh_nodes = [n for n in gltf["nodes"] if "mesh" in n]
    seen_meshes = set()
    for node in mesh_nodes:
        m = node["mesh"]
        if m in seen_meshes:
            gltf["meshes"].append(dict(gltf["meshes"][m]))
            node["mesh"] = len(gltf["meshes"]) - 1
        seen_meshes.add(node["mesh"])

    # One instance-translation accessor per node: the world grid pre-rotated into the node's local
    # frame (R_node^-1 * G). Nodes with the same rotation could share, but a per-node accessor keeps
    # this simple and the buffers are tiny.
    for node in mesh_nodes:
        rot = tuple(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
        inv = quat_conj(rot)

        data = bytearray()
        mn = [float("inf")] * 3
        mx = [float("-inf")] * 3
        for g in grid:
            t = quat_rotate(inv, g)
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
        node.setdefault("extensions", {})["EXT_mesh_gpu_instancing"] = {
            "attributes": {"TRANSLATION": len(gltf["accessors"]) - 1}
        }

    used = set(gltf.get("extensionsUsed", []))
    used.add("EXT_mesh_gpu_instancing")
    gltf["extensionsUsed"] = sorted(used)

    dst.write_text(json.dumps(gltf, separators=(",", ":")))
    print(f"{dst}: {count} clusters ({side}x{side}, step {step}), {len(mesh_nodes)} instanced nodes")


if __name__ == "__main__":
    main()
