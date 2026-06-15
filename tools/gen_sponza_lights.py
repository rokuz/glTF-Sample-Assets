#!/usr/bin/env python3
"""Generate a many-lights stress-test scene from Sponza.

Produces a glTF with a single Sponza and N lights (KHR_lights_punctual) scattered across its
interior volume. Half the lights are point lights, half are spot lights, each with a small emissive
marker in the light's color so positions are visible: a sphere for point lights, a cone (apex at the
light, opening along the beam) for spot lights. The markers read as "unlit" in renderers without
KHR_materials_unlit (base color black, only emissive contributes).

The markers are drawn with EXT_mesh_gpu_instancing (to exercise instancing on Sponza). Instances of
one node share a material, so colors are quantized to an 8-hue palette: a light's color is its
bucket color, and each (type, bucket) becomes one instanced node (TRANSLATION, plus ROTATION for
spot cones).

Output is written next to Sponza.gltf and reuses Sponza.bin + textures via the same relative URIs;
only marker geometry + instance transforms are embedded (base64 data-URI buffers). Deterministic for
a given --seed, so screenshot-test references are stable.

Usage:
  tools/gen_sponza_lights.py --num-lights 256 \
      --sponza Models/Sponza/glTF/Sponza.gltf \
      --out    Models/Sponza/glTF/SponzaManyLights.gltf
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import json
import math
import random
import struct
from pathlib import Path

NUM_BUCKETS = 8  # hue buckets (EXT_mesh_gpu_instancing shares one material per instanced node).


def make_uv_sphere(radius, rings, sectors):
    positions, normals, indices = [], [], []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        sp, cp = math.sin(phi), math.cos(phi)
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors
            n = (sp * math.cos(theta), cp, sp * math.sin(theta))
            normals.append(n)
            positions.append((n[0] * radius, n[1] * radius, n[2] * radius))
    row = sectors + 1
    for r in range(rings):
        for s in range(sectors):
            a, b = r * row + s, r * row + s + row
            indices += [a, b, a + 1, a + 1, b, b + 1]
    return positions, normals, indices


def make_cone(height, half_angle, segments):
    """Cone with apex at the origin opening along -Z (glTF spot lights aim down -Z)."""
    r = height * math.tan(half_angle)
    positions, normals = [(0.0, 0.0, 0.0)], [(0.0, 0.0, 1.0)]  # apex
    for s in range(segments):
        theta = 2.0 * math.pi * s / segments
        x, y = r * math.cos(theta), r * math.sin(theta)
        positions.append((x, y, -height))
        nz = r * math.sin(half_angle)
        ln = math.sqrt(x * x + y * y + nz * nz) or 1.0
        normals.append((x / ln, y / ln, nz / ln))
    indices = []
    for s in range(segments):
        indices += [0, 1 + (s + 1) % segments, 1 + s]
    center = len(positions)
    positions.append((0.0, 0.0, -height))
    normals.append((0.0, 0.0, -1.0))
    for s in range(segments):
        indices += [center, 1 + s, 1 + (s + 1) % segments]
    return positions, normals, indices


def _data_uri(blob):
    return "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()


def add_marker_geometry(d, meshes):
    """Pack meshes [(pos, nrm, idx), ...] into one new buffer; return [(acc_pos, acc_nrm, acc_idx)]."""
    n_buf, blob, out = len(d["buffers"]), bytearray(), []

    def pad4():
        while len(blob) % 4:
            blob.append(0)

    for positions, normals, indices in meshes:
        pos = b"".join(struct.pack("<3f", *p) for p in positions)
        nrm = b"".join(struct.pack("<3f", *n) for n in normals)
        idx = b"".join(struct.pack("<H", i) for i in indices)
        pad4(); o_pos = len(blob); blob += pos
        pad4(); o_nrm = len(blob); blob += nrm
        pad4(); o_idx = len(blob); blob += idx
        pmin = [min(p[j] for p in positions) for j in range(3)]
        pmax = [max(p[j] for p in positions) for j in range(3)]
        bv = len(d["bufferViews"])
        d["bufferViews"] += [
            {"buffer": n_buf, "byteOffset": o_pos, "byteLength": len(pos), "target": 34962},
            {"buffer": n_buf, "byteOffset": o_nrm, "byteLength": len(nrm), "target": 34962},
            {"buffer": n_buf, "byteOffset": o_idx, "byteLength": len(idx), "target": 34963},
        ]
        ac = len(d["accessors"])
        d["accessors"] += [
            {"bufferView": bv, "componentType": 5126, "count": len(positions),
             "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": bv + 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": bv + 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ]
        out.append((ac, ac + 1, ac + 2))
    pad4()
    d["buffers"].append({"uri": _data_uri(blob), "byteLength": len(blob)})
    return out


def add_instance_streams(d, streams):
    """streams: [('VEC3'|'VEC4', [tuple, ...]), ...]. Pack into one buffer; return accessor indices."""
    n_buf, blob, accs = len(d["buffers"]), bytearray(), []

    def pad4():
        while len(blob) % 4:
            blob.append(0)

    for typ, data in streams:
        comps = 3 if typ == "VEC3" else 4
        pad4(); off = len(blob)
        for t in data:
            blob += struct.pack("<%df" % comps, *t)
        bv = len(d["bufferViews"])
        d["bufferViews"].append({"buffer": n_buf, "byteOffset": off, "byteLength": len(data) * comps * 4})
        accs.append(len(d["accessors"]))
        d["accessors"].append({"bufferView": bv, "componentType": 5126, "count": len(data), "type": typ})
    pad4()
    d["buffers"].append({"uri": _data_uri(blob), "byteLength": len(blob)})
    return accs


def add_skin_attributes(d, vert_counts):
    """For each marker (by vertex count) emit JOINTS_0 (all bone 0) + WEIGHTS_0 (all 1.0), so the mesh
    binds 100% to a single root bone. Returns [(joints_acc, weights_acc), ...]."""
    n_buf, blob, out = len(d["buffers"]), bytearray(), []

    def pad4():
        while len(blob) % 4:
            blob.append(0)

    for count in vert_counts:
        pad4(); o_j = len(blob)
        blob += b"".join(struct.pack("<4B", 0, 0, 0, 0) for _ in range(count))  # bone 0
        pad4(); o_w = len(blob)
        blob += b"".join(struct.pack("<4f", 1.0, 0.0, 0.0, 0.0) for _ in range(count))  # full weight
        bv = len(d["bufferViews"])
        d["bufferViews"] += [
            {"buffer": n_buf, "byteOffset": o_j, "byteLength": count * 4, "target": 34962},
            {"buffer": n_buf, "byteOffset": o_w, "byteLength": count * 16, "target": 34962},
        ]
        ac = len(d["accessors"])
        d["accessors"] += [
            {"bufferView": bv, "componentType": 5121, "count": count, "type": "VEC4"},      # u8  JOINTS_0
            {"bufferView": bv + 1, "componentType": 5126, "count": count, "type": "VEC4"},  # f32 WEIGHTS_0
        ]
        out.append((ac, ac + 1))
    pad4()
    d["buffers"].append({"uri": _data_uri(blob), "byteLength": len(blob)})
    return out


def add_identity_ibm(d):
    """One identity MAT4 inverse-bind matrix (single bone bound at the origin); shared by all skins."""
    blob = struct.pack("<16f", 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    n_buf = len(d["buffers"])
    bv = len(d["bufferViews"])
    d["bufferViews"].append({"buffer": n_buf, "byteOffset": 0, "byteLength": len(blob)})
    ac = len(d["accessors"])
    d["accessors"].append({"bufferView": bv, "componentType": 5126, "count": 1, "type": "MAT4"})
    d["buffers"].append({"uri": _data_uri(blob), "byteLength": len(blob)})
    return ac


def add_animation_streams(d, period, nseg, radius):
    """Looping keyframes (one period): bone translation around a small XZ circle (point spheres) and
    bone rotation a full turn about +Y (spot cones — about the apex, so the light position is fixed).
    Returns (time_acc, sphere_translation_acc, cone_rotation_acc)."""
    times = [period * k / nseg for k in range(nseg + 1)]
    orbit = [(radius * math.cos(2.0 * math.pi * k / nseg), 0.0, radius * math.sin(2.0 * math.pi * k / nseg))
             for k in range(nseg + 1)]
    spin = []
    for k in range(nseg + 1):
        a = 2.0 * math.pi * k / nseg
        spin.append((0.0, math.sin(a * 0.5), 0.0, math.cos(a * 0.5)))  # quaternion about +Y

    blob = bytearray()

    def pad4():
        while len(blob) % 4:
            blob.append(0)

    n_buf = len(d["buffers"])
    pad4(); o_t = len(blob)
    blob += b"".join(struct.pack("<f", t) for t in times)
    pad4(); o_s = len(blob)
    blob += b"".join(struct.pack("<3f", *v) for v in orbit)
    pad4(); o_c = len(blob)
    blob += b"".join(struct.pack("<4f", *q) for q in spin)
    bv = len(d["bufferViews"])
    d["bufferViews"] += [
        {"buffer": n_buf, "byteOffset": o_t, "byteLength": 4 * len(times)},
        {"buffer": n_buf, "byteOffset": o_s, "byteLength": 12 * len(orbit)},
        {"buffer": n_buf, "byteOffset": o_c, "byteLength": 16 * len(spin)},
    ]
    ac = len(d["accessors"])
    d["accessors"] += [
        {"bufferView": bv, "componentType": 5126, "count": len(times), "type": "SCALAR",
         "min": [0.0], "max": [period]},
        {"bufferView": bv + 1, "componentType": 5126, "count": len(orbit), "type": "VEC3"},
        {"bufferView": bv + 2, "componentType": 5126, "count": len(spin), "type": "VEC4"},
    ]
    pad4()
    d["buffers"].append({"uri": _data_uri(blob), "byteLength": len(blob)})
    return ac, ac + 1, ac + 2


def quat_from_neg_z(direction):
    """Quaternion [x, y, z, w] rotating local -Z onto the unit `direction`."""
    dot = -direction[2]
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    ax, ay, az = -direction[1], direction[0], 0.0  # cross((0,0,-1), direction)
    s = math.sqrt((1.0 + dot) * 2.0)
    q = [ax / s, ay / s, az / s, s * 0.5]
    ln = math.sqrt(sum(c * c for c in q)) or 1.0
    return [c / ln for c in q]


def rand_dir(rng):
    while True:
        v = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 0.2), rng.uniform(-1.0, 1.0))
        ln = math.sqrt(sum(c * c for c in v))
        if ln > 1e-3:
            return (v[0] / ln, v[1] / ln, v[2] / ln)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sponza", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--num-lights", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--animate", action="store_true",
                    help="skin each marker to a single root bone and animate it: point spheres orbit a "
                         "small radius, spot cones spin. Per-instance phase desyncs them with --anim-cycle.")
    ap.add_argument("--orbit-radius", type=float, default=0.3, help="point-sphere orbit radius (animate).")
    ap.add_argument("--anim-period", type=float, default=4.0, help="animation loop period in seconds.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    d = json.loads(Path(args.sponza).read_text())

    root = d["nodes"][d["scenes"][d.get("scene", 0)]["nodes"][0]]
    scale = (root.get("scale") or [1, 1, 1])[0]
    rmin, rmax = [1e30] * 3, [-1e30] * 3
    for mesh in d["meshes"]:
        for prim in mesh["primitives"]:
            acc = d["accessors"][prim["attributes"]["POSITION"]]
            for j in range(3):
                rmin[j], rmax[j] = min(rmin[j], acc["min"][j]), max(rmax[j], acc["max"][j])
    wmin = [rmin[j] * scale for j in range(3)]
    wmax = [rmax[j] * scale for j in range(3)]

    outer, inner = math.radians(30.0), math.radians(20.0)
    sphere = make_uv_sphere(0.04, 10, 14)
    cone = make_cone(0.15, outer, 16)
    (sphere_acc, cone_acc) = add_marker_geometry(d, [sphere, cone])
    palette = [colorsys.hsv_to_rgb(k / NUM_BUCKETS, 0.9, 1.0) for k in range(NUM_BUCKETS)]

    # Animated variant: skin each marker to one root bone (skin_attr holds its JOINTS_0/WEIGHTS_0;
    # ibm_acc the shared identity inverse-bind). Each instanced marker node gets its own bone + skin,
    # mirroring FoxCrowd (a skinned mesh under EXT_mesh_gpu_instancing animates per instance).
    skin_attr = ibm_acc = None
    if args.animate:
        skin_attr = add_skin_attributes(d, [len(sphere[0]), len(cone[0])])  # [(j,w)_sphere, (j,w)_cone]
        ibm_acc = add_identity_ibm(d)
        d.setdefault("skins", [])

    for ext in ("KHR_lights_punctual", "EXT_mesh_gpu_instancing"):
        d.setdefault("extensionsUsed", [])
        if ext not in d["extensionsUsed"]:
            d["extensionsUsed"].append(ext)

    scene_nodes = list(d["scenes"][d.get("scene", 0)]["nodes"])
    lights = []
    lo = [wmin[0] + 0.5, wmin[1] + 1.0, wmin[2] + 0.5]
    hi = [wmax[0] - 0.5, wmax[1] - 1.5, wmax[2] - 0.5]
    buckets = {}  # (is_spot, bucket) -> [(translation, rotation-quat-or-None)]

    # Each light is its own node (lights cannot be instanced); markers are instanced separately.
    for i in range(args.num_lights):
        is_spot = (i % 2 == 1)
        bucket = (i // 2) % NUM_BUCKETS
        cr, cg, cb = palette[bucket]
        color = [round(cr, 4), round(cg, 4), round(cb, 4)]
        pos = [round(rng.uniform(lo[j], hi[j]), 4) for j in range(3)]
        intensity = round(rng.uniform(2.0, 6.0), 3)
        rng_range = round(rng.uniform(3.0, 6.0), 3)

        node = {"translation": pos, "name": f"Light{i}",
                "extensions": {"KHR_lights_punctual": {"light": i}}}
        if is_spot:
            quat = [round(c, 5) for c in quat_from_neg_z(rand_dir(rng))]
            node["rotation"] = quat
            lights.append({"type": "spot", "color": color, "intensity": intensity, "range": rng_range,
                           "spot": {"innerConeAngle": round(inner, 5), "outerConeAngle": round(outer, 5)},
                           "name": f"L{i}"})
            buckets.setdefault((True, bucket), []).append((pos, quat))
        else:
            lights.append({"type": "point", "color": color, "intensity": intensity,
                           "range": rng_range, "name": f"L{i}"})
            buckets.setdefault((False, bucket), []).append((pos, None))
        scene_nodes.append(len(d["nodes"]))
        d["nodes"].append(node)

    keys = sorted(buckets.keys())
    streams, stream_map = [], {}
    for key in keys:
        insts = buckets[key]
        ti = len(streams); streams.append(("VEC3", [p for p, _ in insts]))
        ri = None
        if key[0]:
            ri = len(streams); streams.append(("VEC4", [q for _, q in insts]))
        stream_map[key] = (ti, ri)
    accs = add_instance_streams(d, streams)

    n_mat, n_mesh = len(d["materials"]), len(d["meshes"])
    anim_joints = []  # (joint_node_index, is_spot) for the animated variant.
    for key in keys:
        is_spot, bucket = key
        cr, cg, cb = palette[bucket]
        d["materials"].append({
            "name": f"Marker_{'spot' if is_spot else 'point'}_{bucket}",
            "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1],
                                     "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "emissiveFactor": [round(cr, 4), round(cg, 4), round(cb, 4)], "doubleSided": True,
        })
        geom = cone_acc if is_spot else sphere_acc
        prim_attrs = {"POSITION": geom[0], "NORMAL": geom[1]}
        if args.animate:
            j_acc, w_acc = skin_attr[1 if is_spot else 0]
            prim_attrs["JOINTS_0"] = j_acc
            prim_attrs["WEIGHTS_0"] = w_acc
        d["meshes"].append({"primitives": [{
            "attributes": prim_attrs, "indices": geom[2], "material": n_mat,
        }]})
        ti, ri = stream_map[key]
        attrs = {"TRANSLATION": accs[ti]}
        if ri is not None:
            attrs["ROTATION"] = accs[ri]
        node = {"mesh": n_mesh, "name": f"Markers_{'spot' if is_spot else 'point'}_{bucket}",
                "extensions": {"EXT_mesh_gpu_instancing": {"attributes": attrs}}}
        if args.animate:
            # One root bone (at the marker origin) per instanced node; the renderer animates it per
            # instance. The instance TRANSLATION/ROTATION places each marker after skinning, so the bone
            # motion is local: spheres orbit their light, cones spin about their apex (the light).
            joint = len(d["nodes"])
            d["nodes"].append({"translation": [0.0, 0.0, 0.0],
                               "name": f"Bone_{'spot' if is_spot else 'point'}_{bucket}"})
            scene_nodes.append(joint)
            node["skin"] = len(d["skins"])
            d["skins"].append({"inverseBindMatrices": ibm_acc, "joints": [joint], "skeleton": joint})
            anim_joints.append((joint, is_spot))
        scene_nodes.append(len(d["nodes"]))
        d["nodes"].append(node)
        n_mat += 1
        n_mesh += 1

    if args.animate:
        t_acc, orbit_acc, spin_acc = add_animation_streams(d, args.anim_period, 32, args.orbit_radius)
        samplers = [{"input": t_acc, "output": orbit_acc, "interpolation": "LINEAR"},
                    {"input": t_acc, "output": spin_acc, "interpolation": "LINEAR"}]
        channels = [{"sampler": 1 if is_spot else 0,
                     "target": {"node": j, "path": "rotation" if is_spot else "translation"}}
                    for (j, is_spot) in anim_joints]
        d.setdefault("animations", []).append({"name": "LightDance", "samplers": samplers,
                                               "channels": channels})

    d["extensions"] = d.get("extensions", {})
    d["extensions"]["KHR_lights_punctual"] = {"lights": lights}
    d["scenes"][d.get("scene", 0)]["nodes"] = scene_nodes

    Path(args.out).write_text(json.dumps(d))
    n_spot = args.num_lights // 2
    print(f"wrote {args.out}: Sponza + {args.num_lights} lights "
          f"({args.num_lights - n_spot} point/sphere, {n_spot} spot/cone), "
          f"{len(keys)} instanced marker nodes"
          + (f", animated (spheres orbit r={args.orbit_radius}, cones spin, {args.anim_period}s loop)"
             if args.animate else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
