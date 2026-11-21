"""5,000-scenario diagnostic sandbox for GPT-6.2 Astra weight-space programming."""

import json
import math
import random
from pathlib import Path

MODES = ("ENGAGE", "KITE", "LOOT", "ROTATE")


def tanh(x: float) -> float:
    return math.tanh(max(-10.0, min(10.0, x)))


def generate_5000_scenarios(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    scenarios = []

    # Feature order:
    # [0:hp, 1:en, 2:out_zone, 3:zone_urg, 4:en_vis, 5:en_close,
    #  6:hp_adv, 7:tri_adv, 8:loot, 9:drop, 10:debuff, 11:sound]
    specs = [
        ("01_storm_outside_critical", (3,), lambda r: [
            r.uniform(0.15, 0.85), r.uniform(0.2, 1.0), 1.0, r.uniform(0.75, 1.0),
            r.choice([0.0, 1.0]), r.uniform(0.2, 0.8), r.uniform(-0.3, 0.3), 0.0,
            r.choice([0.0, 1.0]), 0.0, 0.0, 0.0
        ]),
        ("02_zone_edge_closing_fast", (3,), lambda r: [
            r.uniform(0.4, 0.95), r.uniform(0.3, 1.0), 0.0, r.uniform(0.85, 1.0),
            0.0, 0.0, 0.0, 0.0,
            r.choice([0.0, 1.0]), 0.0, 0.0, r.uniform(0.0, 0.4)
        ]),
        ("03_laser_vs_mortar_cookoff", (0,), lambda r: [
            r.uniform(0.7, 1.0), r.uniform(0.6, 1.0), 0.0, r.uniform(0.0, 0.45),
            1.0, r.uniform(0.3, 0.85), r.uniform(0.05, 0.4), 0.6,
            0.0, 0.0, 0.0, 0.0
        ]),
        ("04_mortar_vs_lightning_push", (0,), lambda r: [
            r.uniform(0.7, 1.0), r.uniform(0.6, 1.0), 0.0, r.uniform(0.0, 0.45),
            1.0, r.uniform(0.4, 0.75), r.uniform(0.05, 0.4), 0.6,
            0.0, 0.0, 0.0, 0.0
        ]),
        ("05_lightning_vs_laser_short", (0,), lambda r: [
            r.uniform(0.7, 1.0), r.uniform(0.65, 1.0), 0.0, r.uniform(0.0, 0.45),
            1.0, r.uniform(0.7, 1.0), r.uniform(0.0, 0.35), 0.6,
            0.0, 0.0, 0.0, 0.0
        ]),
        ("06_finish_low_hp_enemy", (0,), lambda r: [
            r.uniform(0.6, 1.0), r.uniform(0.4, 1.0), 0.0, r.uniform(0.0, 0.5),
            1.0, r.uniform(0.5, 0.95), r.uniform(0.45, 0.85), r.uniform(-0.2, 0.6),
            0.0, 0.0, 0.0, 0.0
        ]),
        ("07_countered_by_enemy_weapon", (1,), lambda r: [
            r.uniform(0.35, 0.75), r.uniform(0.4, 1.0), 0.0, r.uniform(0.0, 0.5),
            1.0, r.uniform(0.55, 1.0), r.uniform(-0.4, 0.0), -0.6,
            0.0, 0.0, 0.0, 0.0
        ]),
        ("08_enemy_charging_laser_close", (1,), lambda r: [
            r.uniform(0.3, 0.75), r.uniform(0.5, 1.0), 0.0, r.uniform(0.0, 0.5),
            1.0, r.uniform(0.6, 1.0), r.uniform(-0.3, 0.1), -0.4,
            0.0, 0.0, 0.0, 0.0
        ]),
        ("09_hit_by_short_or_concuss", (1,), lambda r: [
            r.uniform(0.25, 0.8), r.uniform(0.1, 0.6), 0.0, r.uniform(0.0, 0.5),
            1.0, r.uniform(0.5, 1.0), r.uniform(-0.4, 0.1), r.uniform(-0.6, 0.2),
            0.0, 0.0, 1.0, 0.0
        ]),
        ("10_out_of_energy_close_fight", (1,), lambda r: [
            r.uniform(0.3, 0.75), r.uniform(0.0, 0.15), 0.0, r.uniform(0.0, 0.5),
            1.0, r.uniform(0.6, 1.0), r.uniform(-0.3, 0.1), r.uniform(-0.6, 0.0),
            0.0, 0.0, 0.0, 0.0
        ]),
        ("11_low_hp_need_5s_regen", (1, 3), lambda r: [
            r.uniform(0.12, 0.35), r.uniform(0.3, 1.0), 0.0, r.uniform(0.1, 0.6),
            r.choice([0.0, 1.0]), r.uniform(0.2, 0.7), r.uniform(-0.6, -0.2), r.uniform(-0.6, 0.0),
            0.0, 0.0, 0.0, r.uniform(0.0, 0.8)
        ]),
        ("12_safe_cube_or_weapon_loot", (2,), lambda r: [
            r.uniform(0.55, 1.0), r.uniform(0.4, 1.0), 0.0, r.uniform(0.0, 0.45),
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, r.uniform(0.0, 0.25)
        ]),
        ("13_supply_drop_uncontested", (2,), lambda r: [
            r.uniform(0.65, 1.0), r.uniform(0.5, 1.0), 0.0, r.uniform(0.0, 0.5),
            0.0, 0.0, 0.0, 0.0,
            0.0, r.uniform(0.6, 1.0), 0.0, r.uniform(0.0, 0.3)
        ]),
        ("14_greedy_loot_trap_near_enemy", (1,), lambda r: [
            r.uniform(0.35, 0.7), r.uniform(0.4, 1.0), 0.0, r.uniform(0.0, 0.5),
            1.0, r.uniform(0.65, 1.0), r.uniform(-0.35, 0.0), r.uniform(-0.6, 0.0),
            1.0, 0.0, 0.0, 0.0
        ]),
        ("15_hurt_hearing_near_gunfire", (1, 3), lambda r: [
            r.uniform(0.18, 0.42), r.uniform(0.3, 1.0), 0.0, r.uniform(0.2, 0.7),
            0.0, 0.0, 0.0, 0.0,
            r.choice([0.0, 1.0]), 0.0, 0.0, r.uniform(0.75, 1.0)
        ]),
        ("16_quiet_safe_zone_rotation", (3,), lambda r: [
            r.uniform(0.5, 1.0), r.uniform(0.5, 1.0), 0.0, r.uniform(0.55, 0.8),
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]),
    ]

    for i in range(5000):
        name, valid, fn = specs[i % len(specs)]
        scenarios.append({
            "arch": name,
            "valid": valid,
            "features": [round(v, 3) for v in fn(rng)],
        })
    return scenarios


def run_diagnostic() -> float:
    w_path = Path(__file__).with_name("weights.json")
    data = json.loads(w_path.read_text(encoding="utf-8"))
    w1, b1, w2, b2 = data["W1"], data["B1"], data["W2"], data["B2"]

    scenarios = generate_5000_scenarios()
    total_pass = 0
    stats = {}
    sample_failures = {}

    for sc in scenarios:
        x = sc["features"]
        hidden = [round(tanh(sum(w * xi for w, xi in zip(row, x)) + b), 3) for row, b in zip(w1, b1)]
        scores = [round(sum(w * hi for w, hi in zip(row, hidden)) + b, 3) for row, b in zip(w2, b2)]
        pred = max(range(4), key=lambda i: scores[i])
        ok = pred in sc["valid"]

        rec = stats.setdefault(sc["arch"], [0, 0])
        rec[1] += 1
        if ok:
            total_pass += 1
            rec[0] += 1
        elif sc["arch"] not in sample_failures:
            sample_failures[sc["arch"]] = {
                "features": x,
                "hidden": hidden,
                "scores": dict(zip(MODES, scores)),
                "picked": MODES[pred],
                "expected": [MODES[m] for m in sc["valid"]],
            }

    acc = 100.0 * total_pass / len(scenarios)
    print(f"OVERALL SCORE: {total_pass}/5000 ({acc:.2f}%)\n")
    print("BREAKDOWN BY SCENARIO ARCHETYPE:")
    for arch, (ok_cnt, tot_cnt) in sorted(stats.items()):
        print(f"  {arch:<32}: {ok_cnt:>4}/{tot_cnt:<4} ({100.0 * ok_cnt / tot_cnt:5.1f}%)")

    if sample_failures:
        print("\nSAMPLE FAILURES FOR GPT-6.2 ASTRA TO INSPECT:")
        for arch, fail in sorted(sample_failures.items()):
            print(f"\n--- {arch} ---")
            print(f"  Inputs  : {fail['features']}")
            print(f"  Hidden  : {fail['hidden']}")
            print(f"  Scores  : {fail['scores']}")
            print(f"  Picked  : {fail['picked']} | Expected: {fail['expected']}")
    return acc


if __name__ == "__main__":
    run_diagnostic()
