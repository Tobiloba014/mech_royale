"""Battle Royale survival bot for mechfight/1.

This is a compact neuro-symbolic bot: a small MLP chooses ENGAGE, KITE,
LOOT, or ROTATE, while deterministic helpers handle movement and aiming.
Place weights.json beside this file. The mechfight runtime supplies the
`mechfight` package and calls Bot through mf.run(Bot).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mechfight as mf


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_diff(target: float, current: float) -> float:
    return (target - current + math.pi) % (2.0 * math.pi) - math.pi


def distance3(a, b) -> float:
    return math.dist(tuple(a), tuple(b))


def unit2(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    return (x / length, y / length) if length > 1e-9 else (0.0, 0.0)


def tanh(value: float) -> float:
    return math.tanh(clamp(value, -10.0, 10.0))


class TacticalBrain:
    """12 -> 8 -> 4 MLP; forward_with_hidden supports league telemetry."""

    def __init__(self, weights_path: Path):
        data = json.loads(weights_path.read_text(encoding="utf-8"))
        self.W1 = data["W1"]
        self.B1 = data["B1"]
        self.W2 = data["W2"]
        self.B2 = data["B2"]
        if len(self.W1) != 8 or any(len(row) != 12 for row in self.W1):
            raise ValueError("weights.json W1 must have shape 8 x 12")
        if len(self.W2) != 4 or any(len(row) != 8 for row in self.W2):
            raise ValueError("weights.json W2 must have shape 4 x 8")

    def forward_with_hidden(self, features: list[float]):
        if len(features) != 12:
            raise ValueError(f"expected 12 features, got {len(features)}")
        hidden = [
            tanh(sum(w * x for w, x in zip(row, features)) + bias)
            for row, bias in zip(self.W1, self.B1)
        ]
        scores = [
            sum(w * h for w, h in zip(row, hidden)) + bias
            for row, bias in zip(self.W2, self.B2)
        ]
        return scores, hidden

    def forward(self, features: list[float]) -> list[float]:
        scores, _ = self.forward_with_hidden(features)
        return scores


class Bot:
    name = "br-survival-v5"
    MODE_NAMES = ("ENGAGE", "KITE", "LOOT", "ROTATE")

    def __init__(self, hello: dict[str, Any]):
        self.cfg = hello.get("config", {})
        self.slot = int(hello.get("slot", 0))
        loot_cfg = self.cfg.get("loot", {})
        weapon_cfg = self.cfg.get("weapons", {})
        self.pickup_radius = float(loot_cfg.get("pickup_radius", 4.0))
        self.mortar_g = float(weapon_cfg.get("mortar", {}).get("gravity", 22.0))
        self.upgrade_defs = {u.get("id"): u for u in self.cfg.get("upgrades", [])}
        self.brain = TacticalBrain(Path(__file__).with_name("weights.json"))

        self.current_mode = 3
        self.checkpoint = None
        self.detour_until = 0.0
        self.detour_side = 1
        self.stuck_ticks = 0
        self.last_damage_t = -10.0
        self.shorted_until = 0.0
        self.concussed_until = 0.0
        self.alert_yaw = None
        self.alert_until = 0.0
        self.scan_yaw = (self.slot * 0.4) % (2.0 * math.pi) - math.pi
        self.sound_memory: list[tuple[float, float, float]] = []
        self.starter_target = None

        self.obstacles = []
        try:
            map_dir = Path(hello["map"]["dir"])
            world = json.loads((map_dir / "v2" / "world.json").read_text(encoding="utf-8"))
            for prism in world.get("prisms", []):
                ring = prism.get("ring", [])
                xs, ys = ring[::2], ring[1::2]
                if xs and ys:
                    self.obstacles.append((
                        min(xs), max(xs), min(ys), max(ys),
                        float(prism.get("base", 0.0)), float(prism.get("top", 0.0)),
                    ))
        except (KeyError, OSError, ValueError, TypeError):
            # The bot can still run if a map description is unavailable.
            self.obstacles = []

    def weapon_advantage(self, my_weapons, enemy_weapons) -> float:
        mine = {w.get("kind") for w in my_weapons if isinstance(w, dict)}
        theirs = {w.get("kind") if isinstance(w, dict) else w for w in enemy_weapons}
        counters = {("laser", "mortar"), ("mortar", "lightning"), ("lightning", "laser")}
        score = sum(0.6 if (m, e) in counters else -0.6 if (e, m) in counters else 0.0
                    for m in mine for e in theirs)
        return clamp(score, -1.0, 1.0)

    def nearest_enemy(self, obs):
        me = obs.get("self", {})
        enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        return min(enemies, key=lambda m: distance3(me.get("pos", (0, 0, 0)), m.get("pos", (0, 0, 0))), default=None)

    def remember_sounds(self, obs):
        now = float(obs.get("t", 0.0))
        packets = obs.get("sounds")
        if packets is None:
            packets = [e for e in obs.get("events", []) if e.get("type") in ("heard", "sound")]
        weights = {"near": 0.70, "mid": 0.45, "far": 0.12}
        fresh = [(now - max(0.0, p.get("age", 0.0)), p["bearing"],
                  weights.get(p.get("dist"), p.get("weight", 1.0)))
                 for p in packets if p.get("bearing") is not None and p.get("age", 0.0) < 1.5]
        # Refresh an old bearing on a new report. Two sounds in the same packet still count.
        self.sound_memory = [(t, b, w) for t, b, w in self.sound_memory
                             if now - t < 1.5 and all(abs(angle_diff(b, a)) > 0.22 for _, a, _ in fresh)]
        self.sound_memory.extend(fresh)

    def active_sounds(self, now):
        return [(b, w * (1.0 - (now - t) / 1.5)) for t, b, w in self.sound_memory
                if 0.0 <= now - t < 1.5]

    def sound_level(self, now):
        return min(1.0, sum(w for _, w in self.active_sounds(now)))

    def in_take_range(self, pos, item):
        px, py, pz = item["pos"]
        return math.hypot(pos[0] - px, pos[1] - py) <= self.pickup_radius and abs(pos[2] - pz) <= 4.0

    def is_in_safe_circle(self, pos, zone, margin=4.0):
        """Return whether a map position is safely inside the current circle."""
        center = zone.get("center")
        radius = zone.get("radius")
        if center is None or radius is None:
            return True
        return math.hypot(pos[0] - center[0], pos[1] - center[1]) <= max(0.0, radius - margin)

    def loot_target(self, obs):
        me = obs["self"]
        pos = me.get("pos", (0.0, 0.0, 0.0))
        stats = me.get("stats", {})
        default_speed = self.cfg.get("mech", {}).get("walk_speed", 11.7)
        speed = max(1.0, float(stats.get("walk_speed", default_speed)))
        weapons = me.get("weapons", [])
        held = {w.get("kind") for w in weapons if isinstance(w, dict)}
        hp_ratio = me.get("hp", 1.0) / max(1.0, me.get("max_hp", 1.0))
        candidates = []
        loot_nearby = 0.0
        drop_opp = 0.0
        zone = obs.get("zone", {})

        for cube in obs.get("cubes", []):
            if cube.get("pos") is None or not self.is_in_safe_circle(cube["pos"], zone):
                continue
            d = distance3(pos, cube.get("pos", pos))
            if d < 130.0:
                candidates.append((d / speed, cube.get("pos")))
                loot_nearby = max(loot_nearby, clamp(1.0 - d / 260.0, 0.0, 1.0))

        free_slot = len(weapons) < 2 or None in weapons
        quiet = self.sound_level(float(obs.get("t", 0.0))) < 0.4
        visible = any(m.get("kind") == "mech" for m in obs.get("mechs", []))
        seeking_first_weapon = held == {"gun"} and free_slot and me.get("in_zone", True)
        priority = []
        if free_slot:
            for item in obs.get("pickups", []):
                kind = item.get("kind")
                if item.get("pos") is None or not self.is_in_safe_circle(item["pos"], zone):
                    continue
                d = distance3(pos, item.get("pos", pos))
                if seeking_first_weapon and kind not in (None, "gun"):
                    in_range = self.in_take_range(pos, item)
                    near_weapon = 1.0 - d / 260.0 > 0.65
                    pursue = ((not visible and (quiet or (hp_ratio > 0.75 and near_weapon)))
                              or (visible and hp_ratio >= 0.55 and d <= 70.0))
                    if in_range or pursue:
                        priority.append((not in_range, d, item["id"], item))
                if d < 130.0 and kind not in (None, "gun") and kind not in held:
                    value = 2.0 if len(held) < 2 else 1.2
                    candidates.append((d / (speed * value), item.get("pos")))
                    loot_nearby = max(loot_nearby, clamp(1.0 - d / 260.0, 0.0, 1.0))

        if hp_ratio > 0.50 and held:
            for crate in obs.get("crates", []):
                if crate.get("pos") is None or not self.is_in_safe_circle(crate["pos"], zone):
                    continue
                d = distance3(pos, crate.get("pos", pos))
                if d < 100.0:
                    hp_left = crate.get("hp", 300.0) / 300.0
                    candidates.append(((d / speed + 3.0 * hp_left) / 1.5, crate.get("pos")))
                    loot_nearby = max(loot_nearby, 0.65)

            npc_cfg = self.cfg.get("npc", {"hp": 350.0, "cubes": 2})
            for npc in obs.get("mechs", []):
                if npc.get("kind") != "npc":
                    continue
                if npc.get("pos") is None or not self.is_in_safe_circle(npc["pos"], zone):
                    continue
                d = distance3(pos, npc.get("pos", pos))
                if d < 100.0:
                    hp_left = npc.get("hp", npc_cfg.get("hp", 350.0)) / max(1.0, npc_cfg.get("hp", 350.0))
                    reward = max(1.0, npc_cfg.get("cubes", 2))
                    candidates.append(((d / speed + 4.0 * hp_left) / reward, npc.get("pos")))
                    loot_nearby = max(loot_nearby, 0.65)

        for drop in obs.get("drops", []):
            if drop.get("pos") is None or not self.is_in_safe_circle(drop["pos"], zone):
                continue
            d = distance3(pos, drop.get("pos", pos))
            if d >= 180.0 or drop.get("lands_in", 999.0) >= 18.0:
                continue
            arrival = max(0.0, d - drop.get("radius", 8.0)) / speed
            remaining = max(0.0, drop.get("capture_s", 4.0) - drop.get("progress", 0.0))
            holder = drop.get("holder")
            if holder is not None and holder != me.get("id") and remaining <= arrival:
                continue
            capture = remaining if holder == me.get("id") else drop.get("capture_s", 4.0)
            cubes = self.cfg.get("drops", {}).get("cubes", 3.0)
            reward = cubes + 2.0 * (1.0 - hp_ratio)
            cost = (max(arrival, drop.get("lands_in", 0.0)) + capture) / max(1.0, reward)
            candidates.append((cost, drop.get("pos")))
            drop_opp = max(drop_opp, clamp(1.0 - d / 180.0, 0.0, 1.0))

        self.starter_target = min(priority, key=lambda p: p[:3])[3] if priority else None
        if self.starter_target is not None:
            pickup_signal = 1.0 if quiet and not visible else loot_nearby
            return self.starter_target["pos"], pickup_signal, drop_opp
        candidates = [(cost, p) for cost, p in candidates if p is not None]
        goal = min(candidates, key=lambda c: c[0])[1] if candidates else None
        return goal, loot_nearby, drop_opp

    def extract_features(self, obs, loot_signals=None) -> list[float]:
        """Build features in the exact order expected by Astra's weights."""
        me = obs["self"]
        now = float(obs.get("t", 0.0))
        x, y, _ = me.get("pos", (0.0, 0.0, 0.0))
        hp = me.get("hp", 1.0) / max(1.0, me.get("max_hp", 1.0))
        energy = me.get("energy", 0.0) / max(1.0, me.get("max_energy", 1.0))
        outside = 0.0 if me.get("in_zone", True) else 1.0

        zone = obs.get("zone", {})
        cx, cy = zone["center"]
        radius = max(1.0, zone["radius"])
        urgency = clamp((math.hypot(x - cx, y - cy) / radius) ** 2, 0.0, 1.0)

        enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        nearest = min(
            enemies,
            key=lambda m: distance3(me.get("pos", (0, 0, 0)), m.get("pos", (0, 0, 0))),
            default=None,
        )
        if nearest is not None:
            enemy_distance = distance3(me.get("pos", (0, 0, 0)), nearest.get("pos", (0, 0, 0)))
            enemy_close = clamp(1.0 - enemy_distance / 150.0, 0.0, 1.0)
            enemy_hp = nearest.get("hp", nearest.get("max_hp", 1.0)) / max(1.0, nearest.get("max_hp", 1.0))
            hp_adv = clamp(hp - enemy_hp, -1.0, 1.0)
            tri_adv = self.weapon_advantage(me.get("weapons", []), nearest.get("weapons", []))
        else:
            enemy_close = hp_adv = tri_adv = 0.0

        if loot_signals is None:
            _, loot_nearby, drop_opp = self.loot_target(obs)
        else:
            loot_nearby, drop_opp = loot_signals

        debuffed = float(now < self.shorted_until or now < self.concussed_until)
        sound_threat = self.sound_level(now)

        return [
            clamp(hp, 0.0, 1.0), clamp(energy, 0.0, 1.0), outside, urgency,
            float(nearest is not None), enemy_close, hp_adv, tri_adv,
            clamp(loot_nearby, 0.0, 1.0), clamp(drop_opp, 0.0, 1.0),
            debuffed, sound_threat,
        ]

    def choose_mode(self, scores) -> int:
        best = max(range(4), key=lambda i: scores[i])
        # A small hysteresis prevents rapid mode switching on near ties.
        if self.current_mode < 4 and best != self.current_mode:
            if scores[best] < scores[self.current_mode] + 0.22:
                best = self.current_mode
        self.current_mode = best
        return best

    def safe_edge_anchor(self, obs):
        zone = obs.get("zone", {})
        cx, cy = zone.get("next_center", zone.get("center", (0.0, 0.0)))
        radius = max(0.0, zone.get("next_radius", zone.get("radius", 0.0)))
        theta = self.slot * 2.39996 + float(obs.get("t", 0.0)) * 0.08
        ring = min(max(12.0, radius * 0.68), max(12.0, radius - 15.0))
        return [cx + ring * math.cos(theta), cy + ring * math.sin(theta)]

    def retreat_goal(self, obs, outside):
        x, y = obs["self"]["pos"][:2]
        zone = obs["zone"]
        cx, cy = zone["center"]
        sources = []
        for enemy in obs.get("mechs", []):
            if enemy.get("kind") != "mech":
                continue
            ex, ey = enemy["pos"][:2]
            d = math.hypot(ex - x, ey - y)
            if d > 1e-8:
                sources.append((math.atan2(ey - y, ex - x), 1.0 / (1.0 + d / 60.0)))
        sources.extend(self.active_sounds(float(obs.get("t", 0.0))))
        if not sources:
            return ((cx, cy) if outside else self.safe_edge_anchor(obs)), None

        tx = sum(w * math.cos(b) for b, w in sources)
        ty = sum(w * math.sin(b) for b, w in sources)
        if math.hypot(tx, ty) < 0.05 * sum(w for _, w in sources):
            bearing = max(sources, key=lambda s: s[1])[0]
            tx, ty = math.cos(bearing), math.sin(bearing)
        tx, ty = unit2(tx, ty)
        side = self.detour_side * (1 if self.slot % 2 == 0 else -1)
        dx, dy = unit2(-tx - 0.45 * side * ty, -ty + 0.45 * side * tx)
        if outside:
            ix, iy = unit2(cx - x, cy - y)
            dx, dy = unit2(2.0 * ix + dx, 2.0 * iy + dy)
        gx, gy = x + 32.0 * dx, y + 32.0 * dy
        if not outside and math.hypot(gx - cx, gy - cy) > zone["radius"]:
            ux, uy = unit2(gx - cx, gy - cy)
            gx, gy = cx + 0.99 * zone["radius"] * ux, cy + 0.99 * zone["radius"] * uy
        return (gx, gy), math.atan2(ty, tx)

    def projectile_dodge_vector(self, obs):
        me = obs["self"]
        mx, my, mz = me.get("pos", (0.0, 0.0, 0.0))
        dx = dy = 0.0
        urgent = False
        for proj in obs.get("projectiles", []):
            pos = proj.get("pos")
            vel = proj.get("vel", (0.0, 0.0, 0.0))
            if not pos or len(pos) < 3 or len(vel) < 3:
                continue
            px, py, pz = pos
            vx, vy, vz = vel
            speed2 = vx * vx + vy * vy + vz * vz
            if speed2 < 1.0:
                continue
            rx, ry, rz = mx - px, my - py, (mz + 4.5) - pz
            t = (rx * vx + ry * vy + rz * vz) / speed2
            if not 0.0 <= t <= 0.75:
                continue
            miss = math.hypot(mx - (px + vx * t), my - (py + vy * t))
            kind = proj.get("kind", "gun")
            if miss >= (9.0 if kind == "mortar" else 4.5):
                continue
            h = max(1.0, math.hypot(vx, vy))
            pxn, pyn = -vy / h, vx / h
            side = 1.0 if rx * pxn + ry * pyn >= 0.0 else -1.0
            dx += side * pxn * 1.5
            dy += side * pyn * 1.5
            urgent = urgent or kind == "mortar"
        return dx, dy, urgent

    def move(self, obs, goal):
        me = obs["self"]
        x, y, z = me.get("pos", (0.0, 0.0, 0.0))
        gx, gy = goal[0], goal[1]
        now = float(obs.get("t", 0.0))
        distance = math.hypot(gx - x, gy - y)
        if self.checkpoint is None:
            self.checkpoint = (now, x, y)
        old_t, old_x, old_y = self.checkpoint
        if now - old_t >= 0.8:
            if distance > 4.0 and math.hypot(x - old_x, y - old_y) < 1.8:
                self.detour_side *= -1
                self.detour_until = now + 1.6
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0
            self.checkpoint = (now, x, y)

        dodge_x, dodge_y, urgent = self.projectile_dodge_vector(obs)
        if distance < 1.5 and not urgent:
            return [0.0, 0.0], False, False
        heading = math.atan2(gy - y + dodge_y * 15.0, gx - x + dodge_x * 15.0)
        radius = me.get("stats", {}).get("radius", 2.5)
        height = me.get("stats", {}).get("height", 9.0)
        boxes = [(l - radius, r + radius, b - radius, top + radius)
                 for l, r, b, top, base, roof in self.obstacles
                 if roof > z + 1.2 and base < z + height
                 and l - 20 - radius <= x <= r + 20 + radius
                 and b - 20 - radius <= y <= top + 20 + radius]

        def clearance(dx, dy):
            for step in (2, 4, 8, 12, 16):
                xx, yy = x + step * dx, y + step * dy
                if any(l <= xx <= r and b <= yy <= t and not (l <= x <= r and b <= y <= t)
                       for l, r, b, t in boxes):
                    return step
            return 20

        best_score = -math.inf
        best_move = [0.0, 0.0]
        zone = obs.get("zone", {})
        side = self.detour_side
        for offset in (0, 30 * side, -30 * side, 60 * side, -60 * side,
                       90 * side, -90 * side, 135 * side, -135 * side, 180):
            angle = heading + math.radians(offset)
            dx, dy = math.cos(angle), math.sin(angle)
            # Keep the next movement segment inside the current circle, including
            # combat strafes when an opponent is close to the boundary.
            if me.get("in_zone", True) and not self.is_in_safe_circle(
                (x + 8.0 * dx, y + 8.0 * dy), zone, margin=1.0
            ):
                continue
            room = clearance(dx, dy)
            if room <= 2:
                continue
            travel = min(10.0, room - 1.0)
            remaining = math.hypot(gx - (x + travel * dx), gy - (y + travel * dy))
            score = distance - remaining + 0.45 * room + 8.0 * (dx * dodge_x + dy * dodge_y)
            if now < self.detour_until:
                score += 12.0 * math.sin(math.radians(offset)) * side
            if score > best_score:
                best_score, best_move = score, [dx, dy]
        if best_score == -math.inf and me.get("in_zone", True):
            cx, cy = zone.get("center", (x, y))
            inward = math.atan2(cy - y, cx - x)
            best_move = [math.cos(inward), math.sin(inward)]
        jump = bool(self.stuck_ticks >= 2 and me.get("grounded", False)
                    and me.get("energy", 0.0) > 35.0)
        return best_move, jump, urgent

    def has_line_of_sight(self, x1, y1, z1, x2, y2, z2):
        """Check sampled muzzle-to-target points against map building prisms."""
        for fraction in (0.25, 0.5, 0.75):
            px = x1 + (x2 - x1) * fraction
            py = y1 + (y2 - y1) * fraction
            pz = z1 + (z2 - z1) * fraction
            for left, right, bottom, top, base, roof in self.obstacles:
                if base <= pz <= roof and left <= px <= right and bottom <= py <= top:
                    return False
        return True

    def target_for_combat(self, obs):
        me = obs["self"]
        enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        if enemies:
            return min(enemies, key=lambda m: distance3(me["pos"], m.get("pos", me["pos"])))
        # Farm a nearby crate/NPC only when no player target is visible.
        targets = list(obs.get("crates", [])) + [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]
        return min(targets, key=lambda t: distance3(me["pos"], t.get("pos", me["pos"])), default=None)

    def compute_aim_and_fire(self, obs, target):
        me = obs["self"]
        now = float(obs.get("t", 0.0))
        mx, my, mz = me["pos"]
        cur_yaw, _ = me["aim"]
        muzzle_z = mz + me.get("stats", {}).get("muzzle_height", 7.0)

        tx, ty, tz = target["pos"]
        tvx, tvy, tvz = target.get("vel", (0.0, 0.0, 0.0))
        target_center_z = tz + (4.5 if target.get("kind") == "mech" else 1.5)

        dx, dy, dz = tx - mx, ty - my, target_center_z - muzzle_z
        dist_2d = math.hypot(dx, dy)
        dist_3d = math.hypot(dist_2d, dz)
        aim_yaw = math.atan2(dy, dx)
        aim_pitch = math.atan2(dz, max(0.1, dist_2d))
        weapons = me.get("weapons", [])
        fire = [False] * len(weapons)

        priority = {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}
        ordered_slots = sorted(
            [i for i, weapon in enumerate(weapons) if weapon is not None],
            key=lambda i: priority.get(weapons[i].get("kind"), 0),
            reverse=True,
        )

        aim_locked = False
        mortar_lobbing = False
        direct_los = self.has_line_of_sight(mx, my, muzzle_z, tx, ty, target_center_z)

        for slot in ordered_slots:
            weapon = weapons[slot]
            kind = weapon["kind"]
            stats = weapon.get("stats", {})

            if kind == "laser":
                if now < self.shorted_until:
                    fire[slot] = False
                    continue
                charge = float(weapon.get("charge", 0.0))
                full_charge = max(0.01, float(stats.get("charge", 0.75)))
                # Release well before the 1.6 s cookoff, even if the target
                # moved behind cover or outside range while the laser charged.
                if charge >= 1.20:
                    fire[slot] = False
                    continue
                if charge > 0.0 and not direct_los:
                    fire[slot] = False
                    continue
                if dist_3d <= stats.get("range", 0.0) and direct_los:
                    if not aim_locked:
                        aim_yaw = math.atan2(dy, dx)
                        aim_pitch = math.atan2(dz, max(0.1, dist_2d))
                        aim_locked = True
                    yaw_error = abs(angle_diff(aim_yaw, cur_yaw))
                    if charge < full_charge * 0.85:
                        fire[slot] = True
                    elif yaw_error >= 0.12:
                        fire[slot] = True

            elif kind == "lightning":
                cone_rad = math.radians(stats.get("cone_deg", 30.0) * 0.5)
                if (dist_3d <= stats.get("range", 0.0) and direct_los
                        and me.get("energy", 0.0) >= stats.get("energy", 15.0)):
                    if not aim_locked:
                        aim_yaw = math.atan2(dy, dx)
                        aim_pitch = math.atan2(dz, max(0.1, dist_2d))
                        aim_locked = True
                    yaw_error = abs(angle_diff(aim_yaw, cur_yaw))
                    if yaw_error <= cone_rad:
                        fire[slot] = True

            elif kind == "mortar":
                speed = max(1.0, float(stats.get("speed", 72.0)))
                gravity = max(0.1, float(self.mortar_g))
                # Lead the target for the slower mortar flight time.
                t_est = dist_2d / max(1.0, speed * 0.85)
                lead_x = tx + tvx * t_est - mx
                lead_y = ty + tvy * t_est - my
                lead_dist_2d = math.hypot(lead_x, lead_y)
                disc = speed**4 - gravity * (gravity * lead_dist_2d**2 + 2.0 * dz * speed**2)
                if 0.0 <= disc and 10.0 < lead_dist_2d <= stats.get("range", 0.0) and not aim_locked:
                    aim_yaw = math.atan2(lead_y, lead_x)
                    aim_pitch = math.atan(
                        (speed**2 - math.sqrt(disc)) / (gravity * max(1.0, lead_dist_2d))
                    )
                    aim_locked = True
                    mortar_lobbing = True
                    if abs(angle_diff(aim_yaw, cur_yaw)) < 0.2:
                        fire[slot] = True

            elif kind == "gun":
                # Avoid firing the gun into the sky while the mortar uses a high arc.
                if mortar_lobbing:
                    continue
                if (dist_3d <= stats.get("range", 0.0) and direct_los
                        and weapon.get("reload", 0) == 0):
                    speed = max(1.0, float(stats.get("speed", 250.0)))
                    flight_time = dist_3d / speed
                    lead_x = tx + tvx * flight_time - mx
                    lead_y = ty + tvy * flight_time - my
                    lead_z = target_center_z + tvz * flight_time - muzzle_z
                    if not aim_locked:
                        aim_yaw = math.atan2(lead_y, lead_x)
                        aim_pitch = math.atan2(lead_z, max(0.1, math.hypot(lead_x, lead_y)))
                        aim_locked = True
                    if abs(angle_diff(aim_yaw, cur_yaw)) < 0.25:
                        fire[slot] = True

        return [aim_yaw, aim_pitch], fire

    def pick_upgrade(self, choices, me):
        if not choices:
            return None
        hp_missing = 1.0 - me.get("hp", 1.0) / max(1.0, me.get("max_hp", 1.0))
        energy_missing = 1.0 - me.get("energy", 0.0) / max(1.0, me.get("max_energy", 1.0))
        weights = {"hp": 1.2 + 0.7 * hp_missing, "armor": 1.2,
                   "regen_per_s": 0.2 + 0.3 * hp_missing, "walk_speed": 1.0,
                   "boost_speed": 0.6, "energy": 0.4 + 0.4 * energy_missing,
                   "energy_regen": 0.6 + 0.4 * energy_missing,
                   "turn_rate": 0.3, "range": 0.3, "damage": 0.5,
                   "cooldown": 0.3, "speed": 0.4, "splash_radius": 0.45}
        best_i, best_score = 0, -math.inf
        for i, choice in enumerate(choices):
            if isinstance(choice, str):
                upgrade_id, value = choice, 1.0
            else:
                upgrade_id = choice.get("id")
                value = float(choice.get("value", 1.0))
            definition = self.upgrade_defs.get(upgrade_id, {})
            path = str(definition.get("path", upgrade_id or "")).lower()
            score = value * max((weight for key, weight in weights.items() if key in path), default=0.25)
            if score > best_score:
                best_i, best_score = i, score
        return best_i

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        me = obs["self"]
        now = float(obs.get("t", 0.0))
        damage_alert_yaw = None
        current_yaw = me.get("aim", [0.0, 0.0])[0]
        for event in obs.get("events", []):
            kind = event.get("type")
            if kind == "damaged":
                self.last_damage_t = now
                # League damage events report the attacker's bearing relative to our facing.
                if event.get("bearing") is not None:
                    damage_alert_yaw = (current_yaw + float(event["bearing"]) + math.pi) % (2.0 * math.pi) - math.pi
            elif kind == "shorted":
                self.shorted_until = max(self.shorted_until, now + event.get("duration", 2.0))
            elif kind == "concussed":
                self.concussed_until = max(self.concussed_until, now + event.get("duration", 1.5))
        self.remember_sounds(obs)

        if damage_alert_yaw is not None:
            self.alert_yaw = damage_alert_yaw
            self.alert_until = now + 3.0
        else:
            sounds = obs.get("sounds", [])
            if sounds:
                nearest_sound = min(
                    sounds,
                    key=lambda sound: (
                        0 if sound.get("dist") == "near" else 1,
                        float(sound.get("age", 0.0)),
                    ),
                )
                if nearest_sound.get("bearing") is not None:
                    self.alert_yaw = float(nearest_sound["bearing"])
                    self.alert_until = max(self.alert_until, now + max(0.5, 1.5 - float(nearest_sound.get("age", 0.0))))

        loot_goal, loot_nearby, drop_opp = self.loot_target(obs)
        features = self.extract_features(obs, (loot_nearby, drop_opp))
        scores, hidden = self.brain.forward_with_hidden(features)
        outside_zone = features[2]
        zone_urgency = features[3]
        must_rotate = outside_zone > 0.5 or zone_urgency > 0.92
        hp_ratio = features[0]
        enemy_visible = bool(features[4])
        opening_loot = hp_ratio > 0.75 and not enemy_visible and features[8] > 0.65
        losing_duel = features[6] < -0.15 or features[7] < 0.0
        need_cover = hp_ratio < 0.55 or bool(features[10]) or losing_duel
        threatened = features[11] >= 0.8 or (enemy_visible and features[5] >= 0.5)
        if must_rotate:
            mode = 3
        elif opening_loot:
            mode = 2
        elif need_cover:
            mode = 1
        elif self.starter_target is not None and not enemy_visible:
            mode = 2
        else:
            mode = self.choose_mode(scores)
        # Do not abandon cover for loot while critically wounded and under
        # pressure. This final guard also protects against later mode overrides.
        if (mode == 2 and hp_ratio < 0.45
                and (enemy_visible or bool(self.active_sounds(now)))):
            mode = 1
        self.current_mode = mode
        nearest = self.nearest_enemy(obs)
        me_pos = me.get("pos", (0.0, 0.0, 0.0))
        enemy_distance = distance3(me_pos, nearest.get("pos", me_pos)) if nearest else math.inf
        zone = obs.get("zone", {})
        threat_yaw = None
        if mode == 1:
            goal, threat_yaw = self.retreat_goal(obs, must_rotate)
        elif must_rotate:
            goal = zone["center"]
        elif mode == 3:
            goal = self.safe_edge_anchor(obs)
        elif mode == 2 and loot_goal is not None:
            goal = loot_goal[:2]
        elif (mode == 0 and self.starter_target is not None and nearest is not None
              and 30.0 <= distance3(self.starter_target["pos"], nearest["pos"]) <= 210.0):
            goal = self.starter_target["pos"][:2]
        elif nearest is not None:
            target_range = 55.0
            kinds = {w.get("kind") for w in me.get("weapons", []) if isinstance(w, dict)}
            if "laser" in kinds:
                target_range = 115.0
            elif "mortar" in kinds:
                target_range = 95.0
            elif "lightning" in kinds:
                target_range = 28.0
            ex, ey, _ = nearest.get("pos", me_pos)
            dx, dy = me_pos[0] - ex, me_pos[1] - ey
            nx, ny = unit2(dx, dy)
            if mode == 0 and features[6] > 0.25 and features[4] > 0.45:
                goal = [ex + nx * max(18.0, target_range * 0.7), ey + ny * max(18.0, target_range * 0.7)]
            else:
                goal = [ex + nx * target_range, ey + ny * target_range]
        elif loot_goal is not None and mode == 2:
            goal = loot_goal[:2]
        else:
            goal = self.safe_edge_anchor(obs)

        movement, jump, urgent_dodge = self.move(obs, goal)
        if (me.get("grounded") is False and self.stuck_ticks >= 2
                and me.get("energy", 0.0) > 25.0):
            # In this game, holding jump in the air activates the upward jets.
            jump = True
        weapons = me.get("weapons", [])
        held = {w.get("kind") for w in weapons if isinstance(w, dict)}
        need_first_weapon = len(held) < 2 and "gun" in held
        enemy_charging = nearest is not None and nearest.get("charging", False)
        should_boost = (
            (not me.get("in_zone", True) and any(movement))
            or (urgent_dodge and me.get("energy", 0.0) > 30.0)
            or (need_first_weapon and (mode == 2 or (mode == 0 and self.starter_target is not None))
                and loot_goal is not None and me.get("energy", 0.0) > 35.0)
            or (mode == 1 and threatened and me.get("energy", 0.0) > 0.0)
            or (mode == 1 and me.get("energy", 0.0) > 40.0
                and ((now - self.last_damage_t) < 1.2 or enemy_charging))
            or (not me.get("grounded", True) and me.get("energy", 0.0) > 30.0)
        )
        action = {
            "tick": obs.get("tick", 0),
            "move": movement,
            "boost": bool(should_boost and any(movement)),
        }
        if jump:
            action["jump"] = True

        if obs.get("offer"):
            picked = self.pick_upgrade(obs["offer"], me)
            if picked is not None:
                action["pick"] = picked

        # Pick up a nearby unowned weapon when a slot is available.
        if None in weapons or len(weapons) < 2:
            take_candidates = [self.starter_target] if self.starter_target is not None else obs.get("pickups", [])
            for item in take_candidates:
                if item.get("kind") in held or item.get("kind") == "gun":
                    continue
                if self.in_take_range(me_pos, item):
                    action["take"] = item.get("id")
                    break

        target = self.target_for_combat(obs)
        if target is not None:
            action["aim"], action["fire"] = self.compute_aim_and_fire(obs, target)
            tx, ty, _ = target.get("pos", me_pos)
            self.alert_yaw = math.atan2(ty - me_pos[1], tx - me_pos[0])
            self.alert_until = now + 1.0
        else:
            action["fire"] = [False] * len(weapons)
            if self.alert_yaw is not None and (now < self.alert_until or mode == 1):
                # Hold the front armor toward a recent attacker or nearby sound while evading.
                action["aim"] = [self.alert_yaw, 0.0]
            else:
                current_yaw = me.get("aim", [self.scan_yaw, 0.0])[0]
                self.scan_yaw = (current_yaw + 0.35 + math.pi) % (2.0 * math.pi) - math.pi
                action["aim"] = [self.scan_yaw, 0.0]

        if threat_yaw is not None:
            if abs(angle_diff(action["aim"][0], threat_yaw)) > 0.08:
                action["fire"] = [False] * len(weapons)
            action["aim"] = [threat_yaw, action["aim"][1]]
        return action


if __name__ == "__main__":
    mf.run(Bot)



