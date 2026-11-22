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

        for cube in obs.get("cubes", []):
            d = distance3(pos, cube.get("pos", pos))
            if d < 130.0:
                candidates.append((d / speed, cube.get("pos")))
                loot_nearby = max(loot_nearby, clamp(1.0 - d / 260.0, 0.0, 1.0))

        can_take = len(weapons) < 2 or any(w is None for w in weapons) or "gun" in held
        if can_take:
            for item in obs.get("pickups", []):
                kind = item.get("kind")
                d = distance3(pos, item.get("pos", pos))
                if d < 130.0 and kind not in (None, "gun") and kind not in held:
                    value = 2.0 if len(held) < 2 else 1.2
                    candidates.append((d / (speed * value), item.get("pos")))
                    loot_nearby = max(loot_nearby, clamp(1.0 - d / 260.0, 0.0, 1.0))

        if hp_ratio > 0.50 and held:
            for crate in obs.get("crates", []):
                d = distance3(pos, crate.get("pos", pos))
                if d < 100.0:
                    hp_left = crate.get("hp", 300.0) / 300.0
                    candidates.append(((d / speed + 3.0 * hp_left) / 1.5, crate.get("pos")))
                    loot_nearby = max(loot_nearby, 0.65)

            npc_cfg = self.cfg.get("npc", {"hp": 350.0, "cubes": 2})
            for npc in obs.get("mechs", []):
                if npc.get("kind") != "npc":
                    continue
                d = distance3(pos, npc.get("pos", pos))
                if d < 100.0:
                    hp_left = npc.get("hp", npc_cfg.get("hp", 350.0)) / max(1.0, npc_cfg.get("hp", 350.0))
                    reward = max(1.0, npc_cfg.get("cubes", 2))
                    candidates.append(((d / speed + 4.0 * hp_left) / reward, npc.get("pos")))
                    loot_nearby = max(loot_nearby, 0.65)

        for drop in obs.get("drops", []):
            d = distance3(pos, drop.get("pos", pos))
            if d >= 180.0 or drop.get("lands_in", 999.0) >= 18.0:
                continue
            arrival = max(0.0, d - drop.get("radius", 8.0)) / speed
            remaining = max(0.0, drop.get("capture_s", 4.0) - drop.get("progress", 0.0))
            holder = drop.get("holder")
            visible_enemies = any(m.get("kind") == "mech" for m in obs.get("mechs", []))
            if (holder == me.get("id")
                    and (visible_enemies or float(obs.get("t", 0.0)) - self.last_damage_t <= 2.5)):
                continue
            if holder is not None and holder != me.get("id") and remaining <= arrival:
                continue
            capture = remaining if holder == me.get("id") else drop.get("capture_s", 4.0)
            cubes = self.cfg.get("drops", {}).get("cubes", 3.0)
            reward = cubes + 2.0 * (1.0 - hp_ratio)
            cost = (max(arrival, drop.get("lands_in", 0.0)) + capture) / max(1.0, reward)
            candidates.append((cost, drop.get("pos")))
            drop_opp = max(drop_opp, clamp(1.0 - d / 180.0, 0.0, 1.0))

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
        cx, cy = zone.get("center", zone.get("next_center", (x, y)))
        radius = max(1.0, zone.get("radius", zone.get("next_radius", 1.0)))
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
        sound_threat = sum(
            weight * max(0.0, 1.0 - (now - timestamp) / 3.0)
            for timestamp, _bearing, weight in self.sound_memory
            if now - timestamp <= 3.0
        )
        sound_threat = clamp(sound_threat, 0.0, 1.0)

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
        local_boxes = [(l, r, b, top)
                       for l, r, b, top, base, roof in self.obstacles
                       if roof > z + 1.2 and base < z + height
                       and l - 30.0 <= x <= r + 30.0
                       and b - 30.0 <= y <= top + 30.0]
        enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        threat_pos = min(
            enemies,
            key=lambda enemy: distance3(me["pos"], enemy.get("pos", me["pos"])),
            default=None,
        )

        def clearance(dx, dy):
            for step in (2, 4, 8, 12, 16):
                xx, yy = x + step * dx, y + step * dy
                if any(l <= xx <= r and b <= yy <= t and not (l <= x <= r and b <= y <= t)
                       for l, r, b, t in boxes):
                    return step
            return 20

        best_score = -math.inf
        best_move = [0.0, 0.0]
        side = self.detour_side
        for offset in (0, 30 * side, -30 * side, 60 * side, -60 * side,
                       90 * side, -90 * side, 135 * side, -135 * side, 180):
            angle = heading + math.radians(offset)
            dx, dy = math.cos(angle), math.sin(angle)
            room = clearance(dx, dy)
            if room <= 2:
                continue
            travel = min(10.0, room - 1.0)
            remaining = math.hypot(gx - (x + travel * dx), gy - (y + travel * dy))
            score = distance - remaining + 0.45 * room + 8.0 * (dx * dodge_x + dy * dodge_y)
            if self.current_mode == 1 and threat_pos is not None:
                next_x, next_y = x + travel * dx, y + travel * dy
                threat_x, threat_y = threat_pos.get("pos", me["pos"])[:2]
                midpoint_x = 0.5 * (next_x + threat_x)
                midpoint_y = 0.5 * (next_y + threat_y)
                if any(left <= midpoint_x <= right and bottom <= midpoint_y <= top
                       for left, right, bottom, top in local_boxes):
                    score += 6.0
            if now < self.detour_until:
                score += 12.0 * math.sin(math.radians(offset)) * side
            if score > best_score:
                best_score, best_move = score, [dx, dy]
        jump = bool(self.stuck_ticks >= 2 and me.get("grounded", False)
                    and me.get("energy", 0.0) > 35.0)
        return best_move, jump, urgent

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
        aim = me.get("aim", [0.0, 0.0])
        cur_yaw = aim[0] if aim else 0.0
        muzzle_z = mz + me.get("stats", {}).get("muzzle_height", 7.0)
        tx, ty, tz = target.get("pos", (mx, my, mz))
        # Crates and some NPC records have no vel/kind; both are safe here.
        tvx, tvy, tvz = target.get("vel", (0.0, 0.0, 0.0))
        kind_target = target.get("kind", "")
        center_z = tz + (4.5 if kind_target == "mech" else 1.5)
        weapons = me.get("weapons", [])
        fire = [False] * len(weapons)
        dx, dy, dz = tx - mx, ty - my, center_z - muzzle_z
        d2 = math.hypot(dx, dy)
        d3 = math.hypot(d2, dz)
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, max(0.1, d2))
        priorities = {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}
        slots = sorted((i for i, w in enumerate(weapons) if isinstance(w, dict)),
                       key=lambda i: priorities.get(weapons[i].get("kind"), 0), reverse=True)
        locked = False
        mortar_lobbing = False
        for i in slots:
            weapon = weapons[i]
            wkind = weapon.get("kind")
            stats = weapon.get("stats", {})
            reach = float(stats.get("range", self.cfg.get("weapons", {}).get(wkind, {}).get("range", 0.0)))
            if wkind == "gun" and not mortar_lobbing and d3 <= reach and weapon.get("reload", 0) == 0:
                speed = max(1.0, float(stats.get("speed", 250.0)))
                flight = d3 / speed
                ax = tx + tvx * flight - mx
                ay = ty + tvy * flight - my
                az = center_z + tvz * flight - muzzle_z
                if not locked:
                    yaw, pitch = math.atan2(ay, ax), math.atan2(az, max(0.1, math.hypot(ax, ay)))
                    locked = True
                if abs(angle_diff(yaw, cur_yaw)) < 0.25:
                    fire[i] = True
            elif wkind == "laser" and d3 <= reach and now >= self.shorted_until:
                if not locked:
                    yaw, pitch = math.atan2(dy, dx), math.atan2(dz, max(0.1, d2))
                    locked = True
                charge = weapon.get("charge", 0.0)
                if charge >= 1.20:
                    continue
                full_charge = max(0.01, stats.get("charge", 0.75))
                fire[i] = charge < full_charge * 0.85 or abs(angle_diff(yaw, cur_yaw)) >= 0.12
            elif wkind == "lightning" and d3 <= reach and me.get("energy", 0.0) >= stats.get("energy", 15.0):
                if not locked:
                    yaw, pitch = math.atan2(dy, dx), math.atan2(dz, max(0.1, d2))
                    locked = True
                cone = math.radians(stats.get("cone_deg", 30.0) * 0.5)
                fire[i] = abs(angle_diff(yaw, cur_yaw)) <= cone
            elif wkind == "mortar" and d2 > 10.0 and not locked:
                speed = max(1.0, float(stats.get("speed", 72.0)))
                gravity = max(0.1, self.mortar_g)
                flight = d2 / (0.85 * speed)
                lead_x, lead_y, lead_z = dx + tvx * flight, dy + tvy * flight, dz + tvz * flight
                distance = math.hypot(lead_x, lead_y)
                disc = speed**4 - gravity * (gravity * distance**2 + 2.0 * lead_z * speed**2)
                if disc >= 0.0 and 10.0 < distance <= reach:
                    yaw = math.atan2(lead_y, lead_x)
                    pitch = math.atan((speed**2 - math.sqrt(disc)) / (gravity * distance))
                    locked = mortar_lobbing = True
                    fire[i] = abs(angle_diff(yaw, cur_yaw)) < 0.20
        return [yaw, pitch], fire

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
            elif kind in ("heard", "sound"):
                self.sound_memory.append((now, event.get("bearing", 0.0), event.get("weight", 1.0)))

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
        self.sound_memory = [s for s in self.sound_memory if now - s[0] <= 3.0]

        loot_goal, loot_nearby, drop_opp = self.loot_target(obs)
        features = self.extract_features(obs, (loot_nearby, drop_opp))
        scores, hidden = self.brain.forward_with_hidden(features)
        mode = self.choose_mode(scores)
        nearest = self.nearest_enemy(obs)
        me_pos = me.get("pos", (0.0, 0.0, 0.0))
        enemy_distance = distance3(me_pos, nearest.get("pos", me_pos)) if nearest else math.inf
        zone = obs.get("zone", {})
        must_rotate = not me.get("in_zone", True)

        weapons = me.get("weapons", [])
        held = {w.get("kind") for w in weapons if isinstance(w, dict)}
        opening_pickup = None
        if not must_rotate and now < 6.0 and held == {"gun"} and features[0] > 0.75:
            candidates = (
                (distance3(me_pos, item["pos"]), index, item)
                for index, item in enumerate(obs.get("pickups", []))
                if item.get("kind") not in (None, "gun") and item.get("pos") is not None
            )
            closest = min(candidates, default=None)
            if closest is not None and closest[0] <= 40.0:
                opening_pickup = closest[2]
                mode = 2
                loot_goal = opening_pickup["pos"]

        enemies = [enemy for enemy in obs.get("mechs", []) if enemy.get("kind") == "mech"]
        capturing_drop = next(
            (
                drop for drop in obs.get("drops", ())
                if drop.get("holder") == me.get("id") and drop.get("progress", 0.0) > 0.0
                and me.get("in_zone", True) and me.get("hp", 0.0) > 500.0
                and not enemies and now - self.last_damage_t > 2.5
            ),
            None,
        )

        if capturing_drop is not None:
            mode = 2
            goal = capturing_drop["pos"][:2]
        elif must_rotate or mode == 3:
            goal = self.safe_edge_anchor(obs)
        elif mode == 2 and loot_goal is not None:
            goal = loot_goal[:2]
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
        if me.get("grounded") is False and self.stuck_ticks >= 2 and me.get("energy", 0.0) > 25.0:
            jump = True
        enemy_charging = nearest is not None and nearest.get("charging", False)
        should_boost = (
            (not me.get("in_zone", True) and any(movement))
            or (urgent_dodge and me.get("energy", 0.0) > 30.0)
            or (opening_pickup is not None and me.get("energy", 0.0) > 30.0)
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
            for item in obs.get("pickups", []):
                if item.get("kind") in held or item.get("kind") == "gun":
                    continue
                d = distance3(me_pos, item.get("pos", me_pos))
                if d <= self.pickup_radius:
                    action["take"] = item.get("id")
                    break

        target = self.target_for_combat(obs)
        if target is not None:
            action["aim"], action["fire"] = self.compute_aim_and_fire(obs, target)
            if ((mode in (1, 3) or must_rotate)
                    and distance3(me["pos"], target.get("pos", me["pos"])) > 55.0):
                action["fire"] = [False] * len(weapons)
        else:
            action["fire"] = [False] * len(weapons)
            if self.alert_yaw is not None and (now < self.alert_until or mode == 1):
                # Hold the front armor toward a recent attacker or nearby sound while evading.
                action["aim"] = [self.alert_yaw, 0.0]
            else:
                current_yaw = me.get("aim", [self.scan_yaw, 0.0])[0]
                self.scan_yaw = (current_yaw + 0.35 + math.pi) % (2.0 * math.pi) - math.pi
                action["aim"] = [self.scan_yaw, 0.0]

        return action


if __name__ == "__main__":
    mf.run(Bot)



