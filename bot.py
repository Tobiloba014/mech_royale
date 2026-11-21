"""Mech Royale bot with a tactical neural network loaded from weights.json."""

import json
import math
from pathlib import Path
import mechfight as mf


def tanh(x: float) -> float:
    return math.tanh(max(-10.0, min(10.0, x)))


class TacticalBrain:
    """12 -> 8 -> 4 MLP decision engine loaded from weights.json."""

    def __init__(self, weights_path: Path):
        data = json.loads(weights_path.read_text(encoding="utf-8"))
        self.W1 = data["W1"]
        self.B1 = data["B1"]
        self.W2 = data["W2"]
        self.B2 = data["B2"]

    def forward(self, features: list[float]) -> list[float]:
        hidden = [
            tanh(sum(w * x for w, x in zip(row, features)) + b)
            for row, b in zip(self.W1, self.B1)
        ]
        return [
            sum(w * h for w, h in zip(row, hidden)) + b
            for row, b in zip(self.W2, self.B2)
        ]


class Bot:
    name = "neuro-mech-v2"
    MODE_NAMES = ("ENGAGE", "KITE", "LOOT", "ROTATE")

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.pickup_radius = self.cfg["loot"]["pickup_radius"]
        self.mortar_g = self.cfg["weapons"]["mortar"]["gravity"]
        self.upgrade_defs = {u["id"]: u for u in self.cfg.get("upgrades", [])}
        self.brain = TacticalBrain(Path(__file__).with_name("weights.json"))

        self.checkpoint = None
        self.detour_until = 0.0
        self.detour_side = 1
        self.last_damage_t = -10.0
        self.shorted_until = 0.0
        self.concussed_until = 0.0

        world_path = Path(hello["map"]["dir"]) / "v2" / "world.json"
        world = json.loads(world_path.read_text(encoding="utf-8"))

        self.obstacles = []
        for prism in world["prisms"]:
            xs = prism["ring"][::2]
            ys = prism["ring"][1::2]
            self.obstacles.append(
                (min(xs), max(xs), min(ys), max(ys), prism["base"], prism["top"], prism["kind"])
            )

    def weapon_advantage(self, my_weapons, enemy_weapons) -> float:
        my_kinds = {w["kind"] for w in my_weapons if w is not None}
        en_kinds = set(enemy_weapons)
        score = 0.0
        counters = {("laser", "mortar"), ("mortar", "lightning"), ("lightning", "laser")}
        for m in my_kinds:
            for e in en_kinds:
                if (m, e) in counters:
                    score += 0.6
                elif (e, m) in counters:
                    score -= 0.6
        return max(-1.0, min(1.0, score))

    def loot_target(self, obs):
        """Return the goal with the lowest seconds per reward and both loot signals."""
        me = obs["self"]
        pos = me["pos"]
        speed = max(1.0, me["stats"].get("walk_speed", self.cfg["mech"]["walk_speed"]))
        held = {w["kind"] for w in me["weapons"] if w is not None}
        hp_ratio = me["hp"] / max(1.0, me["max_hp"])
        energy_ratio = me["energy"] / max(1.0, me["max_energy"])
        candidates = []
        loot_nearby = 0.0
        drop_opportunity = 0.0

        for cube in obs["cubes"]:
            distance = math.dist(pos, cube["pos"])
            if distance < 50.0:
                candidates.append((distance / speed, cube["pos"]))
                loot_nearby = 1.0

        can_take = None in me["weapons"] or "gun" in held
        if can_take:
            weapon_value = 1.8 if len(held) < 2 else 1.1
            for item in obs["pickups"]:
                distance = math.dist(pos, item["pos"])
                if distance < 60.0 and item["kind"] != "gun" and item["kind"] not in held:
                    candidates.append((distance / (speed * weapon_value), item["pos"]))
                    loot_nearby = 1.0

        if hp_ratio > 0.60 and held:
            for npc in obs["mechs"]:
                if npc["kind"] != "npc":
                    continue
                distance = math.dist(pos, npc["pos"])
                if distance < 65.0:
                    drone = self.cfg["npc"]
                    hp_left = npc.get("hp", drone["hp"]) / max(1.0, drone["hp"])
                    cost = (distance / speed + 4.0 * hp_left) / max(1.0, drone["cubes"])
                    candidates.append((cost, npc["pos"]))
                    loot_nearby = max(loot_nearby, 0.65)

        for drop in obs.get("drops", ()):
            distance = math.dist(pos, drop["pos"])
            if distance >= 140.0 or drop["lands_in"] >= 15.0:
                continue
            arrival = max(0.0, distance - drop["radius"]) / speed
            remaining = max(0.0, drop["capture_s"] - drop["progress"])
            holder = drop["holder"]
            if holder is not None and holder != me["id"] and remaining <= arrival:
                continue
            capture = remaining if holder == me["id"] else drop["capture_s"]
            cubes = self.cfg.get("drops", {}).get("cubes", 3.0)
            reward = cubes + 2.0 * (1.0 - hp_ratio) + 0.5 * (1.0 - energy_ratio)
            cost = (max(arrival, drop["lands_in"]) + capture) / max(1.0, reward)
            candidates.append((cost, drop["pos"]))
            drop_opportunity = max(drop_opportunity, 1.0 - distance / 140.0)

        best = min(candidates, key=lambda entry: entry[0])[1] if candidates else None
        return best, loot_nearby, drop_opportunity

    def extract_features(self, obs, loot_signals=None) -> list[float]:
        me = obs["self"]
        now = obs["t"]
        x, y, _ = me["pos"]

        hp_ratio = me["hp"] / max(1.0, me["max_hp"])
        en_ratio = me["energy"] / max(1.0, me["max_energy"])
        outside_zone = 0.0 if me["in_zone"] else 1.0

        zc = obs["zone"]["center"]
        zr = max(1.0, obs["zone"]["radius"])
        dist_center = math.hypot(x - zc[0], y - zc[1])
        zone_urgency = max(0.0, min(1.0, (dist_center / zr) ** 2))

        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        nearest = min(enemies, key=lambda m: math.dist(me["pos"], m["pos"]), default=None)

        if nearest:
            d_en = math.dist(me["pos"], nearest["pos"])
            enemy_vis = 1.0
            enemy_close = max(0.0, min(1.0, 1.0 - d_en / 150.0))
            en_hp = 1.0
            if "hp" in nearest and "max_hp" in nearest:
                en_hp = nearest["hp"] / max(1.0, nearest["max_hp"])
            hp_adv = hp_ratio - en_hp
            tri_adv = self.weapon_advantage(me["weapons"], nearest.get("weapons", []))
            if nearest.get("charging"):
                tri_adv = max(-1.0, tri_adv - 0.4)
        else:
            enemy_vis, enemy_close, hp_adv, tri_adv = 0.0, 0.0, 0.0, 0.0

        if loot_signals is None:
            _, loot_nearby, drop_opp = self.loot_target(obs)
        else:
            loot_nearby, drop_opp = loot_signals

        debuffed = 1.0 if (now < self.shorted_until or now < self.concussed_until) else 0.0
        sound_threat = 0.0
        if not nearest:
            for s in obs["sounds"]:
                if s["kind"] in ("gun", "mortar", "laser", "lightning", "step", "boost"):
                    if s["dist"] == "near":
                        sound_threat = max(sound_threat, 1.0)
                    elif s["dist"] == "mid":
                        sound_threat = max(sound_threat, 0.5)

        return [
            hp_ratio, en_ratio, outside_zone, zone_urgency,
            enemy_vis, enemy_close, hp_adv, tri_adv,
            loot_nearby, drop_opp, debuffed, sound_threat,
        ]

    def move(self, obs, goal):
        me = obs["self"]
        x, y, z = me["pos"]
        now = obs["t"]
        distance = math.dist((x, y), goal)

        if self.checkpoint is None:
            self.checkpoint = (now, x, y)
        old_time, old_x, old_y = self.checkpoint
        if now - old_time >= 1.0:
            if distance > 5 and math.hypot(x - old_x, y - old_y) < 2:
                self.detour_side *= -1
                self.detour_until = now + 1.5
            self.checkpoint = (now, x, y)

        if distance < 2.0:
            return [0.0, 0.0]

        heading = math.atan2(goal[1] - y, goal[0] - x)
        radius = me["stats"].get("radius", 2.5)
        height = me["stats"].get("height", 9.0)

        local_boxes = [
            (left - radius, right + radius, bottom - radius, top + radius)
            for left, right, bottom, top, base, roof, kind in self.obstacles
            if roof > z + 1.2
            and base < z + height
            and left - 20 - radius <= x <= right + 20 + radius
            and bottom - 20 - radius <= y <= top + 20 + radius
        ]

        def clearance(dx, dy):
            for step in (2, 4, 8, 12, 16):
                px, py = x + step * dx, y + step * dy
                if any(
                    left <= px <= right and bottom <= py <= top
                    for left, right, bottom, top in local_boxes
                    if not (left <= x <= right and bottom <= y <= top)
                ):
                    return step
            return 20

        best_score = -math.inf
        best_move = [0.0, 0.0]
        side = self.detour_side

        for offset in (
            0, 30 * side, -30 * side, 60 * side, -60 * side,
            90 * side, -90 * side, 135 * side, -135 * side, 180,
        ):
            angle = heading + math.radians(offset)
            dx, dy = math.cos(angle), math.sin(angle)
            room = clearance(dx, dy)
            if room == 2:
                continue
            travel = min(10, room - 1)
            remaining = math.dist((x + travel * dx, y + travel * dy), goal)
            score = (distance - remaining) + 0.45 * room
            if now < self.detour_until:
                score += 12 * math.sin(math.radians(offset)) * side
            if score > best_score:
                best_score = score
                best_move = [dx, dy]

        return best_move

    def compute_aim_and_fire(self, obs, target):
        me = obs["self"]
        now = obs["t"]
        mx, my, mz = me["pos"]
        muzzle_z = mz + me["stats"].get("muzzle_height", 7.0)
        tx, ty, tz = target["pos"]
        tvx, tvy, tvz = target["vel"]
        target_center_z = tz + (4.5 if target["kind"] == "mech" else 1.75)

        dx, dy, dz = tx - mx, ty - my, target_center_z - muzzle_z
        dist_2d = math.hypot(dx, dy)
        dist_3d = math.hypot(dist_2d, dz)

        aim_yaw = math.atan2(dy, dx)
        aim_pitch = math.atan2(dz, max(0.1, dist_2d))
        fire = [False, False]

        priority = {"laser": 4, "lightning": 3, "mortar": 2, "gun": 1}
        ordered_slots = sorted(
            [i for i, w in enumerate(me["weapons"]) if w is not None],
            key=lambda i: priority.get(me["weapons"][i]["kind"], 0),
            reverse=True,
        )

        aim_locked = False
        for slot in ordered_slots:
            w = me["weapons"][slot]
            kind = w["kind"]
            stats = w["stats"]

            if kind == "laser":
                if now < self.shorted_until:
                    continue
                if dist_3d <= stats["range"]:
                    if not aim_locked:
                        aim_yaw = math.atan2(dy, dx)
                        aim_pitch = math.atan2(dz, max(0.1, dist_2d))
                        aim_locked = True
                    yaw_err = abs((aim_yaw - me["aim"][0] + math.pi) % (2 * math.pi) - math.pi)
                    charge = w.get("charge", 0.0)
                    full_charge = stats.get("charge", 0.75)
                    if charge < full_charge * 0.85:
                        fire[slot] = True
                    elif yaw_err < 0.12:
                        fire[slot] = False
                    else:
                        fire[slot] = True

            elif kind == "lightning":
                cone_rad = math.radians(stats.get("cone_deg", 30.0) * 0.5)
                if dist_3d <= stats["range"] and me["energy"] >= 15:
                    if not aim_locked:
                        aim_yaw = math.atan2(dy, dx)
                        aim_pitch = math.atan2(dz, max(0.1, dist_2d))
                        aim_locked = True
                    yaw_err = abs((aim_yaw - me["aim"][0] + math.pi) % (2 * math.pi) - math.pi)
                    if yaw_err <= cone_rad:
                        fire[slot] = True

            elif kind == "mortar":
                v = stats["speed"]
                g = self.mortar_g
                disc = v**4 - g * (g * dist_2d**2 + 2 * dz * v**2)
                if disc >= 0 and dist_2d > 10.0 and not aim_locked:
                    aim_yaw = math.atan2(dy, dx)
                    aim_pitch = math.atan((v**2 - math.sqrt(disc)) / (g * max(1.0, dist_2d)))
                    aim_locked = True
                    yaw_err = abs((aim_yaw - me["aim"][0] + math.pi) % (2 * math.pi) - math.pi)
                    if yaw_err < 0.2:
                        fire[slot] = True

            elif kind == "gun":
                if dist_3d <= stats["range"] and w.get("reload", 0) == 0:
                    speed = stats["speed"]
                    t_flight = dist_3d / max(1.0, speed)
                    px = tx + tvx * t_flight - mx
                    py = ty + tvy * t_flight - my
                    pz = target_center_z + tvz * t_flight - muzzle_z
                    if not aim_locked:
                        aim_yaw = math.atan2(py, px)
                        aim_pitch = math.atan2(pz, max(0.1, math.hypot(px, py)))
                        aim_locked = True
                    yaw_err = abs((aim_yaw - me["aim"][0] + math.pi) % (2 * math.pi) - math.pi)
                    if yaw_err < 0.25:
                        fire[slot] = True

        return [aim_yaw, aim_pitch], fire

    def pick_upgrade(self, choices, me):
        """Compare each offer's projected stat changes with the equipped loadout."""
        equipped = {w["kind"]: w["stats"] for w in me["weapons"] if w is not None}
        current = {"mech": {**self.cfg["mech"], **me["stats"]}}
        for kind, stats in equipped.items():
            current["weapons." + kind] = {**self.cfg["weapons"][kind], **stats}

        missing_hp = 1.0 - me["hp"] / max(1.0, me["max_hp"])
        missing_energy = 1.0 - me["energy"] / max(1.0, me["max_energy"])
        energy_value = 0.32 + 0.40 * missing_energy + 0.15 * ("lightning" in equipped)
        mobility = 1.0 if me["in_zone"] else 1.8
        repair = (0.16 + 0.22 * missing_hp) * (1.0 if me["in_zone"] else 0.6)
        mech_weights = {
            "hp": 1.30 + 0.65 * missing_hp,
            "armor": 1.80 if me["in_zone"] else 0.95,
            "regen_per_s": repair,
            "regen_delay": -repair,
            "walk_speed": 0.75 * mobility,
            "boost_speed": 0.40 * mobility,
            "energy": energy_value,
            "energy_regen": 0.90 * energy_value,
            "boost_drain": -0.70 * energy_value,
            "jet_drain": -0.05 * energy_value,
            "turn_rate": 0.32,
            "pitch_rate": 0.14,
            "near_range_bonus": 0.20,
            "vision_bonus": 0.18,
            "hearing": -0.06,
        }
        weapon_weights = {
            "speed": 0.35,
            "range": 0.32,
            "splash_radius": 0.65,
            "cone_deg": 0.26,
            "pierce": 0.12,
            "chain": 0.18,
            "energy": -(0.28 + 0.40 * missing_energy),
        }
        offsets = {
            "near_range_bonus": self.cfg["vision"]["near_range"],
            "vision_bonus": self.cfg["vision"]["range"],
            "pierce": 1.0,
            "chain": 1.0,
        }

        def sustained_damage(stats):
            interval = stats["cooldown"]
            if "mag" in stats:
                interval += stats["reload"] / max(1.0, stats["mag"])
            elif "charge" in stats:
                interval += stats["charge"]
            return max(0.01, stats["damage"]) / max(0.01, interval)

        def score(choice):
            definition = self.upgrade_defs.get(choice["id"], {})
            required = definition.get("requires")
            if required and required not in equipped:
                return -math.inf

            projected = {}
            for effect in definition.get("effects", ()):
                scope, _, stat = effect["path"].rpartition(".")
                if scope not in current or stat not in current[scope]:
                    continue
                if scope not in projected:
                    projected[scope] = current[scope].copy()
                stats = projected[scope]
                stats[stat] = stats[stat] * effect.get("mul", 1.0) + effect.get("add", 0.0)

            total = 0.0
            for scope, after in projected.items():
                before = current[scope]
                weights = mech_weights if scope == "mech" else weapon_weights
                gain = 0.0
                for stat, weight in weights.items():
                    if stat not in before or before[stat] == after[stat]:
                        continue
                    if stat == "armor":
                        old = 1.0 / max(0.01, 1.0 - before[stat])
                        new = 1.0 / max(0.01, 1.0 - after[stat])
                    else:
                        base = offsets.get(stat, 0.0)
                        old, new = base + before[stat], base + after[stat]
                    gain += weight * math.log(max(0.01, new) / max(0.01, old))
                if scope != "mech":
                    gain += math.log(sustained_damage(after) / sustained_damage(before))
                    gain *= 1.0 if len(equipped) == 1 else 0.75
                total += gain
            return total

        return max(range(len(choices)), key=lambda i: score(choices[i]), default=0)

    def act(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        now = obs["t"]

        for ev in obs["events"]:
            etype = ev["type"]
            if etype == "damaged":
                self.last_damage_t = now
            elif etype == "effect":
                eff = ev.get("effect")
                if eff == "short":
                    self.shorted_until = now + self.cfg["weapons"]["lightning"].get("short_s", 2.0)
                elif eff == "concuss":
                    self.concussed_until = now + 1.5

        # 1. Neural tactical decision
        loot_goal, loot_nearby, drop_opp = self.loot_target(obs)
        features = self.extract_features(obs, (loot_nearby, drop_opp))
        scores = self.brain.forward(features)
        mode = max(range(4), key=lambda i: scores[i])

        # 2. Movement goal from tactical mode
        goal = list(obs["zone"]["next_center"])
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        nearest_enemy = min(enemies, key=lambda m: math.dist(me["pos"], m["pos"]), default=None)

        if mode == 0 and nearest_enemy:
            ex, ey, _ = nearest_enemy["pos"]
            ang = math.atan2(y - ey, x - ex)
            goal = [ex + 35.0 * math.cos(ang), ey + 35.0 * math.sin(ang)]

        elif mode == 1 and nearest_enemy:
            ex, ey, _ = nearest_enemy["pos"]
            away = math.atan2(y - ey, x - ex)
            zc = obs["zone"]["next_center"]
            to_zone = math.atan2(zc[1] - y, zc[0] - x)
            blend = math.atan2(
                math.sin(away) + 0.6 * math.sin(to_zone),
                math.cos(away) + 0.6 * math.cos(to_zone),
            )
            goal = [x + 40.0 * math.cos(blend), y + 40.0 * math.sin(blend)]

        elif mode == 2 and loot_goal is not None:
            goal = loot_goal[:2]

        move = self.move(obs, goal)

        enemy_charging = nearest_enemy is not None and nearest_enemy.get("charging", False)
        should_boost = (
            (not me["in_zone"] and any(move))
            or (mode == 1 and me["energy"] > 45 and ((now - self.last_damage_t) < 0.8 or enemy_charging))
        )

        action = {
            "type": "action",
            "tick": obs["tick"],
            "move": move,
            "boost": should_boost,
            "aim": list(me["aim"]),
            "fire": [False, False],
            "debug": {
                "label": f"{self.MODE_NAMES[mode]} ({scores[mode]:.2f})",
                "lines": [{"a": [x, y, z + 2.0], "b": [goal[0], goal[1], z + 2.0], "color": "#40c0ff"}],
            },
        }

        # 3. Upgrades & weapon pickups/swaps
        if obs["offer"]:
            action["pick"] = self.pick_upgrade(obs["offer"]["choices"], me)

        my_kinds = {w["kind"] for w in me["weapons"] if w is not None}
        can_take = None in me["weapons"] or "gun" in my_kinds
        for item in obs["pickups"]:
            px, py, pz = item["pos"]
            if can_take and item["kind"] != "gun" and item["kind"] not in my_kinds:
                if abs(pz - z) <= 4.0 and math.hypot(px - x, py - y) <= self.pickup_radius:
                    action["take"] = item["id"]
                    if all(w is not None for w in me["weapons"]):
                        for slot_idx, w in enumerate(me["weapons"]):
                            if w and w["kind"] == "gun":
                                action["drop"] = slot_idx
                                break
                    break

        # 4. Combat, sound scanning & rear-armor defense
        target = nearest_enemy or min(npcs, key=lambda n: math.dist(me["pos"], n["pos"]), default=None)
        if target is not None:
            action["aim"], action["fire"] = self.compute_aim_and_fire(obs, target)
        else:
            snapped = False
            for ev in obs["events"]:
                if ev["type"] == "damaged" and ev.get("rear") and ev.get("bearing") is not None:
                    action["aim"] = [ev["bearing"], 0.0]
                    snapped = True
                    break
            if not snapped and obs["sounds"]:
                recent_sound = min(obs["sounds"], key=lambda s: s["age"])
                action["aim"] = [recent_sound["bearing"], 0.0]

        for w in me["weapons"]:
            if w and w["kind"] == "gun" and w["ammo"] < w["stats"]["mag"] // 2 and w["reload"] == 0:
                action["reload"] = True
                break

        return action

if __name__ == "__main__":
    mf.run(Bot)
