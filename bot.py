"""A Python Mech Royale bot with local obstacle avoidance."""

import json
import math
from pathlib import Path

import mechfight as mf


class Bot:
    name = "python-mech-starter"

    def __init__(self, hello):
        # The wrapper calls this once when the match sends its settings.
        self.pickup_radius = hello["config"]["loot"]["pickup_radius"]
        self.checkpoint = None
        self.detour_until = 0.0
        self.detour_side = 1

        world_path = Path(hello["map"]["dir"]) / "v2" / "world.json"
        world = json.loads(world_path.read_text(encoding="utf-8"))

        # Store a bounding rectangle and height range for each solid.
        self.obstacles = []
        for prism in world["prisms"]:
            xs = prism["ring"][::2]
            ys = prism["ring"][1::2]
            self.obstacles.append(
                (min(xs), max(xs), min(ys), max(ys), prism["base"], prism["top"])
            )

    def move(self, obs, goal):
        me = obs["self"]
        x, y, z = me["pos"]
        now = obs["t"]
        distance = math.dist((x, y), goal)

        # Compare positions about once a second to detect getting stuck.
        if self.checkpoint is None:
            self.checkpoint = (now, x, y)

        old_time, old_x, old_y = self.checkpoint
        if now - old_time >= 1.0:
            if distance > 5 and math.hypot(x - old_x, y - old_y) < 2:
                self.detour_side *= -1
                self.detour_until = now + 1.5
            self.checkpoint = (now, x, y)

        if distance < 2:
            return [0.0, 0.0]

        heading = math.atan2(goal[1] - y, goal[0] - x)
        radius = me["stats"].get("radius", 2.5)
        height = me["stats"].get("height", 9.0)

        # Expand each nearby rectangle by the mech's radius for clearance.
        local_boxes = [
            (left - radius, right + radius, bottom - radius, top + radius)
            for left, right, bottom, top, base, roof in self.obstacles
            if roof > z + 1.2
            and base < z + height
            and left - 20 - radius <= x <= right + 20 + radius
            and bottom - 20 - radius <= y <= top + 20 + radius
        ]

        def clearance(dx, dy):
            # Sample several points ahead along this heading.
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
        offsets = (
            0, 30 * side, -30 * side, 60 * side, -60 * side,
            90 * side, -90 * side, 135 * side, -135 * side, 180,
        )

        for offset in offsets:
            angle = heading + math.radians(offset)
            dx, dy = math.cos(angle), math.sin(angle)
            room = clearance(dx, dy)
            if room == 2:
                continue

            travel = min(10, room - 1)
            remaining = math.dist((x + travel * dx, y + travel * dy), goal)
            progress = distance - remaining
            score = progress + 0.45 * room

            if now < self.detour_until:
                score += 12 * math.sin(math.radians(offset)) * side

            if score > best_score:
                best_score = score
                best_move = [dx, dy]

        return best_move

    def act(self, obs):
        # The wrapper calls this for each new observation.
        me = obs["self"]
        x, y, z = me["pos"]
        goal = obs["zone"]["next_center"]
        move = self.move(obs, goal)

        action = {
            "tick": obs["tick"],
            "move": move,
            "boost": not me["in_zone"] and any(move),
            "aim": me["aim"],
        }

        if obs["offer"]:
            action["pick"] = 0

        if me["weapons"][1] is None:
            for item in obs["pickups"]:
                px, py, pz = item["pos"]
                if item["kind"] != "gun" and abs(pz - z) <= 4:
                    if math.hypot(px - x, py - y) <= self.pickup_radius:
                        action["take"] = item["id"]
                        break

        # Use the starter kit's basic combat behavior.
        enemy = min(
            (target for target in obs["mechs"] if target["kind"] == "mech"),
            key=lambda target: math.dist(me["pos"], target["pos"]),
            default=None,
        )

        if enemy is not None:
            tx, ty, tz = enemy["pos"]
            action["aim"] = [
                math.atan2(ty - y, tx - x),
                math.atan2(tz - z - 2.5, math.hypot(tx - x, ty - y)),
            ]
            action["fire"] = [True, True]

        return action


if __name__ == "__main__":
    mf.run(Bot)