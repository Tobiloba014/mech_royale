"""Replay inspector: runs Bot inside check/check.py and logs decisions to stderr."""

import json
import sys
import time
from bot import Bot

MODES = ("ENGAGE", "KITE", "LOOT", "ROTATE")


def main():
    bot = None
    mode_counts = {m: 0 for m in MODES}
    fire_ticks = 0
    take_ticks = 0
    jump_ticks = 0
    total_obs = 0
    latencies_ms = []

    def log(msg: str):
        print(msg, file=sys.stderr, flush=True)

    log(f"{'TICK':<6} {'TIME':<6} {'HP':<5} {'ZONE':<5} {'EN':<3} {'MODE':<8} {'SCORES (E/K/L/R)':<26} {'FIRE':<6} {'MS':<5}")
    log("-" * 82)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mtype = msg.get("type")

        if mtype == "hello":
            bot = Bot(msg)
            print(json.dumps({"type": "ready", "name": bot.name}), flush=True)

        elif mtype == "obs" and bot is not None:
            total_obs += 1
            t0 = time.perf_counter()
            action = bot.act(msg)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            # Respond on stdout immediately so check.py measures true latency
            print(json.dumps(action), flush=True)

            # Inspect neural activations for stderr log
            loot_goal, loot_nearby, drop_opp = bot.loot_target(msg)
            feats = bot.extract_features(msg, (loot_nearby, drop_opp))
            scores = bot.brain.forward(feats)
            mode_idx = max(range(4), key=lambda i: scores[i])
            mode_name = MODES[mode_idx]
            mode_counts[mode_name] += 1

            if any(action.get("fire", [False, False])):
                fire_ticks += 1
            if "take" in action:
                take_ticks += 1
            if action.get("jump"):
                jump_ticks += 1

            interesting = (
                total_obs % 10 == 1
                or msg.get("events")
                or "take" in action
                or "pick" in action
                or action.get("jump")
            )
            if interesting:
                me = msg["self"]
                enemies = sum(1 for m in msg["mechs"] if m["kind"] == "mech")
                in_z = "IN" if me["in_zone"] else "OUT"
                sc_str = "/".join(f"{s:+.1f}" for s in scores)
                f_str = "".join("1" if f else "0" for f in action.get("fire", [False, False]))
                extra = ""
                if "take" in action:
                    extra += f" [TAKE {action['take']}]"
                if "pick" in action:
                    extra += f" [PICK {action['pick']}]"
                if action.get("jump"):
                    extra += " [JUMP]"
                if msg.get("events"):
                    ev_types = ",".join(e["type"] for e in msg["events"])
                    extra += f" ({ev_types})"

                log(
                    f"{msg['tick']:<6} {msg['t']:<6.1f} {int(me['hp']):<5} {in_z:<5} "
                    f"{enemies:<3} {mode_name:<8} {sc_str:<26} {f_str:<6} {dt_ms:<5.2f}{extra}"
                )

        elif mtype == "end":
            break

    log("\n" + "=" * 82)
    log("SUMMARY OVER RECORDED MATCH:")
    log(f"  Mode Distribution : {mode_counts}")
    log(f"  Firing Ticks      : {fire_ticks}/{total_obs}")
    log(f"  Weapon Pickups    : {take_ticks} | Jumps: {jump_ticks}")
    if latencies_ms:
        log(f"  Decision Latency  : avg {sum(latencies_ms)/len(latencies_ms):.2f} ms | max {max(latencies_ms):.2f} ms (budget: 50.0 ms)")


if __name__ == "__main__":
    main()
