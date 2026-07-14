"""Drive the real chase-tui headlessly with Textual's Pilot and capture
SVG frames at each interaction step. v2 (post-review): no pipeline-run /
log sections — the walkthrough covers the customisation steps only, and
lingers on each option so it can be followed.

  python capture_tui.py
"""
import asyncio, json, os, hashlib

OUT = "/tmp/tui_rec"
FR = f"{OUT}/frames"
os.makedirs(FR, exist_ok=True)

DATA = "/tmp/CHASE/20230329_X12/fits"


async def main():
    from chase.tui import ChaseTUI
    from textual.widgets import Input, Checkbox, Select

    app = ChaseTUI()
    manifest = []
    idx = [0]

    def snap(seg, dur=0.3):
        idx[0] += 1
        fn = f"{FR}/{idx[0]:04d}.svg"
        with open(fn, "w") as f:
            f.write(app.export_screenshot())
        manifest.append({"file": fn, "segment": seg, "dur": dur})

    async with app.run_test(size=(118, 38)) as pilot:
        await pilot.pause()
        snap("intro", 2.0); snap("intro", 1.2)

        # 1 · Data source — type the path, then Discover, hold on the result
        src = app.query_one("#src", Input); app.set_focus(src)
        for i in range(1, 9):
            src.value = DATA[: max(1, len(DATA) * i // 8)]
            await pilot.pause(); snap("data", 0.18)
        src.value = DATA; await pilot.pause(); snap("data", 1.0)
        app._discover(); await pilot.pause()
        snap("data", 2.6); snap("data", 1.6)

        # 2 · Field of view — untick Full FOV, then type the patch, hold so
        # the numbers can be read
        ff = app.query_one("#full_fov", Checkbox); app.set_focus(ff)
        ff.value = False
        await pilot.pause(); snap("fov", 1.4)
        patch = app.query_one("#patch", Input); app.set_focus(patch)
        ptxt = "940 1080 1870 2040"
        for i in range(1, 7):
            patch.value = ptxt[: max(1, len(ptxt) * i // 6)]
            await pilot.pause(); snap("fov", 0.22)
        patch.value = ptxt; await pilot.pause(); snap("fov", 2.4)

        # 3 · Calibration — tick each option deliberately
        for cid in ["contrast", "freeze_clim"]:
            cb = app.query_one(f"#{cid}", Checkbox); app.set_focus(cb); cb.value = True
            await pilot.pause(); snap("calib", 1.6)
        snap("calib", 1.8)

        # 4 · Temperature — double, hold on the selection
        sel = app.query_one("#temperature", Select); app.set_focus(sel); sel.value = "double"
        await pilot.pause(); snap("temp", 2.2); snap("temp", 1.2)

        # output dir
        out = app.query_one("#out", Input); app.set_focus(out); out.value = "/tmp/chase_out"
        await pilot.pause(); snap("temp", 1.4)

        # deterministic final state, for the closing "ready to run" beat
        app.query_one("#full_fov", Checkbox).value = False
        for cid in ["track", "resample", "optical_flow", "contrast", "freeze_clim"]:
            app.query_one(f"#{cid}", Checkbox).value = True
        await pilot.pause()
        snap("temp", 2.0)

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=0)
    print(f"frames={idx[0]}")


asyncio.run(main())
