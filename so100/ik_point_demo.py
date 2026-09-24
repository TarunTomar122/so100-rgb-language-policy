"""Interactive side-view SO-100 fingertip IK targets.

Run: source scripts/vulkan_env.sh && PYTHONPATH=. .venv/bin/python -m so100.ik_point_demo
Open: http://127.0.0.1:8768/
"""

from __future__ import annotations

import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from so100.executor import Executor
from so100.sim import TABLE_TOP, Tabletop
from so100.vision import project_xyz

PAGE = Path(__file__).resolve().parents[1] / "demo" / "ik_point_demo.html"
SIZE = 512


class Demo:
    def __init__(self) -> None:
        self.world = Tabletop()
        # Place the existing IK site at the distal end of fixed_jaw_pad_1.
        self.world.model.site_pos[self.world._tcp] = [0.0089, -0.1064, 0.0]
        self.world.home()
        self.world.park_scene()
        self.executor = Executor(self.world)
        self.points = []
        for layer, z in (("table", TABLE_TOP + 0.003), ("above", TABLE_TOP + 0.055)):
            for x in (-0.11, 0.0, 0.11):
                for y in (-0.36, -0.28, -0.20):
                    xyz = np.array([x, y, z])
                    uv = project_xyz(self.world, xyz, SIZE)
                    self.points.append({"id": len(self.points), "layer": layer, "xyz": xyz.tolist(), "uv": uv})
        self.selected: int | None = None
        self.path: list[np.ndarray] = []
        self.step_count = 0
        self.status = "Click a point to move the fingertip"

    def select(self, point_id: int) -> dict:
        if point_id < 0 or point_id >= len(self.points):
            raise ValueError("unknown point")
        self.selected = point_id
        goal = np.asarray(self.points[point_id]["xyz"], dtype=float)
        start = self.world.tcp()
        high = max(float(start[2]), float(goal[2]), 0.12)
        self.path = [
            np.array([start[0], start[1], high]),
            np.array([goal[0], goal[1], high]),
            np.array([goal[0], goal[1], goal[2] + 0.03]),
            goal,
        ]
        self.step_count = 0
        self.status = "Moving to selected 3D point"
        return self.state()

    def step(self) -> dict:
        if self.path:
            goal = self.path.pop(0)
            ok = self.executor.goto(goal)
            self.step_count += 1
            if not ok:
                self.path.clear()
                self.status = "IK could not reach this waypoint"
            elif not self.path:
                error = float(np.linalg.norm(self.world.tcp() - goal))
                self.status = "Fingertip reached the point" if error <= 0.008 else "Fingertip stopped short"
        return self.state()

    def reset(self) -> dict:
        self.world.home()
        self.world.park_scene()
        self.selected = None
        self.path.clear()
        self.step_count = 0
        self.status = "Click a point to move the fingertip"
        return self.state()

    def state(self) -> dict:
        frame = io.BytesIO()
        Image.fromarray(self.world.render(SIZE)).save(frame, format="JPEG", quality=85)
        tip = self.world.tcp()
        goal = None if self.selected is None else np.asarray(self.points[self.selected]["xyz"])
        return {
            "frame": "data:image/jpeg;base64," + base64.b64encode(frame.getvalue()).decode("ascii"),
            "points": self.points,
            "selected": self.selected,
            "tip_xyz": tip.tolist(),
            "tip_uv": project_xyz(self.world, tip, SIZE),
            "error_mm": None if goal is None else float(np.linalg.norm(tip - goal) * 1000),
            "table_penetration_mm": float(self.world.table_penetration() * 1000),
            "moving": bool(self.path),
            "step": self.step_count,
            "status": self.status,
        }


class Handler(BaseHTTPRequestHandler):
    def reply(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = self.server.page.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self.reply(self.server.app.state())
        else:
            self.reply({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if self.path == "/api/select":
                data = self.server.app.select(int(body["id"]))
            elif self.path == "/api/command":
                data = self.server.app.command(str(body["text"]))
            elif self.path == "/api/step":
                data = self.server.app.step()
            elif self.path == "/api/reset":
                data = self.server.app.reset()
            elif self.path == "/api/move" and hasattr(self.server.app, "move_click"):
                data = self.server.app.move_click(float(body["u"]), float(body["v"]))
            else:
                return self.reply({"error": "not found"}, 404)
            self.reply(data)
        except (KeyError, ValueError) as exc:
            self.reply({"error": str(exc)}, 400)
        except Exception as exc:
            self.reply({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, *_: object) -> None:
        return


def serve(app: Demo, page: Path, port: int) -> None:
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.app = app
    server.page = page
    print(f"Demo: http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve(Demo(), PAGE, 8768)
