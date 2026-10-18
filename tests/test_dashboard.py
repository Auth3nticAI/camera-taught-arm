from unittest.mock import MagicMock

import numpy as np

from cta import dashboard


def test_arm_overlay_cleared_only_on_transition(monkeypatch):
    rr = MagicMock()
    monkeypatch.setattr(dashboard, "rr", rr)
    d = dashboard.Dashboard()
    rgb = np.zeros((36, 64, 3), np.uint8)
    arm = np.array([[0.4, 0.3], [0.5, 0.4], [0.6, 0.5]])
    for uv in (arm, None, None, None, arm, None):
        d.log_frame(rgb, None, None, np.zeros(6), 60.0, 10.0, "TRACKING", arm_uv=uv)
    clears = [c for c in rr.log.call_args_list if c.args[0] == "camera/image/arm" and c.args[1] is rr.Clear.return_value]
    assert len(clears) == 2  # once per arm -> no-arm transition, not every frame
