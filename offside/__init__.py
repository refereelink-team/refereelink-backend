"""
offside: 足球越位分析模块。

本目录下代码提供：
- 单帧越位可视化入口（见 `run_var_video.py`）；
- 越位 / 出界 / 门线几何判定逻辑（见 `judgement.py`）；
- 与 `core` 状态中台的适配层（见 `offside_core_integration.py`）；
- 关键帧判罚结果输出（frame/map/json）。

推荐通过 `pip install -e .` 安装后使用：

    from offside.offside_core_integration import process_frame_with_core
"""

from .offside_core_integration import process_frame_with_core
from .judgement import TEAM_GOAL_SIDE, set_team_goal_sides

__all__ = [
    "process_frame_with_core",
    "TEAM_GOAL_SIDE",
    "set_team_goal_sides",
]
