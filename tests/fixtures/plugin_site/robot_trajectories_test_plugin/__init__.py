"""A plugin package as a real one would be, minus the wheel: its dist-info sits beside it, so ``importlib.metadata`` finds its
entry points when this folder is on sys.path (tests/test_plugins.py runs it that way, in a fresh interpreter).

A real plugin registers the same lines in its pyproject.toml:

    [project.entry-points."robot_trajectories.kinds"]
    test_plugin = "robot_trajectories_test_plugin.kinds"

    [project.entry-points."robot_trajectories.time_laws"]
    smoothstep_test = "robot_trajectories_test_plugin.time_law:SmoothStep"
"""
