"""
Plan-path re-export for :mod:`libero.libero.hierarchical_reward_wrapper`.

Importing this submodule loads ``libero.libero.envs`` (robosuite/MuJoCo). For MuJoCo-free
tests, use ``from libero.libero.hierarchical_reward_wrapper import …`` or the same names
from :mod:`libero.libero.bddlsim_interface`.
"""

from libero.libero.hierarchical_reward_wrapper import *  # noqa: F403
