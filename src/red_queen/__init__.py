"""Red Queen Moderation: an RL attacker that stress-tests a hate-speech classifier."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("red-queen")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"
