"""Exception types raised by red_queen.

Every error the package raises on purpose derives from :class:`RedQueenError`,
so callers (the CLI in particular) can tell expected failures from bugs.
"""


class RedQueenError(Exception):
    """Base class for all expected errors in this package."""


class ConfigError(RedQueenError):
    """The YAML config or an override is invalid."""


class DataError(RedQueenError):
    """A dataset could not be loaded or does not have the expected columns."""


class DownloadError(RedQueenError):
    """A remote resource (dataset, model) could not be fetched after retries."""


class ArtifactError(RedQueenError):
    """A checkpoint, run directory or manifest entry is missing or malformed."""
