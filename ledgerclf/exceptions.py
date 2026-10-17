"""Exception hierarchy for user-facing failures (docs/02 section 10).

Every message follows the pattern: what happened, why, how to fix it.
"""

from __future__ import annotations


class LedgerClfError(Exception):
    """Base class for every expected failure in the package.

    Attributes:
        code: Stable machine-readable code (for example ``FILE_EMPTY``).
        message: Plain-English explanation including the fix.
    """

    default_code = "LEDGERCLF_ERROR"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe representation for APIs and the dashboard."""
        return {"error": type(self).__name__, "code": self.code, "message": self.message}


class FileReadError(LedgerClfError):
    """The file is empty, too large, unsupported, unreadable, or protected."""

    default_code = "FILE_UNREADABLE"


class SchemaError(LedgerClfError):
    """The file structure cannot be mapped to the input schema."""

    default_code = "SCHEMA_ERROR"


class ArtifactError(LedgerClfError):
    """Model artifacts are missing, incompatible, or corrupt."""

    default_code = "ARTIFACT_ERROR"


class HoldoutAccessError(LedgerClfError):
    """Holdout labels were requested outside the evaluate-holdout command."""

    default_code = "HOLDOUT_ACCESS_DENIED"


class ConfigError(LedgerClfError):
    """Configuration or champion.json content is invalid."""

    default_code = "CONFIG_ERROR"
