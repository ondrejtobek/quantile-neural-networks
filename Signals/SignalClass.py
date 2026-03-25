"""Base signal class used by all feature definitions."""


class SignalClass:
    """Base class for all cross-sectional signal definitions."""

    def __init__(self):
        """Initialize input/output metadata used by child signal classes."""
        self.Input = {}
        self.Output = []
        self.DependsOn = []

    def CheckOutput(self, Out):
        """Validate requested output names against the class definition.

        Args:
            Out (list[str]): Requested output signal names. Use `["default"]`
                to select all outputs declared in `self.Output`.

        Returns:
            list[str]: Validated output names.

        Raises:
            Exception: If any requested signal name is not defined in
                `self.Output`.
        """
        if Out[0] == "default":
            Out = self.Output
        if any([not item in self.Output for item in Out]):
            raise Exception(
                "Requested anomaly is not defined in the given class. Out should be a list of characters."
            )
        return Out
