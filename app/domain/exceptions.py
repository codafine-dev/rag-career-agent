class TicketParseError(Exception):
    """Raised when a raw ticket record fails validation or normalization."""

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"{source}: {reason}")
