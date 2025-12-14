import warnings

class ModelWebSearchNotSupportedWarning(Warning):
    """
    Warning raised when the selected model does not support web search.
    """

    def __init__(self, error_code: str, data_field: str, message: str = None):
        self.error_code = error_code
        self.data_field = data_field
        self.message = message or (
            f"Model '{data_field}' does not support web search."
        )

        super().__init__(self.message)