import warnings

class ModelToolNotSupportedWarning(Warning):
    """
    Warning raised when the selected model does not support tool/function calling.
    """

    def __init__(self, error_code: str, data_field: str, message: str = None):
        self.error_code = error_code
        self.data_field = data_field
        self.message = message or (
            f"Model '{data_field}' does not support tool/function calling."
        )

        super().__init__(self.message)
