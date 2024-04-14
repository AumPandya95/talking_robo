

class RagException(Exception):
    def __init__(self, *args: object, errors: str) -> None:
        super().__init__(*args)

        self.errors = errors

    def __str__(self) -> str:
        super().__str__()
        return (
            f"Exception message - {self.errors}"
        )

class LangchainException(Exception):
    def __init__(self, *args: object, errors: str) -> None:
        super().__init__(*args)

        self.errors = errors
    
    def __str__(self) -> str:
        super().__str__()
        return (
            f"Exception message - {self.errors}"
        )


class VectorDBException(Exception):
    def __init__(self, *args: object, errors: str) -> None:
        super().__init__(*args)
        
        self.errors = errors

    def __str__(self) -> str:
        super().__str__()
        return (
            f"Exception message - {self.errors}"
        )
