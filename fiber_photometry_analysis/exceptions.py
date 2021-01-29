class FiberPhotometryException(Exception):
    pass


class FiberPhotometryUtilitiesException(FiberPhotometryException):
    pass


class FiberPhotometryTypeError(FiberPhotometryUtilitiesException):
    def __init__(self, value, required_type=int, message="The value '{}' of type '{}' is not of the required type '{}'"):
        self.value = value
        self.required_type = required_type
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message.format(self.value, type(self.value), self.required_type)


class FiberFotometryIoException(FiberPhotometryException):
    pass


class FiberFotometryIoFileNotFoundError(FiberFotometryIoException, FileNotFoundError):
    def __init__(self, file_path):
        super().__init__('File "{}" could not be found'.format(file_path))


class FiberPhotometryGenericSignalProcessingError(FiberPhotometryException):
    pass


class FiberPhotometryGenericSignalProcessingValueError(ValueError,
                                                       FiberPhotometryGenericSignalProcessingError):
    pass
