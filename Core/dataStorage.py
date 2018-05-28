import self as self


class DataStorage:
    def __init__(self, fileName, id, coreImage, alignedImage, shape, isPositive):
        self.fileName = fileName
        self.id = id
        self.coreImage = coreImage
        self.alignedImage = alignedImage
        self.shape = shape
        self.isPositive = isPositive

    def printer(self):
        print(self.fileName)
