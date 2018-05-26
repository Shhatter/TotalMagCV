import self as self


class DataStorage:
    def __init__(self, fileName, id, coreImage, alignedImage, markedPoints, isPositive):
        self.fileName = fileName
        self.id = id
        self.coreImage = coreImage
        self.alignedImage = alignedImage
        self.MarkedPoints = markedPoints
        self.isPositive = isPositive

    def printer(self):
        print(self.fileName)
