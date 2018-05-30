import self as self


class DataStorage:
    def __init__(self, fileName, id, coreImage, alignedImage, shape, isPositive):
        self.fileName = fileName
        self.id = id
        self.coreImage = coreImage
        self.alignedImage = alignedImage
        self.shape = shape
        self.isPositive = isPositive

    def addImages(self, leftNosePart
                  , rightNosePartREV
                  , leftMounthEdge
                  , rightMounthEdgeREV
                  , leftEyeEdge
                  , rightEyeEdgeREV
                  , leftUnderEye
                  , rightUnderEyeREV
                  , mounthLeftPart
                  , mounthRightPartREV):
        self.leftNosePart = leftNosePart
        self.rightNosePartREV = rightNosePartREV
        self.leftMounthEdge = leftMounthEdge
        self.rightMounthEdgeREV = rightMounthEdgeREV
        self.leftEyeEdge = leftEyeEdge
        self.rightEyeEdgeREV = rightEyeEdgeREV
        self.leftUnderEye = leftUnderEye
        self.rightUnderEyeREV = rightUnderEyeREV
        self.mounthLeftPart = mounthLeftPart
        self.mounthRightPartREV = mounthRightPartREV

    def printer(self):
        print(self.fileName)
