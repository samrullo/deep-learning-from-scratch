class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class SimpleSquareFuncLayer:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.mulLayerX1 = MulLayer()
        self.mulLayerX2 = MulLayer()
        self.addLayer = AddLayer()

    def forward(self):
        x1square = self.mulLayerX1.forward(self.x1, self.x1)
        x2square = self.mulLayerX2.forward(self.x2, self.x2)
        out = self.addLayer.forward(x1square, x2square)
        return out

    def backward(self, dout):
        daddoutx1, daddoutx2 = self.addLayer.backward(dout)
        dx1_first, dx1_second = self.mulLayerX1.backward(daddoutx1)
        dx1 = dx1_first + dx1_second
        dx2_first, dx2_second = self.mulLayerX2.backward(daddoutx2)
        dx2 = dx2_first + dx2_second
        return dx1, dx2


class SimpleOvalSquareFuncLayer:
    """
    This is simple layer that uses loss function of 0.1 * x1^2+x2^2
    """

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.mulLayerX1 = MulLayer()
        self.mulLayerX1Coef = MulLayer()
        self.mulLayerX2 = MulLayer()
        self.addLayer = AddLayer()

    def forward(self):
        x1square = self.mulLayerX1.forward(self.x1, self.x1)
        x1squareCoef = self.mulLayerX1Coef.forward(0.1, x1square)
        x2square = self.mulLayerX2.forward(self.x2, self.x2)
        out = self.addLayer.forward(x1squareCoef, x2square)
        return out

    def backward(self, dout):
        daddoutx1Coef, daddoutx2 = self.addLayer.backward(dout)
        dcoef, dx1square = self.mulLayerX1Coef.backward(daddoutx1Coef)
        dx1_first, dx1_second = self.mulLayerX1.backward(dx1square)
        dx1 = dx1_first + dx1_second
        dx2_first, dx2_second = self.mulLayerX2.backward(daddoutx2)
        dx2 = dx2_first + dx2_second
        return dx1, dx2


class SimpleOvalSquareFuncLayer_v2:
    """
    This is simple layer that uses loss function of (x-3)^2+(2y-1)^2
    """

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.addX1_minus_three_layer = AddLayer()
        self.X1_minus_three_square_layer = MulLayer()
        self.mulX2_by_two_layer = MulLayer()
        self.addTwoX2_minus_one_layer = AddLayer()
        self.twoX2_minus_one_squaer_layer = MulLayer()
        self.lastAdd_layer = AddLayer()

    def forward(self):
        x1_minus_three = self.addX1_minus_three_layer.forward(self.x1, -3)
        x1_minus_three_square = self.X1_minus_three_square_layer.forward(x1_minus_three, x1_minus_three)
        x2_mul_by_two = self.mulX2_by_two_layer.forward(self.x2, 2)
        two_x2_minus_one = self.addTwoX2_minus_one_layer.forward(x2_mul_by_two, -1)
        two_x2_minus_one_square = self.twoX2_minus_one_squaer_layer.forward(two_x2_minus_one, two_x2_minus_one)
        out = self.lastAdd_layer.forward(x1_minus_three_square, two_x2_minus_one_square)
        return out

    def backward(self, dout):
        dx1_minus_three_square, dtwo_x2_minus_one_square = self.lastAdd_layer.backward(dout)
        dtwo_x2_minus_one, dtwo_x2_minus_one = self.twoX2_minus_one_squaer_layer.backward(dtwo_x2_minus_one_square)
        dx2_mul_by_two, d_minus_one = self.addTwoX2_minus_one_layer.backward(dtwo_x2_minus_one)
        dx2, d_two = self.mulX2_by_two_layer.backward(dx2_mul_by_two)
        dx2 = 2 * dx2

        dx1_minus_three, dx1_minus_three = self.X1_minus_three_square_layer.backward(dx1_minus_three_square)
        dx1, d_minus_three = self.addX1_minus_three_layer.backward(dx1_minus_three)
        dx1 = 2 * dx1
        return dx1, dx2


if __name__ == '__main__':
    x1, x2 = 1, 2
    simpleSquareFunc = SimpleSquareFuncLayer(x1, x2)
    out = simpleSquareFunc.forward()
    dx1, dx2 = simpleSquareFunc.backward(1)
    print(f"simpleSquare func result : {out}, dx1,dx2 : {dx1, dx2}")
