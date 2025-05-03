import matgl, torch

class MGNetModel(torch.nn.Module): 
    def __init__(self, bmodel):
        super(MGNetModel, self).__init__()
        self.bmodel = bmodel

    def forward(self, xs):
        ret = []
        for x in xs:
            x = self.bmodel.predict_structure(x)
            ret.append(x)
        return torch.tensor(ret)

def load_model():
    print("*** loading model")
    omodel = matgl.load_model("megnet/model")
    return MGNetModel(omodel)


if __name__ == "__main__":
    mod = load_model()