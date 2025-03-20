import torch
from models import faster_rcnn

def main():
    inputs = torch.randn((2, 3, 800, 608), dtype=torch.float32)
    model = faster_rcnn.FasterRCNN(inputs.shape)
    outputs = model(inputs)
    print(outputs[0].shape)  # [2, 1000, 4]
    print(outputs[1].shape)  # [2, 1000, 21]
    print(outputs[1])



if __name__ == '__main__':
    main()