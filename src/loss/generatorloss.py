"""GENERATOR_LOSS"""

import mindspore.nn as nn


class GeneratorLoss(nn.Cell):
    """Loss for rbpn
    use GeneratorLoss to measure the L1Loss
    Args:
        generator: use the generator to rebuild sr image
    Outputs:
        Tensor
    """

    def __init__(self, generator):
        super(GeneratorLoss, self).__init__()
        self.generator = generator
        self.criterion = nn.L1Loss(reduction='mean')


    def construct(self, target, x, neighbor, flow):
        """compute l1loss loss between prediction and target """
        prediction = self.generator(x, neighbor, flow)
        content_loss = self.criterion(prediction, target)
        return content_loss
