import torch
import logging
import numpy as np
import torch.nn as nn
from os.path import join
from model.unet import UNet, UNetRecurrent, UNetFire


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        assert('num_bins' in config)
        self.num_bins = int(config['num_bins'])  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True


class FireNet(BaseE2VID):
    """
    Model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
        kernel_size = config.get('kernel_size', 3)
        recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
        head_recurrent = config.get('head_recurrent', True)
        G_res = config.get('G_res', True)
        self.net = UNetFire(self.num_bins,
                            num_output_channels=1,
                            skip_type=self.skip_type,
                            recurrent_block_type=self.recurrent_block_type,
                            base_num_channels=self.base_num_channels,
                            num_residual_blocks=self.num_residual_blocks,
                            norm=self.norm,
                            kernel_size=kernel_size,
                            recurrent_blocks=recurrent_blocks,
                            head_recurrent=head_recurrent,
                            G_res=G_res)

    def forward(self, event_tensor, prev_states):
        img, states = self.net.forward(event_tensor, prev_states)
        return {'image': img, 'state': states}