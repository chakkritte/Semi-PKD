from PKD.models.backbone import *
from ptflops import get_model_complexity_info
import logging

def build_model(name, args):
    if name == "eeeac2":
        return EEEAC2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "eeeac1":
        return EEEAC1(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "mbv2":
        return MobileNetV2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "mbv3":
        return MobileNetV3(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "efb0":
        return EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "efb4":
        return EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "efb7":
        return EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "ghostnet":
        return GhostNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "rest":
        return ResT(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "pnas":
        return PNASModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
    elif name == "vgg":
        return VGGModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
    elif name == "tresnet":
        return tresnet(num_channels=3, train_enc=True, load_weight=1, pretrained='1k', output_size=args.output_size)
    elif name == "ofa595":
        return OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
    elif name == "resnest":
        return ResNestModel(num_channels=3, train_enc=True, load_weight=1, model='resnest50', output_size=args.output_size)
    elif name == "densenet":
        return DenseModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
    elif name == "mobilevitv2":
        return MobileViTv2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)

def cal_flops_params(model, args):
    macs, params = get_model_complexity_info(model, (3, 288, 384), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
    logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))