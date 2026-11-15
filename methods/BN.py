from .TTAOps import BNModel, configure_bn_model


def setup(model, cfg):
    model = configure_bn_model(
        model,
        cfg.method.bn.eps,
        cfg.method.bn.momentum,
        cfg.method.bn.reset_stats,
        cfg.method.bn.no_stats,
    )
    return BNModel(model, steps=cfg.method.steps), {'bn_mode': 'bnupdate'}
