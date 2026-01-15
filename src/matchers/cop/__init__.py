from .cop_module.utils.full_config import full_default_cfg
from .cop_module.utils.opt_config import opt_default_cfg

def reparameter(matcher):
    module = matcher.backbone.fine_backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.fine_backbone.layer1, matcher.backbone.fine_backbone.layer2, matcher.backbone.fine_backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher
