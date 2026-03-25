# from .loftr import LoFTR
# from .loftr_InV_FC_orth_circle import LoFTR
# from .loftr_InV_FC_orth import LoFTR
# from .loftr_InV_FC import LoFTR
# from .loftr_InV_backfusion import LoFTR
# from .loftr_matchable import LoFTR
# from .loftr_InV_backfusion_vis import LoFTR
# from .loftr_InV_FC_vis import LoFTR
# from .loftr_vis import LoFTR
# from .loftr_orthloss import LoFTR
# from .loftr_lowfre import LoFTR
# from .loftr_highfre import LoFTR
# from .loftr_dino import LoFTR

from .loftr.utils.full_config import full_default_cfg
from .loftr.utils.opt_config import opt_default_cfg

def reparameter(matcher):
    module = matcher.backbone.fine_backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.fine_backbone.layer1, matcher.backbone.fine_backbone.layer2, matcher.backbone.fine_backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    # for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
    #     for module in modules:
    #         if hasattr(module, 'switch_to_deploy'):
    #             module.switch_to_deploy()
    return matcher