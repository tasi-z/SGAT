import torch

from ..utils.util import center_padding, tokens_to_output,get_self_similarity,draw_torch_image,get_matchable_map,draw_patch_match,get_max_cos_similarity
import time
import torch.nn.functional as F
from torchvision import transforms

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vitb14",
        output="dense",
        layer=-1,
        return_multilayer=False,
        _float16=True,
    ):  
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }
        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"weights/{dino_name}",self.checkpoint_name,source='local')
        # dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self._float16 = _float16
        if self._float16:
            self.vit = dino_vit.eval().to(torch.float16)
        else:
            self.vit = dino_vit.eval().to(torch.float32)
        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward_viewinout(self,view0,view1,up_scale=None):
        if self._float16:
            # rgb_0=(view0["image"]*255).to(torch.float16)
            # rgb_1=(view1["image"]*255).to(torch.float16)
            rgb_0=view0["image"].to(torch.float16)
            rgb_1=view1["image"].to(torch.float16)
        else:
            # rgb_0=(view0["image"]*255)
            # rgb_1=(view1["image"]*255)
            rgb_0=view0["image"]
            rgb_1=view1["image"]
        
        b,c,img0_target_h,img0_target_w=rgb_0.shape
        b,c,img1_target_h,img1_target_w=rgb_1.shape
        if up_scale is not None:
            target_size_0 = (int(img0_target_h * up_scale), int(img0_target_w * up_scale))
            target_size_1 = (int(img1_target_h * up_scale), int(img1_target_w * up_scale))
            rgb_0 = F.interpolate(rgb_0, size=target_size_0, mode='bilinear', align_corners=False)
            rgb_1 = F.interpolate(rgb_1, size=target_size_1, mode='bilinear', align_corners=False)
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize=transforms.Normalize(mean=0.5, std=0.2)
        rgb_0 = normalize(rgb_0)
        rgb_1 = normalize(rgb_1)
        feat_0=self._forward(rgb_0).to(torch.float32)
        feat_1=self._forward(rgb_1).to(torch.float32)

        max_sim_map_list0=[]
        max_sim_map_list1=[]
        max_cos_sim_map_list0=[]
        max_cos_sim_map_list1=[]

        range_list=[16,32,64]
        # range_list=[8,16,32,64]
        # range_list=[64]
        for p in range_list:
            # time_start = time.time()
            feats0 = torch.nn.functional.interpolate(feat_0, (int(img0_target_h/p), int(img0_target_w/p)), mode='bilinear', align_corners=True)
            feats1 = torch.nn.functional.interpolate(feat_1, (int(img1_target_h/p), int(img1_target_w/p)), mode='bilinear', align_corners=True)
            max_sim_map0=get_self_similarity(feats0)
            max_sim_map0_up=torch.nn.functional.interpolate(max_sim_map0, (img0_target_h, img0_target_w), mode='bilinear', align_corners=True)
            # B,1,H,W
            max_sim_map_list0.append(max_sim_map0_up)
            # time_start = time.time()
            max_sim_map1=get_self_similarity(feats1)
            max_sim_map1_up=torch.nn.functional.interpolate(max_sim_map1, (img1_target_h, img1_target_w), mode='bilinear', align_corners=True)
            max_sim_map_list1.append(max_sim_map1_up)
            max_cos_sim_map0,max_cos_sim_map1=get_max_cos_similarity(feats0,feats1)
            max_cos_sim_map0_up=torch.nn.functional.interpolate(max_cos_sim_map0, (img0_target_h, img0_target_w), mode='bilinear', align_corners=True)
            max_cos_sim_map1_up=torch.nn.functional.interpolate(max_cos_sim_map1, (img1_target_h, img1_target_w), mode='bilinear', align_corners=True)
            max_cos_sim_map_list0.append(max_cos_sim_map0_up)
            max_cos_sim_map_list1.append(max_cos_sim_map1_up)

        ######求平均
        ave_max_sim_map0=torch.zeros(max_sim_map_list0[0].shape).cuda()
        for max_sim_map in max_sim_map_list0:
            ave_max_sim_map0+=max_sim_map
        ave_max_sim_map0=ave_max_sim_map0/len(max_sim_map_list0)

        ave_max_sim_map1=torch.zeros(max_sim_map_list1[0].shape).cuda()
        for max_sim_map in max_sim_map_list1:
            ave_max_sim_map1+=max_sim_map
        ave_max_sim_map1=ave_max_sim_map1/len(max_sim_map_list1)

        ave_max_cos_sim_map0=torch.zeros(max_cos_sim_map_list0[0].shape).cuda()
        for max_cos_sim_map in max_cos_sim_map_list0:
            ave_max_cos_sim_map0+=max_cos_sim_map
        ave_max_cos_sim_map0=ave_max_cos_sim_map0/len(max_cos_sim_map_list0)

        ave_max_cos_sim_map1=torch.zeros(max_cos_sim_map_list1[0].shape).cuda()
        for max_cos_sim_map in max_cos_sim_map_list1:
            ave_max_cos_sim_map1+=max_cos_sim_map
        ave_max_cos_sim_map1=ave_max_cos_sim_map1/len(max_cos_sim_map_list1)
        
        # 反向
        ave_max_sim_map0=1-ave_max_sim_map0
        ave_max_sim_map1=1-ave_max_sim_map1

        self_cos_map0=ave_max_sim_map0*ave_max_cos_sim_map0
        self_cos_map1=ave_max_sim_map1*ave_max_cos_sim_map1
        return self_cos_map0,self_cos_map1,ave_max_sim_map0,ave_max_sim_map1,ave_max_cos_sim_map0,ave_max_cos_sim_map1
    
    def forward(self,view0,view1,type="specMask",up_scale=None):
        if self._float16:
            # rgb_0=(view0["image"]*255).to(torch.float16)
            # rgb_1=(view1["image"]*255).to(torch.float16)
            rgb_0=view0["image"].to(torch.float16)
            rgb_1=view1["image"].to(torch.float16)
        else:
            # rgb_0=(view0["image"]*255)
            # rgb_1=(view1["image"]*255)
            rgb_0=view0["image"]
            rgb_1=view1["image"]
        
        b,c,img0_target_h,img0_target_w=rgb_0.shape
        b,c,img1_target_h,img1_target_w=rgb_1.shape
        if up_scale is not None:
            target_size_0 = (int(img0_target_h * up_scale), int(img0_target_w * up_scale))
            target_size_1 = (int(img1_target_h * up_scale), int(img1_target_w * up_scale))
            rgb_0 = F.interpolate(rgb_0, size=target_size_0, mode='bilinear', align_corners=False)
            rgb_1 = F.interpolate(rgb_1, size=target_size_1, mode='bilinear', align_corners=False)
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize=transforms.Normalize(mean=0.5, std=0.2)
        rgb_0 = normalize(rgb_0)
        rgb_1 = normalize(rgb_1)
        feat_0=self._forward(rgb_0).to(torch.float32)
        feat_1=self._forward(rgb_1).to(torch.float32)
        # draw_torch_image(rgb_0,title="view0",type="rgb",save_dir="view0.png")
        # draw_torch_image(feat_0,title="feat_0",type="cmap",save_dir="feat_0.png")
        # draw_torch_image(feat_1,title="feat_1",type="cmap",save_dir="feat_1.png")

        if type=="feat":
            return feat_0,feat_1

        max_similarity_map_list0=[]
        max_similarity_map_list1=[]
        matchable_map_list0=[]
        matchable_map_list1=[]
        range_list=[16,32,64]
        # range_list=[8,16,32,64]
        # range_list=[64]
        for p in range_list:
            # time_start = time.time()
            feats0 = torch.nn.functional.interpolate(feat_0, (int(img0_target_h/p), int(img0_target_w/p)), mode='bilinear', align_corners=True)
            feats1 = torch.nn.functional.interpolate(feat_1, (int(img1_target_h/p), int(img1_target_w/p)), mode='bilinear', align_corners=True)
            max_similarity_map0=get_self_similarity(feats0)
            max_similarity_map0_up=torch.nn.functional.interpolate(max_similarity_map0, (img0_target_h, img0_target_w), mode='bilinear', align_corners=True)
            # B,1,H,W
            max_similarity_map_list0.append(max_similarity_map0_up)
            # time_start = time.time()
            max_similarity_map1=get_self_similarity(feats1)
            max_similarity_map1_up=torch.nn.functional.interpolate(max_similarity_map1, (img1_target_h, img1_target_w), mode='bilinear', align_corners=True)
            max_similarity_map_list1.append(max_similarity_map1_up)
        # 可匹配性
        # TODO: 修正可匹配性的计算
        matchable_map0,matchable_map1,idx_0to1_consistent=get_matchable_map(feat_0,feat_1)
        # draw_patch_match(rgb_0,rgb_1,idx_0to1_consistent,matchable_map0,matchable_map1)
        matchable_map0_up=torch.nn.functional.interpolate(matchable_map0, (img0_target_h, img0_target_w), mode='bilinear', align_corners=True)
        matchable_map_list0.append(matchable_map0_up)
        matchable_map1_up=torch.nn.functional.interpolate(matchable_map1, (img1_target_h, img1_target_w), mode='bilinear', align_corners=True)
        matchable_map_list1.append(matchable_map1_up)
            

        ######求平均
        ave_max_similarity_map0=torch.zeros(max_similarity_map_list0[0].shape).cuda()
        for max_similarity_map in max_similarity_map_list0:
            ave_max_similarity_map0+=max_similarity_map
        ave_max_similarity_map0=ave_max_similarity_map0/len(max_similarity_map_list0)

        ave_max_similarity_map1=torch.zeros(max_similarity_map_list1[0].shape).cuda()
        for max_similarity_map in max_similarity_map_list1:
            ave_max_similarity_map1+=max_similarity_map
        ave_max_similarity_map1=ave_max_similarity_map1/len(max_similarity_map_list1)
        #########图0
        ave_matchable_map0=torch.zeros(matchable_map_list0[0].shape).cuda()
        for matchable_map in matchable_map_list0:
            ave_matchable_map0+=matchable_map
        ave_matchable_map0=ave_matchable_map0/len(matchable_map_list0)
        #########图1
        ave_matchable_map1=torch.zeros(matchable_map_list1[0].shape).cuda()
        for matchable_map in matchable_map_list1:
            ave_matchable_map1+=matchable_map
        ave_matchable_map1=ave_matchable_map1/len(matchable_map_list1)
        # 反向
        ave_max_similarity_map0=1-ave_max_similarity_map0
        ave_max_similarity_map1=1-ave_max_similarity_map1

        return ave_max_similarity_map0,ave_max_similarity_map1,ave_matchable_map0,ave_matchable_map1
    def _forward(self, images):

        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        # features_dict = self.vit.forward_features(images)
        # features = features_dict['x_norm_patchtokens']
        # return features.view(features.size(0),features.size(-1), h,w)
        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
        
