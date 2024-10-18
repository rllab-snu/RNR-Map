from src.models.autoencoder.gsn.generator import SceneGenerator, NerfStyleGenerator, Render
from src.models.autoencoder.gsn.common.encoder import Encoder
from src.models.autoencoder.gsn.common.model_utils import RenderParams
import torch
from src.models.autoencoder.autoenc_old import Embedder as Embedder_old
class Embedder(Embedder_old):
    def __init__(self, pretrained_ckpt='pretrained/autoenc_large.ckpt',
                 img_res=128, w_size=128, coordinate_scale=32, w_ch=32, nerf_res=64, sample_size=None, voxel_res=128, use_rgb=True, use_depth=True):
        super(Embedder_old, self).__init__()
        self.img_res = img_res
        self.sample_size = img_res//2 if sample_size is None else sample_size
        self.w_size, self.w_ch = w_size, w_ch
        self.coordinate_scale = coordinate_scale
        self.nerf_res = nerf_res
        self.voxel_res = voxel_res
        self.z_dim = w_ch

        self.nerf_mlp = NerfStyleGenerator(n_layers=8, channels=128, out_channel=128, z_dim=self.z_dim, omega_coord=10, omega_dir=4, skips=[4])
        self.generator_params = RenderParams(Rt=None, K=None, samples_per_ray=64, near=0, far=8, alpha_noise_std=0, nerf_out_res=self.nerf_res)
        self.generator = SceneGenerator(local_generator=self.nerf_mlp, img_res=self.img_res, feature_nerf=True, global_feat_res=self.voxel_res,
                                        coordinate_scale=self.coordinate_scale, alpha_activation='softplus', hierarchical_sampling=False, density_bias=0)
        self.texture_net = Render(in_channel=128)
        self.use_rgb, self.use_depth = use_rgb, use_depth
        in_ch = 3 * use_rgb + 1 * use_depth
        self.encoder = Encoder(in_channel=in_ch, in_res=self.img_res, out_res=self.sample_size, ch_mul=64, ch_max=128, lasts_ch=self.w_ch)

        if pretrained_ckpt:
            ckpt = torch.load(pretrained_ckpt)
            sd = self.state_dict()
            for k,v in ckpt['state_dict'].items():
                if k in sd and (v.shape == sd[k].shape):
                    sd[k] = v
                elif k in sd:
                    print(k, 'in state_dict, but different size! ', v.shape ,'->', sd[k].shape)
                    continue
                else:
                    print(k, 'not in state_dict')
                    continue
            self.load_state_dict(sd)
