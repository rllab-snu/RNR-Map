import torch.nn as nn
from src.models.autoencoder.gsn.generator import SceneGenerator, NerfStyleGenerator, RenderNet2d
from src.models.autoencoder.gsn.common.encoder import Encoder
from src.models.autoencoder.gsn.common.model_utils import RenderParams
from src.models.autoencoder.gsn.common.nerf_utils import get_ray_bundle_batch
import torch
from einops import rearrange, repeat
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class Embedder(nn.Module):
    def __init__(self, pretrained_ckpt='pretrained/autoenc_small.ckpt', img_res=64, sample_size=None, w_size=128, coordinate_scale=32, w_ch=32, nerf_res=32, voxel_res=128, use_rgb=True, use_depth=True):
        super().__init__()
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
        self.texture_net = RenderNet2d(in_channel=128, in_res=self.sample_size, out_res=self.img_res, mode='blur', deep=False)
        self.use_rgb, self.use_depth = use_rgb, use_depth
        in_ch = 3 * use_rgb + 1 * use_depth
        self.encoder = Encoder(in_channel=in_ch, in_res=self.img_res, out_res=self.sample_size, ch_mul=64, ch_max=128, lasts_ch=self.w_ch)

        if pretrained_ckpt:
            ckpt = torch.load(pretrained_ckpt)#['state_dict']
            sd = self.state_dict()
            for k,v in ckpt.items():
                if k in sd and (v.shape == sd[k].shape):
                    sd[k] = v
                elif k in sd:
                    print(k, 'in state_dict, but different size! ', v.shape ,'->', sd[k].shape)
                    continue
                else:
                    print(k, 'not in state_dict')
                    continue
            self.load_state_dict(sd)

    def embed_obs(self, inputs, T=None, past_w=None, past_w_num_mask=None):
        B = inputs['rgb'].shape[0]
        if T is None:
            T = inputs['rgb'].shape[1]
        w = torch.zeros((B, self.w_ch, self.w_size, self.w_size)).to(inputs['rgb'].device)
        w = w.view(B, self.w_ch, -1)

        ts = list(range(T))
        T = len(ts)
        nerf_out_res = self.sample_size
        images = []
        if self.use_rgb:
            rgb = rearrange(inputs['rgb'][:,ts].float(), 'b t c h w -> (b t) c h w')
            images.append(rgb)
        if self.use_depth:
            depth = rearrange(inputs['depth'][:,ts].float(), 'b t c h w -> (b t) c h w')
            images.append(depth)
        image = torch.cat((images), 1).float()
        w_part = self.encoder(image).permute(0,2,3,1).reshape(B*T,nerf_out_res*nerf_out_res,-1)
        w_part = F.normalize(w_part, dim=-1)

        sorted_indices = rearrange(inputs['sorted_indices'][:,ts], 'b t c -> (b t) c')
        seq_unique_list = rearrange(inputs['seq_unique_list'][:,ts], 'b t c -> (b t) c' )
        seq_unique_counts = rearrange(inputs['seq_unique_counts'][:,ts], 'b t c -> (b t) c')

        expanded_w = rearrange(w.unsqueeze(1).repeat(1, T, 1,1), 'b t c hw -> (b t) c hw')
        w_num_mask = torch.zeros_like(expanded_w)[:,0]

        gathered_embedding = torch.gather(w_part, dim=1, index=sorted_indices.unsqueeze(-1).long().repeat(1,1,w_part.shape[-1]))

        for bt in range(B * T):
            nonzero_elements = (seq_unique_counts[bt] != 0).sum()
            if nonzero_elements > 0:
                mean_embeddings = []
                nums = []
                unique_counts = seq_unique_counts[bt][:nonzero_elements].int().tolist()
                splitted_gathered_embedding = gathered_embedding[bt].split(unique_counts)
                for sv, se in zip(seq_unique_list[bt], splitted_gathered_embedding):
                    mean_embeddings.append(se.mean(dim=0))
                    nums.append(len(se))
                mean_embedding = torch.stack(mean_embeddings, dim=-1)
                nums = torch.tensor(nums)
            else:
                continue
            expanded_w[bt, :, seq_unique_list[bt].long()[:nonzero_elements]] = mean_embedding#[:32]
            w_num_mask[bt, seq_unique_list[bt].long()[:nonzero_elements]] = nums.float().to(w_num_mask.device)

        new_w = rearrange(expanded_w, '(b t) c hw -> b t c hw', t=T).mean(dim=1)
        w_num_mask = rearrange(w_num_mask, '(b t) hw -> b t hw', t=T).sum(dim=1)

        new_w = new_w.view(B, self.w_ch, self.w_size, self.w_size)
        w_num_mask = w_num_mask.view(B, 1, self.w_size, self.w_size)

        if past_w is None:
            w = new_w
            new_w_num_mask = w_num_mask
        else:
            new_w_num_mask = w_num_mask.detach() + past_w_num_mask.detach()
            new_w_num_mask[torch.where(new_w_num_mask == 0)] = 1
            w = ((new_w * w_num_mask) + (past_w * past_w_num_mask))/new_w_num_mask

        return w, new_w_num_mask

    def generate(self, w, camera_params, out_res=None):
        # camera_params should be a dict with Rt and K (if Rt is not present it will be sampled)
        nerf_out_res = self.nerf_res if out_res is None else out_res
        samples_per_ray = self.generator_params.samples_per_ray

        # use EMA weights if in eval mode
        # decoder = self.decoder if self.training else self.decoder_ema
        generator = self.generator
        texture_net = self.texture_net

        # duplicate latent codes along the trajectory dimension
        T = camera_params['Rt'].shape[1]  # trajectory length
        w = repeat(w, 'b c h w -> b t c h w', t=T)
        w = rearrange(w, 'b t c h w -> (b t) c h w')

        indices_chunks = [None]
        rgb, depth = [], []
        Rt = rearrange(camera_params['Rt'], 'b t h w -> (b t) h w').clone()
        K = rearrange(camera_params['K'], 'b t h w -> (b t) h w').clone()
        for indices in indices_chunks:
            render_params = RenderParams(
                Rt=Rt,
                K=K,
                samples_per_ray=samples_per_ray,
                near=self.generator_params.near,
                far=self.generator_params.far,
                alpha_noise_std=self.generator_params.alpha_noise_std,
                nerf_out_res=nerf_out_res,
                mask=indices,
            )

            y_hat = generator(local_latents=w, render_params=render_params)

            rgb.append(y_hat['rgb'])  # shape [BT, HW, C]
            depth.append(y_hat['depth'])

        # combine image patches back into full images
        if self.use_rgb:
            rgb = torch.cat(rgb, dim=1)
            rgb_ = rearrange(rgb, 'b (h w) c -> b c h w', h=nerf_out_res, w=nerf_out_res)
            rgb = texture_net(rgb_)
            rgb = rearrange(rgb, '(b t) c h w -> b t c h w', t=T)
        else:
            rgb = None
        depth = torch.cat(depth, dim=1)
        depth = rearrange(depth, '(b t) (h w) -> b t 1 h w', t=T, h=nerf_out_res, w=nerf_out_res)
        return rgb, depth

    def sample_generate(self, w, camera_params, out_res=None, num_samples=200):
        # camera_params should be a dict with Rt and K (if Rt is not present it will be sampled)
        nerf_out_res = self.nerf_res if out_res is None else out_res
        samples_per_ray = self.generator_params.samples_per_ray

        # use EMA weights if in eval mode
        # decoder = self.decoder if self.training else self.decoder_ema
        generator = self.generator
        texture_net = self.texture_net

        # duplicate latent codes along the trajectory dimension
        T = camera_params['Rt'].shape[1]  # trajectory length
        w = repeat(w, 'b c h w -> b t c h w', t=T)
        w = rearrange(w, 'b t c h w -> (b t) c h w')

        Rt = rearrange(camera_params['Rt'], 'b t h w -> (b t) h w').clone()
        K = rearrange(camera_params['K'], 'b t h w -> (b t) h w').clone()
        render_params = RenderParams(
            Rt=Rt,
            K=K,
            samples_per_ray=samples_per_ray,
            near=self.generator_params.near,
            far=self.generator_params.far,
            alpha_noise_std=self.generator_params.alpha_noise_std,
            nerf_out_res=nerf_out_res,
            mask=None,
        )

        y_hat = generator.sample_forward(local_latents=w, render_params=render_params, num_samples=num_samples)

        rgb = texture_net(y_hat['rgb']) if self.use_rgb else None
        depth = y_hat['depth']

        return rgb, depth, y_hat['indices']

    def calculate_mask_func(self, depth, Rt, K):
        w = h = z = self.w_size
        B = len(depth)

        downsampling_ratio = self.sample_size / self.img_res
        fx, fy = K[0, 0, 0] * downsampling_ratio, K[0, 1, 1] * downsampling_ratio
        ray_origins, ray_directions = get_ray_bundle_batch(self.sample_size, self.sample_size, (fx, fy), Rt.inverse())

        if depth.shape[-1] != self.sample_size:
            depth = TF.resize(depth, (self.sample_size, self.sample_size))

        depth_positions = depth.view(B, -1, 1)
        positions_orig = ray_origins.view(B, -1, 3) + ray_directions.view(B, -1, 3) * depth_positions
        positions = torch.clamp(positions_orig / (self.coordinate_scale / 2), -0.99, 0.99)

        obs_ids = list(range(len(depth)))

        scaled_xx = (positions[obs_ids, :, 0] * (w // 2) + w // 2).int()
        scaled_zz = (positions[obs_ids, :, 2] * (h // 2) + h // 2).int()

        pose_for_ordering = scaled_xx * (self.w_size) + scaled_zz
        sorted_pose, sorted_indices = torch.sort(pose_for_ordering)
        all_unique_list = torch.unique(sorted_pose)

        seq_unique_list, seq_unique_counts = [], []
        for i in range(len(sorted_pose)):
            unique_list, unique_counts = torch.unique(sorted_pose[i], return_counts=True)
            padded_unique_list = torch.cat((unique_list, torch.full(size=(len(all_unique_list) - len(unique_list),),
                                                                    fill_value=0.).to(unique_list.device)))
            padded_unique_counts = torch.cat((unique_counts, torch.full(size=(len(all_unique_list) - len(unique_list),),
                                                                        fill_value=0.).to(unique_list.device)))
            seq_unique_list.append(padded_unique_list)
            seq_unique_counts.append(padded_unique_counts)
        seq_unique_list = torch.stack(seq_unique_list)
        seq_unique_counts = torch.stack(seq_unique_counts)

        return sorted_indices, seq_unique_list, seq_unique_counts, None