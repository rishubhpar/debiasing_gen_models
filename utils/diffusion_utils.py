import numpy as np
import torch
import image_classifier


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs = x_shape[0]
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    out = torch.full((bs, 1, 1, 1), out.item(), device=t.device)
    return out



def denoising_step(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   sampling_type='ddim',
                   eta=0.0,
                   learn_sigma=False,
                   sample = True,
                   male = 1,
                   eyeglasses = 1,
                   scale = [1500],
                   timestep_list = [0,50],
                #    usefancy = False,
                #    gamma_factor = 0.1,
                #    guidance_loss = 'chi_without_5',
                   attribute_list = [0,1,0,0],
                   vanilla_generation = False,
                   t_edit=0,
                   hs_coeff=(1.0),
                   delta_h=None,
                   use_mask=False,
                   dt_lambda=1,
                   ignore_timestep=False,
                   image_space_noise=0,
                   universal_guidance=False,
                   dt_end = 999,
                   warigari=False
                   ):

    # Compute noise and variance
    model = models

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)

    # et, et_modified, delta_h, middle_h = model(xt, t, sample=sample, male = male, eyeglasses=eyeglasses, scale=scale, index=index, t_edit=t_edit, hs_coeff=hs_coeff, delta_h=delta_h, ignore_timestep=ignore_timestep, use_mask=use_mask)
    et, et_modified, delta_h, middle_h = model(xt, t, sample=sample, timestep_list = timestep_list, male = male, eyeglasses=eyeglasses, scale=scale, attribute_list=attribute_list, vanilla_generation=vanilla_generation, bt = bt[0].item(), t_edit=t_edit, hs_coeff=hs_coeff, delta_h=delta_h, ignore_timestep=ignore_timestep, use_mask=use_mask)



    if learn_sigma:
        et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
        # if index is not None:
        #     et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
        logvar = logvar_learned
    # else:
    #     logvar = extract(logvars, t, xt.shape)

    # if type(image_space_noise) != int:
    #     print("-----------inside")
    #     if t[0] >= t_edit:
    #         index = 0
    #         if type(image_space_noise) == torch.nn.parameter.Parameter:
    #             et_modified = et + image_space_noise * hs_coeff[1]
    #         else:
    #             # 
    #             # (type(image_space_noise))
    #             temb = models.module.get_temb(t)
    #             et_modified = et + image_space_noise(et, temb) * 0.01

    # # Compute the next x
    # bt = extract(b, t, xt.shape)
    # at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)
    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

    xt_next = torch.zeros_like(xt)

    # Universal guidance
    if universal_guidance:
        gradients = image_classifier.get_image(et, xt, at, male, scale, attribute_list)
        et = et.detach() + (1 - at).sqrt()*gradients


    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        noise = torch.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':

        # if index is not None:
        #     x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
        # else:
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Deterministic.
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            # noise = torch.randn_like(xt[0:1])
            # print(noise.shape)
            # noise = noise.repeat(xt_next.shape[0],1,1,1)
            # xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * noise
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    if dt_lambda != 1 and t[0] >= dt_end:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

    # # Asyrp & DiffStyle
    # if not warigari or index is None:
    #     return xt_next, x0_t, delta_h, middle_h

    # # Warigari by young-hyun, Not in the paper
    # else:
    #     # will be updated
    return xt_next, x0_t, delta_h, middle_h