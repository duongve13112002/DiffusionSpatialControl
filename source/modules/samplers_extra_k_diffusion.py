import torch
import tqdm
import k_diffusion.sampling
from k_diffusion.sampling import default_noise_sampler,to_d, get_sigmas_karras
from tqdm.auto import trange
@torch.no_grad()
def restart_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., restart_list=None):
    """Implements restart sampling in Restart Sampling for Improving Generative Processes (2023)
    Restart_list format: {min_sigma: [ restart_steps, restart_times, max_sigma]}
    If restart_list is None: will choose restart_list automatically, otherwise will use the given restart_list
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': step_id, 'sigma': new_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(), device=sigmas.device)
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(), device=sigmas.device)[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend([(old_sigma, new_sigma) for (old_sigma, new_sigma) in zip(sigma_restart[:-1], sigma_restart[1:])])

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            x = x + k_diffusion.sampling.torch.randn_like(x) * s_noise * (old_sigma ** 2 - last_sigma ** 2) ** 0.5
        x = heun_step(x, old_sigma, new_sigma)
        last_sigma = new_sigma

    return x


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)


@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
    return x

@torch.no_grad()
def sample_heunpp2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    # From MIT licensed: https://github.com/Carzit/sd-webui-samplers-scheduler/
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    s_end = sigmas[-1]
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == s_end:
            # Euler method
            x = x + d * dt
        elif sigmas[i + 2] == s_end:

            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)

            w = 2 * sigmas[0]
            w2 = sigmas[i+1]/w
            w1 = 1 - w2

            d_prime = d * w1 + d_2 * w2


            x = x + d_prime * dt

        else:
            # Heun++
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            dt_2 = sigmas[i + 2] - sigmas[i + 1]

            x_3 = x_2 + d_2 * dt_2
            denoised_3 = model(x_3, sigmas[i + 2] * s_in, **extra_args)
            d_3 = to_d(x_3, sigmas[i + 2], denoised_3)

            w = 3 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w3 = sigmas[i + 2] / w
            w1 = 1 - w2 - w3

            d_prime = w1 * d + w2 * d_2 + w3 * d_3
            x = x + d_prime * dt
    return x