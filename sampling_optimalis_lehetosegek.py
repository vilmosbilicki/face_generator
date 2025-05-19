#ide írok pár függvényt opciónak samplingre
import torch
@torch.no_grad()
def ddim_sample(model, x, steps=100, eta=0.0):
    """
    DDIM sampling with adjustable steps and stochasticity parameter eta.
    When eta=0.0, it's deterministic; when eta=1.0, it's equivalent to DDPM.
    """
    # Calculate timestep schedule for DDIM
    skip = T // steps
    seq = range(0, T, skip)
    
    for i in tqdm(reversed(range(0, steps))):
        current_t = torch.full((x.shape[0],), seq[i], device=x.device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, current_t)
        
        # Get alpha values for current timestep
        alpha_cumprod_t = get_index_from_list(alphas_cumprod, current_t, x.shape)
        
        # Compute alpha values for next timestep (or the final result)
        next_t = current_t - skip
        if i > 0:
            next_alpha_cumprod = get_index_from_list(alphas_cumprod, next_t, x.shape)
        else:
            next_alpha_cumprod = torch.ones_like(alpha_cumprod_t)
            
        # Extract original x_0 from noisy image
        x_0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Direction to the next point
        direction = torch.sqrt(1 - next_alpha_cumprod) * predicted_noise
        
        # Add noise if eta > 0 (stochastic sampling)
        noise = torch.randn_like(x) if eta > 0 else 0
        noise_strength = eta * torch.sqrt((1 - next_alpha_cumprod) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / next_alpha_cumprod))
        
        # Compute the next noisy sample
        x = torch.sqrt(next_alpha_cumprod) * x_0_pred + direction + noise_strength * noise
        
    return x


