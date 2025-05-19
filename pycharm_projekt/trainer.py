import torch
from matplotlib import pyplot as plt
from torch.optim import Adam

from noise_scheduler import get_index_from_list


class Trainer:
    def __init__(self, model, dataset, noise_scheduler, device):
        self.model = model
        self.dataset = dataset
        self.ns = noise_scheduler
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=0.001)

        self.model.to(self.device)

    def train(self):
        epochs = 10  # Try more! #itt erdetileg 100 volt csak az rengeteg ideig fut
        for epoch in range(epochs):
            for step, batch in enumerate(self.dataset.dataloader):
                self.optimizer.zero_grad()

                t = torch.randint(0, self.ns.T, (self.dataset.batch_size,), device=self.device).long()
                loss = self.ns.get_loss(self.model, batch[0], t)
                loss.backward()
                self.optimizer.step()

                if step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    if epoch == 9:
                        self.sample_plot_image()

    @torch.no_grad()
    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = get_index_from_list(self.ns.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.ns.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.ns.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.ns.posterior_variance, t, x.shape)

        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_plot_image(self):
        # Sample noise
        img_size = self.dataset.img_size
        img = torch.randn((1, 3, img_size, img_size), device=self.device)
        plt.figure(figsize=(15, 2))
        plt.axis('off')
        num_images = 4
        stepsize = int(self.ns.T / num_images)

        for i in range(0, self.ns.T)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(img, t)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i / stepsize) + 1)
                self.dataset.show_tensor_image(img.detach().cpu())
        plt.show()