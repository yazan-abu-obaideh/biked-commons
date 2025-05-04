from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np

class Down_Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=400, num_hidden_layers=1):
        super(Down_Model, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class Up_Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=400, num_hidden_layers=1):
        super(Up_Model, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

def diversity_loss(x):
    # Compute pairwise squared Euclidean distances
    r = torch.sum(x ** 2, dim=1, keepdim=True)
    D = r - 2 * torch.matmul(x, x.T) + r.T
    
    # Compute the similarity matrix using RBF
    S = torch.exp(-0.5 * D ** 2)
    
    # Compute the eigenvalues of the similarity matrix
    try:
        eig_val = torch.linalg.eigvalsh(S)
    except:
        eig_val = torch.ones(x.size(0), device=x.device)
    
    # Compute the loss as the negative mean log of the eigenvalues
    loss = -torch.mean(torch.log(torch.clamp(eig_val, min=1e-7)))
    
    return loss

def GAN_step(D, G, D_opt, G_opt, P_batch, cond_batch, noise_batch, batch_size, device, objective_weight=0, diversity_weight=0):
    criterion = nn.BCEWithLogitsLoss()
    D.zero_grad()
    real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

    output = D(P_batch).view(-1)
    L_D_real = criterion(output, real_label)

    fake_data = G(noise_batch)
    output = D(fake_data.detach()).view(-1)
    L_D_fake = criterion(output, fake_label)

    L_D_tot = L_D_real + L_D_fake
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data).view(-1)
    L_G = criterion(output, real_label)

    if diversity_weight > 0:
        _, L_div = diversity_loss(fake_data)
        L_G_tot = L_G + diversity_weight * L_div
    else:
        L_G_tot = L_G
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    report = {"L_D_real": L_D_real.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report


def VAE_step(D, G, D_opt, G_opt, P_batch, cond_batch, noise_batch, batch_size, device, objective_weight=0, diversity_weight=0):
    
    D.zero_grad()
    G.zero_grad()
    
    alpha = 0.2

    encoded = D(P_batch)
    latent_dim = encoded.shape[1] // 2
    mu = encoded[:, :latent_dim]
    logvar = encoded[:, latent_dim:]
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # z = mu + sigma * epsilon
    
    # Forward pass through decoder (G)
    reconstructed = G(z)
    
    # Compute losses
    L_R = nn.MSELoss()(reconstructed, P_batch)
    L_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / P_batch.size(0)

    if diversity_weight > 0:
        L_div = diversity_loss(z)
        L_tot = alpha * L_KL + L_R + diversity_weight * L_div
    else:
        L_div = None
        L_tot = alpha * L_KL + L_R

    L_tot.backward()
    
    D_opt.step()
    G_opt.step()
    
    report = {"L_KL": L_KL.item(), "L_R": L_R.item(), "L_tot": L_tot.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report


class NoiseScheduler:
    def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02, device="cpu"):

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device(device)

        # Linear beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device, dtype=self.betas.dtype), self.alpha_cumprod[:-1]])

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(torch.clamp(1.0 - self.alpha_cumprod, min=1e-8))

    def get_variance(self, t):
        if isinstance(t, torch.Tensor) and t.ndim > 0:  # Batched timesteps
            return torch.index_select(self.betas, 0, t).to(self.device)
        return self.betas[t].to(self.device)  # Single timestep


def DDPM_step_wrapper(scheduler):
    def DDPM_step(D, G, D_opt, G_opt, P_batch, cond_batch, noise_batch, batch_size, device, objective_weight=0, diversity_weight=0):
        t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(P_batch).to(device)

        sqrt_alpha_cumprod_t = scheduler.sqrt_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(alpha_t_bar)
        sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - alpha_t_bar)

        x_t = sqrt_alpha_cumprod_t * P_batch + sqrt_one_minus_alpha_cumprod_t * noise

        t_embedded = t.unsqueeze(-1).float() / scheduler.num_timesteps
        x_input = torch.cat([x_t, t_embedded], dim=-1)

        noise_pred = D(x_input)

        beta_t = scheduler.betas[t].unsqueeze(-1).to(device)  # Variance (beta_t)
        loss_weights = (1 / beta_t) / (1 / beta_t).mean()  # Normalize loss weights
        loss = (loss_weights * nn.MSELoss(reduction="none")(noise_pred, noise)).mean()

        D.zero_grad()
        loss.backward()
        D_opt.step()

        return {"loss": loss.item()}
    return DDPM_step

def DDPM_step_cond_wrapper(scheduler):
    def DDPM_step_cond(D, G, D_opt, G_opt, P_batch, cond_batch, noise_batch, batch_size, device, objective_weight=0, diversity_weight=0):
        P_labels = torch.ones(P_batch.size(0), 1, device=device)  # Class 1 for P_batch
        N_labels = torch.zeros(N_batch.size(0), 1, device=device)  # Class 0 for N_batch
        
        data_batch = torch.cat([P_batch, N_batch], dim=0)
        labels = torch.cat([P_labels, N_labels], dim=0)

        perm = torch.randperm(data_batch.size(0))
        data_batch = data_batch[perm]
        labels = labels[perm]

        t = torch.randint(0, scheduler.num_timesteps, (data_batch.size(0),), device=device)
        noise = torch.randn_like(data_batch).to(device)

        sqrt_alpha_cumprod_t = scheduler.sqrt_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(alpha_t_bar)
        sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - alpha_t_bar)

        x_t = sqrt_alpha_cumprod_t * data_batch + sqrt_one_minus_alpha_cumprod_t * noise

        t_embedded = t.unsqueeze(-1).float() / scheduler.num_timesteps

        x_input = torch.cat([x_t, labels, t_embedded], dim=-1)

        noise_pred = D(x_input)

        beta_t = scheduler.betas[t].unsqueeze(-1).to(device)  # Variance (beta_t)
        loss_weights = (1 / beta_t) / (1 / beta_t).mean()  # Normalize loss weights
        loss = (loss_weights * nn.MSELoss(reduction="none")(noise_pred, noise)).mean()

        # Backpropagation and optimization
        D.zero_grad()
        loss.backward()
        D_opt.step()

        return {"loss": loss.item()}
    return DDPM_step_cond


class ReusableDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.previous_indices = []

    def _shuffle_indices(self):
        self.indices = torch.randperm(len(self.dataset)).tolist()

    def get_batch(self):
        queued = self.previous_indices
        while len(queued) < self.batch_size:
            if self.shuffle:
                self._shuffle_indices()
            queued.extend(self.indices)  # Add individual elements to queued list
        
        self.previous_indices = queued[self.batch_size:]  # Store remaining indices for the next batch
        batch_indices = queued[:self.batch_size]  # Get the batch of the correct size
        return torch.stack([self.dataset[i][0] for i in batch_indices])


def train(D, G, D_opt, G_opt, P_loader, N_loader, num_steps, batch_size, noise_dim, train_step_fn, device, objective_weight, diversity_weight=0):
    # Loss function
    
    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch = P_loader.get_batch().to(device)
        N_batch = N_loader.get_batch().to(device)
        # noise_batch = torch.randn(batch_size, noise_dim).to(device)
        cond_batch = ... #TODO

        report = train_step_fn(D, G, D_opt, G_opt, P_batch, cond_batch, noise_batch, batch_size, device, objective_weight=objective_weight, diversity_weight=diversity_weight)
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)
    return D, G


def get_DDPM_generate_cond(scheduler, data_dim, batch_size=64):
    def DDPM_generate_cond(D, G, cond_batch, latent_dim, device, batch_size=batch_size):
        results = []
        #TODO incorporate guidance and conditioning
        for start_idx in range(0, numgen, batch_size):
            end_idx = min(start_idx + batch_size, numgen)
            current_batch_size = end_idx - start_idx

            x = torch.randn(current_batch_size, data_dim).to(device)

            class_label = torch.ones((current_batch_size, 1), device=device)

            for t in reversed(range(scheduler.num_timesteps)):
                t_embedded = torch.full((current_batch_size, 1), t, device=device).float() / scheduler.num_timesteps
                x_input = torch.cat([x, class_label, t_embedded], dim=-1)

                with torch.no_grad():
                    noise_pred = D(x_input)

                beta_t = scheduler.betas[t].to(device)  # Variance (beta_t)
                alpha_t = scheduler.alphas[t].to(device)  # Current alpha_t (not cumulative)
                sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)  # sqrt(alpha_t)
                sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - cumprod(alpha))

                z = torch.randn_like(x) if t > 0 else 0  # Add noise only if t > 0
                x = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * noise_pred) + torch.sqrt(beta_t) * z

            results.append(x.detach().cpu().numpy())

        return np.concatenate(results, axis=0)
    return DDPM_generate_cond

def get_DDPM_generate_guidance(scheduler, data_dim, guidance_scale=1.0, batch_size=64):
    def DDPM_generate_guidance(D, G, cond_batch, latent_dim, device, batch_size=batch_size):
        results = []

        for start_idx in range(0, numgen, batch_size):
            end_idx = min(start_idx + batch_size, numgen)
            current_batch_size = end_idx - start_idx

            # Start with pure noise for the current batch
            x = torch.randn(current_batch_size, data_dim).to(device)

            # Reverse diffusion process
            for t in reversed(range(scheduler.num_timesteps)):
                t_embedded = torch.full((current_batch_size, 1), t, device=device).float() / scheduler.num_timesteps
                x_input = torch.cat([x, t_embedded], dim=-1)

                with torch.no_grad():
                    noise_pred = D(x_input)

                beta_t = scheduler.betas[t].to(device)  # Variance (beta_t)
                alpha_t = scheduler.alphas[t].to(device)  # Current alpha_t (not cumulative)
                sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)  # sqrt(alpha_t)
                sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - cumprod(alpha))

                # Compute classifier guidance
                x.requires_grad_(True)  # Enable gradient computation for x
                class_prob = A(x).squeeze(-1)  # Binary classifier output P(A=1|x)
                class_grad = torch.autograd.grad(outputs=class_prob.sum(), inputs=x)[0]  # ∇x P(A=1|x)

                # Incorporate classifier guidance
                guided_noise_pred = noise_pred - guidance_scale * class_grad

                # Compute the denoised sample
                z = torch.randn_like(x) if t > 0 else 0  # Add noise only if t > 0
                x = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * guided_noise_pred) + torch.sqrt(beta_t) * z

            results.append(x.detach().cpu().numpy())

        return np.concatenate(results, axis=0)
    return DDPM_generate_guidance


def VAE_generate(D, G, cond_batch, latent_dim, device):
    z = torch.randn(numgen, latent_dim).to(device)
    generated_data = G(z).detach().cpu().numpy()
    return generated_data

def VAE_generate_cond(D, G, cond_batch, numgen, latent_dim, device):
    z = torch.randn(numgen, latent_dim).to(device)
    labels = torch.ones(numgen, 1).to(device)
    z = torch.cat([z, labels], dim=1)
    generated_data = G(z).detach().cpu().numpy()
    return generated_data
    

def GAN_generate(D, G, cond_batch, noise_dim, device):
    noise = torch.randn(numgen, noise_dim).to(device)
    generated_data = G(noise).detach().cpu().numpy()
    return generated_data

def GAN_generate_cond(D, G, cond_batch, noise_dim, device):
    noise = torch.randn(numgen, noise_dim).to(device)
    labels = torch.ones(numgen, 1).to(device)
    noise = torch.cat([noise, labels], dim=1)
    generated_data = G(noise).detach().cpu().numpy()
    return generated_data


def train_model(X, N, Y, C, numgen, numanim, condition, train_params=None, config_params=None, savedir=None):
    batch_size, disc_lr, disc_aux_lr, gen_lr, noise_dim, num_epochs, n_hidden, layer_size= train_params
    aux_setting, validity_weight, diversity_weight = config_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_fn = None

    data_dim = X.shape[1]
    cond_dim = 0

    mode = aux_setting
    if mode.startswith("GAN"):
        
    elif mode.startswith("VAE"):
        
    elif mode.startswith("DDPM"):
        

    if mode in ["GAN"]:
        train_step = GAN_step
        generate_fn = GAN_generate
        D_in = data_dim
        D_out = 1
        G_in = noise_dim
        G_out = data_dim
    elif mode in ["VAE"]:
        train_step = VAE_step
        generate_fn = VAE_generate
        D_in = data_dim
        D_out = 2*noise_dim
        G_in = noise_dim
        G_out = data_dim
    elif mode in ["DDPM_guidance"]:
        train_step = DDPM_step_wrapper(scheduler)
        scheduler = NoiseScheduler(1000, device = device)
        generate_fn = get_DDPM_generate_guidance(scheduler, data_dim, validity_weight, batch_size=batch_size)
        D_in = data_dim + 1
        D_out = data_dim
        G_in = 1 #unused
        G_out = 1 #unused
    elif mode in ["DDPM_conditional"]:
        train_step = DDPM_step_cond_wrapper(scheduler)
        generate_fn = get_DDPM_generate_cond(scheduler, data_dim, batch_size=batch_size)
        scheduler = NoiseScheduler(1000, device = device)
        generate_fn = get_DDPM_generate(scheduler, data_dim, batch_size=batch_size)
        D_in = data_dim + 2
        D_out = data_dim
        G_in = 1 #unused
        G_out = 1 #unused
    else:
        raise ValueError("Invalid mode")


    D = Down_Model(D_in, D_out, layer_size, n_hidden)
    G = Up_Model(G_in, G_out, layer_size, n_hidden)
    A = Down_Model(data_dim, 1, layer_size, n_hidden)

    
    D.to(device)
    G.to(device)
    A.to(device)
    D_opt = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=(0.5,0.999))
    G_opt = torch.optim.Adam(G.parameters(), lr=gen_lr, betas=(0.5,0.999))
    A_opt = torch.optim.Adam(A.parameters(), lr=disc_aux_lr, betas=(0.5,0.999))

    P = torch.tensor(X).float()
    N = torch.tensor(N).float()

    P_loader = ReusableDataLoader(TensorDataset(P), batch_size)
    N_loader = ReusableDataLoader(TensorDataset(N), batch_size)

    if num_epochs>0:
        num_steps = num_epochs*len(P)//batch_size
    else:
        num_steps = -num_epochs #hacky way to specify fixed number of steps rather than epochs

    if pretrain_fn is not None:
        A = pretrain_fn(A, A_opt, P_loader, N_loader, num_steps, batch_size, device)
    
    cond_loader = ... 

    train(D, G, D_opt, G_opt, P_loader, N_loader, num_steps, batch_size, noise_dim, train_step, device, validity_weight, diversity_weight)

    generated_data = generate_fn(D, G, A, numgen, noise_dim, device)
    return [generated_data], [num_epochs]

def train_wrapper(train_params=None, config_params=None):
    def model(X, N, Y=None, C=None, numgen=None, numanim=None, condition=None, savedir=None):
        return train_model(X, N, Y, C, numgen=numgen, numanim=numanim, condition=condition, train_params=train_params, config_params=config_params, savedir=savedir)
    return model