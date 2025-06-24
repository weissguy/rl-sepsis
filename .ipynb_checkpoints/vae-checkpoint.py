import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self.model(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x_hat = self.model(x)
        return x_hat


class VAEModel(nn.Module):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
    ):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder(encoder_input, hidden_dim, latent_dim)
        self.Decoder = Decoder(decoder_input, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)

        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encode(self, s1, s2, y):
        s1_ = s1.view(s1.shape[0], s1.shape[1], -1)
        s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
        y = y.reshape(s1.shape[0], s1.shape[1], -1)

        encoder_input = torch.cat([s1_, s2_, y], dim=-1).view(
            s1.shape[0], -1
        )  # Batch x Ann x (2*T*State + 1)
        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var

    def decode(self, obs, z):
        r = torch.cat([obs, z], dim=-1)  # Batch x Ann x T x (State + Z)
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def get_reward(self, r):
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def transform(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = -torch.sum(
                1
                + (log_var - self.log_var)
                - (log_var - self.log_var).exp()
                - (mean - self.mean).pow(2) / (self.log_var.exp())
            )
        else:
            kl = -torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def forward(self, s1, s2, y):  # Batch x Ann x T x State, Batch x Ann x 1

        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None
        z = z.repeat((1, self.annotation_size * self.size_segment)).view(
            -1, self.annotation_size, self.size_segment, z.shape[1]
        )

        r0 = self.decode(s1, z)
        r1 = self.decode(s2, z)

        r_hat1 = r0.sum(axis=2) / self.scaling
        r_hat2 = r1.sum(axis=2) / self.scaling

        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2).view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
        }

        return loss, metrics

    def sample_prior(self, size):
        z = torch.randn(size, self.latent_dim).to(next(self.parameters()).device) # sample a latent variable from the prior
        if self.learned_prior:
            z = z * torch.exp(0.5 * self.log_var) + self.mean
        elif self.flow_prior:
            z, _ = self.flow(z) # TODO what is this?
        return z

    def sample_posterior(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents


class VAEClassifier(VAEModel):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
    ):
        super(VAEClassifier, self).__init__(
            encoder_input,
            decoder_input,
            latent_dim,
            hidden_dim,
            annotation_size,
            size_segment,
            kl_weight,
            learned_prior,
            flow_prior,
            annealer,
            reward_scaling,
        )

    def forward(self, s1, s2, y):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None
        z = z.repeat((1, self.annotation_size * self.size_segment)).view(
            -1, self.annotation_size, self.size_segment, z.shape[1]
        )

        p_hat = self.Decoder(torch.cat([s1, s2, z], dim=-1)).view(-1, 1)
        p_hat = torch.nn.functional.sigmoid(p_hat).view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
        }

        return loss, metrics

    def decode(self, x, y, z):  # B x S, N x S, B x Z
        x = x[:, None].repeat(1, y.shape[0], 1)  # B x N x S
        z = z[:, None].repeat(1, y.shape[0], 1)  # B x N x Z
        y = y[None].repeat(x.shape[0], 1, 1)  # B x N x S
        x = torch.cat([x, y, z], dim=-1)  # B x N x (2S + Z)
        x = torch.nn.functional.sigmoid(self.Decoder(x))  # B x N x 1
        return x[:, :, 0].mean(dim=-1)  # (B, )



### Flows are used to transform the latent space, capturing the distribution better ###



class PlanarFlow(nn.Module):
    def __init__(self, dim):

        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):

        def m(x):
            return F.softplus(x) - 1.0

        def h(x):
            return torch.tanh(x)

        def h_prime(x):
            return 1.0 - h(x) ** 2

        inner = (self.w * self.u).sum()
        u = self.u + (m(inner) - inner) * self.w / self.w.norm() ** 2
        activation = (self.w * x).sum(dim=1, keepdim=True) + self.b
        x = x + u * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1.0 + (u * psi).sum(dim=1, keepdim=True)))

        return x, log_det


class RadialFlow(nn.Module):
    def __init__(self, dim):

        super(RadialFlow, self).__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):

        def m(x):
            return F.softplus(x)

        def h(r):
            return 1.0 / (a + r)

        def h_prime(r):
            return -h(r) ** 2

        a = torch.exp(self.a)
        b = -a + m(self.b)
        r = (x - self.c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - self.c)
        log_det = (self.d - 1) * torch.log(1.0 + tmp) + torch.log(
            1.0 + tmp + b * h_prime(r) * r
        )

        return x, log_det


class HouseholderFlow(nn.Module):
    def __init__(self, dim):

        super(HouseholderFlow, self).__init__()

        self.v = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):

        outer = self.v.t() * self.v
        v_sqr = self.v.norm() ** 2
        H = torch.eye(self.d).cuda() - 2.0 * outer / v_sqr
        x = torch.mm(H, x.t()).t()

        return x, 0


class NiceFlow(nn.Module):
    def __init__(self, dim, mask, final=False):

        super(NiceFlow, self).__init__()

        self.final = final
        if final:
            self.scale = nn.Parameter(torch.zeros(1, dim))
        else:
            self.mask = mask
            self.coupling = nn.Sequential(
                nn.Linear(dim // 2, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim // 2),
            )

    def forward(self, x):
        if self.final:
            x = x * torch.exp(self.scale)
            log_det = torch.sum(self.scale)

            return x, log_det
        else:
            [B, W] = list(x.size())
            x = x.reshape(B, W // 2, 2)

            if self.mask:
                on, off = x[:, :, 0], x[:, :, 1]
            else:
                off, on = x[:, :, 0], x[:, :, 1]

            on = on + self.coupling(off)

            if self.mask:
                x = torch.stack((on, off), dim=2)
            else:
                x = torch.stack((off, on), dim=2)

            return x.reshape(B, W), 0


class Flow(nn.Module):
    def __init__(self, dim, type, length):

        super(Flow, self).__init__()

        if type == "planar":
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])
        elif type == "radial":
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        elif type == "householder":
            self.flow = nn.ModuleList([HouseholderFlow(dim) for _ in range(length)])
        elif type == "nice":
            self.flow = nn.ModuleList(
                [NiceFlow(dim, i // 2, i == (length - 1)) for i in range(length)]
            )
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x):

        [B, _] = list(x.size())
        log_det = torch.zeros(B, 1).cuda()
        for i in range(len(self.flow)):
            x, inc = self.flow[i](x)
            log_det = log_det + inc

        return x, log_det