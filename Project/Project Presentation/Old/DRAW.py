import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal

class DRAW(nn.Module):

    def __init__(self):
        super(DRAW, self).__init__()

        # Parameters setting:
        self.input_image_size = 28 * 28;
        self.features = 100

        self.T = 3

        self.h_dec_0 = torch.zeros(1, self.features)
        self.c0 = torch.zeros(1, self.input_image_size)

        # self.W = torch.randn((self.input_image_size, self.features)).sigmoid_()

        # DRAW network structure
        #
        # Encoder:
        self.encode_lstm = nn.LSTMCell((self.input_image_size + self.features), self.features)

        self.linear_mu = nn.Linear(self.features, self.features)
        self.linear_sigma = nn.Linear(self.features, self.features)

        # Decoder:
        self.decode_lstm = nn.LSTMCell(self.features, self.input_image_size)

        self.linear_write = nn.Linear(self.input_image_size, self.input_image_size)

    def encoder(self, x, c, h_dec_prev):

        x = x - c.sigmoid_()

        h_e, k_e = self.encode_lstm(torch.cat((x, h_dec_prev), 3), ())

        mu = self.linear_mu(h_e)
        sigma = self.linear_sigma(h_e)

        z = h_e.new(h_e.size()).normal_().mul(sigma.exp_()).add(mu)

        return z, mu, sigma

    def decoder(self, z, c):

        h_dec, mu_t, sigma_t = self.decode_lstm(z)

        c_ = c.add(self.linear_write(h_dec))

        return c_, h_dec

    def forward(self, x):

        z, mu_t, sigma_t = self.encoder(x, self.c0, self.h_dec_0)
        c_t, h_t = self.decoder(z, self.c0)

        mu_t = torch.sum(mu_t.pow(2))
        sigma_t = torch.sum(sigma_t.pow(2)) - torch.log(torch.sum(sigma_t.pow(2)))

        for i in range(1, T):

            z, m_, s_ = self.encoder(x, c_t, h_t)
            c_t, h_t = self.decoder(z, c_t)

            mu_t = mu_t + torch.sum(m_.pow(2))
            sigma_t = sigma_t + torch.sum(s_.pow(2)) - torch.log(torch.sum(s_.pow(2)))

        return c_t, mu_t, sigma_t
