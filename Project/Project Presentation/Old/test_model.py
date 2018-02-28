from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn

from DRAW import DRAW

d = DRAW()

x = torch.randn(1, 28*28)

print(torch.is_tensor(x))
print(d.c0.size())
print(d.h_dec_0 .size())

y = torch.cat((x, d.h_dec_0 ), 1)

print(y.size())

print(d.encode_lstm)

z = Variable(torch.randn(1, 28*28+100))

c0 = Variable(torch.randn(100))
h0 = Variable(torch.randn(100))


h, c = d.decoder(z, (h0, c0))




rnn = nn.LSTMCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
cx = Variable(torch.randn(3, 20))

hx, cx = rnn(input[1], (hx, cx))

# y = d.encoder(Variable(x), Variable(d.c0), Variable(d.h_dec_0))
