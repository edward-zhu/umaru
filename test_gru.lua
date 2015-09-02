require 'nn'
require 'nngraph'
require 'rnn'
require 'GRU'
GRU = require 'GRU_char-rnn'

local myGRU = nn.GRU(10, 20)

local module = myGRU.recurrentModule

local input = torch.rand(10)
local output_ = torch.rand(20)

print(module)
print(module:forward({input, output_}))

gru = GRU.gru(10, 20, 1)cl

graph.dot(gru.fg, "GRU")

print(gru:forward({input, output_})[1])