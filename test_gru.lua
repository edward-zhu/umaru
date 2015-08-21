require 'nn'
require 'rnn'
require 'GRU'

local GRU = nn.GRU(10, 20)

local module = GRU.recurrentModule

local input = torch.rand(10)
local output_ = torch.rand(20)

print(module)
print(module:forward({input, output_}))