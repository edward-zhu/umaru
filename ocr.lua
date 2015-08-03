require 'nn'
require 'rnn'
require 'cltorch'
require 'clnn'

require 'image'

local im = image.load("1.png", 1)[1]:t()

local size = im:size()
local im_w = size[1]
local im_h = size[2]

local input_size = im_h
local hidden_size = 400
local class_num = 10


local net = nn.Sequential()

im = im:cl()

net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.LSTM(input_size, hidden_size)))
output = nn.Sequential()
output:add(nn.Linear(hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
output:cl()
net:add(nn.Sequencer(output))

print(net:forward(im))



