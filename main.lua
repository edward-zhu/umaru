require 'nn'
require 'rnn'
require 'image'

require 'ctc_log'

mnist = require 'mnist'



-- local sample = mnist.traindataset()[1]

-- local im = sample.x:double():t()

local im = image.load("1.png", 1)[1]:t()

local target = "123"

local size = im:size()
local im_w = size[1]
local im_h = size[2]

local input_size = im_h
local hidden_size = 10
local class_num = 10

local net = nn.Sequential()

net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.LSTM(input_size, hidden_size)))


output = nn.Sequential()
output:add(nn.Linear(hidden_size * 2, class_num + 1))
output:add(nn.LogSoftMax())
net:add(nn.Sequencer(output))

outputTable = net:forward(im)

ctc.getCTCCostAndGrad(outputTable, target)


