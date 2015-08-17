require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'loader'
require 'ctc_log'
require 'utils/decoder'

mnist = require 'mnist'

DROPOUT_RATE = 0.4

local input_size = 64
local hidden_size = 100
local class_num = 10

local net = nn.Sequential()

net:add(nn.Dropout(DROPOUT_RATE))
net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.LSTM(input_size, hidden_size)))

output = nn.Sequential()
output:add(nn.Linear(hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
net:add(nn.Sequencer(output))

-- net:remember('both')

torch.manualSeed(450)

params, grad_params = net:getParameters()

state = {
	learningRate = 1e-3,
	momentum = 0.5
}

loader = Loader()
loader:load("1.txt")
codec = loader:codec()

local sample = loader:pick()
local im = sample.img
local target = codec:encode(sample.gt)

raw = image.load(sample.src, 1)

print(raw[1])

print(im)

--[[
for i = 1, 100000 do
	local sample = loader:pick()
	local im = sample.img
	local target = codec:encode(sample.gt)

	print(im)
	
	local feval = function(params)
		net:forget()
	
		outputTable = net:forward(im)
	
		loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
	
		if i % 20 == 0 then
			print(sample.gt)
			print(decoder.best_path_decode(outputTable))
			print(loss)
		end
	
		-- net:zeroGradParameters()
	
		net:backward(im, grad)
		
		grad_params:cmul(torch.eq(grad_params, grad_params):double())
	
		return loss, grad_params
	end
	
	optim.sgd(feval, params, state)
end


]]
