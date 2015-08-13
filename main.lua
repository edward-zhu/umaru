require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'ctc_log'
require 'utils/decoder'

mnist = require 'mnist'

DROPOUT_RATE = 0.4

local input_size = 28
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

for i = 1, 100000 do
	local no = torch.random() % 100 + 1
	local sample = mnist.traindataset()[no]
	local im = sample.x:double():t()
	local target = torch.Tensor{sample.y + 1}
	
	
	local feval = function(params)
		net:forget()
	
		outputTable = net:forward(im)
	
		loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
	
		if i % 20 == 0 then
			print(target[1] - 1)
			print(decoder.decodeTable(outputTable))
			print(loss)
		end
	
		-- net:zeroGradParameters()
	
		net:backward(im, grad)
		
		grad_params:cmul(torch.eq(grad_params, grad_params):double())
	
		return loss, grad_params
	end
	
	optim.sgd(feval, params, state)
end
