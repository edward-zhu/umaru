require 'nn'
require 'cnn'
require 'rnn'
require 'image'

require 'ctc_log'
require 'utils/decoder'

mnist = require 'mnist'

DROPOUT_RATE = 0.4
MOMENTUM = 0.9
LEARNING_RATE = 1e-3
MAXOUTNORM = 2

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

net:cuda()

-- net:remember('both')

torch.manualSeed(450)

params, grad_params = net:getParameters()

for i = 1, 100000 do
	local no = torch.random() % 10 + 1
	local sample = mnist.traindataset()[no]
	local im = sample.x:double():t():cuda()
	local target = torch.Tensor{sample.y + 1}:cuda()
	
	
	local result = torch.zeros(im:size()[2])
	
	net:forget()
	
	outputTable = net:forward(im)
	
	
	loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
	
	
	if i % 20 == 0 then
		print(target[1] - 1)
		print(decoder.decodeTable(outputTable))
		print(loss)
	end
	
	-- net:zeroGradParameters()
	

	net:updateGradInput(im, grad)
	net:accGradParameters(im, grad, MOMENTUM)
	grad_params:clamp(-10, 10)
	
	
	-- print(gradParams)
	
	net:updateParameters(LEARNING_RATE)
	-- net:maxParamNorm(2) 
	
	
	
end
