require 'nn'
require 'rnn'
require 'image'

require 'ctc_log'

mnist = require 'mnist'

DROPOUT_RATE = 0
MOMENTUM = 1
LEARNING_RATE = 1e-3
MAXOUTNORM = 2





local input_size = 19
local hidden_size = 20
local class_num = 4

local net = nn.Sequential()

net:add(nn.Dropout(DROPOUT_RATE))
net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.LSTM(input_size, hidden_size)))

output = nn.Sequential()
output:add(nn.Linear(hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
net:add(nn.Sequencer(output))

net:remember('both')

torch.manualSeed(450)



for i = 1, 1 do
	local no = torch.random() % 2 + 1
	local sample = mnist.traindataset()[no]
	local im = image.load("1/1.png", 1)[1]:t()
	local target = torch.Tensor{0, 1, 2, 0, 3}
	
	local result = torch.zeros(im:size()[2])
	
	-- print(target)
	
	outputTable = net:forward(im)
	
	
	loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
	
	result = torch.zeros(#outputTable)
	
	for i = 1, #outputTable do
		max = -1e100
		max_j = -1
		for j = 1, (#(outputTable[1]))[1] do
			if outputTable[i][j] > max then
				max = outputTable[i][j]
				max_j = j
			end
		end
		result[i] = max_j - 1
	end
	print(loss)
	print(target)
	


	net:updateGradInput(im, grad)
	net:accGradParameters(input, grad, MOMENTUM)
	net:updateParameters(LEARNING_RATE)
	net:maxParamNorm(2) 
	net:zeroGradParameters()
	
	
end
