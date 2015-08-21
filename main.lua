require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'loader'
require 'ctc_log'
require 'utils/decoder'

torch.setdefaulttensortype('torch.FloatTensor')

base = 0

DEBUG = true

timer = torch.Timer()

function show_log(log)
	local now = timer:time().real
	local cost = now - base
	base = now
	print(string.format("[%.4f][%.4f]%s", now, cost, log))
end 

DROPOUT_RATE = 0.4

local input_size = 32
local hidden_size = 200

show_log("Loading samples...")

loader = Loader()
loader:load("wwr.txt")
codec = loader:codec()

show_log(string.format("Loading finished. Got %d samples, %d classes of characters.", #loader.samples, codec.codec_size))

local class_num = codec.codec_size

show_log("Building networks...")

local net = nn.Sequential()

net:add(nn.Dropout(DROPOUT_RATE))
net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.FastLSTM(input_size, hidden_size)))

output = nn.Sequential()
output:add(nn.Linear(hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
net:add(nn.Sequencer(output))
net:float()

-- net:remember('both')

torch.manualSeed(450)

params, grad_params = net:getParameters()

state = {
	learningRate = 1e-4,
	momentum = 0.9
}

show_log(string.format("Start training with learning rate = %.4f, momentum = %.4f", state.learningRate, state.momentum))


for i = 1, 1000000 do
	local sample = loader:pick()
	local im = sample.img
	local target = codec:encode(sample.gt)
	

	local feval = function(params)
		net:forget()
	
		local b1 = timer:time().real
		
		outputTable = net:forward(im)
		
		print("forward time: " .. timer:time().real - b1)
	
	
		local b1 = timer:time().real
		
		loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
		
		-- print("ctc time: " .. timer:time().real - b1)
		
		if i % 10 == 0 then
			print("")
			show_log("EPOCH\t" .. i)
			show_log("TARGET\t" .. sample.gt)
			show_log("OUTPUT\t" .. decoder.best_path_decode(outputTable, codec))
			show_log("LOSS\t" .. loss)
		end
		
		if i % 10000 == 0 then
			print("")
			show_log("Saving model...")
			local filename = string.format("umaru_model_%s_%d.uma", os.date("%y-%m-%d_%X"), i)
			torch.save(filename, net)
			show_log(string.format("Saving finished, saved model file is at %s.", filename))
		end
	
		-- net:zeroGradParameters()
	
		-- print(grad_params)
		
		-- ctc_lua = true
		
		-- print(grad)
		
		-- print("backward")
		
		net:backward(im, grad)
		
		-- print("backward finish")
		
		grad_params:cmul(torch.eq(grad_params, grad_params):float())
		grad_params:clamp(-5, 5)
	
		return loss, grad_params
	end
	
	optim.sgd(feval, params, state)
end


