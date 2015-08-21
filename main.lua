require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'loader'
require 'ctc_log'
require 'utils/decoder'

torch.setdefaulttensortype('torch.FloatTensor')

-- debug switch
DEBUG = false

-- timer initialize
base = 0
timer = torch.Timer()

function show_log(log)
	local now = timer:time().real
	local cost = now - base
	base = now
	print(string.format("[%.4f][%.4f]%s", now, cost, log))
end 

-- initialize

DROPOUT_RATE = 0.4
torch.manualSeed(450)

local input_size = 32
local hidden_size = 200

GPU_ENABLED = 1

-- GPU

if GPU_ENABLED then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

loader = Loader()
loader:load("wwr.txt")
codec = loader:codec()

show_log(string.format("Loading finished. Got %d samples, %d classes of characters.", #loader.samples, codec.codec_size))

local class_num = codec.codec_size

-- build network

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

if GPU_ENABLED then
	net:cuda()
end

-- prepare prarmeters and training method

params, grad_params = net:getParameters()

state = {
	learningRate = 1e-4,
	momentum = 0.9
}

show_log(string.format("Start training with learning rate = %.4f, momentum = %.4f", state.learningRate, state.momentum))

-- training

begin_time = 0

for i = 1, 1000000 do
	local b1 = timer:time().real
	
	local sample = loader:pick()
	--print("pick time: " .. timer:time().real - b1)

	local im = sample.img
	local target = codec:encode(sample.gt)
	

	local feval = function(params)
		net:forget()
	
		--local b1 = timer:time().real
		
		outputTable = net:forward(im)
		
		--print("forward time: " .. timer:time().real - b1)
	
	
		--b1 = timer:time().real
		
		loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
		
		--print("ctc time: " .. timer:time().real - b1)
		
		if i % 10 == 0 then
			print("")
			show_log("EPOCH   " .. i)
			show_log("TARGET  " .. sample.gt)
			show_log("OUTPUT  " .. decoder.best_path_decode(outputTable, codec))
			show_log("LOSS    " .. loss)
			show_log("sec/ep  " .. (timer:time().real - begin_time) / i)
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
		
		--b1 = timer:time().real

		net:backward(im, grad)

		--print("backward time: " .. timer:time().real - b1)
		
		-- print("backward finish")

		--b1 = timer:time().real

		
		grad_params:cmul(grad_params:eq(grad_params))
		grad_params:clamp(-5, 5)

		--print("optim time: " .. timer:time().real - b1)
	
		return loss, grad_params
	end
	
	optim.sgd(feval, params, state)
end


