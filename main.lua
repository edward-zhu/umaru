require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'loader'
require 'ctc_log'
require 'utils.decoder'
require 'utils.levenshtein'

-- initialize
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(450)

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

-- settings

DROPOUT_RATE = 0.4
GPU_ENABLED = false
local input_size = 48
local hidden_size = 200
show_every = 0
save_every = 10000
test_every = 1000

-- configuration
training_list_file = "1.txt"
using_model_file = nil

-- curriculum training settings

curriculum_training = false
weight_change_iter_span = 10000
lambda = 3
lambda_change_every = 1000
lambda_grad = lambda / (weight_change_iter_span / lambda_change_every)

-- GPU

if GPU_ENABLED then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

loader = Loader()
loader:targetHeight(input_size)

loader:load(training_list_file, 0.8)
codec = loader:codec()

show_log(string.format("Loading finished. Got %d samples, %d classes of characters.", #loader.samples, codec.codec_size))
show_log(string.format("Splited into %d for training, %d for testing", #loader.training, #loader.testing))

local class_num = codec.codec_size

-- build network

show_log("Building networks...")

local net

if using_model_file then
	net = torch.load(using_model_file)
else
	net = nn.Sequential()

	-- net:add(nn.Dropout(DROPOUT_RATE))
	net:add(nn.SplitTable(1))
	net:add(nn.BiSequencer(nn.FastLSTM(input_size, hidden_size)))
	output = nn.Sequential()
	-- 
	output:add(nn.Linear(hidden_size * 2, class_num + 1))
	output:add(nn.SoftMax())
	net:add(nn.Sequencer(output))
	net:float()
end

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
	
	local sample
	
	if not curriculum_training then
		sample = loader:pick()
	else
		sample = loader:pickWithWeight()
		if i % lambda_change_every == 0 then
			lambda = lambda - lambda_grad
			loader:updateWeight(lambda)
			show_log("lambda was changed to " .. lambda)
		end
	end
	--print("pick time: " .. timer:time().real - b1)

	local im = sample.img
	local target = codec:encode(sample.gt)
	

	local feval = function(params)
		-- net:forget()
	
		--local b1 = timer:time().real
		
		outputTable = net:forward(im)
		
		--print("forward time: " .. timer:time().real - b1)
	
	
		--b1 = timer:time().real
		
		loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
		
		--print("ctc time: " .. timer:time().real - b1)
		
		if show_every > 0 and i % show_every == 0 then
			print("")
			show_log("EPOCH   " .. i)
			show_log("TARGET  " .. sample.gt)
			show_log("OUTPUT  " .. decoder.best_path_decode(outputTable, codec))
			show_log("LOSS    " .. loss)
			show_log("sec/ep  " .. (timer:time().real - begin_time) / i)
		end
		
		if save_every > 0 and i % save_every == 0 then
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

		
		grad_params:cmul(grad_params:eq(grad_params):float())
		
		grad_params:clamp(-5, 5)

		--print("optim time: " .. timer:time().real - b1)
	
		return loss, grad_params
	end
	
	optim.sgd(feval, params, state)
	
	net:maxParamNorm(2)
	
	-- testing
	
	if test_every > 0 and i % test_every == 0 and #loader.testing > 0 then
		show_log("testing...")
		net:evaluate()
		local dist, len = 0, 0
		loader:reset()
		local s = loader:pickInSequential("testing")
		
		while s do
			local out = decoder.best_path_decode(net:forward(s.img), codec)
			dist = dist + utf8.levenshtein(s.gt, out)
			len = len + utf8.len(s.gt)
			
			-- print("")
			-- show_log("FILE     " .. s.src)
			-- show_log("TARGET   " .. s.gt)
			-- show_log("OUTPUT   " .. out)
			-- show_log("ERROR    " .. string.format("%.2f%%", dist / len * 100))
			
			s = loader:pickInSequential("testing")
		end
		
		show_log("testing finished, error rate:  " .. string.format("%.2f%% at epoch %d.", dist / len * 100, i))
		
		
		
		
		net:training()
	end
	
end


