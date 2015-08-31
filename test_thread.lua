require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'loader'
require 'ctc_log'
require 'utils/decoder'

local threads = require 'threads'


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
local input_size = 32
local hidden_size = 100
clamp_size = 5
learning_rate = 1e-4
momentum = 0.9

-- threading

local nthread = 3

-- configuration
training_list_file = "wwr.txt"

-- GPU

if GPU_ENABLED then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

local loader = Loader()
loader:load(training_list_file)
local codec = loader:codec()

show_log(string.format("Loading finished. Got %d samples, %d classes of characters.", #loader.samples, codec.codec_size))
show_log(string.format("lr = %f, momentum = %.4f clamp = %.2f", learning_rate, momentum, clamp_size))
show_log(string.format("using %d threads.", nthread))

local class_num = codec.codec_size

-- build network

show_log("Building networks...")

local net

net = nn.Sequential()

net:add(nn.Dropout(DROPOUT_RATE))
net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.FastLSTM(input_size, hidden_size)))

output = nn.Sequential()
output:add(nn.Linear(hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
net:add(nn.Sequencer(output))
net:float()

-- prepare prarmeters and training method

local params, grad_params

params, grad_params = net:getParameters()


state = {
	learningRate = 1e-4,
	momentum = 0.9
}

a = "123"

local orig = nil




threads.serialization('threads.sharedserialize')

local pool = threads(nthread, 
	function(id)
		require 'nn'
		require 'rnn'
		require 'ctc_log'
	end,
	
	function()
		torch.setdefaulttensortype('torch.FloatTensor')
		local n = net:clone()
		local p, gp = n:getParameters()
		
		n:zeroGradParameters()
		torch.manualSeed(450)
		
		local loss, grad
		
		function eval(id, ps, im, target)
			-- n:zeroGradParameters()
			p:copy(ps)
			n:forget()
			-- print(p:sum())
			outputTable = n:forward(im)
			loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
			n:backward(im, grad)
			
			-- print("loss " .. loss)
			return outputTable, loss, gp
		end
	end
)

-- training

begin_time = 0

state = {
	learningRate = learning_rate,
	momentum = momentum
}



for i = 1, 1000000 do
	local totalerr = 0
	local totalgrad = nil
	
	local feval = function(params)
		grad_params:zero()
		
		for j = 1, nthread do
			local sample = loader:pick()
			local im = sample.img
			local target = codec:encode(sample.gt)
			pool:addjob(
				function(idx)
					local im = im
					local target = target
					local ps = params
					
					return eval(idx, ps, im, target)
				end,
	
				function(out, loss, gp)
					totalerr = totalerr + loss
					
					if i % 10 == 0 and j == 1 then
						show_log("LOSS " .. loss)
						print(sample.gt)
						print(decoder.best_path_decode(out, codec))
						print(string.format("%.2f sec/ep.", timer:time().real / (i * nthread)))
					end
					
					gp:cmul(gp:eq(gp):float())
					gp:clamp(-clamp_size, clamp_size)
					
					-- print(gp:sum())
					grad_params = grad_params + gp
						
				end,
				idx
			)
		end
		
		pool:synchronize()

		return totalerr, grad_params
	end
	
	optim.sgd(feval, params, state)
	
	
end

pool:terminate()


