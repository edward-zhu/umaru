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

DROPOUT_RATE = 0
GPU_ENABLED = false
local input_size = 32
local hidden_size = 100
clamp_size = 5

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


-- threading

local nthread = 3

threads.serialization('threads.sharedserialize')

local pool = threads(nthread, 
	function(id)
		print("new thread " .. id)
		require 'nn'
		require 'rnn'
		require 'ctc_log'
	end,
	
	function()
		print("function()")
		torch.setdefaulttensortype('torch.FloatTensor')
		local n = net:clone()
		local p, gp = n:getParameters()
		
		n:zeroGradParameters()
		torch.manualSeed(450)
		
		local loss, grad
		
		function eval(id, ps, im, target)
			-- n:zeroGradParameters()
			p:copy(ps)
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
	learningRate = 1e-4,
	momentum = 0.9
}



for i = 1, 1000000 do
	local totalerr = 0
	local totalgrad = nil
	
	
	
	net:zeroGradParameters()
	
	local feval = function(params)
		
		
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
					
					if j == 3 then
						print(decoder.best_path_decode(out, codec))
					end
				
					grad_params = grad_params + gp
				
				end,
				idx
			)
		end
		
		pool:synchronize()
		
		return totalerr, grad_params
	end
	
	-- grad_params = grad_params / nthread
	
	grad_params:cmul(grad_params:eq(grad_params):float())
	grad_params:clamp(-clamp_size, clamp_size)
	
	
	
	optim.sgd(feval, params, state)
	show_log("LOSS " .. totalerr / nthread)
end

pool:terminate()


