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
local hidden_size = 200

-- configuration
training_list_file = "1.txt"

-- GPU

if GPU_ENABLED then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

loader = Loader()
loader:load(training_list_file)
codec = loader:codec()

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

params, grad_params = net:getParameters()

state = {
	learningRate = 1e-4,
	momentum = 0.9
}


-- threading

local nthread = 4

local pool = threads.Threads(nthread, 
	function(id)
		print("new thread " .. id)
	end
)

-- training

begin_time = 0

for i = 1, 1000000 do
	local b1 = timer:time().real
	
	local samples = {}
	
	for i = 1, nthread do
		local sample = loader:pick()
		table.insert(samples, {img = sample.img, target = codec:encode(sample.gt)})
	end

	local feval = function(params)
		net:forget()
		for i = 1, nthread do
			pool:addjob(
			
			)
		end
		
		
		
		outputTable = net:forward(im)
		loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
		
		if i % 1 == 0 then
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
			torch.save(filename, net:float())
			show_log(string.format("Saving finished, saved model file is at %s.", filename))
		end

		net:backward(im, grad)
		grad_params:cmul(grad_params:eq(grad_params):float())
		grad_params:clamp(-1, 1)

	
		return loss, grad_params
	end
	
	optim.sgd(feval, params, state)
end


