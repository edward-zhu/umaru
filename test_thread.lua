require 'nn'
require 'rnn'
require 'image'
require 'optim'
require 'lfs'
require 'json'

require 'loader'
require 'ctc_log'
require 'utils/decoder'

local threads = require 'threads'

-- timer initialize
base = 0
timer = torch.Timer()

-- initialize
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(os.time())

-- debug switch
DEBUG = false



function show_log(log)
	local now = timer:time().real
	local cost = now - base
	base = now
	print(string.format("[%.4f][%.4f]%s", now, cost, log))
end 

-- settings

opt = {
	-- project
	project_name = os.date("%y-%m-%d_") .. torch.random() % 10000,

	-- hyperparameters
	input_size = 48,
	hidden_size = 200,
	clamp_size = 1,
	learning_rate = 1e-4,
	momentum = 0.9,
	dropout_rate = 0.5,

	-- configurations
	gpu = false,

	-- threading
	nthread = 3,

	-- samples
	training_list_file = "wwr.txt",
	codec_file = "",

	-- miscellaneous
	max_iter = 1e10
}

cmd = torch.CmdLine()



show_log("======== UMARU ========")
show_log("project: " .. opt.project_name)
show_log("")

-- preparation for model saving and logging

lfs.mkdir("models")

local project_dir = "models/" .. opt.project_name .. "/"

lfs.mkdir("models/" .. opt.project_name)

json.save(project_dir .. "settings.json", opt)

-- GPU

if opt.gpu then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

local loader = Loader()
loader:targetHeight(opt.input_size)
loader:load(opt.training_list_file)
local codec = loader:codec()

show_log(string.format("Loading finished. Got %d samples, %d classes of characters.", #loader.samples, codec.codec_size))
show_log(string.format("lr = %f, opt.momentum = %.4f clamp = %.2f", opt.learning_rate, opt.momentum, opt.clamp_size))
show_log(string.format("using %d threads.", opt.nthread))

local class_num = codec.codec_size

-- build network

show_log("Building networks...")

local net

net = nn.Sequential()

net:add(nn.Dropout(opt.dropout_rate))
net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.FastLSTM(opt.input_size, opt.hidden_size)))

output = nn.Sequential()
output:add(nn.Dropout(opt.dropout_rate))
output:add(nn.Linear(opt.hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
net:add(nn.Sequencer(output))
net:float()

-- prepare prarmeters and training method

local params, grad_params

params, grad_params = net:getParameters()


state = {
	learningRate = opt.learning_rate,
	momentum = opt.momentum
}

threads.serialization('threads.sharedserialize')

local pool = threads(opt.nthread, 
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
	learningRate = opt.learning_rate,
	momentum = opt.momentum
}



for i = 1, opt.max_iter do
	local totalerr = 0
	local totalgrad = nil
	
	local feval = function(params)
		grad_params:zero()
		
		for j = 1, opt.nthread do
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
						print(string.format("%.2f sec/ep.", timer:time().real / (i * opt.nthread)))
					end
					
					gp:cmul(gp:eq(gp):float())
					gp:clamp(-opt.clamp_size, opt.clamp_size)
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


