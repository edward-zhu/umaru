require 'nn'
require 'rnn'
require 'GRU'
require 'image'
require 'optim'
require 'json'

require 'loader'
require 'ctc_log'
require 'utils.decoder'
require 'utils.levenshtein'

-- initialize
torch.setdefaulttensortype('torch.DoubleTensor')
torch.manualSeed(os.time())

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

opt = {
	-- project
	project_name = os.date("%y-%m-%d_") .. torch.random() % 10000,
	using_model_file = nil,

	recurrent_unit = "gru",

	-- hyperparameters
	input_size = 48,
	hidden_size = 200,
	clamp_size = 1,
	learning_rate = 1e-4,
	momentum = 0.9,
	dropout_rate = 0,
	max_param_norm = false,

	-- configurations
	gpu = false,

	-- threading
	nthread = 3,
	omp_threads = 1,

	-- samples
	training_list_file = "1.txt",
	testing_list_file = nil,
	codec_file = nil,
	testing_ratio = 1, -- is valid unless testing_list_file == nil

	-- miscellaneous
	max_iter = 1e10,
	show_every = 10,
	save_every = 10000,
	test_every = 1000,
	ctc_lua = false,
}

-- load settings

cmd = torch.CmdLine()
cmd:option("-setting", "", "setting file")
params = cmd:parse(arg)

if params.setting ~= "" then
	show_log("loading setting file " .. params.setting)
	opt = json.load(params.setting)
	show_log("setting file loaded successfully.")
end

run_dir = "experiments/" .. opt.project_name .. "/"
paths.mkdir(run_dir)

-- logging

cmd:log(run_dir .. "log.txt", opt)
json.save(run_dir .. "setting.json", opt)



-- apply settings
if opt.omp_threads then
	torch.setnumthreads(opt.omp_threads)
end

ctc_lua = opt.ctc_lua

-- curriculum training settings

curriculum_training = false
weight_change_iter_span = 10000
lambda = 3
lambda_change_every = 1000
lambda_grad = lambda / (weight_change_iter_span / lambda_change_every)

-- GPU

if opt.gpu then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

loader = Loader()
loader:targetHeight(opt.input_size)

if opt.testing_list_file ~= nil then
	loader:load(opt.training_list_file, 1)
	loader:loadTesting(opt.testing_list_file)
else
	loader:load(opt.training_list_file, opt.testing_ratio)
end


-- load codec

if opt.codec_file then
	codec = loader:loadCodec(opt.codec_file)
else
	codec = loader:codec()
	torch.save(opt.training_list_file:gsub("[.].*", ".codec"), codec)
end


show_log(string.format("Loading finished. Got %d samples, %d classes of characters.", #loader.samples, codec.codec_size))
show_log(string.format("Splited into %d for training, %d for testing", #loader.training, #loader.testing))



local class_num = codec.codec_size

-- build network

show_log("Building networks...")

local net, recurrent

if opt.using_model_file then
	net = torch.load(opt.using_model_file)
else
	net = nn.Sequential()

	net:add(nn.SplitTable(1))
	
	if opt.recurrent_unit == "gru" then
		recurrent = nn.GRU(opt.input_size, opt.hidden_size)
	elseif opt.recurrent_unit == "lstm" then
		recurrent = nn.LSTM(opt.input_size, opt.hidden_size)
	elseif opt.recurrent_unit == "lstm_nopeephole" then
		recurrent = nn.LSTM(opt.input_size, opt.hidden_size, 9999, false)
	elseif opt.recurrent_unit == "lstm_fast" then
		recurrent = nn.FastLSTM(opt.input_size, opt.hidden_size)
	end

	net:add(nn.BiSequencer(recurrent))
	output = nn.Sequential()
	output:add(nn.Dropout(opt.dropout_rate))
	output:add(nn.Linear(opt.hidden_size * 2, class_num + 1))
	output:add(nn.SoftMax())
	net:add(nn.Sequencer(output))
	net:double()
end

if opt.gpu then
	net:cuda()
end

-- prepare prarmeters and training method

params, grad_params = net:getParameters()

state = {
	learningRate = opt.learning_rate,
	momentum = opt.momentum
}

show_log(string.format("Start training. umaru~~"))

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

	local im

	if opt.gpu then
		im = sample.img:cuda()
	else
		im = sample.img
	end


	local target = codec:encode(sample.gt)

	local feval = function(params)
		net:forget()



		outputTable = net:forward(im)
		
		loss, grad = ctc.getCTCCostAndGrad(outputTable, target)
		
		if opt.show_every > 0 and i % opt.show_every == 0 then
			print("")
			show_log("EPOCH   " .. i)
			show_log("TARGET  " .. sample.gt)
			show_log("OUTPUT  " .. decoder.best_path_decode(outputTable, codec))
			show_log("LOSS    " .. loss)
			show_log("sec/ep  " .. (timer:time().real - begin_time) / i)
		end

		net:backward(im, grad)

		grad_params:cmul(grad_params:eq(grad_params):double())
		grad_params:clamp(-opt.clamp_size, opt.clamp_size)
		

		return loss, grad_params
	end

	optim.sgd(feval, params, state)

	if opt.max_param_norm then
		net:maxParamNorm(2)
	end

	-- model saving

	if opt.save_every > 0 and i % opt.save_every == 0 then
		print("")
		show_log("Saving model...")
		local filename = string.format("umaru_model_%s_%d.uma", os.date("%y-%m-%d_%X"), i)
		torch.save(run_dir .. filename, net)
		show_log(string.format("Saving finished, saved model file is at %s.", filename))
	end
		
	
	-- testing
	
	if opt.test_every > 0 and i % opt.test_every == 0 and #loader.testing > 0 then
		show_log("testing...")
		net:evaluate()
		local dist, len = 0, 0
		loader:reset()
		local s = loader:pickInSequential("testing")
		
		while s do
			local im
			if opt.gpu then
				im = s.img:cuda()
			else
				im = s.img
			end
			local out = decoder.best_path_decode(net:forward(im), codec)
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


