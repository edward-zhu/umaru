require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'loader'
require 'ctc_log'
require 'utils/decoder'

-- initialize
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

-- settings

GPU_ENABLED = false
local input_size = 32

-- configuration
list_file = "1.txt"
using_model_file = "models/umaru_model_15-08-24_09_12_13_210000.uma"

-- GPU

if GPU_ENABLED then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

loader = Loader()
loader:load(list_file)
codec = loader:codec()

show_log(string.format("Loading finished. Got %d samples, %d classes of characters.", #loader.samples, codec.codec_size))

local class_num = codec.codec_size

-- build network

show_log("Building networks...")

local net

if using_model_file then
	net = torch.load(using_model_file)
	net:evaluate()
else
	error("There must be a model file.")
end

if GPU_ENABLED then
	net:cuda()
end


show_log(string.format("Start solving with model file: %s", using_model_file))

-- solving

local sample = loader:pickInSequential()

begin_time = timer:time().real

while sample do
	local im = sample.img
	local target = codec:encode(sample.gt)
	
	net:forget()
	
	outputTable = net:forward(im)

	if i % 1 == 0 then
		print("")
		show_log("EPOCH   " .. i)
		show_log("TARGET  " .. sample.gt)
		show_log("OUTPUT  " .. decoder.best_path_decode(outputTable, codec))
		show_log("sec/ep  " .. (timer:time().real - begin_time) / i)
	end
end


