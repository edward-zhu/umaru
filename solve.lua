require 'nn'
require 'rnn'
require 'GRU'
require 'image'
require 'optim'

require 'loader'
require 'ctc_log'
require 'utils.decoder'
require 'utils.levenshtein'

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
	-- print(string.format("[%.4f][%.4f]%s", now, cost, log))
	print(string.format("%s", log))
end 

-- settings

GPU_ENABLED = false
local input_size = 48

-- configuration
list_file = "wwr.txt"
using_model_file = "umaru_model_15-09-09_07:26:31_60000.uma"

-- GPU

if GPU_ENABLED then
	require 'cutorch'
	require 'cunn'
end

-- load samples

show_log("Loading samples...")

loader = Loader()
loader:load(list_file)
loader:targetHeight(input_size)
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
local dist, tmp_dist, out = 0, 0, 0
local len, tmp_len = 0, 0

while sample do
	local im = sample.img
	local target = codec:encode(sample.gt)
	
	net:forget()
	
	outputTable = net:forward(im)

	out = decoder.best_path_decode(outputTable, codec)

	tmp_dist = utf8.levenshtein(out, sample.gt)
	tmp_len = utf8.len(sample.gt)
	dist = dist + tmp_dist
	len = len + tmp_len

	print("")
	show_log("FILE     " .. sample.src)
	show_log("TARGET   " .. sample.gt)
	show_log("OUTPUT   " .. out)
	show_log("DISTANCE " .. tmp_dist)
	show_log("ERROR    " .. string.format("%.2f%%", dist / len * 100))

	sample = loader:pickInSequential()
end


