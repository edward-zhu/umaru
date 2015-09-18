require 'slider'
require 'loader'
require 'image'
RBM = require 'rbm'
require 'cutorch'
require 'cunn'

opt = {
	input_size = 48,
	epoch = 30,

	-- sliding window
	window_size = 10,
	stride = 5,

	-- RBM hyperparameters
	hidden_size = 48 * 5,

	-- miscellaneous
	output_file = "wwr.rbm"
}



-- load samples

loader = Loader()
loader:load('wwr.txt')
loader:targetHeight(opt.input_size)

torch.setdefaulttensortype('torch.CudaTensor')

-- setup RBM

local n_visible = opt.input_size * opt.window_size

local rbm = RBM.new{n_visible=n_visible, n_hidden=opt.hidden_size, CDsteps=1, momentum={0.5, 0.9},
                  momentumAfter={5}, v_activation='binary', h_activation='relu',
                  learningRate=0.01}

-- train

for i = 1, opt.epoch do
	-- for each sample
	local im, p, total = loader:pickInSequential()
	local input
	while im do
		xlua.progress(p, total)
		im = im.img
		slider = Slider()
		slider:load(im:t())
		
		-- for each window
		input =  slider:genSequence()
		inputMatrix = nn.JoinTable(1):forward(input):reshape(slider.total, input[1]:size(1)):cuda()

		rbm:updateParameters(inputMatrix)

		im, p, total = loader:pickInSequential()
	end

	loader:reset()

	print(string.format("total progress %d / %d eps.", i, opt.epoch))
end

-- save

rbm_data = {
	n_visible = n_visible,
	n_hidden = opt.hidden_size,
	encoder = rbm.encoder:double(),
	decoder = rbm.decoder:double()
}

paths.mkdir("rbm")

local output_path = "rbm/" .. opt.output_file
torch.save(output_path, rbm_data)
print("RBM network saved at " .. output_path)

-- test

mlp = nn.Sequential()
mlp:add(rbm.encoder)
mlp:add(rbm.decoder)
mlp:cuda()

loader:reset()
local im = loader:pickInSequential().img:cuda()
slider = Slider()
slider:load(im:t())

local input = slider:slide()
local output = mlp:forward(input:reshape(input:nElement()))

input = input:double()
output = output:double()

torch.setdefaulttensortype('torch.DoubleTensor')

image.save("1.png", input:reshape(opt.input_size, opt.window_size))
image.save("2.png", output:reshape(opt.input_size, opt.window_size))