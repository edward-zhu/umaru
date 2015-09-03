require 'nn'
require 'rnn'
require 'image'
require 'optim'

require 'loader'

opts = {
	gpu 			=	false,
	dropout_rate	=	0.4,
	input_size 		=	64,
	hidden_size 	=	400,
	learning_rate	=	1e-4,
	momentum		=	0.9
}

local class_num = 100

local net = nn.Sequential()

net:add(nn.Dropout(DROPOUT_RATE))
net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.LSTM(opts.input_size, opts.hidden_size)))
net:add(nn.BiSequencer(nn.LSTM(opts.hidden_size * 2, opts.hidden_size)))
output = nn.Sequential()
output:add(nn.Linear(opts.hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
net:add(nn.Sequencer(output))

if opts.gpu then
	require 'cutorch'
	require 'cunn'
	net:cuda()
	cutorch.setDevice(1)
	cutorch.manualSeed(450)
else
	torch.manualSeed(450)
end


timer = torch.Timer()




im = torch.randn(64, 64)


if opts.gpu then
	im = im:cuda()
end

base = timer:time().real

outputTable = net:forward(im)

print(timer:time().real - base)



ims = {}

for i = 1, 10 do
	table.insert(ims, torch.randn(64 * i, 64))
end



if opts.gpu then
	for i = 1, 10 do
		ims[i] = ims[i]:cuda()
	end
end

base = timer:time().real

last = base

for i = 1, 10 do

	outputTable = net:forward(ims[i])
	last = timer:time().real
end

base = last

for i = 1, 10 do

	outputTable = net:forward(ims[i])
	print((timer:time().real - last))
	last = timer:time().real
end

print(last - base)



