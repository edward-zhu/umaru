require 'nn'
require 'rnn'
require 'logs'

mnist = require 'mnist'



local sample = mnist.traindataset()[1]

local im = sample.x:double():t()
local target = tostring(sample.y)

local size = im:size()
local im_w = size[1]
local im_h = size[2]

local input_size = im_h
local hidden_size = 10
local class_num = 10

function getOnehotMatrix(target)
	onehot = torch.zeros((#target)[1], class_num + 1)
	for i = 1, (#target)[1] do
		c = target[i]
		if c > 0 then
			onehot[i][c] = 1
		else
			onehot[i][class_num + 1] = 1
		end
	end
	return onehot
end

function getFilledTarget(target)
	local filled = torch.zeros(#target * 2 + 1)
	for i = 1, (#filled)[1] do
		if i % 2 == 0 then
			filled[i] = string.sub(target, i / 2, i / 2)
		end
	end
	return filled
end

function toMatrix(outputTable)
	local net = nn.Sequential()
	net:add(nn.JoinTable(1))
	net:add(nn.Reshape(#outputTable, 11))
	return net:forward(outputTable)
end


--[[
	getForwardVariable 
	
	calculate ForwardVariable for any (t, u)

	- outputTable: a T * (2C + 1) matrix
	- alignedTable: a T * L matrix
	- target: a (2L + 1) * (2C + 1) matrix
]]--
function getForwardVariable(outputTable, alignedTable, target)
	local T = (#outputTable)[1]
	-- create a T * (2L + 1) Matrix
	
	local L = (#target)[1]
	local fvs = torch.zeros(T, L)
	
	-- calculate using dynamic programming
	
	-- initialize
	
	fvs[1][1] = alignedTable[1][1]
	fvs[1][2] = alignedTable[1][2]
	
	local upper_bound = 2

	-- calculate
	for i = 2, T do
		upper_bound = upper_bound + 2
		if upper_bound > L then
			upper_bound = L
		end
		for u = 1, upper_bound do
		
			-- if l'[u] is not blank
		
			if u % 2 == 1 then
				fvs[i][u] = fvs[i][u] + fvs[i - 1][u]
				if u > 1 then fvs[i][u] = fvs[i][u] + fvs[i - 1][u - 1] end
				fvs[i][u] = fvs[i][u] * alignedTable[i][u]
			else
				if u > 2 and target[u - 2] ~= target[u] then fvs[i][u] = fvs[i][u] + fvs[i - 1][u - 2] end
				if u > 1 then fvs[i][u] = fvs[i][u] + fvs[i - 1][u - 1] end
				fvs[i][u] = fvs[i][u] + fvs[i - 1][u]
				fvs[i][u] = fvs[i][u] * alignedTable[i][u]
			end
		end
		
		
	end
	
	return fvs
end

function getBackwardVariable(outputTable, alignedTable, target)
	local T = (#outputTable)[1]
	-- create a T * (2L + 1) Matrix
	
	local L = (#target)[1]
	local bvs = torch.zeros(T, L)
	
	-- initialize
	
	bvs[T][L] = 1
	bvs[T][L - 1] = 1
	
	-- calculate using dynamic programming
	
	for i = T - 1, 1, -1 do
		for u = L, 1, -1 do
			if i % 2 == 1 then
				bvs[i][u] = bvs[i][u] + alignedTable[i + 1][u] * bvs[i + 1][u]
				if u < L then bvs[i][u] = bvs[i][u] + alignedTable[i + 1][u + 1] * bvs[i + 1][u + 1] end
			else
				bvs[i][u] = bvs[i][u] + alignedTable[i + 1][u] * bvs[i + 1][u]
				if u < L then bvs[i][u] = bvs[i][u] + alignedTable[i + 1][u + 1] * bvs[i + 1][u + 1] end
				if u < L - 1 and target[u + 2] ~= target[u] then
					bvs[i][u] = bvs[i][u] + alignedTable[i + 1][u + 2] * bvs[i + 1][u + 2]
				end
			end
		end
	end
	
	return bvs
end


local net = nn.Sequential()

net:add(nn.SplitTable(1))
net:add(nn.BiSequencer(nn.LSTM(input_size, hidden_size)))


output = nn.Sequential()
output:add(nn.Linear(hidden_size * 2, class_num + 1))
output:add(nn.SoftMax())
net:add(nn.Sequencer(output))

outputTable = net:forward(im)

-- convert target to one-hot Matrix (class + 1 * len(target))

target = getFilledTarget(target)
target = getOnehotMatrix(target)

outputTable = toMatrix(outputTable)

-- get alienged_table
	-- outputTable: Tx(cls+1)
	-- target: L'x(cls+1) --> targetT : (cls+1)xL'
	-- alienged_table = TxL'
local alignedTable = outputTable * target:t()

fvs = getForwardVariable(outputTable, alignedTable, target)

-- calculate backwardVariable

bvs = getBackwardVariable(outputTable, alignedTable, target)

print(bvs)

-- calculate loss

-- output = net:forward(im)


-- print(net:forward(im))
