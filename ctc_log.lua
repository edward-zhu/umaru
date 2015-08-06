require 'nn'
require 'utils/logs'

ctc = {}

--[[
	getOnehotMatrix

	target -  a vector of number of class

	return a L * C onehot Matrix, C is the number of kinds of classes.
]]

function ctc.__getOnehotMatrix(target, class_num)
	onehot = torch.zeros((#target)[1], class_num)
	for i = 1, (#target)[1] do
		c = target[i]
		if c > 0 then
			onehot[i][c] = 1
		else
			onehot[i][class_num] = 1
		end
	end
	return onehot
end

--[[
	getFilledTarget

	target - a unicode string of ground truth

	return a 2L + 1 vector of number of class.
]]

function ctc.__getFilledTarget(target)
	local filled = torch.zeros(#target * 2 + 1)
	for i = 1, (#filled)[1] do
		if i % 2 == 0 then
			filled[i] = string.sub(target, i / 2, i / 2)
		end
	end
	return filled
end

function ctc.__toMatrix(outputTable)
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
function ctc.__getForwardVariable(outputTable, alignedTable, target)
	local T = (#outputTable)[1]
	-- create a T * (2L + 1) Matrix
	
	local L = (#target)[1]
	local fvs = torch.ones(T, L) * logs.LOG_ZERO
	
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
				fvs[i][u] = logs.log_add(fvs[i][u], fvs[i - 1][u])
				if u > 1 then fvs[i][u] = logs.log_add(fvs[i][u], fvs[i - 1][u - 1]) end
				fvs[i][u] = logs.log_mul(fvs[i][u], alignedTable[i][u])
			else
				if u > 2 and target[u - 2] ~= target[u] then fvs[i][u] = logs.log_add(fvs[i][u], fvs[i - 1][u - 2]) end
				if u > 1 then fvs[i][u] = logs.log_add(fvs[i][u], fvs[i - 1][u - 1]) end
				fvs[i][u] = logs.log_add(fvs[i][u], fvs[i - 1][u])
				fvs[i][u] = logs.log_mul(fvs[i][u], alignedTable[i][u])
			end
		end
		
	end
	
	return fvs
end

function ctc.__getBackwardVariable(outputTable, alignedTable, target)
	local T = (#outputTable)[1]
	-- create a T * (2L + 1) Matrix
	
	local L = (#target)[1]
	local bvs = torch.ones(T, L) * logs.LOG_ZERO
	
	-- initialize
	
	bvs[T][L] = 0
	bvs[T][L - 1] = 0
	
	-- calculate using dynamic programming
	
	for i = T - 1, 1, -1 do
		for u = L, 1, -1 do
			if i % 2 == 1 then
				bvs[i][u] = logs.log_mul(alignedTable[i + 1][u], bvs[i + 1][u])
				if u < L then 
					bvs[i][u] = logs.log_add(bvs[i][u], logs.log_mul(alignedTable[i + 1][u + 1], bvs[i + 1][u + 1])) 
				end
			else
				bvs[i][u] = logs.log_mul(alignedTable[i + 1][u], bvs[i + 1][u])
				if u < L then 
					bvs[i][u] = logs.log_add(bvs[i][u], logs.log_mul(alignedTable[i + 1][u + 1], bvs[i + 1][u + 1])) 
				end
				if u < L - 1 and target[u + 2] ~= target[u] then
					bvs[i][u] = logs.log_add(bvs[i][u], logs.log_mul(alignedTable[i + 1][u + 2], bvs[i + 1][u + 2]))
				end
			end
		end
	end
	
	return bvs
end

-- calculate cost matrix (Tx(cls+1))

function ctc.__getGrad(fb, pzx, class_num, outputTable, target)
	local T = (#fb)[1]
	local grad = torch.zeros(T, class_num)
	local temp_sum = 0
	local u = 0
	for t = 1, T do
		for k = 1, class_num do
			temp_sum = 0
			grad[t][k] = logs.log_mul(-pzx, -outputTable[t][k])
			u = k
			
			-- if current label is blank
			if u == class_num then u = 0 end
			for i = 1, (#target)[1] do
				if target[i] == u then
					temp_sum = logs.log_add(temp_sum, fb[t][i])
				end
			end
			grad[t][k] = logs.log_mul(grad[t][k], temp_sum)
			grad[t][k] = -logs.safe_exp(grad[t][k])
		end
	end
	return grad
end

function ctc.__getCost(fb, target)
	local cost = 0.0
	
	for i = 1, (#target)[1] do
		cost = logs.log_add(fb)
	end
end

function ctc.getCTCCostAndGrad(outputTable, target)
	-- convert target to one-hot Matrix (class + 1 * len(target))
	local class_num = (#(outputTable[1]))[1]
	local T = #outputTable
	
	
	targetClasses = ctc.__getFilledTarget(target)
	targetMatrix = ctc.__getOnehotMatrix(targetClasses, class_num)

	outputTable = ctc.__toMatrix(outputTable)

	-- get aligned_table
		-- outputTable: Tx(cls+1)
		-- target: L'x(cls+1) --> targetT : (cls+1)xL'
		-- alienged_table = TxL'
	local alignedTable = outputTable * targetMatrix:t()

	-- calculate forwardVariable (in log space)
	fb = ctc.__getForwardVariable(outputTable, alignedTable, targetMatrix)
	
	
	local L_1 = #target * 2 + 1
	
	-- calculate log(p(z|x))
	pzx = logs.log_sum(fb[T][L_1], fb[T][L_1-1])
	
	-- calculate backwardVariable (in log space)
	fb = fb + ctc.__getBackwardVariable(outputTable, alignedTable, targetMatrix)
	
	-- calculate gradient matrix (Tx(cls+1))
	local grad = ctc.__getGrad(fb, pzx, class_num, outputTable, targetClasses)
	
	print(grad)
	
end
