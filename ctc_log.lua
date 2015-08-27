require 'nn'
require 'utils/logs'
require 'libctc'

ctc = {}

local base = 0
local timer = nil;



function print_timestamp(msg) 
	if not DEBUG then
		return
	end
	
	print(msg .. " " .. timer:time().real - base)
	base = timer:time().real
end

--[[
	getOnehotMatrix

	target -  a vector of number of class

	return a (2L + 1) * C onehot Matrix, C is the number of kinds of classes.
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

function ctc.__getFilledTargetFromString(target)
	local filled = torch.zeros(#target * 2 + 1)
	for i = 1, (#filled)[1] do
		if i % 2 == 0 then
			filled[i] = string.sub(target, i / 2, i / 2) + 1
		end
	end
	return filled
end

function ctc.__getFilledTarget(target)
	local filled = torch.zeros(#target * 2 + 1)
	for i = 1, (#filled)[1] do
		if i % 2 == 0 then
			filled[i] = target[i / 2]
		end
	end
	return filled
end

function ctc.__toMatrix(outputTable, class_num)
	local net = nn.Sequential()
	net:add(nn.JoinTable(1))
	net:add(nn.Reshape(#outputTable, class_num))
	return net:forward(outputTable)
end

--[[
	getForwardVariable 
	
	calculate ForwardVariable for any (t, u)

	- outputTable: a T * (C + 1) matrix
	- alignedTable: a T * L matrix
	- target: a (2L + 1) * (C + 1) matrix
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
	
	local lower_bound = 0
	local upper_bound = 2

	-- calculate
	for i = 2, T do
		upper_bound = upper_bound + 2
		if upper_bound > L then
			upper_bound = L
		end
		lower_bound = L - 2 * (T - i) - 1
		if lower_bound < 1 then
			lower_bound = 1
		end
		for u = lower_bound, upper_bound do
			-- if l'[u] is blank
		
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
	
	local upper_bound = L - 2
	local lower_bound
	
	for i = T - 1, 1, -1 do
		upper_bound = upper_bound - 2
		if upper_bound < 1 then
			upper_bound = 1
		end
		
		lower_bound = 2 * i
		if lower_bound > L - 1 then
			lower_bound = L - 1
		end
		
		-- print(lower_bound, upper_bound)
		
		for u = lower_bound, upper_bound, -1 do
			
			if u % 2 == 1 then
				bvs[i][u] = logs.log_mul(alignedTable[i + 1][u], bvs[i + 1][u])
				bvs[i][u] = logs.log_add(bvs[i][u], logs.log_mul(alignedTable[i + 1][u + 1], bvs[i + 1][u + 1])) 
			else
				bvs[i][u] = logs.log_mul(alignedTable[i + 1][u], bvs[i + 1][u]) 
				bvs[i][u] = logs.log_add(bvs[i][u], logs.log_mul(alignedTable[i + 1][u + 1], bvs[i + 1][u + 1])) 
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
			temp_sum = logs.LOG_ZERO
			grad[t][k] = logs.log_mul(-pzx, -outputTable[t][k])
			u = k
			
			
			
			-- if current label is blank
			if u == class_num then u = 0 end
			for i = 1, (#target)[1] do
				if target[i] == u then
					-- print(fb[t][i])
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
	if DEBUG then
		print("debug")
		timer = torch.Timer()
		base = 0;
	end
	
	-- convert target to one-hot Matrix (class + 1 * len(target))
	local class_num = (#(outputTable[1]))[1]
	local T = #outputTable
	
	print_timestamp("	CTC begin")
	
	targetClasses = ctc.__getFilledTarget(target)
	
	-- print(targetClasses)
	
	
	targetMatrix = ctc.__getOnehotMatrix(targetClasses, class_num)
	
	outputTable = ctc.__toMatrix(outputTable, class_num)

	if torch.type(outputTable) ~= "torch.FloatTensor" then
		outputTable = outputTable:float()
	end
	
	outputTable = outputTable:cmax(1e-4)
	local total = outputTable:sum(2):expand(outputTable:size()[1], outputTable:size()[2])
	outputTable = torch.cdiv(outputTable, total)

	print_timestamp("	perpare")
	
	-- print(outputTable)
	
	outputTable:apply(function (x)
		x = logs.safe_log(x)
		return x
	end)
	
	print_timestamp("	log")
	
	-- get aligned_table
		-- outputTable: Tx(cls+1)
		-- target: L'x(cls+1) --> targetT : (cls+1)xL'
		-- alienged_table = TxL'
	local alignedTable = outputTable * targetMatrix:t()

	-- calculate forwardVariable (in log space)
	
	local fvs, bvs, fb, grad
	
	if ctc_lua then
		fvs = ctc.__getForwardVariable(outputTable, alignedTable, targetClasses)
	else
		fvs = libctc.get_forward_variable(outputTable, alignedTable, targetClasses)
	end
	
	
	
	local L_1 = (#targetClasses)[1]
	
	-- calculate log(p(z|x))
	local pzx = logs.log_add(fvs[T][L_1], fvs[T][L_1-1])

	
	-- calculate backwardVariable (in log space)
	if ctc_lua then
		bvs = ctc.__getBackwardVariable(outputTable, alignedTable, targetClasses)
	else
		bvs = libctc.get_backward_variable(outputTable, alignedTable, targetClasses)
	end
	
	-- print(torch.dist(bvs, bvs1))

	print_timestamp("	fw bw")
	
	fb = fvs + bvs
	

	-- calculate gradient matrix (Tx(cls+1))
	if ctc_lua then
		grad = ctc.__getGrad(fb, pzx, class_num, outputTable, targetClasses)
	else
		grad = libctc.get_grad(fb, outputTable, targetClasses, pzx)
	end
	
	print_timestamp("	get grad")
	
	--[[
	print("=========FVS=========")
	print(fvs:t())
	print("=========BVS=========")
	print(bvs:t())
	print("=========GRAD=========")
	print(grad)
	]]
	
	if GPU_ENABLED then
		grad = grad:cuda()
	end
	

	grad = nn.SplitTable(1):forward(grad)
	
	
	return -pzx, grad
	
end

function ctc.test()
	return gpu
end
