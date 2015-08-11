decoder = {}

function decoder.decode(x)
	local res = ""
	last = -1
	for i = 1, (#x)[1] do
		if x[i] ~= last then
			res = res .. " " .. x[i]
			last = x[i]
		end
	end
	return res
end

function decoder.decodeTable(outputTable)
	local result = torch.zeros(#outputTable)
	
	for i = 1, #outputTable do
		max = -1e100
		max_j = -1
		for j = 1, (#(outputTable[1]))[1] do
			if outputTable[i][j] > max then
				max = outputTable[i][j]
				max_j = j
			end
		end
		result[i] = max_j - 1
	end
	
	return decoder.decode(result)
end