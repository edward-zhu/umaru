decoder = {}

function decoder.best_path_decode(outputTable, codec)
	local result = {}
	
	local class_num = #(outputTable[1])[1]
	
	local last_max_class = nil;
	local last_max = -1;
	
	for i = 1, #outputTable do
		local max_val, max = torch.max(outputTable[i], 1)
		max = max[1]
		
		if max = class_num then
			if last_max ~= -1 and last_max_class ~= nil then
				table.insert(result, last_max_class)
				last_max = -1
				last_max_class = nil
			end
		else
			if max_val > last_max then
				last_max = max_val
				last_max_class = max
			end
		end
	end
	
	return codec:decode(result)
end