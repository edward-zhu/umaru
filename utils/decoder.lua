decoder = {}

function decoder.best_path_decode(outputTable, codec)
	local result = {}
	
	
	local class_num = outputTable[1]:size()[1]
	local last_max_class = nil;
	local last_max = -1;
	
	for i = 1, #outputTable do
		local max_val, max = torch.max(outputTable[i], 1)
		max = max[1]
		
		max_val = max_val[1]
		
		if max ~= last_max_class then
			if max ~= class_num then
				table.insert(result, max)
			end
			last_max_class = max
		end

			
	end
	
	return codec:decode(result)
end