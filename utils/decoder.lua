decoder = {}

function decoder.decode(x)
	last = -1
	for i = 1, (#x)[1] do
		if x[i] ~= last then
			io.write(x[i] .. " ")
			last = x[i]
		end
	end
	print("\n")
end