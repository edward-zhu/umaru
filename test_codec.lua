require 'loader'

l = Loader()
l:load("1.txt")

INIT_LAMBDA = 3.0

function getLambda(i, total)
	return (1 - i / total) * INIT_LAMBDA 
end

lambda = 3.0

for i = 1, 100 do
	print(l:pickWithWeight())
	
	if i % 10 == 0 then
		lambda = getLambda(i, 100)
		print(">>> updated lambda = " .. lambda)
		l:updateWeight(lambda)
	end
end

