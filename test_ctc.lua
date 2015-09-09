require 'nn'
require 'ctc_log'




torch.setdefaulttensortype('torch.DoubleTensor')

outputTable = torch.Tensor{
	{0.0684907, 0.0683173, 0.0682402, 0.0682124, 0.0682041, 0.068242, 0.0682717},
	{0.067452, 0.0671584, 0.066958, 0.0667768, 0.0665642, 0.0662235, 0.0656839},
	{0.0722717, 0.0725465, 0.0726959, 0.0727905, 0.072852, 0.0729304, 0.0730594},
	{0.0719666, 0.0716504, 0.0714893, 0.0713959, 0.0713595, 0.0713568, 0.0714108},
	{0.0744622, 0.0744583, 0.0744827, 0.0745185, 0.0745582, 0.0746348, 0.0747697}, 
	{0.0660699, 0.0658052, 0.0656008, 0.0654654, 0.0653909, 0.0653646, 0.0653229},
	{0.0736831, 0.0741657, 0.0743352, 0.0743431, 0.0742691, 0.0740992, 0.0738383},
	{0.0771784, 0.0769233, 0.0768525, 0.0768478, 0.0768212, 0.0767496, 0.076588},
	{0.0770828, 0.0770411, 0.0770657, 0.0770728, 0.0770434, 0.0769496, 0.0767167},
	{0.0658117, 0.0656928, 0.0656175, 0.0656082, 0.0656804, 0.0658425, 0.0661498}, 
	{0.0691583, 0.0690236, 0.0690129, 0.069102, 0.0692818, 0.0695909, 0.0701721},
	{0.0700088, 0.0704332, 0.0706438, 0.0707642, 0.0708752, 0.0710007, 0.0711254},
	{0.0727488, 0.0726681, 0.0726766, 0.0727124, 0.0727471, 0.0728149, 0.0730135}, 
	{0.0736152, 0.0741163, 0.074329, 0.07439, 0.0743531, 0.0742006, 0.0738778},
}

target = {4, 3, 13, 1, 10, 7}

function toMatrix(outputTable)
	local net = nn.Sequential()
	net:add(nn.JoinTable(1))
	net:add(nn.Reshape(#outputTable, outputTable[1]:size(1)))
	return net:forward(outputTable)
end

-- outputTable = nn.Log():forward(outputTable:t())

nrow = outputTable:size(2)

splitedOutputTable = nn.SplitTable(1):forward(outputTable:t())

c_pzx, c_grad = ctc.getCTCCostAndGrad(splitedOutputTable, target)

c_m = toMatrix(c_grad):float()


eps = 1e-6

ctc_lua = false

est_grad = torch.Tensor(nrow)

for i = 1, nrow do
	outputTable[1][i] = outputTable[1][i] + eps

	splitedOutputTable = nn.SplitTable(1):forward(outputTable:t())
	loss1, _ = ctc.getCTCCostAndGrad(splitedOutputTable, target)

	outputTable[1][i] = outputTable[1][i] - 2 * eps
	splitedOutputTable = nn.SplitTable(1):forward(outputTable:t())
	loss2, _ = ctc.getCTCCostAndGrad(splitedOutputTable, target)
	
	est_grad[i] = (loss1 - loss2) / eps
end

print(est_grad)



