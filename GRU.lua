local GRU, parent

GRU, parent = torch.class('nn.GRU', 'nn.AbstractRecurrent')

function GRU:__init(inputSize, outputSize, rho)
	parent.__init(self, rho or 9999)
	self.inputSize = inputSize
	self.outputSize = outputSize
	
	self.recurrentModule = self:buildModel()
	
	self.modules[1] = self.recurrentModule
	self.sharedClones[1] = self.recurrentModule
	
	self.zeroTensor = torch.Tensor()
	
	self.cells = {}
	self.gradCells = {}
end

function GRU:buildGate()
	local gate = nn.Sequential()
	local i2g = nn.Linear(self.inputSize, self.outputSize)
	local o2g = nn.Linear(self.outputSize, self.outputSize)
	local para = nn.ParallelTable()
	para:add(i2g):add(o2g)
	gate:add(para)
	gate:add(nn.CAddTable())
	gate:add(nn.Sigmoid())
	
	return gate
end

function GRU:buildResetGate()
	self.resetGate = (self.resetGate == nil and self:buildGate() or self.resetGate)
	return self.resetGate
end

function GRU:buildUpdateGate()
	self.updateGate = (self.updateGate == nil and self:buildGate() or self.updateGate)
	return self.updateGate
end


-- outputCandidate = tanh(W * x + U(r . h[t - 1])))
function GRU:buildOutputCandidate()
	local hiddenCandidate = nn.Sequential()
	local left = nn.Sequential()
	-- select x
	left:add(nn.SelectTable(1))
	left:add(nn.Linear(self.inputSize, self.outputSize))
	local right = nn.Sequential()
	-- select (r, y[t - 1])
	right:add(nn.NarrowTable(2, 2))
	right:add(nn.CMulTable())
	right:add(nn.Linear(self.outputSize, self.outputSize))
	local para = nn.ConcatTable()
	para:add(left):add(right)
	
	hiddenCandidate:add(para)
	hiddenCandidate:add(nn.CAddTable())
	hiddenCandidate:add(nn.Tanh())
	
	return hiddenCandidate
end

-- input {input, output[t - 1]}

function GRU:buildModel()
	self.resetGate = self:buildResetGate()
	self.updateGate = self:buildUpdateGate()
	self.outputCandidate = self:buildOutputCandidate()
	
	local cell = nn.Sequential()
	
	local concat = nn.ConcatTable()
	concat:add(nn.Identity()):add(self.resetGate):add(self.updateGate)
	
	
	-- input: {x , h}
	-- output: {{x, h[t - 1]}, r, z}
	
	
	cell:add(concat)
	cell:add(nn.FlattenTable())
	
	
	local seq1 = nn.Sequential()
	
	
	-- select output[t - 1]
	seq1:add(nn.SelectTable(2))
	
	local seq2 = nn.Sequential()
	seq2:add(nn.SelectTable(4))
	seq2:add(nn.MulConstant(-1, false))
	seq2:add(nn.AddConstant(1, false))
	
	local seq3 = nn.Sequential()
	seq3:add(nn.NarrowTable(1, 3))
	seq3:add(self.outputCandidate)
	
	local concat2 = nn.ConcatTable()
	-- input: {x, h[t - 1], r, z}
	-- output: h[t - 1] (1 - z)  z ~h
	concat2:add(seq1)
	concat2:add(seq2)
	concat2:add(nn.SelectTable(4))
	concat2:add(seq3)
	
	cell:add(concat2)
	
	
	-- cell:add(nn.FlattenTable())
	
	
	local seq4 = nn.Sequential()
	seq4:add(nn.NarrowTable(1, 2))
	seq4:add(nn.CMulTable())
	
	local seq5 = nn.Sequential()
	seq5:add(nn.NarrowTable(3, 2))
	seq5:add(nn.CMulTable())
	
	-- input: {(1 - z) h[t - 1] z ~h}
	-- output: {(1 - z) * h[t - i], z * ~h}
	

	local concat3 = nn.ConcatTable()
	concat3:add(seq4):add(seq5)
	
	cell:add(concat3)
	
	
	cell:add(nn.CAddTable())

	
	return cell
end