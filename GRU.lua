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

function GRU:updateOutput(input)
	local prevOutput, prevCell
	if self.step == 1 then
		prevOutput = self.zeroTensor
		
		assert(input:dim() == 1, "only support input with dimension == 1")
		
		self.zeroTensor:resize(self.outputSize):zero()
	else
		prevOutput = self.output
	end
	
	local output
	if self.train ~= false then
		self:recycle()
		local recurrentModule = self:getStepModule(self.step)
		
		-- print{input, prevOutput}
		
		output = recurrentModule:updateOutput{input, prevOutput}
	else
		output = self.recurrentModule:updateOutput{input, prevOutput}
	end
	
	if self.train ~= false then
		local input_ = self.inputs[self.step]
		self.inputs[self.step] = self.copyInputs
			and nn.rnn.recursiveCopy(input_, input)
			or nn.rnn.recursiveSet(input_, input)
	end
	
	self.outputs[self.step] = output
	
	self.output = output
	
	self.step = self.step + 1
	self.gradParametersAccumulated = false
	
	return self.output
end

function GRU:backwardThroughTime()
	assert(self.step > 1, "expecting at least one updateOutput")
	self.gradInputs = {}
	local rho = math.min(self.rho, self.step - 1)
	local stop = self.step - rho
	if self.fastBackward then
		local gradPrevOutput
		for step = self.step - 1, math.max(stop, 1), -1 do
			local recurrentModule = self:getStepModule(step)
			
			local gradOutput= self.gradOutputs[step]
			if gradPrevOutput then
				self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradPrevOutput)
				nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
				gradOutput = self._gradOutputs[step]
			end
			
			local scale = self.scales[step]
			local output = (step == 1) and self.zeroTensor or self.outputs[step - 1]
			
			local inputTable = {self.inputs[step], output}
			local gradInputTable = recurrentModule:backward(inputTable, gradOutput, scale)
			
			local gradInput, gradPrevOutput = unpack(gradInputTable)
			
			table.insert(self.gradInputs, 1, gradInput)
		end
		return gradInput
	else
		local gradInput = self:updateGradInputThroughTime()
		self:accGradParametersThroughTime()
		return gradInput
	end
end

function GRU:updateGradInputThroughTime()
	assert(self.step > 1, "expecting at least one updateOutput")
	self.gradInputs = {}
	local gradInput, gradPrevOutput
	local rho = math.min(self.rho, self.step - 1)
	local stop = self.step - rho
	
	for step = self.step - 1, math.max(stop, 1), -1 do
		local recurrentModule = self:getStepModule(step)
		
		local gradOutput = self.gradOutputs[step]
		if gradPrevOutput then
			self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradPrevOutput)
			nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
			gradOutput = self._gradOutputs[step]
		end
		
		local output = (step == 1) and self.zeroTensor or self.outputs[step - 1]
		local inputTable = {self.inputs[step], output}
		local gradInputTable = recurrentModule:updateGradInput(inputTable, gradOutput)
		
		gradInput, gradPrevOutput = unpack(gradInputTable)
		
		table.insert(self.gradInputs, 1, gradInput)
	end
	
	return gradInput
end

function GRU:accGradParametersThroughTime()
	local rho = math.min(self.rho, self.step - 1)
	local stop = self.step - rho
	for step = self.step - 1, math.max(stop, 1), -1 do
		local recurrentModule = self:getStepModule(step)
		
		local scale = self.scales[step]
		local output = (step == 1) and self.zeroTensor or self.outputs[step - 1]
		local inputTable = {self.inputs[step], output}
		local gradOutput = (step == self.step - 1) and self.gradOutputs[step] or self._gradOutputs[step]
		
		
		
		recurrentModule:accGradParameters(inputTable, gradOutput, scale)
	end
	
	self.gradParametersAccumulated = true
	return gradInput
end

function GRU:accUpdateGradParametersThroughTime(lr)
	local rho = math.min(self.rho, self.step - 1)
	local stop = self.step - rho
	
	for step = self.step - 1, math.max(stop, 1), -1 do
		local recurrentModule = self:getStepModule(step)
		
		local scale = self.scales[step]
		local output = (step == 1) and self.zeroTensor or self.outputs[step - 1]
		local inputTable = {self.inputs[step], output}
		local gradOutput = (step == self.step - 1) and self.gradOutputs[step] or self._gradOutputs[step]
		local gradOutputTable = {self.gradOutputs[step]}
		
		recurrentModule:accUpdateGradParameters(inputTable, gradOutput, lr * scale)
	end
	
	return gradInput
end