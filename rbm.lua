--[[
Copyright (c) 2015, Nils Hammerla
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
--]]

local Rbm = {}
Rbm.__index = Rbm

function Rbm.new(arg)
    local self = setmetatable({}, Rbm)

    -- parse parameters
    -- network
    self.n_visible = arg.n_visible
    self.n_hidden = arg.n_hidden

    -- unit type, default: sigmoids
    self.v_activation = arg.v_activation or 'binary'
    self.h_activation = arg.v_activation or 'binary'
    self.useStates = arg.useStates or true

    -- learning
    self.learningRate = arg.learningRate or 0.1
    self.minibatchSize = arg.minibatchSize or 100
    self.momentum = arg.momemtum
    self.momentumAfter = arg.momentumAfter or 5
    self.CDsteps = arg.CDsteps or 1

    -- regularisation
    self.weightCost = arg.weightCost or -0.0000001

    -- some variables to save progress
    self.epoch = 0
    self.epochError = 0

    -- initialise weights
    self:initWeights()

    -- set up sampling functions for visible and hidden units
    self.binarySampler = function(input)
        local a = nn.Sigmoid()(input)
        local s = torch.gt(a, torch.Tensor(a:size()):rand(a:size())):type(torch.getdefaulttensortype())
        return a,s
    end
    self.reluSampler = function(input)
        local n = torch.Tensor(input:size()):randn(input:size())
        local a = nn.ReLU()(input+n)
        return a,a
    end
    self.gaussSampler = function(input)
        return input, input+torch.Tensor(input:size()):randn(input:size())
    end

    if self.h_activation == 'binary' then
        self.h_sampler = self.binarySampler
    elseif self.h_activation == 'relu' then
        self.h_sampler = self.reluSampler
    elseif self.h_activation == 'gaussian' then
        self.h_sampler = self.gaussSampler
    end

    if self.v_activation == 'binary' then
        self.v_sampler = self.binarySampler
    elseif self.v_activation == 'relu' then
        self.v_sampler = self.reluSampler
    elseif self.v_activation == 'gaussian' then
        self.v_sampler = self.gaussSampler
    end

    -- Set up an encoder and decoder (nn)
    -- Weights in these networks point to the weight tensor in the rbm
    self.encoder, self.decoder = self:getNN()

    return self
end

function Rbm.initWeights(self)
    -- Initialise weights
    -- Basically reset the whole thing
    local nV, nH = self.n_visible, self.n_hidden

    self.W     = torch.Tensor(nV,nH):randn(nV, nH):mul(0.1)
    self.hbias = torch.Tensor(1,nH):zeros(1,nH)
    self.vbias = torch.Tensor(1,nV):zeros(1,nV)

    self.Winc = torch.Tensor(nV,nH):zeros(nV, nH)
    self.hbias = torch.Tensor(1,nH):zeros(1, nH)
    self.vbias = torch.Tensor(1,nV):zeros(1, nV)
end

function Rbm.HgivenV(self, v_sample)
    -- sample hidden layer based on visible layer
    local pre, post, states
    -- h_in = v*W + h_bias
    pre  = torch.mm(v_sample, self.W):add(self.hbias:repeatTensor(v_sample:size(1),1))
    post, states = self.h_sampler(pre)

    if self.useStates == false then
        states = post
    end

    return pre, post, states
end

function Rbm.VgivenH(self, h_sample)
    -- sample visible layer given hidden layer
    local pre, post, states
    -- v_in = h*W' + v_bias
    pre  = torch.mm(h_sample, self.W:t()):add(self.vbias:repeatTensor(h_sample:size(1),1))
    post, states = self.v_sampler(pre)

    if self.useStates == false then
        states = post
    end

    return pre, post, states
end

function Rbm.sampleChain(self, h, CDsteps)
    -- sample markov chain for contrastive divergence training (starting from hidden state h)
    start = h
    local v_mean, v_sample, h_mean, h_sample
    for i=1, CDsteps do
        _, v_mean, v_sample = self:VgivenH(start)
        _, h_mean, h_sample = self:HgivenV(v_sample)
        start = h_sample -- reset
    end
    return v_mean, v_sample, h_mean, h_sample
end

function Rbm.freeEnergy(self,sample)
    -- calculate free energy (for convergence check if required)
    -- This is just for binary-binary rbms! (so far)
    local wx_b = torch.mm(sample, self.W):add(self.hbias:repeatTensor(sample:size(1),1))
    local vbias_term = torch.mm(sample, self.vbias:t())
    local hidden_term = torch.log(torch.add(wx_b:exp(),1)):sum(2)
    local e = -hidden_term - vbias_term
    return e
end

function Rbm.updateParameters(self, v0)
    -- calculate gradients for W, vbias, hbias and update weight matrices

    local momentum

    -- if we use momentum, then check which we want to use
    if self.momemtum then
        momentum = self.momentum[1]
    end
    if self.momentum and self.epoch > self.momentumAfter then
        momentum = self.momentum[2]
    end

    -- sample first hidden layer
    local _, h0_mean, h0_sample = self:HgivenV(v0)

    -- get sample from markov chain
    local v_model_mean, v_model_sample, h_model_mean, h_model_sample = self:sampleChain(h0_sample, self.CDsteps)

    if momentum then
        -- if momentum is set then memorise weights (soo much memory for this)
        local ww = self.Winc:clone()
        local vb = self.vbias:clone()
        local hb = self.hbias:clone()
    end

    -- calculate derivatives and update matrices

    -- calculate weight derivatives.
    -- This looks a bit weird but splitting up the calculations is apparently
    -- more memory efficient (still a mystery to me).

    -- formulas (from Geoff Hinton and Ruslan Salakhutdinov's matlab code):
    -- vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    -- visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    -- hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

    self.Winc = torch.mm(v0:t(),h0_mean)                            -- posprods
    self.Winc:add(torch.mm(v_model_mean:t(), h_model_mean):mul(-1)) -- -negprods
    self.Winc:div(v0:size(1))                                       -- / numsamples
    self.Winc:add(torch.mul(self.W, self.weightCost))               -- regularisation
    self.Winc:mul(self.learningRate)                                -- * learning rate

    -- visible bias
    self.vbiasinc = v0:sum(1)
    self.vbiasinc:add(-v_model_mean:sum(1))
    self.vbiasinc:mul(self.learningRate)
    self.vbiasinc:div(v0:size(1))

    -- hidden bias
    self.hbiasinc = h0_mean:sum(1)
    self.hbiasinc:add(-h_model_mean:sum(1))
    self.hbiasinc:mul(self.learningRate)
    self.hbiasinc:div(v0:size(1))

    if self.momentum and self.epoch > 1 then
        -- momentum? if so add derivatives*momentum
        self.Winc:add(torch.mul(ww, momentum))
        self.vbiasinc:add(torch.mul(vb, momentum))
        self.hbiasinc:add(torch.mul(hb, momentum))
    end

    -- update params
    self.W:add(self.Winc)
    self.vbias:add(self.vbiasinc)
    self.hbias:add(self.hbiasinc)
end

function Rbm.train(self, data, epochs)
    -- train an rbm on data for a number of epochs. If data is large this may run
    -- into memory issues. If so use updateParameters()
    local e, a
    for e=1,epochs do
        xlua.progress(e, epochs)

        self.epoch = self.epoch + 1
        for i=1,data:size(1),self.minibatchSize do
            self:updateParameters(data[{{i,i+self.minibatchSize-1}, {}}])
        end

        collectgarbage() -- this is needed, not sure where the leak is
    end
end

function Rbm.getNN(self)
    -- construct an encoder and decoder network
    -- these share the memory with the rbm so no just little overhead. This should
    -- make stacking rbms and fine-tuning with backprop much easier.

    local encoder, decoder

    -- encoder: visible -> hidden
    encoder = nn.Sequential()
    encoder:add(nn.Linear(self.n_visible, self.n_hidden))
    if self.h_activation == 'binary' then
        encoder:add(nn.Sigmoid())
    elseif self.h_activation == 'relu' then
        encoder:add(nn.ReLU())
    elseif self.h_activation == 'gaussian' then
        -- linear is fine
    end
    encoder:get(1).weight = self.W:t() -- weight matrix is flipped in nn
    encoder:get(1).bias = self.hbias[1]

    -- decoder: hidden -> visible
    decoder = nn.Sequential()
    decoder:add(nn.Linear(self.n_hidden, self.n_visible))
    if self.v_activation == 'binary' then
        decoder:add(nn.Sigmoid())
    elseif self.v_activation == 'relu' then
        decoder:add(nn.ReLU())
    elseif self.v_activation == 'gaussian' then
        -- linear is fine
    end
    decoder:get(1).weight = self.W
    decoder:get(1).bias = self.vbias[1]

    return encoder, decoder
end

function Rbm.fromNN(self, encoder, decoder)
    -- Parsing encoder (linear layer + non-linearity) for weights and hidden bias,
    -- and hidden unit actiation. Parsing decoder for visible bias (and activation).
    --
    -- Assumes:
    --  encoder = nn.Sequential()
    --  encoder:add(nn.Linear())
    --  encoder:add(nn.Sigmoid())
    --
    self.W = encoder.get(1).weight:t()
    self.hbias[1] = encoder.get(1).bias
    self.h_activation = encoder.get(2)
    self.vbias[1] = decoder.get(1).bias
    self.v_activation = decoder.get(2)
end

return Rbm
