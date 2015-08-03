require 'torch'
require 'optim'
require 'nn'

data = torch.Tensor{
    {40, 6, 4},
    {44, 10, 4},
    {46, 12, 5},
    {48, 14, 7},
    {52, 16, 9},
    {58, 18, 12},
    {60, 22, 14},
    {68, 24, 20},
    {74, 26, 21},
    {80, 32, 24}
}

model = nn.Sequential()
ninputs = 2;
noutputs = 1;

model:add(nn.Linear(ninputs, noutputs))

criterion = nn.MSECriterion()

x, dl_dx = model:getParameters()

feval = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end

    _nidx_ = (_nidx_ or 0) + 1
    if _nidx_ > (#data)[1] then _n_idx_ = 1 end

    local sample = data[_nidx_]
    local target = sample[{ {1} }]
    local inputs = sample[{ {2, 3} }]

    dl_dx:zero()

    local loss_x = criterion:forward(model:forward(inputs), target)
    model:backward(inputs, criterion:backward(model.output, tatget))

    return loss_x, dl_dx

end

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

for i = 1, 1e4 do
    current_loss = 0

    for i = 1, (#data)[1] do
