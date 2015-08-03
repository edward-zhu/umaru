require 'nn'
require 'cltorch'
require 'clnn'
-- require 'rnn'

net = nn.Sequential()

net:add(nn.Linear(3, 5))
net:add(nn.LogSoftMax())

net:cl()

x = torch.randn(3, 3):cl()
print(net:forward(x):exp())
