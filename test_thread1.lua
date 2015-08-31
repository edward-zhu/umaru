require 'nn'

net = nn.Linear(10, 10)

params, grad_params = net:getParameters()

n = net:clone()

p, g = n:getParameters()

p = params

net:share(n, 'weight', 'bias')


inp = torch.randn(10)

n:forward(inp)

grad = torch.randn(10)

n:backward(inp, grad)

n:updateParameters(1e-3)

print(p:sum())
print(params:sum())









