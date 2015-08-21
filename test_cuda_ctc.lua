require 'cutorch'
require 'ctc'

t = torch.randn(100000):cuda()
t2 = torch.randn(100000)

timer = torch.Timer()

last = 0

for i = 1, (#t)[1] do
	t[i] = t[i] + 1
end

now = timer:time().real
print(now - last)
last = now


for i = 1, (#t2)[1] do
	t2[i] = t2[i] + 1
end


now = timer:time().real
print(now - last)
last = now

t3 = t:float()

for i = 1, (#t3)[1] do
	t3[i] = t3[i] + 1
end


now = timer:time().real
print(now - last)
last = now

print(t3)