require 'normalizer'
a = torch.range(1, 50):reshape(5, 10)
normalizer.gauss2d(a, 3.0, 4.0)
print(a)