require 'image'
require 'normalizer/normalizer'

im = image.load("1.png", 1)

if im:dim() == 3 then
	im = im[1]
end

output = torch.Tensor()

w = im:size()[2]
h = im:size()[1]

ones = torch.ones(h, w)

im = ones - im

normalizer.normalize(im, output)

image.save("out.png", output)


