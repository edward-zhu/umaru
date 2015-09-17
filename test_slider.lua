require 'image'
require 'slider'


im = image.load('scaled.png')[1]

slider = Slider()
slider:load(im)

-- print(im)

local s = slider:slide()

while s do
	image.display(s)
	s = slider:slide()
end

-- image.display(im[{{1, im:size(1)}, {50, 150}}])

