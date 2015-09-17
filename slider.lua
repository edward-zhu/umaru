Slider = {
	pos			= 0, 
	total 		= 0,
	stride 		= 0,
	win_width 	= 0,
	height		= 0,
	width 		= 0,
	im = nil
}

setmetatable(Slider, {
	__call = 
		function (cls, ...)
			return cls:new(...)
		end
})

function Slider:new(o, win_width, stride)
	local o = o or {}
	self.win_width = win_width or 10
	self.stride = stride or self.win_width / 2
	setmetatable(o, self)
	self.__index = self
	return o
end

function Slider:load(im)
	assert(im:dim() == 2, "[Slider] a 2-dimensional tensor expected.")

	self.height = im:size(1)
	self.width = im:size(2)

	self.total = self.width / self.stride -- total # of windows

	self.pos = 0
	self.im = im
end

function Slider:slide()
	assert(self.im, "[Slider] need to load a image before slide")

	if self.pos >= self.total then
		return nil
	end

	local _start = self.pos * self.stride + 1
	local _end = _start + self.win_width
	local ret

	if _end > self.width then
		ret = torch.zeros(self.height, self.win_width)
		ret[{{1, self.height}, {1, self.width - _start + 1}}] = self.im[{{1, self.height}, {_start, self.width}}]

	else
		ret = self.im[{{1, self.height}, {_start, _start + self.win_width - 1}}]
	end

	self.pos = self.pos + 1

	return ret
end

function Slider:genSequence()
	local seq = {}

	local s = self:slide()

	while s do
		s = s:reshape(s:nElement())

		table.insert(seq, s)
		s = self:slide()
	end

	return seq
end