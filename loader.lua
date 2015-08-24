require 'image'
require 'codec'
require 'normalizer'
utf8 = require 'utf8'

Loader = {
	samples = {},
	weights = nil,
	p = nil,
	codec_table = {},
	codec_inv = {},
	codec_size = 0,
	codec_obj = nil,
	threshold = 3,
	lambda = 3.0,
	pos = 0
}

setmetatable(Loader, {
	__call = 
		function (cls, ...)
			return cls:new(...)
		end
})

function Loader:new(o)
	o = o or {}
	setmetatable(o, self)
	self.__index = self
	return o
end

function Loader.__getNormalizedImage(src)
	local im = image.load(src, 1)

	if im:dim() == 3 then
		im = im[1]
	end

	output = torch.DoubleTensor()

	w = im:size()[2]
	h = im:size()[1]

	ones = torch.ones(h, w)

	im = ones - im
	normalizer.normalize(im:double(), output)
	return output:float()
end

function Loader:load(file)
	self.samples = {}
	local f = assert(io.open(file, "r"))
	for line in f:lines() do
		local src = line
		local gt = src:gsub("[.].*", ".gt.txt")
		local cf = io.open(gt, "r")
		local gt = cf:read("*line")
		cf:close()
		
		for _, c, _ in utf8.iter(gt) do
			if self.codec_table[c] == nil then
				self.codec_size = self.codec_size + 1
				self.codec_table[c] = self.codec_size
			end
			
		end
		
		table.insert(self.samples, {src = src, gt = gt, img = nil})
	end
	f:close()
	
	for k, v in pairs(self.codec_table) do
		self.codec_inv[v] = k
	end
	
	self.codec_obj = nil
	self.weights = nil
	
	-- return self.samples
end

function Loader:__pick(index)
	if self.samples[index].img == nil then
		self.samples[index].img = Loader.__getNormalizedImage(self.samples[index].src):t()
		if GPU_ENABLED then
			self.samples[index].img = self.samples[index].img:cuda()
		end
	end
	
	return self.samples[index]
end

function Loader:pick()
	local index = torch.random(#self.samples)
	
	return self:__pick(index)
end

function Loader:pickWithWeight()
	if self.weights == nil then
		self.weights = torch.zeros(#self.samples)
		for i, v in ipairs(self.samples) do
			self.weights[i] = math.pow(1.0 / math.max(utf8.len(v.gt), self.threshold), self.lambda)
		end
		self.weights = torch.div(self.weights, torch.sum(self.weights))
		
		self.p = torch.zeros(#self.samples)
		local i = 0
		self.p:apply(function()
			i = i + 1
			return torch.normal(1.0 / self.weights[i], 1.0 / self.weights[i] / 3.0) 
		end)
	end
	local _, index = torch.min(self.p, 1)
	index = index[1]
	self.p[index] = torch.normal(1.0 / self.weights[index], 1.0 / self.weights[index] / 3.0) + 1
	
	return self:__pick(index)
end

function Loader:pickInSequential()
	if self.pos <= #self.samples then
		self.pos = self.pos + 1
		return self:__pick(self.pos - 1)
	else
		return nil
	end
end

function Loader:updateWeight(lambda)
	self.lambda = lambda
	self.weights = nil
end

function Loader:codec()
	self.codec_obj = self.codec_obj or Codec:new{
		codec = self.codec_table,
		codec_inv = self.codec_inv,
		codec_size = self.codec_size
	}
	
	return self.codec_obj
end