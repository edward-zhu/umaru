require 'codec'
utf8 = require 'utf8'

Loader = {
	samples = {},
	codec_table = {},
	codec_inv = {},
	codec_size = 0,
	codec_obj = nil
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

function Loader:load(file)
	self.samples = {}
	local f = assert(io.open(file, "r"))
	for line in f:lines() do
		local src = line
		local gt = src:gsub(".png", ".gt.txt")
		local cf = assert(io.open(gt, "r"))
		local gt = cf:read("*line")
		cf:close()
		
		for _, c, _ in utf8.iter(gt) do
			if self.codec_table[c] == nil then
				self.codec_size = self.codec_size + 1
				self.codec_table[c] = self.codec_size
			end
			
		end
		
		table.insert(self.samples, {src = src, gt = gt})
	end
	f:close()
	
	for k, v in pairs(self.codec_table) do
		self.codec_inv[v] = k
	end
	
	return self.samples
end

function Loader:pick()
	local index = torch.random(#self.samples)
	return self.samples[index]
end

function Loader:codec()
	self.codec_obj = self.codec_obj or Codec:new{
		codec = self.codec_table,
		codec_inv = self.codec_inv,
		codec_size = self.codec_size
	}
	
	return self.codec_obj
end