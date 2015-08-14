utf8 = utf8 or require 'utf8'

Codec = {
	codec = {},
	codec_inv = {},
	codec_size = 0
}

setmetatable(Codec, {
	__call = 
		function (cls, ...)
			return cls:new(...)
		end
})

function Codec:new(o)
	o = o or {}
	setmetatable(o, self)
	self.__index = self
	return o
end

function Codec:encode(src)
	local result = {}
	for _, v, _ in utf8.iter(src) do
		table.insert(result, self.codec[v])
	end
	
	return result
end

function Codec:decode(src)
	local result = ""
	for _, v in ipairs(src) do
		result = result .. self.codec_inv[v]
	end
	
	return result
end