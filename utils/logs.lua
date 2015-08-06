logs = {
	EXP_MAX		= 1e100,
	EXP_MIN		= 1e-100,
	LOG_ZERO	= -1e100,
	LOG_INF		= 1e100
}

logs.EXP_LIMIT = math.log(logs.EXP_MAX)

function logs.safe_log(x)
	if x == 0 then
		return logs.LOG_ZERO
	elseif x > 0 then
		return math.log(x)
	else
		error("passing a negtive number to the log function.")
	end
end

function logs.safe_exp(x)
	if x == logs.LOG_ZERO then
		return 0
	end
	if x >= logs.EXP_LIMIT then
		return EXP_MAX
	end
	return math.exp(x)
end

function logs.log_add(x, y)
	if x == logs.LOG_ZERO then
		return y
	end
	if y == logs.LOG_ZERO then
		return x
	end
	if x < y then
		return y + math.log(1.0 + logs.safe_exp(x - y))
	else
		return x + math.log(1.0 + logs.safe_exp(y - x))
	end
end

function logs.log_sub(x, y)
	if y == logs.LOG_ZERO then
		return x
	end
	if y >= x then
		return logs.LOG_ZERO
	end
	return x + math.log(1.0 - logs.safe_exp(y - x))
end

function logs.log_mul(x, y)
	if y == logs.LOG_ZERO or x == logs.LOG_ZERO then
		return logs.LOG_ZERO
	end
	
	return x + y
end

function logs.log_div(x, y)
	if x == logs.LOG_ZERO then
		return logs.LOG_ZERO
	end
	
	if y == logs.LOG_ZERO then
		return logs.LOG_INF
	end
	
	return x - y
end

function logs.log_sum(...)
	local arg = table.pack(...)
	if arg["n"] == 1 then
		return arg[1]
	end
	
	local max = math.max(unpack(arg))
	
	local result = 0.0
	
	for i, v in ipairs(arg) do
		result = result + logs.safe_exp(v - max)
	end
	
	result = max + logs.safe_log(result)
	
	return result
end

