utf8 = require 'utf8'

function utf8.levenshtein(str1, str2)
	local len1 = utf8.len(str1)
	local len2 = utf8.len(str2)

	local matrix = {}
	local cost

	if (len1 == 0) then
		return len2
	elseif (len2 == 0) then
		return len1
	elseif (str1 == str2) then
		return 0
	end

	for i = 0, len1 do
		matrix[i] = {}
		matrix[i][0] = i
	end
	for j = 1, len2 do
		matrix[0][j] = j
	end

	for i = 1, len1 do
		for j = 1, len2 do
			if (utf8.at(str1, i) == utf8.at(str2, j)) then
				cost = 0
			else
				cost = 1
			end

			matrix[i][j] = math.min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)
		end
	end

	return matrix[len1][len2]
end
