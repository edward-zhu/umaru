f = open("data1.txt", "r")
lines = f.readlines()
f.close()

output = "data = torch.Tensor{\n"

for line in lines:
	nums = line.split()
	output += "\t{"
	output += ",".join(nums)
	output += "},\n"

output += "}"

f = open("data.lua", "w")
f.write(output)
f.close()