require 'loader'

l = Loader()
l:load("1.txt")

codec = l:codec()

str = codec:encode("雪之下")

print(str)

str = codec:decode({1, 2, 3})

print(str)

