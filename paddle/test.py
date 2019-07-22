def test():
	train_reader = {'Name':'Alice','Age':15}
	return train_reader
train_reader = lambda:test()
print train_reader()
