mem_view = memoryview(b'a'*10000)
{f'pos{i}':mem_view[i%10000:] for i in range(500000)}
