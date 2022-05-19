a = b'a'*10000
{f'pos{i}':a[i%10000:] for i in range(500000)}
