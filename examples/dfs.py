def find(fv:int, d:dict) -> list:
    for k, v in d.items():
        if v == fv: return [k]
        elif type(v)==dict:
            l = find(fv, v)
            if not l: return None
            l.insert(0,k)
            return l
    return None

d = {
    'a': {
        'b': {
            'c': 3,
            'd': 4
        }
    },
    'e': 5
}
res = find(3, d)
print(res)

# def fl(lst:list, d:dict) -> int:
#     v = lst[0]
#     res = d[v]
#     if len(lst[1:])==0: return res
#     else: return fl(lst[1:], res)
# res = fl(res, d)
# print(res)

fl = lambda lst, d: d[lst[0]] if len(lst[1:])==0 else fl(lst[1:], d[lst[0]])
print(fl(res, d))
