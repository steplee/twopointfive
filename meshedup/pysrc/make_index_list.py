

# Print edges of a cube
l = []
for i in range(8):
    for j in range(i):
        d = i^j
        if (d&(d-1)) == 0 and d>0: l.append(('{{ 0b{:03d} , 0b{:03d} }}'.format(int(bin(i)[2:]),int(bin(j)[2:]))))

