folders = [
    'Baseline/arkansaw',
    'Baseline/ballou',
    'Baseline/denmark',
    'Baseline/eudora',
    'Baseline/hillsdale',
    'Baseline/mesic',
    'Baseline/pablo',
    'Baseline/quantico',
    'Baseline/roane',
    'Baseline/stokes'
]

txt = '/log.txt'
f_o = open("least_pe.txt","w+")

for folder in folders:
    pes = []
    with open(folder + txt) as f:
        for l in f:
            if l[0] + l[1] == 'Ph':
                #print(l[19:-1])
                pes.append(float(l[19:-1]))
    print(folder)
    f_o.write(folder+'\n')
    f_o.write(str(min(pes))+'\n')
    print(min(pes))
    print(pes.index(min(pes)))

f_o.close()
