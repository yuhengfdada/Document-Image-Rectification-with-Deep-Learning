'''from pathlib import Path
p = Path('.')
li = list(p.glob('*.png'))
f = open('test.txt','w')
for i, file in enumerate(li):
    f.write(file.stem)
    f.write('\n')
f.close()'''