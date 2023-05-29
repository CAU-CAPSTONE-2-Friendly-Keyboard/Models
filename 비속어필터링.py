def get_data(filename):
    with open("./DB/" + filename) as f:
        lines = f.readlines()
    data = []
    for d in lines:
        data.append(d.strip())
    data.sort(key= lambda x : len(x))
    return data

def bad2star(line):
    used_badword = []
    for badword in ko_data:
        if badword in line:
            line = line.replace(badword,'*'*len(badword))
            used_badword.append(badword)
    return (line,used_badword)

ko_data = get_data("KO")
result, used_badword = bad2star('이제 그만 닥쳐봐')
print(result)
print(used_badword)
