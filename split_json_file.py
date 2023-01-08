import json

f = open('logs.json')
  
data = json.load(f)
j = 0
  
for i in data:
  name = "logings\logs_" + str(j) + ".json"
  f_tmp = open(name, "w")
  f_tmp.write(str(i))
  f_tmp.close()
  j += 1
  
f.close()