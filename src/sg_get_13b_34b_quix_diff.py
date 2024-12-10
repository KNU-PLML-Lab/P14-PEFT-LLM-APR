import json
import os

def sg_get_13b_34b_quix_diff():
  cl13b = '/home/yglee/WorkspaceLabModels/codellama_13b_v8/quixbugs_finetune_validate.json'
  cl34b = '/home/yglee/WorkspaceLabModels/codellama_34b_v8/quixbugs_finetune_validate.json'
  with open(cl13b, 'r') as f:
    cl13b_raw = json.load(f)
  with open(cl34b, 'r') as f:
    cl34b_raw = json.load(f)

  cl13b_data = {}
  cl34b_data = {}
  for key, data in cl13b_raw.get('data').items():
    cl13b_data[key] = False

    _output = data.get('output')
    for _o in _output:
      if _o.get('correctness') == 'plausible':
        cl13b_data[key] = True
        break
      
  for key, data in cl34b_raw.get('data').items():
    cl34b_data[key] = False
    
    _output = data.get('output')
    for _o in _output:
      if _o.get('correctness') == 'plausible':
        cl34b_data[key] = True
        break

  # create csv
  csv = []
  for key in cl13b_data.keys():
    csv.append({
      'key': key,
      'cl13b': cl13b_data.get(key, None),
      'cl34b': cl34b_data.get(key, None),
    })
  with open('outputs/cl13b_cl34b_diff.csv', 'w') as f:
    f.write('key,cl13b,cl34b\n')
    for row in csv:
      f.write(f"{row['key']},{row['cl13b']},{row['cl34b']}\n")

  with open('outputs/cl13b_cl34b_diff_codes.txt', 'w') as f:
    for key in cl13b_data.keys():
      if cl13b_data.get(key) != cl34b_data.get(key) and cl13b_data.get(key) is not None and cl34b_data.get(key) is not None:
        f.write(f"// PROBLEM: {key}\n")

        clinput = cl13b_raw.get('data').get(key).get('input')
        cl13b_outputs = cl13b_raw.get('data').get(key).get('output')
        cl34b_outputs = cl34b_raw.get('data').get(key).get('output')
        f.write(clinput); f.write('\n// 13B ----------\n')
        for cl13b_output in cl13b_outputs:
          correctness = cl13b_output.get('correctness')
          patch = cl13b_output.get('patch')
          f.write(f"{'✅' if correctness == 'plausible' else '❌'}: {json.dumps(patch)}\n")
        f.write('// 34B ----------\n')
        for cl34b_output in cl34b_outputs:
          correctness = cl34b_output.get('correctness')
          patch = cl34b_output.get('patch')
          f.write(f"{'✅' if correctness == 'plausible' else '❌'}: {json.dumps(patch)}\n")
        f.write('//==========\n')



if __name__ == '__main__':
  sg_get_13b_34b_quix_diff()
