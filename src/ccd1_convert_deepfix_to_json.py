import json
import pandas as pd

# [{input: INCORRECT_CODE, output: STATEMENT}, ...]
def ccd1_convert_deepfix_to_json(filepaths):
  COLUMNS = ['Correct_code', 'Incorrect_code', 'Statement']

  def read_examples(filename):
    """Read examples from filename for DeepFix style training Line stmt Line stmt Line stmt ..."""
    examples = []
    data = pd.read_csv(filename, sep='\t', header=[0]).drop(columns=COLUMNS[0])
    for idx, elem in data.iterrows():
      code = ' '.join(elem[COLUMNS[1]].split('||| '))[:-1].strip()
      stmt = elem[COLUMNS[2]].strip()

      examples.append({
        'input': code,
        'output': stmt
      })
    return examples 
  
  for filepath in filepaths:
    examples = read_examples(filepath)
    with open(filepath.replace('.txt', '.json'), 'w') as f:
      for example in examples:
        f.write(json.dumps(example) + '\n')