import json
import sys
import os
import shutil

import humaneval_command


# def humaneval_codegen_finetune_output(
#   model,
#   model_name,
#   tokenizer,
#   input_file,
#   output_file,
#   args: sg_args.GenerationArguments,
# ):
#   codegen_output = json.load(open(input_file, 'r'))
#   codegen_output['model'] = model_name
#   is_incoder = 'incoder' in model_name.lower()
#   if is_incoder:
#     print('🧂 incoder detected. dedicate EOS token provide.');

#   starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#   timings = []
#   oom = 0
#   memory_allocated, memory_reserved = 0, 0
#   for i, filename in enumerate(codegen_output['data']):
#     text = codegen_output['data'][filename]['input']
#     print(i + 1, 'generating', filename)
    
#     first_input = None
#     eos_id = None
#     if is_incoder:
#       first_input = inputs.input_ids
#       eos_id = tokenizer.convert_tokens_to_ids('<|endofmask|>')
#     else: 
#       # TODO: 전체 다 넣던거에서 input_ids 혹은 첫 인자만 넣도록 바뀌였음
#       first_input = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
#       eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

#     starter.record()
#     try:
#       inputs = tokenizer(text, return_tensors="pt").to('cuda')
#       print(inputs.input_ids.dtype)
#       print(inputs.attention_mask.dtype)
#       generated_ids = model.generate(
#         first_input,
#         max_new_tokens=args.max_new_tokens,
#         num_beams=args.num_beams,
#         num_return_sequences=args.num_beams,
#         early_stopping=True, 

#         pad_token_id=eos_id, eos_token_id=eos_id,

#         generation_config=GenerationConfig(
#           do_sample=args.do_sample,
#           max_new_tokens=args.max_new_tokens,
#           top_p=args.top_p,
#           temperature=args.temperature,
#         )
#       )
#     except Exception as e:
#       print(e)
#       oom += 1
#       continue
#     ender.record()
#     torch.cuda.synchronize()
#     curr_time = starter.elapsed_time(ender)
#     timings.append(curr_time)
    
#     total_allocated, total_reserved = 0, 0
#     total_allocated += torch.cuda.memory_allocated(torch.device(model.device)) / (1024 * 1024)
#     total_reserved += torch.cuda.memory_reserved(torch.device(model.device)) / (1024 * 1024)
#     if total_allocated > memory_allocated:
#       memory_allocated = total_allocated
#     if total_reserved > memory_reserved:
#       memory_reserved = total_reserved

#     print(curr_time, memory_allocated, memory_reserved, oom)

#     output = []
#     for generated_id in generated_ids:
#       output.append(tokenizer.decode(generated_id, skip_special_tokens=False))
#     codegen_output['data'][filename]['output'] = output
#     json.dump(codegen_output, open(output_file, 'w'), indent=2)
#   codegen_output['time'] = int(np.sum(timings) / 1000)
#   json.dump(codegen_output, open(output_file, 'w'), indent=2)

def codegen_output_to_patch(output):
  start_index = 0
  if '// fixed lines: \n' in output:
    start_index = output.index('// fixed lines: \n') + len('// fixed lines: \n')
  output = output[start_index: ]
  end_index = len(output)
  if '<|endoftext|>' in output:
    end_index = output.index('<|endoftext|>')
  output = output[: end_index]
  return output