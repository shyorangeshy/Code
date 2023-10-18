import openai
import json

class ACEExample(object):
    def __init__(self, context, labels,com,com_num) -> None:
        self.context = context
        self.labels = labels
        self.com=com
        self.com_num=com_num
class ACEProcessor():
    def read_example(self, file_name):
        with open(file_name,'r',encoding="utf8") as fin:
            examples = []
            _corpus = json.load(fin)
            for traget in _corpus:
                context, label_str,com,com_num = traget[0], traget[1],traget[-2],traget[-1]
                labels = label_str.split(" ")
                assert len(context.split(" ")) == len(labels)
                example = ACEExample(context, labels,com,com_num)
                examples.append(example)
        return examples
    def convert_examples_to_style(self, examples):
        rel_all_B=["capable of ","causes ","is a ","manner of ","motivated by goal ","receives action "]
        all_need=[]
        for example in examples:
            one_example_need=[]
            one_example_need.append(example.context)
            one_example_need.append(" ".join(example.labels))
            example_tri=set(example.labels)
            choice=[]
            if len(example_tri)>1 or (len(example_tri)==1 and 'O' not in example_tri):
                 for i in example_tri:
                      if i !='O':
                           tri_index=example.labels.index(i)
                           if example.com[tri_index]!="-1":
                                choice.append((tri_index,['['+rel_all_B[int(example.com_num[tri_index].split(" ")[i])]+example.com[tri_index].split(" ")[i]+']' for i in range(1,len(example.com_num[tri_index].split(" ")))]))
            one_example_need.append(choice)
            all_need.append(one_example_need)
        return all_need
def get_response(prompt, temperature=0.1, max_tokens=2048):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=temperature,
    top_p=1,
    max_tokens=max_tokens,
    messages=[
      {"role": "user", "content": f"{prompt}"}
    ]
  )
  return completion
def get_correct_output(gpt_text,sort_option):
    sort_index=[]
    for i in sort_option:
        if i in gpt_text:
            sort_index.append(1)
        else:
            sort_index.append(0)
    
    if sum(sort_index)==1:
        return sort_option[sort_index.index(1)]
    else:
        return "%"

if __name__=="__main__":
    API_KEY=("",
         )
    openai.api_key = API_KEY[0]
    processor = ACEProcessor()
    examples = processor.read_example('./test.json')
    examples_new=processor.convert_examples_to_style(examples)
    for example in examples_new:
        final_comse=[]
        context,choices=example[0],example[-1]
        for count in range(len(choices)):

            prompt_input="Given the sentence after the symbol \"#\" , for the word \""+context.split(" ")[choices[count][0]]+"\" in this sentence, from the multiple meanings separated by the symbol \",\" after the symbol\"@\", select the one that most conforms to the semantic meaning of the word \""+context.split(" ")[choices[count][0]]+"\" in the sentence after \"#\". If there is no meaning that meets the conditions, please output it \"%\", otherwise please box your answer with \"[]\""+"\n" \
                +"#"+context +"\n"+  \
                "@"+",".join(choices[count][-1])
            print('\n',prompt_input)
            while (True):           
                try:
                    gpt_text = get_response(prompt_input)
                    print('\ngpt_oringal',gpt_text["choices"][0]["message"]["content"])
                    final_one=get_correct_output(gpt_text["choices"][0]["message"]["content"],choices[count][-1])
                    final_comse.append((choices[count][0],final_one))
                    print('\ngpt_choose',final_one)
                    break
                except Exception as e:
                    if isinstance(e, openai.error.RateLimitError) and \
                        'You exceeded your current quota, please check your plan and billing details.' in e.user_message:
                        break
                    else:           
                        print('\nerror')
                        continue
        example[-1]=final_comse

    with open('ta.json', 'w') as f:
        json.dump(examples_new, f ,indent=4)
