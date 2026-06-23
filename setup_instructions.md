## Setup instructions for my machine

I ran into a major works-on-my-machine- issue when trying to use this code base. I made the following changes to get the program working.

The original environment.yaml was full of deprecated modules, so I just created a new yaml by trial and error. Additional modules must be installed in addition to those in the new yaml. The following list might be enough.

```bash
pip install git+https://github.com/davidbau/baukit
pip install flax
pip install t5[gcp]
pip install openai
pip install evaluate
```

### In collect_activations.py

New imports to go around the deprecated llama module:

```python
import transformers.models.llama.tokenization_llama as tf_token_llama
import transformers.models.llama.modeling_llama as tf_model_llama

#tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
tokenizer = tf_token_llama.LlamaTokenizer.from_pretrained(MODEL)
print("Tokenizer ok")

#model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
model = tf_model_llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
```

Change the dataset path, and also the local paths to avoid directory-not-found errors

```python
dataset = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')['validation']

# The ACT-part is missing from all paths by default. This is seen in other files as well
ref_df = pd.read_csv('ACT/TruthfulQA/TruthfulQA.csv')
```
The actual activation collection will cause a memory leak in certain conditions (e.g. free Google Colab). A solution to this is splitting the activation collection to chunks, so that the memory does not get filled up. This is the code after the "Getting activations"- print command

### In llama/init.py

Comment out the is_flax_available import. The program does not necessarily need it. The import is used in one place in the init file, and that can just be commented out as follows:

```python
#try:
#    if not is_flax_available():
#        raise OptionalDependencyNotAvailable()
#except OptionalDependencyNotAvailable:
#    pass
#else:
#    _import_structure["modeling_flax_llama"] = ["FlaxLlamaForCausalLM" "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"]
_import_structure["modeling_flax_llama"] = ["FlaxLlamaForCausalLM" "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"]
```
### In utils.py, tokenized_tqa_all()

When iterating through the dataset questions, the program will throw an IndexError if the category is null. This can be fixed by setting the category to "Unknown".

```python
# This takes the row where the currect question is located, and selects the Category-column 
new_category = ref_df.loc[ref_df['Question'] == question, 'Category']

# If category is missing i.e. length is zero, set category to unknown
if len(new_category) == 0:
    category = "Unknown"
else:
    category = ref_df.loc[ref_df['Question'] == question, 'Category'].iloc[0] if ref_df is not None else 'Unknown'
```

### In utils.py, get_llama_activations_bau()

This is the trickies one in my opinion. Changing the imports in ```collect_activations.py``` to go around the llama module caused the head_out attribute to stop working. I fixed the issue by replacing it with o_proj, but the detailed consequences of this are unknown to me. The program ran fine, but this might have contorted the outcome.

```python
#HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
```

Detailed explanation about this from Claude (don't trust blindly): "Note that o_proj captures the output projection (after heads are concatenated), not individual head outputs. If the ACT code downstream expects per-head activations and reshapes them assuming head_out gives one tensor per head, you may need to also adjust how the activations are sliced after collection. Check how head_wise_activations is used after get_llama_activations_bau returns to see if that's an issue"

Another change is that we clear the cache. This is related to the custom batch processing we use with activation collecting.

```python
# Clear cache  
del ret
torch.cuda.empty_cache()
```

### Commands

Evaluate
```bash
python ACT/valid_2_fold.py --model_name llama_7B --num_heads 24 --alpha 12 --n_clusters 3 --probe_base_weight 0 --judge_name "GPT-4.1 nano" --info_name "GPT-4.1 nano"
```