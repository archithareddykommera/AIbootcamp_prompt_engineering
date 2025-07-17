import os
import sys
if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
import types
import json
import adalflow as adal
from typing import Dict ,Union
from adalflow.optim.types import ParameterType
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.datasets.types import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc


log_file = open("console_output.log", "a", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file



few_shot_template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
{# Few shot demos #}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %} 
<END_OF_SYSTEM_PROMPT>
<START_OF_USER> 
{{input_str}}
<END_OF_USER>
"""

model_configs = {
    "gpt_3_model" : {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-3.5-turbo-0125",
            "temperature": 0.0,
            "top_p": 0.99,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        },
    },

    "gpt_o3_mini_model" : {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "o3-mini"           
        },
    },

    "gpt_o1_model" : {
        "model_client": OpenAIClient(), 
        "model_kwargs": {
            "model": "o1",

        },
    },
    "gpt_4_model" : {
        "model_client": OpenAIClient(), 
        "model_kwargs": {
            "model": "gpt-4",
        },
    },
}

class ObjectCountTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

# replace the system prompt with the best prompt based on score from adal flow
        system_prompt = adal.Parameter(
            data="""
You are a helpful assistant that counts how many times a letter appears in the name of a country.You will be given a question that asks for the country with the most repeated letter in its name.
""",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
        )
        few_shot_demos = adal.Parameter(
            data=None,
            role_desc="To provide few shot demos to the language model",
            requires_opt=True,  # Changed to True for few-shot learning
            param_type=adal.ParameterType.DEMOS,
        )

        self.llm_counter = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=few_shot_template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "few_shot_demos": few_shot_demos,
            },
            use_cache=True,
        )

    def bicall(
            self, question: str, id: str = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
        return output

result = {}
question = "What country has the same letter repeated the most in its name?"


# Run the task pipeline for each model configuration to test the system prompt 
# Paste the system prompt with the best score from adal flow above 
# dont need to run the below before autooptimizing the prompts 

        # for model_name, config in model_configs.items():
        #     print(f"\nRunning for model:{model_name}")
        #     task_pipeline = ObjectCountTaskPipeline(**config)
        #     answer = task_pipeline(question, id="model_name")
        #     result[model_name]= answer


        # print(result)

########################################Below is the code for auto-optimizing the prompts########################################

"""### Load Datasets"""

def load_datasets(local_file = "data/letter_counting.json", max_examples: int = None):
    """Load train/val/test datasets from a local JSON file.
The JSON file must contain a list of objects with 'id', 'question', and 'answer' fields.
We'll simply split them equally (or you can customize splitting)"""

    with open(local_file, 'r', encoding="utf-8") as f:
        data = json.load(f)['examples']

    if max_examples is not None:
        data = data[:max_examples]      
    
    count = 0
    for row in data:
        row['id'] = count
        count += 1
    # Convert to Example objects
    examples = [Example(id=item['id'], question=item['input'], answer=item['output']) for item in data]
    n = len(examples)
    train_data = examples[:int(0.6 * n)]
    val_data = examples[int(0.6 * n):int(0.8 * n)]
    test_data = examples[int(0.8 * n):]
    return train_data, val_data, test_data

# check the datasets

train_data, val_data, test_data = load_datasets(max_examples=2)

class ObjectCountAdalComponent(adal.AdalComponent):  # noqa: F811
    def __init__(
            self,
            model_client: adal.ModelClient,
            model_kwargs: Dict,
            backward_engine_model_config: Dict,
            teacher_model_config: Dict,
            text_optimizer_model_config: Dict,
    ):
    
    
        task = ObjectCountTaskPipeline(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
        )
        super().__init__(task=task, eval_fn=eval_fn, loss_fn=loss_fn)

        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config
    

    def prepare_task(self, sample: Example):
        return self.task.bicall, {"question": sample.question, "id": sample.id}

    def prepare_eval(self, sample: Example, y_pred: adal.GeneratorOutput) -> float:
        y_label = -1
        if (y_pred is not None and y_pred.data is not None
        ):
            y_label = y_pred.data
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    def prepare_loss(self, sample: Example, pred: adal.Parameter):
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )

        pred.eval_input = pred.full_response.data
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}
    
def train(task_model_name="gpt_3_model", optimizer_model_name="gpt_o3_mini_model"):


    task_model_config = model_configs[task_model_name]
    optimizer_model_config = model_configs[optimizer_model_name]
    adal_component = ObjectCountAdalComponent(
        **task_model_config,
        teacher_model_config=optimizer_model_config,
        text_optimizer_model_config=optimizer_model_config,
        backward_engine_model_config=optimizer_model_config,
    )


    trainer = adal.Trainer(
        adaltask=adal_component,
        max_steps=12,  # 12 steps of LLM-AutoDiff, and 12 steps of few-shot learning(2)
        raw_shots=1,
        bootstrap_shots=1,
        strategy="random",
    )

    train_dataset, val_dataset, test_dataset = load_datasets()

    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
    optimized_pipeline = adal_component.task
    optimized_system_prompt = optimized_pipeline.llm_counter.prompt_kwargs["system_prompt"].data
    optimized_demos = optimized_pipeline.llm_counter.prompt_kwargs["few_shot_demos"].data

    # ✅ Print them
    print("\n===== Optimized System Prompt =====\n")
    print(optimized_system_prompt)

    # ✅ Optional: Save to JSON
    with open("optimized_prompt.json", "w", encoding="utf-8") as f:
        json.dump({
            "system_prompt": optimized_system_prompt,
            "few_shot_demos": optimized_demos
        }, f, indent=2, ensure_ascii=False)


######### comment the below code to run the task pipeline for each model configuration to test the system prompt #########

try:

    train()
finally:
    log_file.close()

