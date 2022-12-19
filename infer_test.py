import sys
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


def main(prompt: str):

    # Model to use for generation.
    model_name_or_path = "out/"  # default path of fine-tuned model
    # model_name_or_path = "EleutherAI/gpt-j-6B"  # For comparison purposes.

    print("*** Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    print("*** Loading model. (This takes a while.)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16
    )

    print("*** Generating.")
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0
    )
    out = pipe(
        prompt,
        num_return_sequences=4,
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id
    )

    print("*** Outputting.")
    for result in out:
        gen = result['generated_text'][len(prompt):].strip()
        print(prompt, "[", gen, "]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(" ".join(sys.argv[1:]))
    else:
        main("HuggingFace is the")
