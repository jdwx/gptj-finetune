# gptj-finetune

This is a set of scripts designed to fine-tune GPT-J-6B in FP16 using the
HuggingFace Transformers library and Microsoft's DeepSpeed ZeRO-3
optimizations to fit the model in the amount of VRAM available on
consumer-grade GPUs.

There is
[a HuggingFace example for this](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py),
but it mixes a JSON config file with a ton of command-line options, devotes
a lot of code to parameter parsing and data management, and needs some tweaks
to work well in low memory conditions.  All of that is powerful and very
flexible (HuggingFace is awesome!), but makes it difficult to follow what's
going on that's specific to fine-tuning and using DeepSpeed. (At least it
did for me.)

So this implementation is designed to be much more streamlined, focusing on
doing the minimum needed to fine-tune in a way that is hopefully relatively
easy to follow.  All the training-related configuration options are in
Python dictionaries in the finetune.py file. No command line options or JSON
required.

All the defaults are chosen for small tasks.  I use this with a few
megabytes of fine-tuning text.  If you have 100GiB of text files to
train on, this example is probably not for you.

## Tuning

There are two parameters to tune based on the amount of VRAM you have.

### per\_device\_train\_batch\_size

This seems to require VRAM roughly equal to:

* 2.6 GiB per sample in the batch (per\_device\_train\_batch\_size of 4 => 10 GiB)
* 0.7 GiB for each power of two (per\_device\_train\_batch\_size of 4 => 1 GiB)
* 4.0 GiB just for showing up

So on a 3090 or 4090 with 24 GiB of VRAM, you can raise batch\_size as high
as 7, which will barely squeeze in. (calculation 2.6 * 7 + 0.7 * 2 + 4 = 23.6,
observed 23,568 MiB used at peak)

If you have 16 GiB your max batch size is probably 3.  If you have 12 GiB
your max batch size is probably 2.  If you have 8 GiB your max batch size
is probably 1, and it'll be very close (observed 8,104 MiB used).

As the name implies, this parameter is per-device, not aggregated. So if you
are training with multiple GPUs, base your calculations on the one with the
smallest amount of VRAM.

Note that increasing the batch size above 4 does *not* substantially improve
performance unless you can get to 8 (e.g., at least 32 GiB of VRAM on one card).
It's just another hyperparameter to search to figure out what will produce the
best model.  Also, the amount of VRAM used varies slightly from run to run,
so going too close to the line is not advisable.

### per\_device\_eval\_batch\_size

Evaluation takes less memory than training, so this can be set higher than
per\_device\_train\_batch\_size.

For small amounts of VRAM, you may need to turn this down as well.  The default
of 8 squeaks into 24 GiB of VRAM.  The performance increase from larger values
is very, very small.

### gradient\_accumulation\_steps

This parameter doesn't affect your VRAM usage, but it helps you compensate for
small batch sizes.

Setting this so that
per\_device\_train\_batch\_size * gradient\_accumulation\_steps = 128
is a good default, but it's a hyperparameter, so, check it.  (Using 128 is
more of what you would call a guideline than an actual rule.)

**However...** your actual full batch size is:

full\_batch\_size = per\_device\_train\_batch\_size * gradient\_accumulation\_steps * num\_gpus

Your training samples will be batched on this full boundary.  Any leftovers
that don't fill a batch will be dropped.  If you are fine-tuning on a small
number of samples, you can skip nearly half of your training data this
way if you're not careful.  E.g., if you take the defaults that give a full
batch size of 128, and you have 250 samples, you're going to get one batch of
128 samples and the other 122 get dropped.

Training for more epochs will mitigate that by giving all the samples a chance
to be chosen, but that can dramatically increase training time and is
susceptible to bad luck.  Instead, start with the largest size for
gradient_accumulation_steps that gives you a value in the general
vicinity of 128 and for which (number_of_samples % full_batch_size) is as
close to zero as possible.

## Instructions

To use this:

1. Put text files you want to fine-tune on in the text/ directory with .txt extensions. (Whatever cleaning you're going to do to them needs to happen before they hit this directory.)
2. Create and activate a venv based on requirements.txt. (See make-venv.sh.)
3. Tweak the parameters outlined above if you don't have 24 GiB of VRAM (per card).
4. Run: `python token_split.py`
5. Run: `deepspeed finetune.py`
6. Wait a very long time.
7. Run: `python infer_test.py` to test text generation using the fine-tuned model.
8. Enjoy your fine-tuned model, which will be located in the out/ directory.

## Credits

This code is based on the HuggingFace examples (and their
[transformers library](https://huggingface.co/docs/transformers/index)).
Understanding and some parameter values were obtained from various snippets
online pertaining to GPT-J-6B, most prominently
[this repo](https://github.com/mallorbc/Finetune_GPTNEO_GPTJ6B/tree/main/finetuning_repo)
by Blake Mallory and
[this gist](https://gist.github.com/kinoc/dca36b12b5e956688a9b92a87ba7c52c#gistcomment-4398821)
by Kinoc.
