

from lib import get_data_from_pickle
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments


import sys


# All the stuff that says "auto" here will be populated by the HuggingFace
# Trainer based on training_args (below) and its internal computations.
# This way we don't have to manually keep DeepSpeed and Trainer in sync.
deepspeed_args = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 12,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": False
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}


# noinspection PyTypeChecker
training_args = TrainingArguments(
    deepspeed=deepspeed_args,
    do_train=True,
    do_eval=True,
    fp16=True,
    evaluation_strategy="epoch",
    gradient_checkpointing=True,
    output_dir="out",
    eval_steps=1,
    learning_rate=5e-06,
    warmup_steps=10,

    # These control your GPU memory usage. See README.md for details.
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,

    # Gradient accumulation makes up for having a small batch size.
    # See README.md for tuning information. It's tricky on small
    # datasets.
    gradient_accumulation_steps=32,

    # The number of epochs to train for.
    num_train_epochs=1,

    # Since the default is to train for one epoch, there is no need
    # to save intermediate models.
    save_strategy="no",

    # Uncomment these parameters to save checkpoints.
    # Do this if your model takes over an hour to fine-tune.
    # Otherwise, it is a waste of time and disk I/O.
    # save_strategy="epoch",
    # save_total_limit=2,  # Each checkpoint is about 80 GiB on disk.

    # Be careful with this option, as it positively *explodes* DRAM usage
    # for some reason.  It uses nearly 512 GiB of system RAM and swap to
    # load the final model using two 3090s.
    # load_best_model_at_end=True,
)


def get_data(train_path: str, test_path: str):
    train_set = list(get_data_from_pickle(train_path))
    test_set = list(get_data_from_pickle(test_path))
    return train_set, test_set


def get_model(model_name_or_path: str):
    config = AutoConfig.from_pretrained(
        model_name_or_path, cache_dir=None, use_cache=False
    )
    # config.gradient_checkpointing = True
    # print("Config =", config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=None,
    )
    return model


def train(source: str, dest: str) -> None:
    train_set, test_set = get_data("train_chunks.pkl", "test_chunks.pkl")
    model = get_model(source)
    training_args.output_dir = dest
    trainer = Trainer(
        model,
        train_dataset=train_set,
        eval_dataset=test_set,
        args=training_args
    )
    print("*** I AM TRAINING! ***")
    trainer.train()
    print("*** I AM SAVING! ***")
    trainer.save_model(dest)
    print("*** I AM DONE! ***")


def main() -> int:
    # W&B is a great tool and I recommend using the free tier to
    # visualize your training stats, particularly eval loss, as this
    # is a good way to see how much training is worthwhile and when
    # overfitting kicks in. But if you don't have it and don't want it,
    # uncomment this.
    # os.environ["WANDB_DISABLED"] = "true"

    args = []
    for arg in sys.argv[1:]:
        if not arg.startswith("--local_rank="):
            args.append(arg)
    if len(args) == 0:
        train("EleutherAI/gpt-j-6B", "out/")
    elif len(args) == 1:
        train("EleutherAI/gpt-j-6B", args[0])
    elif len(args) == 2:
        train(args[0], args[1])
    else:
        print("Usage: deepspeed", sys.argv[0], "[[input_model] output_model]")
        return 10
    return 0


if __name__ == "__main__":
    sys.exit(main())
