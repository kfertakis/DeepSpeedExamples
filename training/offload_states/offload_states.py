# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import argparse
import math 
import os 
import json

import deepspeed.comm as dist
from deepspeed.ops.adam import FusedAdam
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, get_scheduler, SchedulerType
from transformers.deepspeed import HfDeepSpeedConfig

import torch

import deepspeed
from deepspeed.runtime.zero.offload_config import (
    OffloadDeviceEnum,
    OffloadStateTypeEnum,
)


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False, nlayers=1):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayers)]
        )
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        for l in self.linears:
            x = l(x)
        return self.cross_entropy_loss(x, y)


def random_dataset(total_samples, hidden_dim, device, dtype):
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=dtype)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(
        hidden_dim
    )
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    return train_dataset


def random_dataloader(model, total_samples, hidden_dim, device, dtype):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_dataset = random_dataset(total_samples, hidden_dim, device, dtype=dtype)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def run_model(
    model_name,
    config_dict,
    hidden_dim,
    dtype,
    include,
    pin_memory,
    non_blocking,
    iteration,
    warmup,
):
    # model, _, _, _ = deepspeed.initialize(
    #     model=model, model_parameters=model.parameters(), config=config_dict
    # )
    # data_loader = random_dataloader(
    #     model=model,
    #     total_samples=iteration,
    #     hidden_dim=hidden_dim,
    #     device=model.device,
    #     dtype=dtype,
    # )
    model = init_model(model_name)

    time_offload_list = []
    time_load_list = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_text = ["DeepSpeed is a powerful library", "Hugging Face models are great for NLP tasks"]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    inputs.to('cuda')
    
    model.train()

    dist.barrier()
    # for i, batch in enumerate(data_loader):
    for i in range(10):
        # loss = model(batch[0], batch[1])
        outputs = model(**inputs)

        # decoded_input = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in inputs['input_ids']]
        # print("Decoded input (prompt):", decoded_input)

        # predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        # decoded_output = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in predicted_token_ids]
        # print("Decoded model output:", decoded_output)

        # Generate dummy target (for example, shift the input_ids by one token to simulate next token prediction)
        shifted_input_ids = inputs['input_ids'].clone()
        shifted_input_ids[:, :-1] = inputs['input_ids'][:, 1:]

        # Compute dummy loss (CrossEntropyLoss for language modeling tasks)
        loss_fn = torch.nn.CrossEntropyLoss()
        logits = outputs.logits
        logits = logits.view(-1, logits.size(-1))
        shifted_input_ids = shifted_input_ids.view(-1)

        loss = loss_fn(logits, shifted_input_ids)

        model.backward(loss)
        model.step()

        # Start offloading
        alloc_before_offload = get_accelerator().memory_allocated()
        dist.barrier()

        print(
            f"before offload GPU allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
        )

        time_start = time.time()
        model.offload_states(
            include=include,
            device=OffloadDeviceEnum.cpu,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )
        dist.barrier()
        time_after_offload = time.time()
        alloc_after_offload = get_accelerator().memory_allocated()
        assert (
            alloc_after_offload < alloc_before_offload
        ), f"Allocated memory should decrease after offload"

        print(
            f"after offload GPU allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
        )

        # Load offloaded states back
        model.reload_states()
        dist.barrier()
        time_after_load = time.time()

        time_offload_list.append(time_after_offload - time_start)
        time_load_list.append(time_after_load - time_after_offload)

        assert (
            alloc_after_offload < get_accelerator().memory_allocated()
        ), f"Allocated memory should increase after offload back"

        print(
            f"after load_back GPU allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
        )

        if dist.get_rank() == 0:
            print(
                f"Memory usage ({i}): include={include}, pin_memory={pin_memory}, non_blocking={non_blocking} alloc_before_offload={alloc_before_offload} alloc_after_offload={alloc_after_offload}"
            )
    # remove warmup
    time_offload_list = time_offload_list[warmup:]
    time_load_list = time_load_list[warmup:]

    if dist.get_rank() == 0:
        with open("offload_states.log", "a") as f:
            offload_time = sum(time_offload_list) / len(time_offload_list)
            load_time = sum(time_load_list) / len(time_load_list)
            msg = f"{1 if pin_memory else 0},{1 if non_blocking else 0},{offload_time},{load_time}"
            f.write(f"{msg}\n")
        print(
            f"Summary: pin_memory={pin_memory} non_blocking={non_blocking} offload={offload_time} load={load_time}"
        )

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()

def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path:
        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer

def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    return tokenizer

def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups

def init_model(model_name_or_path):
    # DS Config
    ds_config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 8,
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 10000.0,
            "stage3_max_live_parameters": 30000000.0,
            "stage3_prefetch_bucket_size": 30000000.0,
            "memory_efficient_linear": False,
        },
        "bfloat16": {"enabled": True},
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": True,
            "max_out_tokens": 512,
            "inference_tp_size": 1,
            "release_inference_cache": False,
            "pin_parameters": True,
            "tp_gather_partition_size": 8,
        },
        "tensorboard": {
            "enabled": False,
            "output_path": "step3_tensorboard/ds_tensorboard_logs/",
            "job_name": "step3_actor_tensorboard",
        },
    }

    tokenizer = load_hf_tokenizer(model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=None)
    

    # Model
    model = create_hf_model(
        model_class=AutoModelForCausalLM,
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        ds_config=ds_config,
        dropout=0.0,
    )

    # Optimizer
    model_weight_decay = 0.0
    model_lora_learning_rate = 0.0005
    AdamOptimizer = FusedAdam
    optim_params = get_optimizer_grouped_parameters(
        model, model_weight_decay, model_lora_learning_rate
    )
    optim = AdamOptimizer(
        optim_params, lr=model_lora_learning_rate, betas=(0.9, 0.95)
    )

    num_warmup_steps=100
    total_iters=10
    # LR Scheduler
    lr_scheduler = get_scheduler(
        name=SchedulerType.COSINE,
        optimizer=optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_iters,
    )

    # DeepSpeed Engine
    # TODO: move enable_hybrid_engine and pin_parameters to ds_config
    model_engine, *_ = deepspeed.initialize(
        model=model, optimizer=optim, lr_scheduler=lr_scheduler, config=ds_config
    )

    return model_engine


def main():
    parser = argparse.ArgumentParser(description="Test Offload States")
    parser.add_argument(
        "--included_state",
        type=str,
        choices=[e.name for e in OffloadStateTypeEnum] + [None],
        default=None,
        help="State to include",
    )
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory")
    parser.add_argument("--non_blocking", action="store_true", help="Non blocking")
    parser.add_argument("--nlayers", type=int, default=1, help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument(
        "--dtype",
        choices=["torch.bfloat16", "torch.float16", "torch.float32"],
        default="torch.bfloat16",
        help="Data type",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser.add_argument("--iteration", type=int, default=10, help="Warmup")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup")

    args = parser.parse_args()

    dtype = eval(args.dtype)
    hidden_dim = args.hidden_dim

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-6}},
        "zero_optimization": {
            "stage": 3,
        },
    }

    if dtype == torch.float16:
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    # with deepspeed.zero.Init(config_dict_or_path=config_dict):
        # model = SimpleModel(hidden_dim, nlayers=args.nlayers)
    model_name = 'facebook/opt-1.3b'
    # model = init_model(model_name)

    included_state = (
        None
        if args.included_state is None
        else [OffloadStateTypeEnum[args.included_state]]
    )
    run_model(
        model_name,
        config_dict,
        hidden_dim,
        dtype,
        included_state,
        args.pin_memory,
        args.non_blocking,
        args.iteration,
        args.warmup,
    )


if __name__ == "__main__":
    main()
