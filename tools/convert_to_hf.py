import argparse
import copy
import os
import os.path as osp

import torch
from mmengine.config import Config, ConfigDict
from mmengine.dist import master_only
from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER


def convert_dict2config_dict(value):
    value = ConfigDict(**value)
    for key in value.keys():
        if isinstance(value[key], dict):
            value[key] = convert_dict2config_dict(value[key])
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an InternVL Sa2VA checkpoint to Hugging Face format")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("pth_model", help="checkpoint file or distributed checkpoint directory")
    parser.add_argument("--save-path", type=str, default=None, help="output directory")
    return parser.parse_args()


@master_only
def master_print(message):
    print(message)


def main():
    args = parse_args()
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError as exc:
            raise FileNotFoundError(f"Cannot find {args.config}") from exc

    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    checkpoint = args.pth_model
    if osp.isdir(checkpoint):
        checkpoint = osp.join(checkpoint, "mp_rank_00_model_states.pt")
        if not osp.isfile(checkpoint):
            raise FileNotFoundError(f"No rank-0 model state in {checkpoint}")

    backend = get_file_backend(checkpoint)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        patch_fileio()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state_dict.get("state_dict", state_dict.get("module", state_dict))
    model.load_state_dict(state_dict, strict=False)
    print(f"Load PTH model from {checkpoint}")

    iteration = osp.basename(checkpoint).split(".")[0]
    model._merge_lora()
    model.mllm.use_llm_lora = False
    model.mllm.model.modules_to_save = None
    model.mllm.model.transfer_to_hf = True
    all_state_dict = model.all_state_dict()

    from projects.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
    from projects.hf.models.modeling_sa2va_chat import Sa2VAChatModel

    config = Sa2VAChatConfig.from_pretrained(cfg.path)
    config_dict = config.to_dict()
    config_dict["llm_config"]["vocab_size"] = len(model.mllm.tokenizer)
    config_dict["template"] = cfg.template
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_sa2va_chat.Sa2VAChatConfig",
        "AutoModel": "modeling_sa2va_chat.Sa2VAChatModel",
        "AutoModelForCausalLM": "modeling_sa2va_chat.Sa2VAChatModel",
    }

    converted = {}
    for key, value in all_state_dict.items():
        new_key = copy.deepcopy(key)
        new_key = new_key.replace("mllm.model.", "").replace(".gamma", ".g_weight")
        converted[new_key] = value

    hf_config = Sa2VAChatConfig(**config_dict)
    hf_model = Sa2VAChatModel(
        hf_config,
        vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model,
    )
    missing_keys, unexpected_keys = hf_model.load_state_dict(converted)

    if args.save_path is None:
        args.save_path = f"./{osp.dirname(args.pth_model)}_{iteration}_hf"
    os.makedirs(args.save_path, exist_ok=True)
    hf_model.save_pretrained(args.save_path)
    model.mllm.tokenizer.save_pretrained(args.save_path)
    os.system(f"cp -pr ./projects/hf/models/* {args.save_path}")

    master_print("\n--- Weight Loading Report ---")
    if missing_keys:
        master_print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        master_print(f"Warning: Unexpected keys: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        master_print("All keys matched successfully!")
    print(f"Save the HF model into {args.save_path}")


if __name__ == "__main__":
    main()
