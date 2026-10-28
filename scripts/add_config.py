"""Register the pi0.5-DROID fine-tuning config in an openpi checkout (idempotent).

    python scripts/add_config.py --openpi /path/to/openpi
"""
import argparse
import pathlib
import re

NAME = "pi05_droid_hdh"
CONFIG = f'''    TrainConfig(
        name="{NAME}",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotDROIDDataConfig(repo_id="hdh/droid", base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets", asset_id="droid")),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=5_000,
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=5_000, decay_lr=2.5e-6),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        ema_decay=None,
        save_interval=1_000,
    ),
'''

ap = argparse.ArgumentParser()
ap.add_argument("--openpi", required=True, type=pathlib.Path)
path = ap.parse_args().openpi / "src" / "openpi" / "training" / "config.py"
src = path.read_text()
if f'name="{NAME}"' not in src:
    anchor = re.search(r'name="pi05_droid_finetune".*?\n    \),\n', src, re.S)
    assert anchor, f"pi05_droid_finetune not found in {path}"
    path.write_text(src[:anchor.end()] + CONFIG + src[anchor.end():])
print(f"{NAME} registered in {path}")
