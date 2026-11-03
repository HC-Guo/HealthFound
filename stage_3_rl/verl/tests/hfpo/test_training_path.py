import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_retention import TinyTokenizer, anchor_config, corpus

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.retention_dataset import RetentionBatchStream
from verl.utils.hfpo_retention import RETENTION_VERSION


class InertWorkers:
    def update_actor(self, *data):
        self.received = data
        return DataProto(meta_info={"metrics": {}})

    def save_checkpoint(self, local_path, *args, **kwargs):
        Path(local_path).mkdir(parents=True)

    def load_checkpoint(self, local_path, **kwargs):
        self.loaded_path = local_path


def trainer(tmp_path, path, enabled=True):
    result = RayPPOTrainer.__new__(RayPPOTrainer)
    config = anchor_config([path])
    result.config = OmegaConf.create(dict(
        trainer=dict(default_local_dir=str(tmp_path / "checkpoints"), default_hdfs_dir=None,
                     resume_mode="auto", del_local_ckpt_after_load=False),
        actor_rollout_ref=dict(actor=dict(ppo_mini_batch_size=2, ppo_epochs=2, retention_loss_coef=.2),
                              rollout=dict(n=16)),
        data=dict(retention=OmegaConf.to_container(config)),
    ))
    result.actor_rollout_wg = InertWorkers()
    result.retention_stream = RetentionBatchStream(TinyTokenizer(), config) if enabled else None
    result.train_dataloader = SimpleNamespace(state_dict=lambda: {"rl_progress": 9},
                                             load_state_dict=lambda state: None)
    result.use_critic = False
    result.global_steps = 3
    return result


def test_driver_draws_fresh_anchor_batches_for_each_optimizer_update(tmp_path):
    path = corpus(tmp_path)
    instance = trainer(tmp_path, path)
    batch = DataProto.from_dict(tensors={"input_ids": torch.ones(64, 2, dtype=torch.long)})
    instance._update_actor(batch)
    rl, anchor = instance.actor_rollout_wg.received
    assert len(rl) == 64
    assert len(anchor) == 8  # 2 PPO minibatches * 2 epochs * 2 global anchor samples
    assert len(anchor.meta_info["global_token_num"]) == 8


def test_driver_zero_coefficient_path_passes_only_rl_batch(tmp_path):
    path = corpus(tmp_path)
    instance = trainer(tmp_path, path, enabled=False)
    batch = DataProto.from_dict(tensors={"input_ids": torch.ones(64, 2, dtype=torch.long)})
    instance._update_actor(batch)
    assert len(instance.actor_rollout_wg.received) == 1


def test_trainer_checkpoint_restores_anchor_progress_and_records_configuration(tmp_path):
    path = corpus(tmp_path)
    original = trainer(tmp_path, path)
    original.retention_stream.next_batch(4)
    original._save_checkpoint()
    expected = original.retention_stream.next_batch(5)
    folder = tmp_path / "checkpoints/global_step_3"
    metadata = json.loads((folder / "hfpo_retention.json").read_text())
    assert metadata["implementation"] == RETENTION_VERSION
    assert metadata["retention_loss_coef"] == .2
    assert OmegaConf.load(folder / "hfpo_config.yaml").data.retention.batch_size == 2
    resumed = trainer(tmp_path, path)
    resumed._load_checkpoint()
    actual = resumed.retention_stream.next_batch(5)
    for key in expected.batch.keys():
        torch.testing.assert_close(actual.batch[key], expected.batch[key])


def test_old_checkpoint_without_anchor_state_is_not_silently_resumed(tmp_path):
    path = corpus(tmp_path)
    original = trainer(tmp_path, path, enabled=False)
    original._save_checkpoint()
    with pytest.raises(ValueError, match="anchor_data.pt"):
        trainer(tmp_path, path)._load_checkpoint()


def test_worker_forwards_separate_anchor_batch_before_optimizer_update(tmp_path, monkeypatch):
    import verl.workers.fsdp_workers as workers

    worker = workers.ActorRolloutRefWorker.__new__(workers.ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_offload_param = False
    worker._is_offload_optimizer = False
    worker._world_size = 1
    worker.config = OmegaConf.create({"actor": {"ppo_epochs": 2}})
    worker.ulysses_sharding_manager = nullcontext()
    received = {}

    def update_policy(data, anchor_data):
        received.update(rl=data, anchor=anchor_data)
        return {"actor/retention_loss": [2.]}

    worker.actor = SimpleNamespace(update_policy=update_policy)
    worker.flops_counter = SimpleNamespace(estimate_flops=lambda *args: (10., 100.))
    worker.actor_lr_scheduler = SimpleNamespace(get_last_lr=lambda: [.001], step=lambda: None)
    monkeypatch.setattr(workers, "get_torch_device", lambda: SimpleNamespace(
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0))
    rl = DataProto.from_dict(tensors={"input_ids": torch.ones(4, 2, dtype=torch.long)},
                             meta_info={"global_token_num": [2] * 4})
    anchor = DataProto.from_dict(tensors={"input_ids": torch.ones(6, 2, dtype=torch.long)},
                                 meta_info={"global_token_num": [2] * 6})
    output = inspect.unwrap(workers.ActorRolloutRefWorker.update_actor)(worker, rl, anchor)
    assert len(received["rl"]) == 4
    assert len(received["anchor"]) == 6
    assert output.meta_info["metrics"]["actor/retention_loss"] == [2.]
