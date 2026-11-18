import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.base.decorator import _split_args_kwargs_data_proto
from verl.utils.dataset.retention_dataset import RetentionBatchStream, RetentionDataset
from verl.utils.hfpo_retention import optimizer_updates_for_batch, retention_loss, validate_retention_config


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(character) % 23 for character in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize and add_generation_prompt
        return self.encode("|".join(message["content"] for message in messages)) + [25]


def anchor_config(files, **changes):
    values = dict(train_files=[str(path) for path in files], batch_size=2, seed=17, shuffle=True,
                  prompt_key="prompt", response_key="response", instruction_key=None,
                  max_prompt_length=24, max_response_length=12, apply_chat_template_kwargs={})
    values.update(changes)
    return OmegaConf.create(values)


def corpus(tmp_path, records=None):
    records = records if records is not None else [
        dict(prompt=f"Q{index}", response="answer"[:1 + index % 5]) for index in range(7)
    ]
    path = tmp_path / "anchors.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def composed_config(**changes):
    config_dir = Path(__file__).resolve().parents[2] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=[
            "actor_rollout_ref.actor.retention_loss_coef=0.25",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.name=vllm",
            "data.retention.train_files=[/synthetic/anchors.jsonl]",
            "trainer.n_gpus_per_node=2", "trainer.nnodes=1",
        ])
    for key, value in changes.items():
        OmegaConf.update(config, key, value)
    return config


def test_per_example_normalization_and_mask_gradients():
    log_probs = torch.tensor([[-2., -4., -99.], [-3., -99., -99.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    loss = retention_loss(log_probs, mask)
    assert loss.item() == 3
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([[-.25, -.25, 0.], [-.5, 0., 0.]]))


@pytest.mark.parametrize("mask", [torch.zeros(2, 3), torch.tensor([[1, 0, 0], [0, 0, 0]])])
def test_empty_supervised_response_rejected(mask):
    with pytest.raises(ValueError, match="supervised response"):
        retention_loss(torch.zeros(2, 3), mask)


def test_padding_nan_is_excluded():
    assert retention_loss(torch.tensor([[-2., float("nan")]]), torch.tensor([[1, 0]])).item() == 2


def test_teacher_forcing_alignment_and_padding(tmp_path):
    path = corpus(tmp_path, [dict(prompt="Q", response="ab")])
    item = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))[0]
    length = 24
    assert item["responses"][:3].tolist() == TinyTokenizer().encode("ab") + [1]
    assert item["input_ids"][length:length + 3].tolist() == item["responses"][:3].tolist()
    assert item["input_ids"][length - 1].item() == 25  # assistant prefix predicts the FIRST target token
    assert item["response_mask"].sum().item() == 3
    assert item["attention_mask"].sum().item() == 5  # Q, prefix, a, b, EOS
    assert torch.all(item["attention_mask"][length + 3:] == 0)


def test_target_is_not_silently_truncated(tmp_path):
    path = corpus(tmp_path, [dict(prompt="Q", response="a" * 12)])
    dataset = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))
    with pytest.raises(ValueError, match="no silent target truncation"):
        dataset[0]


@pytest.mark.parametrize("record", [dict(prompt="Q", response=""), dict(prompt="Q"), dict(prompt="", response="A")])
def test_invalid_anchor_targets_fail(tmp_path, record):
    path = corpus(tmp_path, [record])
    with pytest.raises(ValueError):
        RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))[0]


def test_alpaca_and_chat_inputs(tmp_path):
    path = corpus(tmp_path, [dict(instruction="Task", input="Q", output="A")])
    config = anchor_config([path], instruction_key="instruction", prompt_key="input", response_key="output")
    assert RetentionDataset([str(path)], TinyTokenizer(), config)[0]["response_mask"].sum() == 2
    path = corpus(tmp_path, [dict(prompt=[dict(role="user", content="Q")], response="A")])
    assert RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))[0]["response_mask"].sum() == 2


def test_parquet_inputs(tmp_path):
    import pandas as pd

    path = tmp_path / "anchors.parquet"
    pd.DataFrame([dict(prompt="Q", response="A")]).to_parquet(path)
    assert len(RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))) == 1


@pytest.mark.parametrize("draws_before_save", [1, 3, 4, 7])
def test_anchor_resume_across_loader_wraps(tmp_path, draws_before_save):
    path = corpus(tmp_path)
    config = anchor_config([path])
    stream = RetentionBatchStream(TinyTokenizer(), config)
    stream.next_batch(draws_before_save)
    # Exercise the actual torch serialization used by trainer checkpoints.
    buffer = io.BytesIO()
    torch.save(stream.state_dict(), buffer)
    expected = stream.next_batch(5)
    buffer.seek(0)
    resumed = RetentionBatchStream(TinyTokenizer(), config)
    resumed.load_state_dict(torch.load(buffer, weights_only=False))
    actual = resumed.next_batch(5)
    for key in expected.batch.keys():
        torch.testing.assert_close(actual.batch[key], expected.batch[key])


def test_changed_anchor_configuration_cannot_resume(tmp_path):
    path = corpus(tmp_path)
    stream = RetentionBatchStream(TinyTokenizer(), anchor_config([path]))
    state = stream.state_dict()
    resumed = RetentionBatchStream(TinyTokenizer(), anchor_config([path], seed=18))
    with pytest.raises(ValueError, match="does not match"):
        resumed.load_state_dict(state)


def test_hydra_keys_and_actor_dataclass():
    from verl.utils.config import omega_conf_to_dataclass

    config = composed_config()
    assert validate_retention_config(config)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.retention_loss_coef == .25
    assert actor.retention_micro_batch_size_per_gpu == 1
    assert config.data.retention.batch_size == 32


@pytest.mark.parametrize("key,value", [
    ("actor_rollout_ref.actor.retention_loss_coef", -.1),
    ("actor_rollout_ref.actor.retention_loss_coef", float("nan")),
    ("actor_rollout_ref.actor.retention_loss_coef", float("inf")),
    ("actor_rollout_ref.actor.strategy", "megatron"),
    ("actor_rollout_ref.actor.retention_micro_batch_size_per_gpu", 0),
    ("data.retention.train_files", []), ("data.retention.batch_size", 3),
    ("data.retention.max_response_length", 0),
    ("actor_rollout_ref.actor.ppo_epochs", 0),
])
def test_invalid_retention_configuration_rejected(key, value):
    with pytest.raises(ValueError):
        validate_retention_config(composed_config(**{key: value}))


def test_zero_coefficient_does_not_require_anchors():
    config = composed_config(**{"actor_rollout_ref.actor.retention_loss_coef": 0., "data.retention.train_files": []})
    assert not validate_retention_config(config)


def test_anchor_count_follows_optimizer_steps_not_rollout_count():
    actor = SimpleNamespace(ppo_mini_batch_size=2, ppo_epochs=3)
    assert optimizer_updates_for_batch(4, actor, rollout_n=1) == 6
    assert optimizer_updates_for_batch(4 * 16, actor, rollout_n=16) == 6
    with pytest.raises(ValueError):
        optimizer_updates_for_batch(5, actor, rollout_n=1)


def test_independent_rl_and_anchor_dp_dispatch():
    rl = DataProto.from_dict(tensors={"input_ids": torch.arange(8).reshape(4, 2)})
    anchor = DataProto.from_dict(tensors={"input_ids": torch.arange(12).reshape(6, 2)})
    positional, _ = _split_args_kwargs_data_proto(2, rl, anchor)
    assert [len(part) for part in positional[0]] == [2, 2]
    assert [len(part) for part in positional[1]] == [3, 3]


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(26, 8)
        self.projection = torch.nn.Linear(8, 26)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


@pytest.fixture
def actor_factory(tmp_path):
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    initialized_here = not torch.distributed.is_initialized()
    if initialized_here:
        torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)

    def factory(model, coefficient, micro_batch_size=1, ppo_mini_batch_size=2, ppo_epochs=1, dynamic=False, kl=False):
        config = OmegaConf.create(dict(
            use_remove_padding=False, use_fused_kernels=False, ulysses_sequence_parallel_size=1,
            entropy_from_logits_with_chunking=False, use_torch_compile=False, entropy_coeff=0,
            entropy_checkpointing=False, use_kl_loss=kl, kl_loss_coef=.05, kl_loss_type="low_var_kl",
            tis_imp_ratio_cap=-1,
            ppo_mini_batch_size=ppo_mini_batch_size, ppo_micro_batch_size_per_gpu=1,
            use_dynamic_bsz=dynamic, ppo_max_token_len_per_gpu=1000, ppo_epochs=ppo_epochs,
            policy_loss=dict(loss_mode="vanilla"), loss_agg_mode="token-mean",
            clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3., grad_clip=1e8,
            retention_loss_coef=coefficient, retention_micro_batch_size_per_gpu=micro_batch_size,
        ))
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        return DataParallelPPOActor(config, model, optimizer)

    yield factory
    if initialized_here:
        torch.distributed.destroy_process_group()


def as_data(dataset, indices):
    records = [dataset[index] for index in indices]
    return DataProto.from_dict(tensors={key: torch.stack([record[key] for record in records]) for key in records[0]})


def rl_batch(dataset, indices):
    data = as_data(dataset, indices)
    data.batch["old_log_probs"] = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    # Zero RL advantages isolate whether the NEW supervised path changes weights.
    data.batch["advantages"] = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    data.meta_info["temperature"] = .6
    return data


def test_actual_actor_retention_changes_weights_and_zero_preserves_them(tmp_path, actor_factory):
    path = corpus(tmp_path)
    dataset = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))
    torch.manual_seed(7)
    initial = TinyCausalLM()
    frozen = copy.deepcopy(initial)
    actor_factory(frozen, 0).update_policy(rl_batch(dataset, [0, 1]))
    for before, after in zip(initial.parameters(), frozen.parameters(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    trained = copy.deepcopy(initial)
    actor = actor_factory(trained, .3)
    metrics = actor.update_policy(rl_batch(dataset, [0, 1]), as_data(dataset, [2, 3, 4]))
    assert any(
        not torch.equal(before, after) for before, after in zip(initial.parameters(), trained.parameters(), strict=True)
    )
    assert metrics["actor/retention_loss"][0] > 0
    assert metrics["actor/retention_coef"] == [.3]


def test_actor_gradient_matches_one_combined_objective(tmp_path, actor_factory):
    path = corpus(tmp_path)
    dataset = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))
    torch.manual_seed(10)
    initial = TinyCausalLM()
    expected = copy.deepcopy(initial)
    reference = actor_factory(expected, .3)
    anchor = as_data(dataset, [2, 3, 4])
    _, log_probs = reference._forward_micro_batch(dict(anchor.batch), temperature=1., calculate_entropy=False)
    (.3 * retention_loss(log_probs, anchor.batch["response_mask"])).backward()
    reference.actor_optimizer.step()
    actual = copy.deepcopy(initial)
    actor_factory(actual, .3).update_policy(rl_batch(dataset, [0, 1]), as_data(dataset, [2, 3, 4]))
    for target, observed in zip(expected.parameters(), actual.parameters(), strict=True):
        torch.testing.assert_close(target, observed, rtol=3e-3, atol=3e-5)


def test_policy_kl_and_retention_share_one_optimizer_step(tmp_path, actor_factory):
    from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla, kl_penalty

    path = corpus(tmp_path)
    dataset = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))
    torch.manual_seed(13)
    initial = TinyCausalLM()
    expected, actual = copy.deepcopy(initial), copy.deepcopy(initial)
    reference = actor_factory(expected, .3, kl=True)
    rl = rl_batch(dataset, [0, 0])
    rl.batch["advantages"].fill_(1.)
    _, log_probs = reference._forward_micro_batch(dict(rl.batch), temperature=.6)
    rl.batch["ref_log_prob"] = log_probs.detach() - .2
    pg, *_ = compute_policy_loss_vanilla(
        old_log_prob=log_probs.detach(), log_prob=log_probs, advantages=rl.batch["advantages"],
        response_mask=rl.batch["response_mask"], config=reference.config,
    )
    kl = agg_loss(kl_penalty(log_probs, rl.batch["ref_log_prob"], "low_var_kl"),
                  rl.batch["response_mask"], "token-mean")
    anchor = as_data(dataset, [2, 3, 4])
    _, anchor_log_probs = reference._forward_micro_batch(dict(anchor.batch), temperature=1.)
    combined = pg + .05 * kl + .3 * retention_loss(anchor_log_probs, anchor.batch["response_mask"])
    combined.backward()
    reference.actor_optimizer.step()
    actor = actor_factory(actual, .3, kl=True)
    actor.update_policy(rl, as_data(dataset, [2, 3, 4]))
    for target, observed in zip(expected.parameters(), actual.parameters(), strict=True):
        torch.testing.assert_close(target, observed, rtol=3e-3, atol=3e-5)


@pytest.mark.parametrize("dynamic", [False, True])
def test_anchor_microbatch_partition_preserves_update(tmp_path, actor_factory, dynamic):
    path = corpus(tmp_path)
    dataset = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))
    torch.manual_seed(4)
    initial = TinyCausalLM()
    first, second = copy.deepcopy(initial), copy.deepcopy(initial)
    actor_factory(first, .4, micro_batch_size=1, dynamic=dynamic).update_policy(
        rl_batch(dataset, [0, 1]), as_data(dataset, [2, 3, 4]))
    actor_factory(second, .4, micro_batch_size=2, dynamic=dynamic).update_policy(
        rl_batch(dataset, [0, 1]), as_data(dataset, [2, 3, 4]))
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-3, atol=3e-5)


def test_actual_actor_requires_anchor_for_every_optimizer_update(tmp_path, actor_factory):
    path = corpus(tmp_path)
    dataset = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))
    actor = actor_factory(TinyCausalLM(), .3, ppo_mini_batch_size=1, ppo_epochs=2)
    with pytest.raises(ValueError, match="every actor optimizer update"):
        actor.update_policy(rl_batch(dataset, [0, 1]), as_data(dataset, [2, 3, 4]))
    metrics = actor.update_policy(rl_batch(dataset, [0, 1]), as_data(dataset, [2, 3, 4, 5]))
    assert len(metrics["actor/retention_loss"]) == 4


def test_simulated_equal_dp_shards_match_global_anchor_mean(tmp_path, actor_factory):
    path = corpus(tmp_path)
    dataset = RetentionDataset([str(path)], TinyTokenizer(), anchor_config([path]))
    torch.manual_seed(19)
    initial = TinyCausalLM()
    global_model = copy.deepcopy(initial)
    actor_factory(global_model, .4, ppo_mini_batch_size=4).update_policy(
        rl_batch(dataset, [0, 1, 2, 3]), as_data(dataset, [0, 1, 2, 3, 4, 5]))
    rank_models = [copy.deepcopy(initial), copy.deepcopy(initial)]
    actor_factory(rank_models[0], .4).update_policy(rl_batch(dataset, [0, 1]), as_data(dataset, [0, 1, 2]))
    actor_factory(rank_models[1], .4).update_policy(rl_batch(dataset, [2, 3]), as_data(dataset, [3, 4, 5]))
    for global_parameter, rank0, rank1 in zip(
        global_model.parameters(), rank_models[0].parameters(), rank_models[1].parameters(), strict=True
    ):
        torch.testing.assert_close(global_parameter, (rank0 + rank1) / 2, rtol=3e-3, atol=3e-5)
