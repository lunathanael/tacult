import argparse
import pathlib
import random
import time
from typing import List, Optional

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn

from tacult.utac_game import UtacGame
from tacult.base.nn_wrapper import load_network


def run_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_value_net_path", type=pathlib.Path, required=True)
    parser.add_argument("--policy_value_net_onnx_path", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return args


def load_policy_value_net(state_dict_path: pathlib.Path, device: torch.device) -> nn.Module:
    directory = pathlib.Path(state_dict_path).parent
    filename = pathlib.Path(state_dict_path).name
    policy_value_net = load_network(folder=directory, filename=filename)
    policy_value_net = policy_value_net.nnet
    policy_value_net.to(device=device)
    checkpoint = torch.load(state_dict_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    policy_value_net.load_state_dict(state_dict)
    policy_value_net.eval()
    return policy_value_net


def get_random_uttt(
    depth: int, seed: Optional[int] = None, max_num_retries: int = 100
) -> UtacGame:
    random.seed(seed)
    game = UtacGame()
    uttt = game.getInitBoard()
    d = 0
    num_retries = 0
    player = 1
    while d < depth:
        if game.getGameEnded(uttt, player) != 0:
            uttt = game.getInitBoard()
            player = 1
            d = 0
            num_retries += 1
            if num_retries >= max_num_retries:
                raise RuntimeError(f"max_num_retries={max_num_retries} exceeded!")
        valids = game.getValidMoves(uttt, player)
        action = random.choice(np.where(valids)[0])
        uttt, player = game.getNextState(uttt, player, action)
        d += 1
    uttt = game.getCanonicalForm(uttt, 1)
    return uttt


def get_random_input_arrays(num_input_arrays: int) -> List[np.ndarray]:
    input_arrays = []
    game = UtacGame()
    for i in range(num_input_arrays):
        uttt = get_random_uttt(depth=i % 70, seed=i)
        input_array = game._get_obs(uttt)
        input_array = np.expand_dims(input_array, axis=0)
        input_arrays.append(input_array)
    return input_arrays


def export_policy_value_net_onnx(
    policy_value_net: nn.Module,
    input_tensor: torch.Tensor,
    policy_value_net_onnx_path: pathlib.Path,
) -> None:
    torch.onnx.export(
        policy_value_net,
        input_tensor,
        policy_value_net_onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy_logits", "state_value"],
        verbose=True,
    )


def check_policy_value_net_onnx(policy_value_net_onnx_path: pathlib.Path) -> None:
    policy_value_net_onnx = onnx.load(str(policy_value_net_onnx_path))
    onnx.checker.check_model(policy_value_net_onnx)


def compute_torch_outputs(
    policy_value_net: nn.Module, input_arrays: List[np.ndarray]
) -> List[List[np.ndarray]]:
    torch_outputs = []
    runtime, runcount = 0.0, 0
    for i, input_array in enumerate(input_arrays):
        input_tensor = torch.from_numpy(input_array).float()
        with torch.no_grad():
            start_time = time.perf_counter()
            policy_logits, state_value = policy_value_net(input_tensor)
            end_time = time.perf_counter()
            if i >= 10:
                runtime += end_time - start_time
                runcount += 1
            torch_output = [policy_logits, state_value]
            torch_output = [tensor.numpy().copy() for tensor in torch_output]
            torch_outputs.append(torch_output)
    avg_runtime = runtime / runcount
    print(f"compute_torch_outputs: runtime={avg_runtime:.6f}")
    return torch_outputs


def compute_onnxruntime_outputs(
    policy_value_net_onnx_path: pathlib.Path, input_arrays: List[np.ndarray]
) -> List[List[np.ndarray]]:
    onnxruntime_outputs = []
    runtime, runcount = 0.0, 0
    ort_session = onnxruntime.InferenceSession(str(policy_value_net_onnx_path))
    for i, input_array in enumerate(input_arrays):
        ort_inputs = {"input": input_array.astype(np.float32)}
        start_time = time.perf_counter()
        ort_outputs = ort_session.run(None, ort_inputs)
        policy_logits, state_value = ort_outputs
        end_time = time.perf_counter()
        if i >= 10:
            runtime += end_time - start_time
            runcount += 1
        onnxruntime_output = [policy_logits, state_value]
        onnxruntime_outputs.append(onnxruntime_output)
    avg_runtime = runtime / max(runcount, 1)
    print(f"compute_onnxruntime_outputs: runtime={avg_runtime:.6f}")
    return onnxruntime_outputs


def compare_outputs(
    torch_outputs: List[List[np.ndarray]],
    onnxruntime_outputs: List[List[np.ndarray]],
    rtol: float,
    atol: float,
) -> None:
    assert type(torch_outputs) == type(onnxruntime_outputs)
    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        assert type(torch_output) == type(onnxruntime_output)
        assert len(torch_output) == len(onnxruntime_output)
        for torch_array, onnxruntime_array in zip(torch_output, onnxruntime_output):
            assert torch_array.shape == onnxruntime_array.shape
            np.testing.assert_allclose(torch_array, onnxruntime_array, rtol=rtol, atol=atol)


def main(_args=None) -> None:
    if _args is None:
        args = run_argparse()
    else:
        args = _args
    print(args)

    policy_value_net = load_policy_value_net(
        state_dict_path=args.policy_value_net_path,
        device=torch.device("cpu"),
    )

    input_arrays = get_random_input_arrays(num_input_arrays=700)

    export_policy_value_net_onnx(
        policy_value_net=policy_value_net,
        input_tensor=torch.from_numpy(input_arrays[0]).float(),
        policy_value_net_onnx_path=args.policy_value_net_onnx_path,
    )

    check_policy_value_net_onnx(
        policy_value_net_onnx_path=args.policy_value_net_onnx_path,
    )

    torch_outputs = compute_torch_outputs(
        policy_value_net=policy_value_net,
        input_arrays=input_arrays,
    )

    print(f"len(torch_outputs) = {len(torch_outputs)}")

    onnxruntime_outputs = compute_onnxruntime_outputs(
        policy_value_net_onnx_path=args.policy_value_net_onnx_path,
        input_arrays=input_arrays,
    )
    co = compute_onnxruntime_outputs(
        policy_value_net_onnx_path=args.policy_value_net_onnx_path,
        input_arrays=[np.expand_dims(UtacGame()._get_obs(UtacGame().getInitBoard()), axis=0)],
    )[0]
    print(co[0].reshape(9, 9).round(2))
    print(co[1])

    print(f"len(onnxruntime_outputs) = {len(onnxruntime_outputs)}")

    compare_outputs(
        torch_outputs=torch_outputs,
        onnxruntime_outputs=onnxruntime_outputs,
        rtol=2e-5,
        atol=2e-5,
    )

    print("All comparison tests passed successfully!")


def export_model(_args=None):
    main(_args)


if __name__ == "__main__":
    export_model()
