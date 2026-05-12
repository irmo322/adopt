import unittest
import torch
from adopt import ADOPT


class TestADOPTIgnoresZeroGradients(unittest.TestCase):
    def _make_params_and_optimizer(self, param_data, foreach=None):
        """Create a fresh set of params and optimizer from given data tensors."""
        params = [torch.nn.Parameter(d.clone()) for d in param_data]
        optimizer = ADOPT(params, lr=1e-3, foreach=foreach)
        return params, optimizer

    def _run_steps(self, params, optimizer, grad_sequence):
        """
        Run optimizer steps given a sequence of gradients.
        grad_sequence: list of (grad_for_param0, grad_for_param1, ...) per step.
                       A None gradient means "skip this param this step" (zero grad).
        """
        for grads in grad_sequence:
            optimizer.zero_grad()
            for param, grad in zip(params, grads):
                if grad is not None:
                    param.grad = grad.clone()
            optimizer.step()

    def test_zero_gradients_ignored(self):
        torch.manual_seed(322)

        # Two parameters of different shapes
        param_data = [
            torch.randn(4, 4),
            torch.randn(3, 5),
        ]

        # --- Sequence A: non-zero gradients at every step ---
        # Define the "real" gradients for each parameter at each step
        # (only the steps where the gradient is non-zero)
        grads_param0 = [torch.randn(4, 4) for _ in range(4)]
        grads_param1 = [torch.randn(3, 5) for _ in range(3)]

        # Sequence A: param0 active at steps 0,1,2,3 ; param1 active at steps 0,1,2
        # Step 0: both active
        # Step 1: both active
        # Step 2: both active
        # Step 3: only param0 active

        grad_sequence_A = [
            (grads_param0[0], grads_param1[0]),  # step 0
            (grads_param0[1], grads_param1[1]),  # step 1
            (grads_param0[2], grads_param1[2]),  # step 2
            (grads_param0[3], None),             # step 3: param1 inactive
        ]

        # --- Sequence B: same gradients, but with zeros interspersed ---
        # param0 receives the same grads_param0[0..3] but with zeros in between
        # param1 receives the same grads_param1[0..2] but with zeros in between
        # The actual updates must be identical since zeros are ignored.

        grad_sequence_B = [
            (torch.zeros(4, 4), torch.zeros(3, 5)),   # step 0: all zero, nothing happens
            (grads_param0[0], grads_param1[0]),       # step 1
            (torch.zeros(4, 4), grads_param1[1]),     # step 2: param0 zero
            (grads_param0[1], torch.zeros(3, 5)),     # step 3: param1 zero
            (grads_param0[2], grads_param1[2]),       # step 4
            (torch.zeros(4, 4), torch.zeros(3, 5)),   # step 5: all zero
            (grads_param0[3], None),                  # step 6: param1 inactive
            (None, None),                             # step 7: both inactive
        ]

        # Run sequences

        params_A, optim_A = self._make_params_and_optimizer(param_data)
        self._run_steps(params_A, optim_A, grad_sequence_A)

        params_B, optim_B = self._make_params_and_optimizer(param_data)
        self._run_steps(params_B, optim_B, grad_sequence_B)

        params_B_multi, optim_B_multi = self._make_params_and_optimizer(param_data, foreach=True)
        self._run_steps(params_B_multi, optim_B_multi, grad_sequence_B)

        # --- Compare parameters ---
        for i, (p_A, p_B, p_B_m) in enumerate(zip(params_A, params_B, params_B_multi)):
            torch.testing.assert_close(
                p_A, p_B,
                msg=f"Parameter {i} differs between run A and run B"
            )
            torch.testing.assert_close(
                p_A, p_B_m,
                msg=f"Parameter {i} differs between run A and run B multi"
            )

        # --- Compare internal optimizer states (exp_avg, exp_avg_sq, step) ---
        for i, (p_A, p_B, p_B_m) in enumerate(zip(params_A, params_B, params_B_multi)):
            state_A = optim_A.state[p_A]
            state_B = optim_B.state[p_B]
            state_B_multi = optim_B_multi.state[p_B_m]
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    state_A[key], state_B[key],
                    msg=f"State '{key}' for parameter {i} differs between run A and run B"
                )
                torch.testing.assert_close(
                    state_A[key], state_B_multi[key],
                    msg=f"State '{key}' for parameter {i} differs between run A and run B multi"
                )
