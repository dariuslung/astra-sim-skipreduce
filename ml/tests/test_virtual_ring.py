import unittest
import torch
from ml.simulation.virtual_ring import simulate_skipreduce_ring
from ml.simulation.error_feedback import ErrorFeedbackBuffer


class TestVirtualRing(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_ranks = 4
        self.tensor_len = 1024

    def test_zero_skip_matches_allreduce(self):
        grads = [torch.randn(self.tensor_len, device=self.device) for _ in range(self.num_ranks)]
        g_true = sum(grads) / self.num_ranks

        g_sim, stats = simulate_skipreduce_ring(
            grads=grads,
            num_ranks=self.num_ranks,
            s=0,
            transform_type="none",
            retention_ratio=0.0
        )

        self.assertTrue(torch.allclose(g_sim, g_true, atol=1e-6))
        self.assertAlmostEqual(stats["cos_sim"], 1.0, places=5)
        self.assertAlmostEqual(stats["rel_l2_error"], 0.0, places=5)
        self.assertAlmostEqual(stats["payload_ratio"], 1.0, places=5)

    def test_full_retention_matches_allreduce(self):
        grads = [torch.randn(self.tensor_len, device=self.device) for _ in range(self.num_ranks)]
        g_true = sum(grads) / self.num_ranks

        for transform in ["dct", "hadamard"]:
            g_sim, stats = simulate_skipreduce_ring(
                grads=grads,
                num_ranks=self.num_ranks,
                s=1,
                transform_type=transform,
                retention_ratio=1.0
            )
            self.assertTrue(torch.allclose(g_sim, g_true, atol=1e-4), f"Failed for {transform}")
            self.assertGreater(stats["cos_sim"], 0.9999)

    def test_partial_retention_improves_over_pure_skip(self):
        grads = [torch.randn(self.tensor_len, device=self.device) for _ in range(self.num_ranks)]

        # Pure SkipReduce (s=1, r=0)
        _, stats_pure = simulate_skipreduce_ring(
            grads=grads,
            num_ranks=self.num_ranks,
            s=1,
            transform_type="none",
            retention_ratio=0.0
        )

        # DCT Transformed SkipReduce (s=1, r=0.2)
        _, stats_dct = simulate_skipreduce_ring(
            grads=grads,
            num_ranks=self.num_ranks,
            s=1,
            transform_type="dct",
            retention_ratio=0.2
        )

        # Transformed SkipReduce should have higher cosine similarity and lower relative L2 error
        self.assertGreater(stats_dct["cos_sim"], stats_pure["cos_sim"])
        self.assertLess(stats_dct["rel_l2_error"], stats_pure["rel_l2_error"])

    def test_error_feedback_tracking(self):
        ef = ErrorFeedbackBuffer(num_ranks=self.num_ranks)
        
        # Run 5 simulated steps with a persistent gradient trend
        for step in range(5):
            grads = [torch.randn(self.tensor_len, device=self.device) + 1.0 for _ in range(self.num_ranks)]
            g_sim, stats = simulate_skipreduce_ring(
                grads=grads,
                num_ranks=self.num_ranks,
                s=1,
                transform_type="dct",
                retention_ratio=0.15,
                param_id=0,
                ef_buffer=ef
            )
            self.assertIn("cos_sim", stats)
            self.assertGreater(stats["cos_sim"], 0.0)


if __name__ == "__main__":
    unittest.main()
