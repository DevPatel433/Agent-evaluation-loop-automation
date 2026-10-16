import unittest
import agent_evals as ae


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(ae.accuracy("You have 30 days.", ["30 days"]), 1.0)
        self.assertEqual(ae.accuracy("Two weeks.", ["30 days"]), 0.0)

    def test_hallucination(self):
        q = "How long to return?"
        self.assertEqual(ae.hallucination("Returns are accepted within 30 days of delivery.", q, ae.CONTEXT), 0.0)
        self.assertGreater(ae.hallucination("A 2021 study shows customers love store credit.", q, ae.CONTEXT), 0.0)

    def test_quality(self):
        self.assertEqual(ae.quality("30 days."), 1.0)
        self.assertLess(ae.quality("I think, maybe it is 30 days. Let me know if anything else."), 0.6)


class TestLoop(unittest.TestCase):
    def test_mock_agent_reaches_targets(self):
        agent, system = ae.MockAgent(), ae.BASE_PROMPT
        for _ in range(5):
            m = ae.summarize(ae.run_suite(agent, system))
            if ae.passed(m):
                break
            _, bad = ae.worst_metric(m)
            todo = [k for k in sorted(bad, key=bad.get, reverse=True) if bad[k] > 0 and ae.RULES[k] not in system]
            system += " " + ae.RULES[todo[0]]
        self.assertTrue(ae.passed(m))


if __name__ == "__main__":
    unittest.main()
