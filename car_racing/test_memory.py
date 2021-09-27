import unittest

from memory import Memory


class TestMemory(unittest.TestCase):

    def setUp(self):
        self.memory = Memory(capacity=10, batch_size=2, init_capacity=2)

    def test_insertion(self):
        """ Verify insertion to memory is working properly. """

        self.memory.store(1, 2, 3)
        self.assertEqual(self.memory.num_of_samples(), 1, "Should be 1")
        self.memory.store(4, 5, 6, 7)
        self.assertEqual(self.memory.num_of_samples(), 2, "Should be 2")

    def test_is_init(self):
        """ Check is_initialized function is working as expected. """

        # case 1: samples less than init_capacity
        self.memory.store(1, 2, 3)
        self.assertFalse(self.memory.is_initialized())
        # case 2: samples equal init_capacity
        self.memory.store(4, 5, 6)
        self.assertTrue(self.memory.is_initialized())
        # case 3: samples more than init_capacity
        self.memory.store(7, 8, 9)
        self.assertTrue(self.memory.is_initialized())

    def test_sample(self):
        """ Validate multiple scenarios about sampling from memory. """

        self.memory.store(1, 2, 3)
        self.assertRaises(ValueError, self.memory.sample)  # samples in memory less than sample batch size
        self.memory.store(4, 5, 6)
        res = self.memory.sample()
        self.assertEqual(len(res), self.memory.batch_size, "Should be 2")
        self.assertEqual(self.memory.num_of_samples(), 2, "Should be 2")
        self.memory.store(7, 8, 9)
        res = self.memory.sample()
        self.assertEqual(len(res), self.memory.batch_size, "Should be 2")
        self.assertEqual(self.memory.num_of_samples(), 3, "Should be 3")

    def test_full(self):
        """ Test the operation of memory when it's full. """

        for i in range(self.memory.capacity):
            self.memory.store(i)
        self.assertEqual(self.memory.num_of_samples(), self.memory.capacity, "Should be 10")
        self.memory.store(self.memory.capacity + 1)
        self.assertEqual(self.memory.num_of_samples(), self.memory.capacity, "Should be 10")

    def test_flush(self):
        """ Test that flush method actually empties the memory. """

        self.memory.store(1, 2, 3)
        self.memory.store(4, 5, 6)
        self.assertEqual(self.memory.num_of_samples(), 2, "Should be 2")
        self.memory.flush()
        self.assertEqual(self.memory.num_of_samples(), 0, "Should be 0")


if __name__ == '__main__':
    unittest.main()
