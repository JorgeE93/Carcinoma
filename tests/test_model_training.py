import unittest
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.Carcinoma import MyDataset, train_transform, get_weights, Config


class TestMyDataset(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.sample_data = [([[0] * 512] * 512, 0)] * 10  # Sample data
        self.dataset = MyDataset(self.sample_data, train_transform(self.config))

    def test_len(self):
        self.assertEqual(len(self.dataset), 10)

    def test_getitem(self):
        img, label = self.dataset[0]
        self.assertEqual(img.shape, (3, 512, 512))
        self.assertEqual(label, 0)


class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.sample_data = [([[0] * 512] * 512, 0)] * 10  # Sample data
        self.dataset = MyDataset(self.sample_data, train_transform(self.config))

    def test_dataloader(self):
        train_weights = get_weights(self.dataset)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        trainloader = DataLoader(self.dataset, batch_size=2, sampler=train_sampler, num_workers=0)

        for images, labels in trainloader:
            self.assertEqual(images.shape, (2, 3, 512, 512))
            self.assertEqual(labels.shape, (2,))
            break  # Test one batch


if __name__ == '__main__':
    unittest.main()
