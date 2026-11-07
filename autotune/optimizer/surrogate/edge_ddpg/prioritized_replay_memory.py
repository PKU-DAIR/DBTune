import random
import pickle
import numpy as np
from collections import deque


class UniformReplayMemory(object):
    """
    Tier-balanced replay memory (v4).

    Stores transitions tagged by hardware tier and samples an equal number
    from each tier present in the buffer.  This preserves the cross-tier
    diversity that drives the C2→C3 performance jump (Critic learns a global
    Q-function across all tiers) while avoiding the PER bias toward stale
    high-TD-error transitions that caused C4 regression in v3.

    Buffer persists across phases (like v3) so the Critic retains cross-tier
    knowledge, but each tier contributes equally to every training batch
    regardless of how many steps were spent in that tier.

    Interface identical to PrioritizedReplayMemory — no other code changes
    except callers optionally pass tier= to add().
    """

    def __init__(self, capacity):
        self.capacity = capacity
        # Store (tier_key, sample) tuples in a single ring-buffer.
        # Tier tagging happens at add() time; grouping at sample() time.
        self.buffer = deque(maxlen=capacity)

    def add(self, error, sample, tier=None):
        # error is ignored — no priority tree.
        # tier is an arbitrary string key (e.g. "rpi_2gb"); None = unknown.
        tier_key = tier if tier is not None else '__unknown__'
        self.buffer.append((tier_key, sample))

    def __len__(self):
        return len(self.buffer)

    def sample(self, n):
        n = min(n, len(self.buffer))
        if n == 0:
            return [], []

        # Group available transitions by tier
        by_tier = {}
        for tier_key, sample in self.buffer:
            by_tier.setdefault(tier_key, []).append(sample)

        tiers = list(by_tier.keys())
        random.shuffle(tiers)          # randomise which tier gets the remainder
        num_tiers = len(tiers)
        base = n // num_tiers
        remainder = n % num_tiers

        batch = []
        for i, tier_key in enumerate(tiers):
            k = base + (1 if i < remainder else 0)
            k = min(k, len(by_tier[tier_key]))
            batch.extend(random.sample(by_tier[tier_key], k))

        # Edge-case: some tiers had fewer than base samples → top up randomly
        if len(batch) < n:
            batch_ids = {id(s) for s in batch}
            all_samples = [s for (_, s) in self.buffer if id(s) not in batch_ids]
            extra = min(n - len(batch), len(all_samples))
            if extra > 0:
                batch.extend(random.sample(all_samples, extra))

        random.shuffle(batch)          # avoid tier-ordering bias within the batch
        idxs = [0] * len(batch)        # dummy — update() is a no-op
        return batch, idxs

    def update(self, idx, error):
        pass   # no priority tree to update

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'buffer': list(self.buffer)}, f)

    def load_memory(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if 'buffer' in data:
            # Native format: list of (tier_key, sample) tuples
            entries = data['buffer']
            # Back-compat: plain samples without tier tag
            tagged = []
            for entry in entries:
                if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
                    tagged.append(entry)
                else:
                    tagged.append(('__unknown__', entry))
            self.buffer = deque(tagged, maxlen=self.capacity)
        elif 'tree' in data:
            # Legacy PER SumTree format — extract raw samples
            tree = data['tree']
            for item in tree.data:
                if item != 0 and item is not None:
                    self.buffer.append(('__unknown__', item))


class SumTree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.num_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return [idx, self.tree[idx], self.data[data_idx]]


class PrioritizedReplayMemory(object):

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        # (s, a, r, s, t)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def __len__(self):
        return self.tree.num_entries

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        return batch, idxs

        # sampling_probabilities = priorities / self.tree.total()
        # is_weight = np.power(self.tree.num_entries * sampling_probabilities, -self.beta)
        # is_weight /= is_weight.max()

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump({"tree": self.tree}, f)
        f.close()

    def load_memory(self, path):
        with open(path, 'rb') as f:
            _memory = pickle.load(f)
        self.tree = _memory['tree']

