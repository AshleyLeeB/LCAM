from collections import OrderedDict


class SubgraphCache:
    def __init__(self, max_size=200000):
        self.max_size = max_size
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, node_id, h, max_nodes):
        return (int(node_id), int(h), int(max_nodes))

    def get(self, node_id, h, max_nodes):
        key = self._key(node_id, h, max_nodes)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, node_id, h, max_nodes, value):
        key = self._key(node_id, h, max_nodes)
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return {
            'size': len(self._cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }
