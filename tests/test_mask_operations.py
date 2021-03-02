import numpy as np

high_ranges = ((0, 10), (50, 60), (80, 90), (91, 100))
duration = 100
test_bool_map = np.zeros(duration)
res_test_bool_map = 1
for s, e in high_ranges:
    test_bool_map[s:e] = 1

def test_filter_edge_events(test_bool_map, res_test_bool_map, 10, 10, duration):
    pass
