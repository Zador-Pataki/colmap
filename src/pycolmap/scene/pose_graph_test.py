import numpy as np
import pycolmap


def test_pose_graph_edge_default_init():
    edge = pycolmap.PoseGraphEdge()
    assert edge is not None


def test_pose_graph_edge_init_with_rigid3d():
    rigid = pycolmap.Rigid3d()
    edge = pycolmap.PoseGraphEdge(cam2_from_cam1=rigid)
    assert edge is not None


def test_pose_graph_edge_cam2_from_cam1_readwrite():
    edge = pycolmap.PoseGraphEdge()
    rigid = pycolmap.Rigid3d()
    edge.cam2_from_cam1 = rigid
    assert isinstance(edge.cam2_from_cam1, pycolmap.Rigid3d)


def test_pose_graph_edge_num_matches_readwrite():
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 100
    assert edge.num_matches == 100


def test_pose_graph_edge_valid_readwrite():
    edge = pycolmap.PoseGraphEdge()
    edge.valid = True
    assert edge.valid is True
    edge.valid = False
    assert edge.valid is False


def test_pose_graph_edge_invert():
    edge = pycolmap.PoseGraphEdge()
    edge.valid = True
    edge.num_matches = 50
    edge.invert()


def test_pose_graph_default_init():
    graph = pycolmap.PoseGraph()
    assert graph is not None


def test_pose_graph_empty():
    graph = pycolmap.PoseGraph()
    assert graph.empty is True


def test_pose_graph_num_edges():
    graph = pycolmap.PoseGraph()
    assert graph.num_edges == 0


def test_pose_graph_add_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    assert graph.num_edges == 1


def test_pose_graph_has_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(3, 4)


def test_pose_graph_get_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 42
    edge.valid = True
    graph.add_edge(1, 2, edge)
    retrieved_edge = graph.get_edge(1, 2)
    assert retrieved_edge.num_matches == 42


def test_pose_graph_delete_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    result = graph.delete_edge(1, 2)
    assert result is True
    assert graph.num_edges == 0


def test_pose_graph_update_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    updated_edge = pycolmap.PoseGraphEdge()
    updated_edge.num_matches = 99
    updated_edge.valid = True
    graph.update_edge(1, 2, updated_edge)
    retrieved = graph.get_edge(1, 2)
    assert retrieved.num_matches == 99


def test_pose_graph_clear():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    graph.clear()
    assert graph.num_edges == 0
    assert graph.empty is True


def test_pose_graph_edges_property():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    edges = graph.edges
    assert len(edges) == 1


def test_pose_graph_edge_map_type():
    graph = pycolmap.PoseGraph()
    edges = graph.edges
    assert isinstance(edges, pycolmap.PoseGraphEdgeMap)


# --- Fork Edge field tests ---



def test_pose_graph_edge_rel_depth_scale_default():
    edge = pycolmap.PoseGraphEdge()
    assert np.isclose(edge.rel_depth_scale, 1.0)


def test_pose_graph_edge_rel_depth_scale_roundtrip():
    edge = pycolmap.PoseGraphEdge()
    edge.rel_depth_scale = 2.5
    assert np.isclose(edge.rel_depth_scale, 2.5)


def test_pose_graph_edge_cov_t_default():
    edge = pycolmap.PoseGraphEdge()
    assert np.allclose(edge.cov_t, np.zeros((3, 3)))


def test_pose_graph_edge_cov_t_roundtrip():
    edge = pycolmap.PoseGraphEdge()
    edge.cov_t = np.eye(3)
    assert np.allclose(edge.cov_t, np.eye(3))


def test_pose_graph_edge_are_lc_default_empty():
    edge = pycolmap.PoseGraphEdge()
    assert edge.are_lc == []


def test_pose_graph_edge_are_lc_roundtrip():
    edge = pycolmap.PoseGraphEdge()
    edge.are_lc = [True, False, True]
    assert edge.are_lc == [True, False, True]


def test_pose_graph_edge_is_lc_default_false():
    edge = pycolmap.PoseGraphEdge()
    assert edge.is_LC is False


def test_pose_graph_edge_is_lc_roundtrip():
    edge = pycolmap.PoseGraphEdge()
    edge.is_LC = True
    assert edge.is_LC is True


def test_pose_graph_edge_weight_default():
    edge = pycolmap.PoseGraphEdge()
    assert np.isclose(edge.weight, 1.0)


def test_pose_graph_edge_weight_roundtrip():
    edge = pycolmap.PoseGraphEdge()
    edge.weight = 0.75
    assert np.isclose(edge.weight, 0.75)


# --- PoseGraph batch method tests ---


def test_pose_graph_filter_inlier_num_removes_below_threshold():
    graph = pycolmap.PoseGraph()
    e1, e2, e3 = (pycolmap.PoseGraphEdge() for _ in range(3))
    e1.num_matches = 5
    e1.valid = True
    e2.num_matches = 15
    e2.valid = True
    e3.num_matches = 8
    e3.valid = True
    graph.add_edge(1, 2, e1)
    graph.add_edge(2, 3, e2)
    graph.add_edge(3, 4, e3)
    graph.filter_inlier_num(10)
    assert graph.get_edge(1, 2).valid is False
    assert graph.get_edge(2, 3).valid is True
    assert graph.get_edge(3, 4).valid is False


def test_pose_graph_filter_inlier_ratio_removes_below_threshold():
    graph = pycolmap.PoseGraph()
    e1, e2 = pycolmap.PoseGraphEdge(), pycolmap.PoseGraphEdge()
    e1.num_matches = 5
    e1.total_matches = 100
    e1.valid = True
    e2.num_matches = 60
    e2.total_matches = 100
    e2.valid = True
    graph.add_edge(1, 2, e1)
    graph.add_edge(2, 3, e2)
    graph.filter_inlier_ratio(0.5)
    assert graph.get_edge(1, 2).valid is False
    assert graph.get_edge(2, 3).valid is True


def test_pose_graph_keep_largest_connected_components():
    graph = pycolmap.PoseGraph()
    e = pycolmap.PoseGraphEdge()
    e.valid = True
    e.num_matches = 10
    graph.add_edge(1, 2, e)
    graph.keep_largest_connected_components(1, 1)


def test_pose_graph_assign_mdrp_results_empty():
    graph = pycolmap.PoseGraph()
    counts = graph.assign_mdrp_results({}, 0.1)
    assert counts == (0, 0)


def test_pose_graph_extract_valid_pair_data_empty():
    graph = pycolmap.PoseGraph()
    data = graph.extract_valid_pair_data()
    assert len(data.pair_ids) == 0
