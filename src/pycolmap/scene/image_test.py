import numpy as np

import pycolmap


def test_image_image_id_readwrite():
    image = pycolmap.Image()
    image.image_id = 10
    assert image.image_id == 10


def test_image_camera_id_readwrite():
    image = pycolmap.Image()
    image.camera_id = 5
    assert image.camera_id == 5


def test_image_frame_id_readwrite():
    image = pycolmap.Image()
    image.frame_id = 3
    assert image.frame_id == 3


def test_image_name_readwrite():
    image = pycolmap.Image()
    image.name = "test_image.jpg"
    assert image.name == "test_image.jpg"


def test_image_points2d_readwrite():
    image = pycolmap.Image()
    point_list = pycolmap.Point2DList()
    point_list.append(pycolmap.Point2D(xy=np.array([1.0, 2.0])))
    point_list.append(pycolmap.Point2D(xy=np.array([3.0, 4.0])))
    image.points2D = point_list
    assert len(image.points2D) == 2


def test_image_has_camera_id():
    image = pycolmap.Image()
    assert not image.has_camera_id()
    image.camera_id = 1
    assert image.has_camera_id()


def test_image_has_frame_id():
    image = pycolmap.Image()
    assert not image.has_frame_id()
    image.frame_id = 1
    assert image.has_frame_id()


def test_image_num_points2d():
    image = pycolmap.Image()
    assert image.num_points2D() == 0
    point_list = pycolmap.Point2DList()
    point_list.append(pycolmap.Point2D(xy=np.array([1.0, 2.0])))
    image.points2D = point_list
    assert image.num_points2D() == 1


def test_image_num_points3d_readonly():
    image = pycolmap.Image()
    assert image.num_points3D == 0


def test_image_has_pose_readonly():
    image = pycolmap.Image()
    assert isinstance(image.has_pose, bool)


def test_image_map_empty():
    image_map = pycolmap.ImageMap()
    assert len(image_map) == 0


def test_image_map_insert_and_access():
    image_map = pycolmap.ImageMap()
    image = pycolmap.Image()
    image.image_id = 1
    image.name = "test.jpg"
    image_map[1] = image
    assert len(image_map) == 1
    assert image_map[1].name == "test.jpg"


# --- Fork field tests ---


def test_image_depth_priors_default_empty():
    image = pycolmap.Image()
    assert image.depth_priors == []


def test_image_depth_priors_roundtrip():
    image = pycolmap.Image()
    image.depth_priors = [1.0, 2.0, 3.0]
    assert image.depth_priors == [1.0, 2.0, 3.0]


def test_image_depth_prior_stddevs_roundtrip():
    image = pycolmap.Image()
    image.depth_prior_stddevs = [0.1, 0.2]
    assert image.depth_prior_stddevs == [0.1, 0.2]


def test_image_depth_prior_validity_roundtrip():
    image = pycolmap.Image()
    image.depth_prior_validity = [True, False, True]
    assert image.depth_prior_validity == [True, False, True]


def test_image_is_inlier_roundtrip():
    image = pycolmap.Image()
    image.is_inlier = [True, True]
    assert image.is_inlier == [True, True]


def test_image_is_depth_outlier_roundtrip():
    image = pycolmap.Image()
    image.is_depth_outlier = [False, True]
    assert image.is_depth_outlier == [False, True]


def test_image_is_track_anchor_roundtrip():
    image = pycolmap.Image()
    image.is_track_anchor = [False]
    assert image.is_track_anchor == [False]


def test_image_is_excluded_roundtrip():
    image = pycolmap.Image()
    image.is_excluded = [True]
    assert image.is_excluded == [True]


def test_image_angular_stddevs_roundtrip():
    image = pycolmap.Image()
    val = [np.array([0.01, 0.02])]
    image.angular_stddevs = val
    assert len(image.angular_stddevs) == 1
    assert np.allclose(image.angular_stddevs[0], [0.01, 0.02])


def test_image_angular_cholesky_xy_roundtrip():
    image = pycolmap.Image()
    val = [np.array([1.0, 0.0, 1.0])]
    image.angular_cholesky_xy = val
    assert len(image.angular_cholesky_xy) == 1
    assert np.allclose(image.angular_cholesky_xy[0], [1.0, 0.0, 1.0])


def test_image_angular_stddevs_z_roundtrip():
    image = pycolmap.Image()
    image.angular_stddevs_z = [0.05]
    assert np.isclose(image.angular_stddevs_z[0], 0.05)


def test_image_log_scale_roundtrip():
    image = pycolmap.Image()
    assert image.log_scale == 0.0
    image.log_scale = 0.5
    assert np.isclose(image.log_scale, 0.5)


def test_image_log_scale_stddev_roundtrip():
    image = pycolmap.Image()
    image.log_scale_stddev = 0.1
    assert np.isclose(image.log_scale_stddev, 0.1)


def test_image_gravity_sigma_roundtrip():
    image = pycolmap.Image()
    assert image.gravity_sigma == 0.0
    image.gravity_sigma = 0.035
    assert np.isclose(image.gravity_sigma, 0.035)


def test_image_features_undist_roundtrip():
    image = pycolmap.Image()
    val = [np.array([0.1, 0.2, 1.0])]
    image.features_undist = val
    assert len(image.features_undist) == 1
    assert np.allclose(image.features_undist[0], [0.1, 0.2, 1.0])


def test_image_gravity_info_default():
    image = pycolmap.Image()
    assert image.gravity_info.has_gravity is False


def test_image_gravity_info_set_gravity():
    image = pycolmap.Image()
    g = np.array([0.0, -1.0, 0.0])
    image.gravity_info.SetGravity(g)
    assert image.gravity_info.has_gravity is True
    assert np.allclose(image.gravity_info.GetGravity(), g)


def test_image_gravity_info_get_r_align():
    image = pycolmap.Image()
    g = np.array([0.0, -1.0, 0.0])
    image.gravity_info.SetGravity(g)
    R = image.gravity_info.GetRAlign()
    assert R.shape == (3, 3)
    assert np.allclose(R[:, 1], g)
