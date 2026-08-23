Obsolete. Read [tools/ROADMAP.md](../ROADMAP.md).

cuVSLAM tracker
==============

This is the standalone command line application for image sequence tracking.
Tracker use edex files as input and output. Edex files (cuVSLAM Data EXchange) is
the json-based scene description format. Edex could store image sequences,
camera extrincis, 2d tracks, 3d tracking results (camera curves and 3d tracks positions).


Command Line
------------
Usage:
tracker config_file.cfg

All parameters stored inside the configuration file. Tracker will search it
in the path stored in CUVSLAM_DATASETS environment variable.

If you use defaults during cmake cuVSLAM generation, you find cuvslam_start_cmd
script in cuVSLAM build folder. cuvslam_start_cmd will care about environment.

Config file format
------------------
Config file is simple json like:

{
  "version": "0.1",
  "in_edex": "test/ft_drone_v03.edex",
  "out_edex": "edex_result/result_ft_drone_v03_filter.edex",
  "camera_id": 0,
  "reverse": false,
  "precompute_2d_tracks": true,
  "precompute_keyframes": false,
  "filter_2d_tracks": true,
  "use_cuda": false
}

* version - File version.
* in_edex - Input scene file.
* out_edex - Output scene file.
* camera_id - Choose camera to track.
* reverse - Track frames ordering.
* precompute_2d_tracks - Don't recalculate sparse optical flow. Use already stored in scene.
* precompute_keyframes - Don't recalculate which frame is keyframes. Use already stored in scene.
* filter_2d_tracks - Don't stored all 2d tracks in output scene. Only well triangulated.
* use_cuda - Use CUDA (or VisionWorks) implementations.
