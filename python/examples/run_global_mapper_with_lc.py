"""
Naive baseline that exercises the LC (loop closure) extensions of the global
mapper.

Pipeline:
    feature extraction
        -> sequential matcher (writes ``are_lc=false`` per-inlier into the
           database; treated as tracking-style pairs)
        -> vocab-tree matcher (writes ``are_lc=true``; treated as
           loop-closure pairs)
        -> ``run_global_mapper`` with the LC flags ON

The four LC code paths exercised by the run are:

    track_lc_second_pass
        ``MakeLoopClosureMatchPredicate`` filters LC inliers out of the
        union-find pass. ``AppendLoopClosureObservations`` then attaches the
        LC observations as parallel ``Track::lc_elements`` of the regular
        tracks (so geometry is unchanged but LC residuals can fire).

    use_lc_observations
        Global positioning treats ``Track::lc_elements`` as additional
        observations (with their own optional Cauchy loss / weight). With
        empty LC elements this is a no-op; with vocab-tree-tagged inliers
        the Ceres problem includes those residuals.

    skip_risky_LC_pairs
        Rotation averaging skips pairs whose LC inliers strictly outnumber
        non-LC inliers. Reads ``ImagePair.are_lc`` from the
        ``CorrespondenceGraph``; without ``are_lc`` populated this filter
        silently never triggers.

    MST LC-penalty (rotation averaging init)
        ``InitializeFromMaximumSpanningTree`` subtracts a large penalty from
        edges whose ``are_lc`` true count exceeds non-LC inliers, so the MST
        routes around them. Active when ``use_video_constraints=true``.

Without ``are_lc`` plumbed through the database (older databases or matchers
that do not tag pairs), all four paths silently degrade to vanilla colmap
behaviour: every pair looks tracking-like, no LC residuals, no MST penalty,
no risky-pair skipping.

Usage:
    python run_global_mapper_with_lc.py \\
        --image-path /path/to/images \\
        --database-path /tmp/lc_baseline.db \\
        --output-path /tmp/lc_baseline_recon \\
        --vocab-tree-path /path/to/vocab_tree.bin
"""

import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--vocab-tree-path",
        type=Path,
        required=True,
        help="Pretrained vocab tree (e.g. vocab_tree_flickr100K_words256K.bin).",
    )
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)
    if args.database_path.exists():
        args.database_path.unlink()

    # Feature extraction.
    pycolmap.extract_features(
        database_path=args.database_path,
        image_path=args.image_path,
    )

    # Sequential matcher writes ``are_lc=false`` per-inlier into the
    # ``two_view_geometries.are_lc`` column.
    pycolmap.match_sequential(
        database_path=args.database_path,
    )

    # Vocab-tree matcher writes ``are_lc=true``.
    pycolmap.match_vocabtree(
        database_path=args.database_path,
        vocab_tree_path=args.vocab_tree_path,
    )

    # Global mapper with LC flags ON.
    options = pycolmap.GlobalMapperOptions()
    options.track_lc_second_pass = True
    options.global_positioning.use_lc_observations = True
    options.rotation_averaging.skip_risky_LC_pairs = True
    options.rotation_averaging.use_video_constraints = True

    pycolmap.run_global_mapper(
        database_path=args.database_path,
        image_path=args.image_path,
        output_path=args.output_path,
        options=options,
    )


if __name__ == "__main__":
    main()
