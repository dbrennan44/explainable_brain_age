"""BIDS-oriented command-line wrapper for explainable brain-age inference."""

import argparse
import json
from pathlib import Path


DEFAULT_DERIVATIVE_NAME = "brain_age_salience"


class ArgumentFormatter(
    argparse.RawDescriptionHelpFormatter,
):
    """Show useful defaults while preserving example formatting."""

    def _get_help_string(self, action):
        help_text = action.help
        if help_text is argparse.SUPPRESS:
            return help_text
        if "%(default)" in help_text:
            return help_text
        if not action.option_strings or action.default is argparse.SUPPRESS:
            return help_text
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return help_text
        return f"{help_text} (default: %(default)s)"


def _ensure_bids_label(label, prefix):
    if label is None:
        return None

    label = str(label).strip()
    if not label:
        raise ValueError(f"Empty {prefix} label.")

    return label if label.startswith(f"{prefix}-") else f"{prefix}-{label}"


def _strip_nifti_suffix(path):
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _bids_parts(path):
    return _strip_nifti_suffix(path).split("_")


def _bids_base(subject_label, session_label=None, run_label=None):
    parts = [subject_label]
    if session_label:
        parts.append(session_label)
    if run_label:
        parts.append(run_label)
    return "_".join(parts)


def _resolve_derivatives_dir(bids_dir, derivatives_dir):
    if derivatives_dir is None:
        return bids_dir / "derivatives" / DEFAULT_DERIVATIVE_NAME

    derivatives_dir = Path(derivatives_dir).expanduser()
    if derivatives_dir.is_absolute():
        return derivatives_dir

    return bids_dir / derivatives_dir


def _find_raw_t1w(bids_dir, subject_label, session_label=None, run_label=None):
    anat_parts = [bids_dir, subject_label]
    if session_label:
        anat_parts.append(session_label)
    anat_parts.append("anat")
    anat_dir = Path(*anat_parts)

    if not anat_dir.exists():
        session_hint = ""
        if session_label is None:
            session_hint = " Provide --session for datasets that use ses-* folders."
        raise FileNotFoundError(f"Missing BIDS anat directory: {anat_dir}.{session_hint}")

    candidates = []
    for pattern in ("*_T1w.nii.gz", "*_T1w.nii"):
        for path in anat_dir.glob(pattern):
            parts = _bids_parts(path)
            if subject_label not in parts:
                continue
            if session_label and session_label not in parts:
                continue
            if run_label and run_label not in parts:
                continue
            candidates.append(path)

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "No matching raw T1w image found in BIDS anat directory:\n"
            f"  {anat_dir}\n"
            f"  subject={subject_label}, session={session_label}, run={run_label}"
        )

    if len(candidates) > 1:
        candidate_list = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Multiple T1w images matched. Provide --run or --t1-image:\n"
            f"  {candidate_list}"
        )

    return candidates[0]


def _find_preprocessed_t1(derivatives_dir, base):
    expected = derivatives_dir / f"{base}_desc-preproc_T1w.nii.gz"
    if expected.exists():
        return expected

    legacy_matches = sorted(derivatives_dir.glob(f"{base}*_desc-preproc_T1w.nii.gz"))
    if len(legacy_matches) == 1:
        return legacy_matches[0]
    if len(legacy_matches) > 1:
        candidate_list = "\n  ".join(str(path) for path in legacy_matches)
        raise RuntimeError(
            "Multiple preprocessed T1 images matched. Provide --t1-image:\n"
            f"  {candidate_list}"
        )

    raise FileNotFoundError(
        "Expected preprocessed T1 not found:\n"
        f"  {expected}\n"
        "Run with --do_preprocessing true, or provide --t1-image."
    )


def _analysis_label(args):
    parts = ["mean" if args.mean_head else "median"]
    if args.n_affine:
        parts.append(f"affine{args.n_affine}")
    if args.n_smooth:
        parts.append(f"smooth{args.n_smooth}{args.smoothgrad_mode}")
    if args.mask_noise:
        parts.append("maskedNoise")
    if args.no_slice_norm:
        parts.append("rawSliceGrad")
    return "".join(part[0].upper() + part[1:] for part in parts)


def _jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _str_to_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False

    raise argparse.ArgumentTypeError("expected true or false")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run ANTsPyNet brain-age inference with salience maps on a BIDS T1w image.",
        formatter_class=ArgumentFormatter,
        epilog="""examples:
  Run preprocessing from a raw BIDS T1w:
    python brain_age_salience_bids.py /path/to/bids sub-001 --session ses-01

  Reuse the ANTsPyNet brain_age-preprocessed T1 saved in derivatives:
    python brain_age_salience_bids.py /path/to/bids 001 --session 01 --do_preprocessing false --n-smooth 25 --mask-noise

  Analyze an explicit ANTsPyNet brain_age-preprocessed T1 image:
    python brain_age_salience_bids.py /path/to/bids sub-001 --t1-image /path/to/preproc_T1w.nii.gz --do_preprocessing false

  Use the original ANTsPyNet median-of-slices fallback:
    python brain_age_salience_bids.py /path/to/bids sub-001 --session ses-01 --median-head
""",
    )

    parser.add_argument("bids_dir", help="BIDS dataset root.")
    parser.add_argument(
        "participant_label",
        help="Participant label, with or without the sub- prefix.",
    )
    parser.add_argument(
        "--session",
        "--session-label",
        dest="session_label",
        default=None,
        help="Session label, with or without the ses- prefix.",
    )
    parser.add_argument(
        "--run",
        "--run-label",
        dest="run_label",
        default=None,
        help="Run label, with or without the run- prefix.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--derivatives-dir",
        dest="derivatives_dir",
        default=f"derivatives/{DEFAULT_DERIVATIVE_NAME}",
        help=(
            "Output derivative directory. Relative paths are resolved under bids_dir; "
            f"the default is derivatives/{DEFAULT_DERIVATIVE_NAME}."
        ),
    )

    input_group = parser.add_argument_group("input and preprocessing")
    input_group.add_argument(
        "--do_preprocessing",
        dest="do_preprocessing",
        nargs="?",
        const=True,
        default=True,
        type=_str_to_bool,
        metavar="{true,false}",
        help=(
            "Run ANTsPyNet preprocessing before inference. Set to false only "
            "when the input is already preprocessed by ANTsPyNet brain_age/this "
            "brain-age salience pipeline; arbitrary preprocessing will not work."
        ),
    )
    input_group.add_argument(
        "--preprocess",
        "--do-preprocessing",
        dest="do_preprocessing",
        nargs="?",
        const=True,
        default=argparse.SUPPRESS,
        type=_str_to_bool,
        help=argparse.SUPPRESS,
    )
    input_group.add_argument(
        "--t1-image",
        dest="t1_image",
        default=None,
        help=(
            "Explicit path to a T1 image. By default this image is assumed to be "
            "raw and will be preprocessed; combine with --do_preprocessing false "
            "only for ANTsPyNet brain_age-preprocessed inputs."
        ),
    )
    input_group.add_argument(
        "--t1-manual",
        "--t1_manual",
        dest="t1_image",
        default=None,
        help=argparse.SUPPRESS,
    )

    model_group = parser.add_argument_group("model and salience options")
    parser.set_defaults(mean_head=True)
    model_group.add_argument(
        "--mean-head",
        "--mean_head",
        dest="mean_head",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    model_group.add_argument(
        "--median-head",
        "--median_head",
        dest="mean_head",
        action="store_false",
        help=(
            "Use the ANTsPyNet fallback: median of standard slice-wise predictions. "
            "By default, the differentiable mean head is used."
        ),
    )
    model_group.add_argument(
        "--mask-noise",
        "--mask_noise",
        dest="mask_noise",
        action="store_true",
        help="Restrict SmoothGrad noise to nonzero brain voxels.",
    )
    model_group.add_argument(
        "--no-slice-norm",
        "--no_slice_norm",
        dest="no_slice_norm",
        action="store_true",
        help="Disable per-slice salience normalization.",
    )
    model_group.add_argument(
        "--n-affine",
        "--n_affine",
        dest="n_affine",
        type=int,
        default=0,
        help="Number of affine augmentation simulations to average.",
    )
    model_group.add_argument(
        "--sd-affine",
        "--sd_affine",
        dest="sd_affine",
        type=float,
        default=0.01,
        help="Affine transform standard deviation for augmentation.",
    )
    model_group.add_argument(
        "--n-smooth",
        "--n_smooth",
        dest="n_smooth",
        type=int,
        default=0,
        help="Number of SmoothGrad noise samples. Set >0 to write SmoothGrad output.",
    )
    model_group.add_argument(
        "--sd-noise",
        "--sd_noise",
        dest="sd_noise",
        type=float,
        default=0.20,
        help="SmoothGrad Gaussian noise standard deviation on normalized image intensities.",
    )
    model_group.add_argument(
        "--smoothgrad-mode",
        "--mode",
        dest="smoothgrad_mode",
        choices=["square", "abs"],
        default="square",
        help="Transform applied to SmoothGrad gradients before averaging.",
    )
    model_group.add_argument(
        "--slice-start",
        type=int,
        default=25,
        help="First axial slice index used by the 2D brain-age model.",
    )
    model_group.add_argument(
        "--slice-stop",
        type=int,
        default=145,
        help="Exclusive stop axial slice index used by the 2D brain-age model.",
    )
    model_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for TensorFlow/NumPy stochastic steps.",
    )

    return parser


def validate_args(parser, args):
    if args.n_affine < 0:
        parser.error("--n-affine must be >= 0.")
    if args.n_smooth < 0:
        parser.error("--n-smooth must be >= 0.")
    if args.sd_affine < 0:
        parser.error("--sd-affine must be >= 0.")
    if args.sd_noise < 0:
        parser.error("--sd-noise must be >= 0.")
    if args.slice_start >= args.slice_stop:
        parser.error("--slice-start must be less than --slice-stop.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be >= 0.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    bids_dir = Path(args.bids_dir).expanduser().resolve()
    if not bids_dir.exists():
        parser.error(f"bids_dir does not exist: {bids_dir}")

    subject_label = _ensure_bids_label(args.participant_label, "sub")
    session_label = _ensure_bids_label(args.session_label, "ses")
    run_label = _ensure_bids_label(args.run_label, "run")
    base = _bids_base(subject_label, session_label, run_label)

    derivatives_dir = _resolve_derivatives_dir(bids_dir, args.derivatives_dir)
    out_dir = derivatives_dir / subject_label
    if session_label:
        out_dir = out_dir / session_label
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import ants

        from brain_age_salience import brain_age_with_affine_smoothgrad_unified
    except ImportError as exc:
        parser.exit(
            1,
            "Missing Python dependency needed for inference:\n"
            f"  {exc}\n"
            "Create the project environment with:\n"
            "  conda env create -f environment.yml\n",
        )

    if args.t1_image:
        t1_path = Path(args.t1_image).expanduser().resolve()
        if not t1_path.exists():
            raise FileNotFoundError(f"Explicit T1 image does not exist: {t1_path}")
        print(f"Using explicit T1 image: {t1_path}")
        do_preprocessing = args.do_preprocessing
    elif args.do_preprocessing:
        t1_path = _find_raw_t1w(bids_dir, subject_label, session_label, run_label)
        print(f"Using raw BIDS T1w image for preprocessing: {t1_path}")
        do_preprocessing = True
    else:
        t1_path = _find_preprocessed_t1(out_dir, base)
        print(f"Using existing preprocessed T1 image: {t1_path}")
        do_preprocessing = False

    if do_preprocessing:
        print("ANTsPyNet preprocessing is enabled.")
    else:
        print(
            "Preprocessing is disabled; input is assumed to be ANTsPyNet "
            "brain_age/brain-age salience preprocessed."
        )

    t1_img = ants.image_read(str(t1_path))
    result = brain_age_with_affine_smoothgrad_unified(
        t1_img,
        do_preprocessing=do_preprocessing,
        use_mean_head=args.mean_head,
        number_of_simulations=args.n_affine,
        smooth_samples=args.n_smooth,
        sd_affine=args.sd_affine,
        smooth_noise=args.sd_noise,
        smooth_variant=args.smoothgrad_mode,
        mask_noise=args.mask_noise,
        slice_wise_normalization=not args.no_slice_norm,
        which_slices=range(args.slice_start, args.slice_stop),
        random_seed=args.seed,
        verbose=True,
    )

    analysis_label = _analysis_label(args)
    json_path = out_dir / f"{base}_desc-{analysis_label}_brainage.json"
    vanilla_path = out_dir / f"{base}_desc-{analysis_label}Vanilla_salience.nii.gz"
    smoothgrad_path = out_dir / f"{base}_desc-{analysis_label}SmoothGrad_salience.nii.gz"
    preproc_path = out_dir / f"{base}_desc-preproc_T1w.nii.gz"

    result["vanilla_salience"].to_file(str(vanilla_path))
    if result["smoothgrad_salience"] is not None:
        result["smoothgrad_salience"].to_file(str(smoothgrad_path))
    else:
        smoothgrad_path = None

    result["t1_preprocessed"].to_file(str(preproc_path))

    metadata = {
        "predicted_age": float(result["predicted_age"]),
        "brain_age_per_slice": _jsonable(result["brain_age_per_slice"]),
        "input": {
            "bids_dir": str(bids_dir),
            "t1_image": str(t1_path),
            "participant_label": subject_label,
            "session_label": session_label,
            "run_label": run_label,
        },
        "processing": {
            "preprocessing_run": do_preprocessing,
            "head": "mean" if args.mean_head else "median",
            "n_affine": args.n_affine,
            "sd_affine": args.sd_affine,
            "n_smooth": args.n_smooth,
            "sd_noise": args.sd_noise,
            "smoothgrad_mode": args.smoothgrad_mode,
            "mask_noise": args.mask_noise,
            "slice_wise_normalization": not args.no_slice_norm,
            "slice_start": args.slice_start,
            "slice_stop": args.slice_stop,
            "seed": args.seed,
        },
        "outputs": {
            "preprocessed_t1": str(preproc_path),
            "vanilla_salience": str(vanilla_path),
            "smoothgrad_salience": None if smoothgrad_path is None else str(smoothgrad_path),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Predicted brain age: {float(result['predicted_age']):.2f}")
    print(f"Wrote metadata: {json_path}")
    print(f"Wrote vanilla salience: {vanilla_path}")
    if smoothgrad_path is not None:
        print(f"Wrote SmoothGrad salience: {smoothgrad_path}")
    print(f"Wrote preprocessed T1: {preproc_path}")


if __name__ == "__main__":
    main()
