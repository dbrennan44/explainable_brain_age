"""Standalone and importable explainable brain-age analysis."""

import argparse
import json
from pathlib import Path


DEFAULT_SLICES = range(25, 145)


def brain_age_with_affine_smoothgrad_unified(
    t1,
    do_preprocessing=True,
    use_mean_head=True,
    number_of_simulations=0,
    sd_affine=0.01,
    smooth_samples=0,
    smooth_noise=0.20,
    verbose=False,
    which_slices=None,
    smooth_variant="square",
    mask_noise=False,
    slice_wise_normalization=True,
    random_seed=None,
):
    """
    Unified brain-age pipeline with:
        - optional preprocessing
        - optional differentiable mean-head model
        - optional affine simulations
        - optional SmoothGrad
        - optional noise masking

    Parameters
    ----------
    use_mean_head : bool
        If True (default), use a differentiable subject-level prediction from
        the mean of slice predictions. If False, use the standard ANTsPyNet
        slice-wise model and summarize by median.

    do_preprocessing : bool
        If True, run ANTsPyNet preprocessing. If False, assume input is preprocessed.

    Returns a dict with:
        predicted_age
        brain_age_per_slice
        vanilla_salience
        smoothgrad_salience
        t1_preprocessed

    slice_wise_normalization : bool
        If True (default) normalize gradients per slice so each map is scaled by
        its own max absolute value. Set to False to keep raw gradients.

    random_seed : int or None
        Optional seed for NumPy and TensorFlow stochastic operations.
    """
    import ants
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras as keras
    from antspynet.utilities import get_pretrained_network, preprocess_brain_image

    if number_of_simulations < 0:
        raise ValueError("number_of_simulations must be >= 0.")
    if smooth_samples < 0:
        raise ValueError("smooth_samples must be >= 0.")
    if sd_affine < 0:
        raise ValueError("sd_affine must be >= 0.")
    if smooth_noise < 0:
        raise ValueError("smooth_noise must be >= 0.")
    if smooth_variant not in {"square", "abs"}:
        raise ValueError("smooth_variant must be either 'square' or 'abs'.")
    if random_seed is not None and random_seed < 0:
        raise ValueError("random_seed must be >= 0.")
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    if which_slices is None:
        which_slices = list(DEFAULT_SLICES)
    else:
        which_slices = list(which_slices)
    if not which_slices:
        raise ValueError("which_slices must contain at least one slice index.")

    # -------------------------
    # Preprocessing
    # -------------------------
    if do_preprocessing:
        prep = preprocess_brain_image(
            t1,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            template="croppedMni152",
            template_transform_type="antsRegistrationSyNQuickRepro[a]",
            do_bias_correction=True,
            do_denoising=True,
            verbose=verbose,
        )
        t1_preprocessed = prep["preprocessed_image"] * prep["brain_mask"]
    else:
        t1_preprocessed = t1

    # Normalize
    image_min = t1_preprocessed.min()
    image_range = t1_preprocessed.max() - image_min
    if image_range <= 0:
        raise ValueError("Input T1 image has no intensity range after preprocessing.")
    t1_preprocessed = (t1_preprocessed - image_min) / image_range

    H, W, D = t1_preprocessed.shape
    if min(which_slices) < 0 or max(which_slices) >= D:
        raise ValueError(
            f"Requested slice range {min(which_slices)}..{max(which_slices)} "
            f"is outside image depth 0..{D - 1}."
        )

    # -------------------------
    # Load original DeepBrainNet
    # -------------------------
    weights = get_pretrained_network("brainAgeDeepBrainNet")
    base_model = keras.models.load_model(weights)

    # -------------------------
    # Wrap mean-head if selected
    # -------------------------
    if use_mean_head:
        inp = keras.Input(shape=(H, W, 3))
        slice_pred = base_model(inp)      # (1,1)
        mean_pred = tf.reduce_mean(slice_pred, axis=0, keepdims=True)
        model = keras.Model(inputs=inp, outputs=mean_pred)
    else:
        model = base_model   # standard slice model

    # -------------------------
    # Affine sims
    # -------------------------
    input_list = [[t1_preprocessed]]

    if number_of_simulations > 0:
        sims = ants.randomly_transform_image_data(
            reference_image=t1_preprocessed,
            input_image_list=input_list,
            number_of_simulations=number_of_simulations,
            transform_type="affine",
            sd_affine=sd_affine,
            input_image_interpolator="linear",
        )
    else:
        sims = {"simulated_images": [], "simulated_transforms": []}

    # -------------------------
    # Accumulators
    # -------------------------
    total_sims = number_of_simulations + 1
    all_predictions = None
    vanilla_accum = None
    smooth_accum = None

    # -------------------------
    # Loop over sims
    # -------------------------
    for sim in range(total_sims):

        # pick simulated or original
        if sim == 0:
            batch_img = t1_preprocessed
            transform = None
        else:
            batch_img = sims["simulated_images"][sim - 1][0]
            transform = sims["simulated_transforms"][sim - 1]

        # -------------------------
        # Build batch input (N,H,W,1)
        # -------------------------
        batch_np = np.zeros((len(which_slices), H, W, 1), np.float32)
        for j, idx in enumerate(which_slices):
            batch_np[j, :, :, 0] = ants.slice_image(batch_img, 2, idx).numpy()

        batch_tf = tf.convert_to_tensor(batch_np)

        # -------------------------
        # VANILLA saliency
        # -------------------------
        with tf.GradientTape() as tape:
            tape.watch(batch_tf)
            batch_rgb = tf.repeat(batch_tf, 3, axis=-1)
            preds = model(batch_rgb)

        grads = tape.gradient(preds, batch_tf)[..., 0].numpy()
        vanilla_sal = grads

        # -------------------------
        # SMOOTHGRAD
        # -------------------------
        if smooth_samples > 0:
            sg_sum = None

            # build noise mask for slices
            mask_slice_tf = None
            if mask_noise:
                mask_np = (batch_img > 0).numpy().astype(np.float32)
                mask_batch_np = np.zeros_like(batch_np)
                for j, idx in enumerate(which_slices):
                    mask_batch_np[j, :, :, 0] = mask_np[:, :, idx]
                mask_slice_tf = tf.convert_to_tensor(mask_batch_np, dtype=batch_tf.dtype)

            for k in range(smooth_samples):

                noise = tf.random.normal(tf.shape(batch_tf), stddev=smooth_noise)

                if mask_noise and mask_slice_tf is not None:
                    noise = noise * mask_slice_tf

                noisy = tf.clip_by_value(batch_tf + noise, 0, 1)

                with tf.GradientTape() as sg_tape:
                    sg_tape.watch(noisy)
                    noisy_rgb = tf.repeat(noisy, 3, axis=-1)
                    sg_pred = model(noisy_rgb)

                sg_grad = sg_tape.gradient(sg_pred, noisy)[..., 0]
                sg_map = tf.square(sg_grad) if smooth_variant == "square" else tf.abs(sg_grad)

                sg_sum = sg_map if sg_sum is None else sg_sum + sg_map

            smooth_sal = (sg_sum / smooth_samples).numpy()

        else:
            smooth_sal = None

        # -------------------------
        # Build 3D volumes
        # -------------------------
        vanilla_img = t1_preprocessed * 0
        smooth_img = t1_preprocessed * 0 if smooth_sal is not None else None

        for j, idx in enumerate(which_slices):
            # optional per-slice normalization to keep gradients comparable
            if slice_wise_normalization:
                v = vanilla_sal[j] / (np.max(np.abs(vanilla_sal[j])) + 1e-8)
            else:
                v = vanilla_sal[j]
            vanilla_img[:, :, idx] = v

            if smooth_img is not None:
                if slice_wise_normalization:
                    s = smooth_sal[j] / (np.max(np.abs(smooth_sal[j])) + 1e-8)
                else:
                    s = smooth_sal[j]
                smooth_img[:, :, idx] = s

        # warp back to native
        if transform is not None:
            vanilla_img = ants.apply_ants_transform_to_image(
                transform=transform.invert(), image=vanilla_img, reference=t1_preprocessed
            )
            if smooth_img is not None:
                smooth_img = ants.apply_ants_transform_to_image(
                    transform=transform.invert(), image=smooth_img, reference=t1_preprocessed
                )

        # aggregate sims
        vanilla_accum = vanilla_img if vanilla_accum is None else vanilla_accum + vanilla_img
        if smooth_img is not None:
            smooth_accum = smooth_img if smooth_accum is None else smooth_accum + smooth_img

        # aggregate predictions
        pred_np = np.asarray(preds.numpy()).squeeze()
        all_predictions = (
            pred_np if all_predictions is None else
            all_predictions + (pred_np - all_predictions) / (sim + 1)
        )

    # -------------------------
    # Final prediction
    # -------------------------
    prediction_array = np.asarray(all_predictions)
    if use_mean_head:
        predicted_age = float(prediction_array.reshape(-1)[0])
    else:
        predicted_age = float(np.median(prediction_array))

    return {
        "predicted_age": predicted_age,
        "brain_age_per_slice": all_predictions,
        "vanilla_salience": vanilla_accum / total_sims,
        "smoothgrad_salience": None if smooth_accum is None else smooth_accum / total_sims,
        "t1_preprocessed": t1_preprocessed,
    }


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


def _strip_nifti_suffix(path):
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else value


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


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run explainable ANTsPyNet brain-age inference on a single T1 image."
        ),
        formatter_class=ArgumentFormatter,
        epilog="""examples:
  Analyze an already ANTsPyNet brain_age-preprocessed T1 image:
    python brain_age_salience.py sub-001_desc-preproc_T1w.nii.gz

  Run ANTsPyNet preprocessing first:
    python brain_age_salience.py sub-001_T1w.nii.gz --do_preprocessing

  Add SmoothGrad salience:
    python brain_age_salience.py sub-001_desc-preproc_T1w.nii.gz --n-smooth 25 --mask-noise

  Use the original ANTsPyNet median-of-slices fallback:
    python brain_age_salience.py sub-001_desc-preproc_T1w.nii.gz --median-head

For BIDS datasets, prefer the BIDS helper:
    python brain_age_salience_bids.py /path/to/bids sub-001 --session ses-01 --do_preprocessing
""",
    )

    parser.add_argument("t1_image", help="Path to a T1 image.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory for JSON, salience maps, and the preprocessed T1 image.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Filename prefix for outputs. Defaults to the input image basename.",
    )
    parser.add_argument(
        "--do_preprocessing",
        action="store_true",
        help=(
            "Run ANTsPyNet preprocessing before inference. Without this flag, "
            "the input is assumed to be a T1 image already preprocessed by "
            "ANTsPyNet brain_age/this brain-age salience pipeline; arbitrary "
            "preprocessing will not work."
        ),
    )
    parser.set_defaults(mean_head=True)
    parser.add_argument(
        "--mean-head",
        "--mean_head",
        dest="mean_head",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--median-head",
        "--median_head",
        dest="mean_head",
        action="store_false",
        help=(
            "Use the ANTsPyNet fallback: median of standard slice-wise predictions. "
            "By default, the differentiable mean head is used."
        ),
    )
    parser.add_argument(
        "--mask-noise",
        "--mask_noise",
        dest="mask_noise",
        action="store_true",
        help="Restrict SmoothGrad noise to nonzero brain voxels.",
    )
    parser.add_argument(
        "--no-slice-norm",
        "--no_slice_norm",
        dest="no_slice_norm",
        action="store_true",
        help="Disable per-slice salience normalization.",
    )
    parser.add_argument(
        "--n-affine",
        "--n_affine",
        dest="n_affine",
        type=int,
        default=0,
        help="Number of affine augmentation simulations to average.",
    )
    parser.add_argument(
        "--sd-affine",
        "--sd_affine",
        dest="sd_affine",
        type=float,
        default=0.01,
        help="Affine transform standard deviation for augmentation.",
    )
    parser.add_argument(
        "--n-smooth",
        "--n_smooth",
        dest="n_smooth",
        type=int,
        default=0,
        help="Number of SmoothGrad noise samples. Set >0 to write SmoothGrad output.",
    )
    parser.add_argument(
        "--sd-noise",
        "--sd_noise",
        dest="sd_noise",
        type=float,
        default=0.20,
        help="SmoothGrad Gaussian noise standard deviation on normalized image intensities.",
    )
    parser.add_argument(
        "--smoothgrad-mode",
        "--mode",
        dest="smoothgrad_mode",
        choices=["square", "abs"],
        default="square",
        help="Transform applied to SmoothGrad gradients before averaging.",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=25,
        help="First axial slice index used by the 2D brain-age model.",
    )
    parser.add_argument(
        "--slice-stop",
        type=int,
        default=145,
        help="Exclusive stop axial slice index used by the 2D brain-age model.",
    )
    parser.add_argument(
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

    t1_path = Path(args.t1_image).expanduser().resolve()
    if not t1_path.exists():
        parser.error(f"t1_image does not exist: {t1_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or _strip_nifti_suffix(t1_path)
    analysis_label = _analysis_label(args)

    try:
        import ants
    except ImportError as exc:
        parser.exit(
            1,
            "Missing Python dependency needed for inference:\n"
            f"  {exc}\n"
            "Create the project environment with:\n"
            "  conda env create -f environment.yml\n",
        )

    print(f"Using T1 image: {t1_path}")
    if args.do_preprocessing:
        print("ANTsPyNet preprocessing is enabled.")
    else:
        print(
            "Preprocessing is disabled; input is assumed to be ANTsPyNet "
            "brain_age/brain-age salience preprocessed."
        )

    t1_img = ants.image_read(str(t1_path))
    result = brain_age_with_affine_smoothgrad_unified(
        t1_img,
        do_preprocessing=args.do_preprocessing,
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

    json_path = output_dir / f"{output_prefix}_desc-{analysis_label}_brainage.json"
    vanilla_path = output_dir / f"{output_prefix}_desc-{analysis_label}Vanilla_salience.nii.gz"
    smoothgrad_path = output_dir / f"{output_prefix}_desc-{analysis_label}SmoothGrad_salience.nii.gz"
    preproc_path = output_dir / f"{output_prefix}_desc-preproc_T1w.nii.gz"

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
            "t1_image": str(t1_path),
        },
        "processing": {
            "preprocessing_run": args.do_preprocessing,
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
