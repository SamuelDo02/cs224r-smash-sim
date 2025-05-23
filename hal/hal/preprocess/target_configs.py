from hal.constants import INCLUDED_BUTTONS
from hal.constants import INCLUDED_BUTTONS_NO_SHOULDER
from hal.constants import ORIGINAL_BUTTONS
from hal.constants import ORIGINAL_BUTTONS_NO_SHOULDER
from hal.constants import SHOULDER_CLUSTER_CENTERS_V0
from hal.constants import SHOULDER_CLUSTER_CENTERS_V2
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0_1
from hal.constants import STICK_XY_CLUSTER_CENTERS_V1
from hal.constants import STICK_XY_CLUSTER_CENTERS_V2
from hal.constants import STICK_XY_CLUSTER_CENTERS_V3
from hal.preprocess.registry import TargetConfig
from hal.preprocess.registry import TargetConfigRegistry
from hal.preprocess.transformations import concatenate_main_stick
from hal.preprocess.transformations import encode_buttons_one_hot
from hal.preprocess.transformations import encode_buttons_one_hot_early_release
from hal.preprocess.transformations import encode_buttons_one_hot_no_shoulder
from hal.preprocess.transformations import encode_buttons_one_hot_no_shoulder_early_release
from hal.preprocess.transformations import encode_c_stick_one_hot_coarse
from hal.preprocess.transformations import encode_c_stick_one_hot_coarser
from hal.preprocess.transformations import encode_c_stick_one_hot_fine
from hal.preprocess.transformations import encode_main_stick_one_hot_coarse
from hal.preprocess.transformations import encode_main_stick_one_hot_fine
from hal.preprocess.transformations import encode_main_stick_one_hot_finer
from hal.preprocess.transformations import encode_original_buttons_multi_hot
from hal.preprocess.transformations import encode_original_buttons_one_hot_no_shoulder
from hal.preprocess.transformations import encode_shoulder_one_hot
from hal.preprocess.transformations import encode_shoulder_one_hot_coarse
from hal.preprocess.transformations import get_button_shoulder_l
from hal.preprocess.transformations import get_button_shoulder_r


def baseline_coarse() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_coarse,
            "c_stick": encode_c_stick_one_hot_coarse,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def coarse_shoulder() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_coarse,
            "c_stick": encode_c_stick_one_hot_coarse,
            "shoulder": encode_shoulder_one_hot_coarse,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "shoulder": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V0),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def baseline_fine() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_fine,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def gaussian_coarse() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": concatenate_main_stick,
            "c_stick": concatenate_main_stick,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
        reference_points=STICK_XY_CLUSTER_CENTERS_V0,
        sigma=0.08,
    )


def gaussian_fine() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": concatenate_main_stick,
            "c_stick": concatenate_main_stick,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V1),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
        reference_points=STICK_XY_CLUSTER_CENTERS_V1,
        sigma=0.05,
    )


def fine_main_analog_shoulder() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_buttons_one_hot_no_shoulder,
            "shoulder": encode_shoulder_one_hot_coarse,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
            "shoulder": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(INCLUDED_BUTTONS_NO_SHOULDER),),
            "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V0),),
        },
    )


def fine_main_analog_shoulder_early_release() -> TargetConfig:
    """Combined X/Y button, no L/R button, coarse analog shoulder"""
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_buttons_one_hot_no_shoulder_early_release,
            "shoulder": encode_shoulder_one_hot_coarse,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
            "shoulder": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(INCLUDED_BUTTONS_NO_SHOULDER),),
            "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V0),),
        },
    )


def baseline_finer() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_finer,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V3),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def fine_main_coarser_cstick() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(INCLUDED_BUTTONS),),
        },
    )


def fine_orig_buttons() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_original_buttons_multi_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(ORIGINAL_BUTTONS),),
        },
    )


def fine_orig_buttons_one_hot_shoulder_one_hot() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_original_buttons_one_hot_no_shoulder,
            "shoulder": encode_shoulder_one_hot_coarse,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
            "shoulder": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(ORIGINAL_BUTTONS_NO_SHOULDER),),
            "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V0),),
        },
    )


def separate_digital_shoulders_analog_shoulder_one_hot() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_buttons_one_hot_no_shoulder,
            "analog_shoulder": encode_shoulder_one_hot,
            "shoulder_l": get_button_shoulder_l,
            "shoulder_r": get_button_shoulder_r,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
            "analog_shoulder": 0,
            "shoulder_l": 0,
            "shoulder_r": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(INCLUDED_BUTTONS_NO_SHOULDER),),
            "analog_shoulder": (len(SHOULDER_CLUSTER_CENTERS_V2),),
            "shoulder_l": (1,),
            "shoulder_r": (1,),
        },
    )


def button_fine_main_coarser_cstick_medium_analog_shoulder() -> TargetConfig:
    """Include collapsed X/Y and L/R digital buttons, early release; plus analog shoulder."""
    return TargetConfig(
        transformation_by_target={
            "main_stick": encode_main_stick_one_hot_fine,
            "c_stick": encode_c_stick_one_hot_coarser,
            "buttons": encode_buttons_one_hot_early_release,
            "shoulder": encode_shoulder_one_hot,
        },
        frame_offsets_by_target={
            "main_stick": 0,
            "c_stick": 0,
            "buttons": 0,
            "shoulder": 0,
        },
        target_shapes_by_head={
            "main_stick": (len(STICK_XY_CLUSTER_CENTERS_V2),),
            "c_stick": (len(STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (len(INCLUDED_BUTTONS),),
            "shoulder": (len(SHOULDER_CLUSTER_CENTERS_V2),),
        },
    )


TargetConfigRegistry.register("baseline_coarse", baseline_coarse())
TargetConfigRegistry.register("coarse_shoulder", coarse_shoulder())
TargetConfigRegistry.register("baseline_fine", baseline_fine())
TargetConfigRegistry.register("gaussian_coarse", gaussian_coarse())
TargetConfigRegistry.register("gaussian_fine", gaussian_fine())
TargetConfigRegistry.register("fine_main_analog_shoulder", fine_main_analog_shoulder())
TargetConfigRegistry.register("fine_main_analog_shoulder_early_release", fine_main_analog_shoulder_early_release())
TargetConfigRegistry.register("baseline_finer", baseline_finer())
TargetConfigRegistry.register("fine_main_coarser_cstick", fine_main_coarser_cstick())
TargetConfigRegistry.register("fine_orig_buttons", fine_orig_buttons())
TargetConfigRegistry.register(
    "fine_orig_buttons_one_hot_shoulder_one_hot", fine_orig_buttons_one_hot_shoulder_one_hot()
)
TargetConfigRegistry.register(
    "separate_digital_shoulders_analog_shoulder_one_hot", separate_digital_shoulders_analog_shoulder_one_hot()
)
TargetConfigRegistry.register(
    "button_fine_main_coarser_cstick_medium_analog_shoulder",
    button_fine_main_coarser_cstick_medium_analog_shoulder(),
)
