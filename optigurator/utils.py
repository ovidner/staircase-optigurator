import functools
import json
import os
from hashlib import blake2b
from types import SimpleNamespace

import attr
import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()

ANGLE_VARS_SWEEP_DIRECTION = np.array([1, -1])
DATA_DIR = "data"


def interweave(a, b):
    """
    >>> import numpy as np
    >>> interweave(np.array([1., 3., 5., 7.]), np.array([2., 4., 6., 8.]))
    array([1., 2., 3., 4., 5., 6., 7., 8.])
    >>> interweave(np.array([1., 3., 5., 7.]), np.array([2., 4., 6.]))
    array([1., 2., 3., 4., 5., 6., 7.])
    """
    ret = np.empty((a.size + b.size,), dtype=a.dtype)
    ret[0::2] = a
    ret[1::2] = b

    return ret


def cumulative_spread(array, x):
    """
    >>> import numpy as np
    >>> a = np.array([1., 2., 3., 4.])
    >>> cumulative_spread(a, 0.)
    array([0., 0., 0., 0.])
    >>> cumulative_spread(a, 5.)
    array([1., 2., 2., 0.])
    >>> cumulative_spread(a, 6.)
    array([1., 2., 3., 0.])
    >>> cumulative_spread(a, 12.)
    array([1., 2., 3., 4.])
    """
    # This is probably inefficient.
    cumulative_effect = np.cumsum(array) - array
    b = x - cumulative_effect

    return np.fmin(array, np.fmax(0, b))


def height(sweep_array, pitch_array, angle):
    return np.sum(cumulative_spread(sweep_array, angle) * pitch_array)


def forward_angle_diff(start, end, sweep_direction):
    """
    When sweeping in `sweep_direction` from `start`, how many degrees is it to `end`?

    `sweep_direction` is 1 for counter-clockwise, -1 for clockwise
    """
    return (sweep_direction * (end - start)) % 360


def sweep_angle(start, delta, sweep_direction):
    return (start + delta * sweep_direction) % 360


def start_end_angle(
    min_start,
    max_start,
    min_end,
    max_end,
    min_clearance,
    scale_factor,
    placement_factor,
    sweep_direction,
):
    assert 0 <= scale_factor.all() <= 1
    assert 0 <= placement_factor.all() <= 1

    max_clearance = forward_angle_diff(min_start, max_end, sweep_direction)
    max_clearance[max_clearance == 0.0] = 360.0
    clearance_diff = max_clearance - min_clearance
    assert clearance_diff.all() >= 0

    clearance = min_clearance + clearance_diff * scale_factor
    placement_diff = max_clearance - clearance

    start = sweep_angle(min_start, placement_diff * placement_factor, sweep_direction)
    end = sweep_angle(start, clearance, sweep_direction)

    return start, end, clearance


def with_constants(cls):
    """
    Class decorator for OpenMDAO components that declares a "constants" option
    and also adds the passed in object as an easily accessible property.
    """

    @functools.wraps(cls, updated=tuple())
    class decorated_cls(cls):
        def initialize(self):
            super().initialize()
            self.options.declare(
                "constants",
                desc="Problem-specific constants object",
                types=ProblemConstants,
            )

        @property
        def constants(self):
            return self.options["constants"]

    return decorated_cls


@attr.dataclass(frozen=True)
class DesignVariable:
    """
    Simple data container for specifying the bounds of a design variable.
    """

    lower: np.ndarray
    upper: np.ndarray

    @property
    def mean(self):
        return np.mean([self.lower, self.upper], axis=0)


@attr.dataclass(frozen=True)
class ProblemConstants:
    """
    Convenience data container class to pass around problem-specific constants
    in one single object.
    """

    name: str
    floor_height: DesignVariable
    radii: list
    segment_start_angle: DesignVariable
    segment_end_angle: DesignVariable
    floor_angle_clearance_min: np.ndarray

    # Attributes with defaults must come last
    step_radius_offset: float = 0.250
    # The free-height is theoretical, so we add some margin
    free_height_lower: float = 2.0 + 0.25
    orientation: DesignVariable = DesignVariable(
        lower=np.array([0], dtype=int), upper=np.array([1], dtype=int)  # CCW  # CW
    )
    step_depth: DesignVariable = DesignVariable(
        lower=np.array([0.130]), upper=np.array([0.380])
    )
    step_height: DesignVariable = DesignVariable(
        lower=np.array([0.150]), upper=np.array([0.240])
    )

    @property
    def id(self):
        return blake2b(str(self.__dict__).encode("utf-8"), digest_size=4).hexdigest()

    @property
    def vector_shape(self):
        floor = self.floor_height.lower.shape[0]
        return SimpleNamespace(common=(1,), segment=(floor - 1,), floor=(floor,))

    @property
    def floor_angle_clearance_scale_factor(self):
        upper = np.ones(self.vector_shape.floor, dtype=float)
        upper[0] = 0.0
        upper[-1] = 0.0

        return DesignVariable(
            lower=np.zeros(self.vector_shape.floor, dtype=float), upper=upper
        )

    @property
    def floor_angle_clearance_placement_factor(self):
        return DesignVariable(
            lower=np.zeros(self.vector_shape.floor, dtype=float),
            upper=np.ones(self.vector_shape.floor, dtype=float),
        )

    @property
    def floor_start_angle(self):
        # Padded with accomodation for bottom floor clearance at beginning
        return DesignVariable(
            lower=np.insert(
                self.segment_end_angle.lower,
                0,
                sweep_angle(
                    self.segment_start_angle.lower[:, 0],
                    self.floor_angle_clearance_min[0],
                    -ANGLE_VARS_SWEEP_DIRECTION,
                ),
                axis=1,
            ),
            upper=np.insert(
                self.segment_end_angle.upper,
                0,
                sweep_angle(
                    self.segment_start_angle.upper[:, 0],
                    self.floor_angle_clearance_min[0],
                    -ANGLE_VARS_SWEEP_DIRECTION,
                ),
                axis=1,
            ),
        )

    @property
    def floor_end_angle(self):
        # Padded with accomodation for top floor clearance at end
        return DesignVariable(
            lower=np.append(
                self.segment_start_angle.lower,
                sweep_angle(
                    self.segment_end_angle.lower[:, -1],
                    self.floor_angle_clearance_min[-1],
                    ANGLE_VARS_SWEEP_DIRECTION,
                ).reshape((2, 1)),
                axis=1,
            ),
            upper=np.append(
                self.segment_start_angle.upper,
                sweep_angle(
                    self.segment_end_angle.upper[:, -1],
                    self.floor_angle_clearance_min[-1],
                    ANGLE_VARS_SWEEP_DIRECTION,
                ).reshape((2, 1)),
                axis=1,
            ),
        )

    @property
    def radius_index(self):
        return DesignVariable(
            lower=np.array([0], dtype=int), upper=np.array([len(self.radii)], dtype=int)
        )

    @property
    def extra_sweep_tendency(self):
        return DesignVariable(
            lower=np.zeros(self.vector_shape.segment, dtype=float),
            upper=np.ones(self.vector_shape.segment, dtype=float),
        )

    @property
    def extra_steps_tendency(self):
        return DesignVariable(
            lower=np.zeros(self.vector_shape.segment, dtype=float),
            upper=np.ones(self.vector_shape.segment, dtype=float),
        )

    def radius_availability(self, radius):
        # Let's just imagine that this method would represent a frozen state
        # from an API call made at initialization time
        with open(os.path.abspath(os.path.join(DATA_DIR, "availability.json"))) as fp:
            data = json.load(fp)

        # Woohoo, that's one nasty dict lookup!
        return data.get(str(int(radius * 1000)), -np.inf)

    def radius_price(self, radius):
        # Let's just imagine that this method would represent a frozen state
        # from an API call made at initialization time
        with open(os.path.abspath(os.path.join(DATA_DIR, "price.json"))) as fp:
            data = json.load(fp)

        # Woohoo, that's one nasty dict lookup!
        return data.get(str(int(radius * 1000)), np.inf)


def get_onshape_value_map(value_map, key):
    return next(
        filter(
            lambda x: x["message"]["key"]["message"]["value"] == key,
            value_map["message"]["value"],
        )
    )


def get_onshape_value_map_raw_value(value_map, key):
    return get_onshape_value_map(value_map, key)["message"]["value"]["message"]["value"]


def get_onshape_value_map_value(value_map, key):
    payload = get_onshape_value_map(value_map, key)
    return ureg.Quantity(
        payload["message"]["value"]["message"]["value"],
        payload["message"]["value"]["message"]["unitToPower"][0]["key"].lower(),
    )


def yield_floor_values(top_value_map):
    for f in get_onshape_value_map(top_value_map, "floors")["message"]["value"][
        "message"
    ]["value"]:
        yield (
            get_onshape_value_map_value(f, "height").to(ureg.meter).magnitude,
            get_onshape_value_map_value(f, "angleClearanceLower")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "downwardAngleCcwLower")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "downwardAngleCcwUpper")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "downwardAngleCwLower")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "downwardAngleCwUpper")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "upwardAngleCcwLower")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "upwardAngleCcwUpper")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "upwardAngleCwLower")
            .to(ureg.degree)
            .magnitude,
            get_onshape_value_map_value(f, "upwardAngleCwUpper")
            .to(ureg.degree)
            .magnitude,
        )


def problem_constants_from_onshape_feature(top_value_map):

    floor_parameters = np.array(
        list(yield_floor_values(top_value_map)),
        dtype=np.dtype(
            [
                ("height", float),
                ("angle_clearance_lower", float),
                ("downward_angle_ccw_lower", float),
                ("downward_angle_ccw_upper", float),
                ("downward_angle_cw_lower", float),
                ("downward_angle_cw_upper", float),
                ("upward_angle_ccw_lower", float),
                ("upward_angle_ccw_upper", float),
                ("upward_angle_cw_lower", float),
                ("upward_angle_cw_upper", float),
            ]
        ),
    )
    all_radii = [x * 100 for x in range(6, 16)]  # 600 to 1500 in 100 increments
    min_radius = (
        get_onshape_value_map_value(top_value_map, "radiusMin").to(ureg.meter).magnitude
    )
    max_radius = (
        get_onshape_value_map_value(top_value_map, "radiusMax").to(ureg.meter).magnitude
    )

    return ProblemConstants(
        name="foo",
        floor_height=DesignVariable(
            lower=floor_parameters["height"], upper=floor_parameters["height"]
        ),
        floor_angle_clearance_min=floor_parameters["angle_clearance_lower"],
        segment_start_angle=DesignVariable(
            lower=np.array(
                [
                    floor_parameters["upward_angle_ccw_lower"][:-1],
                    floor_parameters["upward_angle_cw_lower"][:-1],
                ]
            ),
            upper=np.array(
                [
                    floor_parameters["upward_angle_ccw_upper"][:-1],
                    floor_parameters["upward_angle_cw_upper"][:-1],
                ]
            ),
        ),
        segment_end_angle=DesignVariable(
            lower=np.array(
                [
                    floor_parameters["downward_angle_ccw_lower"][1:],
                    floor_parameters["downward_angle_cw_lower"][1:],
                ]
            ),
            upper=np.array(
                [
                    floor_parameters["downward_angle_ccw_upper"][1:],
                    floor_parameters["downward_angle_cw_upper"][1:],
                ]
            ),
        ),
        radii=[r / 1000 for r in all_radii if int(min_radius * 1000) <= r <= int(max_radius * 1000)],
        orientation=DesignVariable(
            lower=np.array(
                [
                    0
                    if get_onshape_value_map_raw_value(top_value_map, "turnCcwAllowed")
                    else 1
                ],
                dtype=int,
            ),  # CCW
            upper=np.array(
                [
                    1
                    if get_onshape_value_map_raw_value(top_value_map, "turnCwAllowed")
                    else 0
                ],
                dtype=int,
            ),  # CW
        ),
    )


def format_degree(num):
    return "{:.1f}\xb0".format(num)


def format_millimeter(num):
    return "{:.1f} mm".format(num * 1000)


def format_percent(num):
    return "{:.1f}%".format(num * 100)


def recording_filename(id_):
    return os.path.abspath(os.path.join(DATA_DIR, "recordings", f"{id_}.sqlite"))
