from itertools import chain

import numpy as np
from openmdao.api import (
    ExplicitComponent,
    Group,
    IndepVarComp,
    Problem,
    SimpleGADriver,
    SqliteRecorder,
)

from optigurator.utils import (
    cumulative_spread,
    forward_angle_diff,
    height,
    interweave,
    recording_filename,
    start_end_angle,
    with_constants,
)


@with_constants
class DerivedDesignVariables(ExplicitComponent):
    def setup(self):
        self.add_input(
            "radius_index", shape=self.constants.vector_shape.common, units=None
        )
        self.add_input(
            "floor_height", shape=self.constants.vector_shape.floor, units="m"
        )

        self.add_output(
            "segment_height", shape=self.constants.vector_shape.segment, units="m"
        )
        self.add_output("radius", shape=self.constants.vector_shape.common, units="m")
        self.add_output(
            "step_radius", shape=self.constants.vector_shape.common, units="m"
        )

    def compute(self, inputs, outputs):
        outputs["radius"] = self.constants.radii[int(inputs["radius_index"].round())]
        outputs["step_radius"] = outputs["radius"] - self.constants.step_radius_offset
        outputs["segment_height"] = np.diff(inputs["floor_height"])


@with_constants
class DesignVariables(Group):
    def setup(self):
        raw = self.add_subsystem("raw", IndepVarComp(), promotes_outputs=["*"])
        raw.add_output("floor_height", val=self.constants.floor_height.lower, units="m")
        raw.add_output(
            "_orientation_index", val=self.constants.orientation.lower, units=None
        )
        raw.add_output(
            "_radius_index", val=self.constants.radius_index.lower, units=None
        )
        raw.add_output(
            "floor_angle_clearance_scale_factor",
            val=self.constants.floor_angle_clearance_scale_factor.mean,
            units=None,
        )
        raw.add_output(
            "floor_angle_clearance_placement_factor",
            val=self.constants.floor_angle_clearance_placement_factor.mean,
            units=None,
        )
        raw.add_output(
            "extra_sweep_tendency",
            val=self.constants.extra_sweep_tendency.mean,
            units=None,
        )
        raw.add_output(
            "extra_steps_tendency",
            val=self.constants.extra_steps_tendency.mean,
            units=None,
        )

        derived = self.add_subsystem(
            "derived",
            DerivedDesignVariables(constants=self.constants),
            promotes_outputs=["*"],
        )

        self.connect("floor_height", ["derived.floor_height"])
        self.connect("_radius_index", ["derived.radius_index"])


@with_constants
class Angles(ExplicitComponent):
    def setup(self):
        self.add_input(
            "orientation_index", shape=self.constants.vector_shape.common, units=None
        )
        self.add_input(
            "floor_angle_clearance_scale_factor",
            shape=self.constants.vector_shape.floor,
            units=None,
        )
        self.add_input(
            "floor_angle_clearance_placement_factor",
            shape=self.constants.vector_shape.floor,
            units=None,
        )

        self.add_output(
            "floor_start_angle", shape=self.constants.vector_shape.floor, units="deg"
        )
        self.add_output(
            "floor_end_angle", shape=self.constants.vector_shape.floor, units="deg"
        )
        self.add_output(
            "floor_angle_clearance",
            shape=self.constants.vector_shape.floor,
            units="deg",
        )
        self.add_output(
            "segment_start_angle",
            shape=self.constants.vector_shape.segment,
            units="deg",
        )
        self.add_output(
            "segment_end_angle", shape=self.constants.vector_shape.segment, units="deg"
        )
        self.add_output(
            "segment_net_sweep", shape=self.constants.vector_shape.segment, units="deg"
        )

    def compute(self, inputs, outputs):
        orientation_index = int(inputs["orientation_index"].round())

        sweep_direction = [1, -1][orientation_index]

        floor_start_angle, floor_end_angle, floor_angle_clearance = start_end_angle(
            min_start=self.constants.floor_start_angle.lower[orientation_index],
            max_start=self.constants.floor_start_angle.upper[orientation_index],
            min_end=self.constants.floor_end_angle.lower[orientation_index],
            max_end=self.constants.floor_end_angle.upper[orientation_index],
            min_clearance=self.constants.floor_angle_clearance_min,
            scale_factor=inputs["floor_angle_clearance_scale_factor"],
            placement_factor=inputs["floor_angle_clearance_placement_factor"],
            sweep_direction=sweep_direction,
        )

        outputs["floor_start_angle"] = floor_start_angle
        outputs["floor_end_angle"] = floor_end_angle
        outputs["floor_angle_clearance"] = floor_angle_clearance
        outputs["segment_start_angle"] = segment_start_angle = floor_end_angle[:-1]
        outputs["segment_end_angle"] = segment_end_angle = floor_start_angle[1:]
        outputs["segment_net_sweep"] = forward_angle_diff(
            segment_start_angle, segment_end_angle, sweep_direction
        )


@with_constants
class SweepSteps(ExplicitComponent):
    def setup(self):
        self.add_input(
            "extra_sweep_tendency",
            shape=self.constants.vector_shape.segment,
            units=None,
        )
        self.add_input(
            "extra_steps_tendency",
            shape=self.constants.vector_shape.segment,
            units=None,
        )
        self.add_input(
            "segment_height", shape=self.constants.vector_shape.segment, units="m"
        )
        self.add_input(
            "segment_net_sweep", shape=self.constants.vector_shape.segment, units="deg"
        )
        self.add_input(
            "step_radius", shape=self.constants.vector_shape.common, units="m"
        )

        self.add_output(
            "step_depth", shape=self.constants.vector_shape.segment, units="m"
        )
        self.add_output(
            "step_height", shape=self.constants.vector_shape.segment, units="m"
        )
        self.add_output(
            "segment_sweep", shape=self.constants.vector_shape.segment, units="deg"
        )
        self.add_output(
            "segment_extra_sweep",
            shape=self.constants.vector_shape.segment,
            units="deg",
        )
        self.add_output(
            "segment_pitch", shape=self.constants.vector_shape.segment, units="m/deg"
        )
        self.add_output(
            "step_angle", shape=self.constants.vector_shape.segment, units="deg"
        )
        self.add_output(
            "step_count", shape=self.constants.vector_shape.segment, units=None
        )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        one = np.ones(self.constants.vector_shape.segment)
        min_step_depth = one * self.constants.step_depth.lower
        max_step_depth = one * self.constants.step_depth.upper
        min_step_height = one * self.constants.step_height.lower
        max_step_height = one * self.constants.step_height.upper

        min_pitch = (
            np.deg2rad(1) * min_step_height * inputs["step_radius"] / max_step_depth
        )
        max_pitch = (
            np.deg2rad(1) * max_step_height * inputs["step_radius"] / min_step_depth
        )
        ideal_pitch = max_pitch - inputs["extra_sweep_tendency"] * (
            max_pitch - min_pitch
        )
        ideal_extra_sweep = (
            inputs["segment_height"] / ideal_pitch - inputs["segment_net_sweep"]
        )
        # Rounds the extra sweep to the nearest 360-multiple greater than or equal to 0
        segment_extra_sweep = np.fmax(((ideal_extra_sweep + 180) // 360) * 360, 0)
        segment_sweep = inputs["segment_net_sweep"] + segment_extra_sweep
        segment_pitch = inputs["segment_height"] / segment_sweep

        min_step_count = np.fmin(
            np.deg2rad(segment_sweep) * inputs["step_radius"] / max_step_depth,
            inputs["segment_height"] / max_step_height,
        )
        max_step_count = np.fmax(
            np.deg2rad(segment_sweep) * inputs["step_radius"] / min_step_depth,
            inputs["segment_height"] / min_step_height,
        )
        # gross_step_count is the number of steps one would have to take when
        # treading the stair while net_step_count is the actual number of step
        # parts to instantiate
        gross_step_count = np.fmax(
            np.round(
                min_step_count
                + inputs["extra_steps_tendency"] * (max_step_count - min_step_count)
            ),
            one,
        )
        net_step_count = gross_step_count - 1

        # Note that the step heights and angles are calculated with different
        # step counts.
        step_angle = segment_sweep / net_step_count
        step_depth = np.deg2rad(step_angle) * inputs["step_radius"]
        step_height = inputs["segment_height"] / gross_step_count

        outputs["step_depth"] = step_depth
        outputs["step_height"] = step_height
        outputs["segment_extra_sweep"] = segment_extra_sweep
        outputs["segment_sweep"] = segment_sweep
        outputs["segment_pitch"] = segment_pitch
        outputs["step_angle"] = step_angle
        outputs["step_count"] = gross_step_count


@with_constants
class Usability(ExplicitComponent):
    def setup(self):
        self.add_input(
            "step_depth", shape=self.constants.vector_shape.segment, units="m"
        )
        self.add_input(
            "step_height", shape=self.constants.vector_shape.segment, units="m"
        )
        self.add_input(
            "floor_angle_clearance",
            shape=self.constants.vector_shape.floor,
            units="deg",
        )
        self.add_input(
            "segment_sweep", shape=self.constants.vector_shape.segment, units="deg"
        )
        self.add_input(
            "segment_pitch", shape=self.constants.vector_shape.segment, units="m/deg"
        )
        self.add_output(
            "min_free_height", shape=self.constants.vector_shape.common, units="m"
        )
        self.add_output(
            "step_comfort_rule_1_value",
            shape=self.constants.vector_shape.segment,
            units="m",
        )
        self.add_output(
            "step_comfort_rule_2_value",
            shape=self.constants.vector_shape.segment,
            units="m",
        )
        self.add_output(
            "max_step_comfort_rule_deviation",
            shape=self.constants.vector_shape.common,
            units=None,
        )
        self.add_output("min_max_step_height", shape=(2,), units="m")
        self.add_output("min_max_step_depth", shape=(2,), units="m")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # https://www.finehomebuilding.com/2016/08/04/2-rules-comfortable-stairs
        step_comfort_rule_1_value = inputs["step_height"] + inputs["step_depth"]
        step_comfort_rule_2_value = 2 * inputs["step_height"] + inputs["step_depth"]
        step_comfort_rule_1_deviation = (
            np.fmax(np.abs(step_comfort_rule_1_value - 0.457) - 0.0254, 0) / 0.457
        )
        step_comfort_rule_2_deviation = (
            np.fmax(np.abs(step_comfort_rule_2_value - 0.635) - 0.0254, 0) / 0.635
        )

        outputs["step_comfort_rule_1_value"] = step_comfort_rule_1_value
        outputs["step_comfort_rule_2_value"] = step_comfort_rule_2_value
        outputs["max_step_comfort_rule_deviation"] = np.max(
            np.fmax(step_comfort_rule_1_deviation, step_comfort_rule_2_deviation)
        )

        outputs["min_max_step_height"] = np.array(
            [np.min(inputs["step_height"]), np.max(inputs["step_height"])]
        )
        outputs["min_max_step_depth"] = np.array(
            [np.min(inputs["step_depth"]), np.max(inputs["step_depth"])]
        )

        outputs["min_free_height"] = self.compute_free_height(
            inputs["floor_angle_clearance"],
            inputs["segment_sweep"],
            inputs["segment_pitch"],
        )

    def compute_free_height(self, floor_angle_clearance, segment_sweep, segment_pitch):
        floor_segment_sweep = interweave(floor_angle_clearance, segment_sweep)
        floor_segment_pitch = interweave(
            # Floors have zero pitch
            np.zeros(floor_angle_clearance.size, dtype=float),
            segment_pitch,
        )

        # All sweep angles where the pitch changes.
        pitch_changing_sweeps = np.cumsum(floor_segment_sweep)
        # Beyond this sweep angle, the free height is infinite from our
        # perspective.
        max_angle = np.fmax(pitch_changing_sweeps[-1] - 360.0, 0.0)

        # Uses all pitch-changing sweep angles (and the points right under
        # them) as samples where we evaluate the free height, as these ought to
        # be the only places where we can find local minima and maxima.
        angle_samples = np.array(
            sorted(
                set(
                    x
                    for x in chain(pitch_changing_sweeps, pitch_changing_sweeps - 360.0)
                    if 0.0 <= x <= max_angle
                )
            )
        )

        # Old brute-force sampling method.
        # angle_samples = np.linspace(0.0, max_angle, int(max_angle / 2) + 1)

        free_heights = [
            (
                height(floor_segment_sweep, floor_segment_pitch, angle + 360.0)
                - height(floor_segment_sweep, floor_segment_pitch, angle)
            )
            for angle in angle_samples
        ]

        return np.min(free_heights)


@with_constants
class PriceAvailability(ExplicitComponent):
    def setup(self):
        self.add_input(
            "step_count", shape=self.constants.vector_shape.segment, units=None
        )
        self.add_input("radius", shape=self.constants.vector_shape.common, units="m")

        self.add_output(
            "total_price", shape=self.constants.vector_shape.common, units=None
        )
        self.add_output(
            "total_delivery_time", shape=self.constants.vector_shape.common, units=None
        )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        total_step_count = np.sum(inputs["step_count"])

        if (
            self.constants.radius_availability(float(inputs["radius"]))
            >= total_step_count
        ):
            outputs["total_delivery_time"] = 7
        else:
            outputs["total_delivery_time"] = 30

        outputs["total_price"] = (
            self.constants.radius_price(float(inputs["radius"])) * total_step_count
        )


@with_constants
class SpiralStaircase(Group):
    def setup(self):
        design_vars = self.add_subsystem(
            "design_vars", DesignVariables(constants=self.constants)
        )
        angles = self.add_subsystem("angles", Angles(constants=self.constants))
        sweep_steps = self.add_subsystem(
            "sweep_steps", SweepSteps(constants=self.constants)
        )
        usability = self.add_subsystem("usability", Usability(constants=self.constants))
        price_availability = self.add_subsystem(
            "price_availability", PriceAvailability(constants=self.constants)
        )

        self.connect("design_vars.segment_height", ["sweep_steps.segment_height"])
        self.connect("design_vars._orientation_index", ["angles.orientation_index"])
        self.connect("design_vars.step_radius", ["sweep_steps.step_radius"])
        self.connect(
            "design_vars.floor_angle_clearance_scale_factor",
            ["angles.floor_angle_clearance_scale_factor"],
        )
        self.connect(
            "design_vars.floor_angle_clearance_placement_factor",
            ["angles.floor_angle_clearance_placement_factor"],
        )
        self.connect(
            "design_vars.extra_sweep_tendency", ["sweep_steps.extra_sweep_tendency"]
        )
        self.connect(
            "design_vars.extra_steps_tendency", ["sweep_steps.extra_steps_tendency"]
        )
        self.connect("design_vars.radius", ["price_availability.radius"])

        self.connect("angles.segment_net_sweep", ["sweep_steps.segment_net_sweep"])
        self.connect(
            "angles.floor_angle_clearance", ["usability.floor_angle_clearance"]
        )

        self.connect("sweep_steps.segment_pitch", ["usability.segment_pitch"])
        self.connect("sweep_steps.segment_sweep", ["usability.segment_sweep"])
        self.connect("sweep_steps.step_height", ["usability.step_height"])
        self.connect("sweep_steps.step_depth", ["usability.step_depth"])
        self.connect("sweep_steps.step_count", ["price_availability.step_count"])

        self.add_design_var(
            "design_vars._orientation_index",
            lower=self.constants.orientation.lower,
            upper=self.constants.orientation.upper,
        )
        self.add_design_var(
            "design_vars.floor_height",
            lower=self.constants.floor_height.lower,
            upper=self.constants.floor_height.upper,
        )
        self.add_design_var(
            "design_vars.floor_angle_clearance_scale_factor",
            lower=self.constants.floor_angle_clearance_scale_factor.lower,
            upper=self.constants.floor_angle_clearance_scale_factor.upper,
        )
        self.add_design_var(
            "design_vars.floor_angle_clearance_placement_factor",
            lower=self.constants.floor_angle_clearance_placement_factor.lower,
            upper=self.constants.floor_angle_clearance_placement_factor.upper,
        )
        self.add_design_var(
            "design_vars.extra_sweep_tendency",
            lower=self.constants.extra_sweep_tendency.lower,
            upper=self.constants.extra_sweep_tendency.upper,
        )
        self.add_design_var(
            "design_vars.extra_steps_tendency",
            lower=self.constants.extra_steps_tendency.lower,
            upper=self.constants.extra_steps_tendency.upper,
        )

        self.add_constraint(
            "usability.min_free_height", lower=self.constants.free_height_lower
        )
        self.add_constraint(
            "usability.min_max_step_depth",
            indices=[0],
            lower=self.constants.step_depth.lower,
        )
        self.add_constraint(
            "usability.min_max_step_height",
            indices=[1],
            upper=self.constants.step_height.upper,
        )

        self.add_objective("price_availability.total_price")
        self.add_objective("price_availability.total_delivery_time")
        self.add_objective("usability.max_step_comfort_rule_deviation")


def run_optimization(problem_constants):
    prob = Problem()
    prob.model = SpiralStaircase(constants=problem_constants)

    prob.model.add_recorder(SqliteRecorder(recording_filename(problem_constants.id)))
    prob.model.recording_options["record_outputs"] = True

    prob.driver = SimpleGADriver(
        penalty_exponent=2.0,
        penalty_parameter=1000.0,
        max_gen=75,
        pop_size=0,  # Automatic
        bits={
            "design_vars.floor_height": 2,
            "design_vars.floor_angle_clearance_scale_factor": 3,
            "design_vars.floor_angle_clearance_placement_factor": 3,
            "design_vars._orientation_index": 1,
            "design_vars.extra_sweep_tendency": 2,
            "design_vars.extra_steps_tendency": 3,
        },
        multi_obj_weights={
            "price_availability.total_price": 1 / 200_000,
            "price_availability.total_delivery_time": 1 / 30,
            "usability.max_step_comfort_rule_deviation": 25 * 1,
        },
    )
    prob.setup()
    prob.run_driver()

    return prob
