# Copyright 2022 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from unittest.mock import Mock, patch

import pytest
import sympy

import cirq


def assert_optimizes(optimized: cirq.AbstractCircuit, expected: cirq.AbstractCircuit):
    # Ignore differences that would be caught by follow-up optimizations.
    followup_transformers: List[cirq.TRANSFORMER] = [
        cirq.drop_negligible_operations,
        cirq.drop_empty_moments,
    ]
    for transform in followup_transformers:
        optimized = transform(optimized)
        expected = transform(expected)

    cirq.testing.assert_same_circuits(optimized, expected)


def test_merge_single_qubit_gates_to_phased_x_and_z():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(a),
        cirq.Y(b) ** 0.5,
        cirq.CZ(a, b),
        cirq.H(a),
        cirq.Z(a),
        cirq.measure(b, key="m"),
        cirq.H(a).with_classical_controls("m"),
    )
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates_to_phased_x_and_z(c),
        expected=cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=1)(a),
            cirq.PhasedXPowGate(phase_exponent=0.5)(b) ** 0.5,
            cirq.CZ(a, b),
            (cirq.PhasedXPowGate(phase_exponent=-0.5)(a)) ** 0.5,
            cirq.measure(b, key="m"),
            cirq.H(a).with_classical_controls("m"),
        ),
    )


def test_merge_single_qubit_gates_to_phased_x_and_z_deep():
    a = cirq.NamedQubit("a")
    c_nested = cirq.FrozenCircuit(cirq.H(a), cirq.Z(a), cirq.H(a).with_tags("ignore"))
    c_nested_merged = cirq.FrozenCircuit(
        cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5).on(a), cirq.H(a).with_tags("ignore")
    )
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tags"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6),
    )
    c_expected = cirq.Circuit(
        c_nested_merged,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags("preserve_tags"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(6),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_new = cirq.merge_single_qubit_gates_to_phased_x_and_z(c_orig, context=context)
    cirq.testing.assert_same_circuits(c_new, c_expected)


def _phxz(a: float | sympy.Symbol, x: float | sympy.Symbol, z: float | sympy.Symbol):
    return cirq.PhasedXZGate(axis_phase_exponent=a, x_exponent=x, z_exponent=z)


def test_merge_single_qubit_gates_to_phxz():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(a),
        cirq.Y(b) ** 0.5,
        cirq.CZ(a, b),
        cirq.H(a),
        cirq.Z(a),
        cirq.measure(b, key="m"),
        cirq.H(a).with_classical_controls("m"),
    )
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates_to_phxz(c),
        expected=cirq.Circuit(
            _phxz(-1, 1, 0).on(a),
            _phxz(0.5, 0.5, 0).on(b),
            cirq.CZ(a, b),
            _phxz(-0.5, 0.5, 0).on(a),
            cirq.measure(b, key="m"),
            cirq.H(a).with_classical_controls("m"),
        ),
    )


def test_merge_single_qubit_gates_to_phxz_deep():
    a = cirq.NamedQubit("a")
    c_nested = cirq.FrozenCircuit(cirq.H(a), cirq.Z(a), cirq.H(a).with_tags("ignore"))
    c_nested_merged = cirq.FrozenCircuit(_phxz(-0.5, 0.5, 0).on(a), cirq.H(a).with_tags("ignore"))
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tags"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6),
    )
    c_expected = cirq.Circuit(
        c_nested_merged,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags("preserve_tags"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(6),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_new = cirq.merge_single_qubit_gates_to_phxz(c_orig, context=context)
    cirq.testing.assert_same_circuits(c_new, c_expected)


def test_merge_single_qubit_moments_to_phxz():
    q = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.X.on_each(*q[:2])),
        cirq.Moment(cirq.T.on_each(*q[1:])),
        cirq.Moment(cirq.Y.on_each(*q[:2])),
        cirq.Moment(cirq.CZ(*q[:2]), cirq.Y(q[2])),
        cirq.Moment(cirq.X.on_each(*q[:2])),
        cirq.Moment(cirq.T.on_each(*q[1:])),
        cirq.Moment(cirq.Y.on_each(*q[:2])),
        cirq.Moment(cirq.Y(q[0]).with_tags("nocompile"), cirq.Z.on_each(*q[1:])),
        cirq.Moment(cirq.X.on_each(q[0])),
        cirq.Moment(cirq.measure(q[0], key="a")),
        cirq.Moment(cirq.X(q[1]).with_classical_controls("a")),
        cirq.Moment(cirq.X.on_each(q[1])),
    )
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───X───────Y───@───X───────Y───Y[nocompile]───X───M───────────
                  │                                  ║
1: ───X───T───Y───@───X───T───Y───Z──────────────────╫───X───X───
                                                     ║   ║
2: ───────T───────Y───────T───────Z──────────────────╫───╫───────
                                                     ║   ║
a: ══════════════════════════════════════════════════@═══^═══════
''',
    )
    context = cirq.TransformerContext(tags_to_ignore=("nocompile",))
    c_new = cirq.merge_single_qubit_moments_to_phxz(c_orig, context=context)
    cirq.testing.assert_has_diagram(
        c_new,
        '''
0: ───PhXZ(a=-0.5,x=0,z=-1)──────@───PhXZ(a=-0.5,x=0,z=-1)──────Y[nocompile]───X───M───────────
                                 │                                                 ║
1: ───PhXZ(a=-0.25,x=0,z=0.75)───@───PhXZ(a=-0.25,x=0,z=0.75)───Z──────────────────╫───X───X───
                                                                                   ║   ║
2: ───PhXZ(a=0.25,x=0,z=0.25)────Y───PhXZ(a=0.25,x=0,z=0.25)────Z──────────────────╫───╫───────
                                                                                   ║   ║
a: ════════════════════════════════════════════════════════════════════════════════@═══^═══════
''',
    )


def test_merge_single_qubit_moments_to_phxz_deep():
    q = cirq.LineQubit.range(3)
    x_t_y = cirq.FrozenCircuit(
        cirq.Moment(cirq.X.on_each(*q[:2])),
        cirq.Moment(cirq.T.on_each(*q[1:])),
        cirq.Moment(cirq.Y.on_each(*q[:2])),
    )
    c_nested = cirq.FrozenCircuit(
        x_t_y,
        cirq.Moment(cirq.CZ(*q[:2]), cirq.Y(q[2])),
        x_t_y,
        cirq.Moment(cirq.Y(q[0]).with_tags("ignore"), cirq.Z.on_each(*q[1:])),
    )

    c_nested_merged = cirq.FrozenCircuit(
        [_phxz(-0.25, 0.0, 0.75)(q[1]), _phxz(0.25, 0.0, 0.25)(q[2]), _phxz(-0.5, 0.0, -1.0)(q[0])],
        [cirq.CZ(q[0], q[1]), cirq.Y(q[2])],
        [_phxz(-0.25, 0.0, 0.75)(q[1]), _phxz(0.25, 0.0, 0.25)(q[2]), _phxz(-0.5, 0.0, -1.0)(q[0])],
        cirq.Moment(cirq.Y(q[0]).with_tags("ignore"), cirq.Z.on_each(*q[1:])),
    )
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tags"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6),
    )
    c_expected = cirq.Circuit(
        c_nested_merged,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags("preserve_tags"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(6),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_new = cirq.merge_single_qubit_moments_to_phxz(c_orig, context=context)
    cirq.testing.assert_allclose_up_to_global_phase(
        c_new.unitary(), c_expected.unitary(), atol=1e-7
    )


def test_merge_single_qubit_moments_to_phxz_global_phase():
    c = cirq.Circuit(cirq.GlobalPhaseGate(1j).on())
    c2 = cirq.merge_single_qubit_gates_to_phxz(c)
    assert c == c2


def test_merge_single_qubit_moments_to_phased_x_and_z_global_phase():
    c = cirq.Circuit(cirq.GlobalPhaseGate(1j).on())
    c2 = cirq.merge_single_qubit_gates_to_phased_x_and_z(c)
    assert c == c2


def test_merge_single_qubit_gates_to_phxz_symbolized():
    """Test case diagram.
    Input circuit:
    # pylint: disable=line-too-long
    0: ───X─────────@──────────H[ignore]───H───X───PhXZ(a=a0,x=x0,z=z0)───X───PhXZ(a=a1,x=x1,z=z1)───
                    │
    1: ───H^h_exp───@^cz_exp─────────────────────────────────────────────────────────────────────────
    Expected output:
    0: ───PhXZ(a=-1,x=1,z=0)─────@──────────H[ignore]───PhXZ(a=a1,x=x1,z=z1)───
                                 │
    1: ───PhXZ(a=a0,x=x0,z=z0)───@^cz_exp──────────────────────────────────────
    # pylint: enable=line-too-long
    """
    a, b = cirq.LineQubit.range(2)
    sa0, sa1 = [sympy.Symbol(a) for a in ["a0", "a1"]]
    sx0, sx1 = [sympy.Symbol(x) for x in ["x0", "x1"]]
    sz0, sz1 = [sympy.Symbol(z) for z in ["z0", "z1"]]
    input_circuit = cirq.Circuit(
        cirq.Moment(cirq.X(a), cirq.H(b) ** sympy.Symbol("h_exp")),
        cirq.Moment(cirq.CZ(a, b) ** sympy.Symbol("cz_exp")),
        cirq.Moment(cirq.H(a).with_tags("ignore")),
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.X(a)),
        cirq.Moment(_phxz(sa0, sx0, sz0).on(a)),
        cirq.Moment(cirq.X(a)),
        cirq.Moment(_phxz(sa1, sx1, sz1).on(a)),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"])
    sweep = cirq.Zip(
        cirq.Points(key="h_exp", points=[0, 1]),
        cirq.Points(key="cz_exp", points=[0, 1]),
        cirq.Points(key="a0", points=[0, 1]),
        cirq.Points(key="x0", points=[0, 1]),
        cirq.Points(key="z0", points=[0, 1]),
        cirq.Points(key="a1", points=[0, 1]),
        cirq.Points(key="x1", points=[0, 1]),
        cirq.Points(key="z1", points=[0, 1]),
    )
    output_circuit, new_sweep = cirq.merge_single_qubit_gates_to_phxz_symbolized(
        input_circuit, context=context, sweep=sweep
    )
    expected = cirq.Circuit(
        cirq.Moment(_phxz(-1, 1, 0).on(a), _phxz(sa0, sx0, sz0).on(b)),
        cirq.Moment(cirq.CZ(a, b) ** sympy.Symbol("cz_exp")),
        cirq.Moment(cirq.H(a).with_tags("ignore")),
        cirq.Moment(_phxz(sa1, sx1, sz1).on(a)),
    )
    assert_optimizes(output_circuit, expected)

    # Check the unitaries are preserved for each set of sweep paramerization.
    for old_resolver, new_resolver in zip(sweep, new_sweep):
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
            cirq.resolve_parameters(input_circuit, old_resolver),
            cirq.resolve_parameters(output_circuit, new_resolver),
            {q: q for q in input_circuit.all_qubits()},
        )


def test_merge_single_qubit_gates_to_phxz_symbolized_non_parameterized_singles():
    """Test merge_single_qubit_gates_to_phxz_symbolized when all single qubit gates are not
    parameterized."""

    a, b = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(cirq.H(a), cirq.H(a), cirq.CZ(a, b) ** sympy.Symbol("exp"))
    expected_circuit = cirq.merge_single_qubit_gates_to_phxz(input_circuit)
    output_circuit, _ = cirq.merge_single_qubit_gates_to_phxz_symbolized(
        input_circuit, sweep=cirq.Points(key="exp", points=[0.1, 0.2, 0.5])
    )
    assert_optimizes(output_circuit, expected_circuit)


def test_merge_single_qubit_gates_to_phxz_symbolized_with_global_phases():
    a = cirq.NamedQubit("a")
    input_circuit = cirq.Circuit(
        cirq.GlobalPhaseGate(1j).on(), cirq.X(a), cirq.Y(a) ** sympy.Symbol("y_exp")
    )
    new_circuit, _ = cirq.merge_single_qubit_gates_to_phxz_symbolized(
        input_circuit, sweep=cirq.Points(key="y_exp", points=[0, 1])
    )
    expected_circuit = cirq.Circuit(
        cirq.GlobalPhaseGate(1j).on(),
        _phxz(sympy.Symbol("a0"), sympy.Symbol("x0"), sympy.Symbol("z0")).on(a),
    )

    assert_optimizes(new_circuit, expected_circuit)


def test_merge_single_qubit_gates_to_phxz_symbolized_different_structures_error():
    """Tests that the function raises a RuntimeError if merged structures of the circuit differ
    for different parameterizations."""
    a = cirq.NamedQubit("a")
    circuit = cirq.Circuit(cirq.H(a) ** sympy.Symbol("exp"))
    sweep = cirq.Points(key="exp", points=[0.1, 0.2])

    with patch(
        "cirq.protocols.resolve_parameters",
        side_effect=[
            cirq.Circuit(cirq.H(a).with_tags("_symbolized_single")),
            cirq.Circuit(cirq.H(a)),
        ],
    ):
        with pytest.raises(
            RuntimeError,
            match="Different resolvers in sweep resulted in different merged structures.",
        ):
            cirq.merge_single_qubit_gates_to_phxz_symbolized(circuit, sweep=sweep)


def test_merge_single_qubit_gates_to_phxz_symbolized_unexpected_gate_error():
    """Tests that the function raises a RuntimeError of unexpected gate."""
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(a) ** sympy.Symbol("exp1"),
        cirq.X(a),
        cirq.CZ(a, b),
        cirq.Y(a),
        cirq.H(a) ** sympy.Symbol("exp2"),
    )
    sweep = cirq.Points(key="exp1", points=[0.1, 0.2]) * cirq.Points(key="exp2", points=[0.1, 0.2])

    mock_iter = Mock()
    mock_iter.__next__ = Mock(return_value=2)

    with patch(
        "cirq.transformers.analytical_decompositions"
        ".single_qubit_decompositions.single_qubit_matrix_to_phxz",
        return_value=cirq.H,
    ):
        with pytest.raises(
            RuntimeError, match="Expected the merged gate to be a PhasedXZGate or IdentityGate."
        ):
            cirq.merge_single_qubit_gates_to_phxz_symbolized(circuit, sweep=sweep)
