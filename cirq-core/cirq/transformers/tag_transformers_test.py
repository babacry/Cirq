# Copyright 2025 The Cirq Developers
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

import cirq


def test_index_tags():
    q0, q1 = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1", "tag2"),
        cirq.Y(q1).with_tags("tag1"),
        cirq.CZ(q0, q1).with_tags("tag2"),
    )
    expected_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1_0", "tag2_0"),
        cirq.Y(q1).with_tags("tag1_1"),
        cirq.CZ(q0, q1).with_tags("tag2_1"),
    )
    cirq.testing.assert_equivalent_op_tree(
        cirq.index_tags(input_circuit, target_tags={"tag1", "tag2"}), expected_circuit
    )


def test_remove_tags():
    q0, q1 = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag1", "tag2"),
        cirq.Y(q1).with_tags("tag1"),
        cirq.CZ(q0, q1).with_tags("tag2"),
    )
    expected_circuit = cirq.Circuit(
        cirq.X(q0).with_tags("tag2"), cirq.Y(q1), cirq.CZ(q0, q1).with_tags("tag2")
    )
    cirq.testing.assert_equivalent_op_tree(
        cirq.remove_tags(input_circuit, target_tags={"tag1"}), expected_circuit
    )
