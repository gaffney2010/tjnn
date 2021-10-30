import random
from typing import Any, Iterator, List, Optional, Type

import attr
import numpy as np


Vector = List[float]


class NnException(Exception):
    pass


class Node(object):
    def __init__(self, name: str):
        self.incoming: List["Edge"] = list()
        self.outgoing: List["Edge"] = list()
        self.value: float = 0.0
        self.path_product: float = 1.0
        self.type = "Bare Node"
        self._name = name

    @property
    def name(self) -> str:
        return f"{self.type}: {self._name}"

    def __hash__(self) -> int:
        return hash(self.name)

    def activate(self) -> None:
        # Applies some activation layer to the value
        pass


class InputNode(Node):
    def __init__(self, name: str):
        super().__init__(name)

    def set_value(self, value: float) -> None:
        self.value = value


# Always assume it's ReLu
class BodyNode(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.type = "Body"

    def activate(self) -> None:
        if self.value < 0:
            self.value = 0


class Edge(object):
    def __init__(self, nfrom: Node, nto: Node):
        self.weight = random.uniform(-0.01, 0.01)
        self.gradiant = 0
        self.nfrom = nfrom
        self.nto = nto

    def update_gradient(self, gamma: float) -> None:
        self.weight += self.gradiant * gamma


# Always assume it's a logit
class OutputNode(Node):
    def __init__(self, name: str):
        self.actual: Optional[float] = None
        super().__init__(name)
        self.type = "Output"

    def set_value(self, value: float) -> None:
        self.actual = value

    def activate(self) -> None:
        # Some conditionals to avoid overflow
        if self.value > 5.0:
            self.value = 1.0
            return
        if self.value < -5.0:
            self.value = 0
            return

        exp = np.exp(self.value)
        self.value = exp / (exp + 1.0)

    def derivative_error(self) -> float:
        # Dumb hack, you know why
        if 0.99999 < self.value < 1.00001:
            self.value = 1.00001
        if -0.00001 < self.value < 0.00001:
            self.value = 0.00001

        if self.actual == 1:
            return 1.0 / self.value
        if self.actual == 0:
            return 1.0 / (1.0 - self.value)
        raise NnException


class Graph(object):
    def __init__(self):
        self.layers: List[List[Node]] = list()
        self.edges: List[Edge] = list()

    @property
    def inputs(self) -> List[Node]:
        return self.layers[0]

    @property
    def outputs(self) -> List[Node]:
        return self.layers[-1]

    def put_in_out(self, input: Vector, output: Vector) -> None:
        if len(self.inputs) != len(input):
            raise NnException
        if len(self.outputs) != len(output):
            raise NnException

        # Just do the translation node-by-node
        for i in range(len(self.inputs)):
            self.inputs[i].set_value(input[i])
        for i in range(len(self.outputs)):
            self.outputs[i].set_value(output[i])

    def _forward_walk(self) -> Iterator[Node]:
        for layer in self.layers:
            for node in layer:
                yield node

    def _back_walk_from(self, end_node: Node) -> Iterator[Node]:
        yield end_node
        for layer in self.layers[-2::-1]:
            for node in layer:
                yield node

    def forward_prop(self) -> None:
        # First zero out all nodes, except the input nodes
        input_set = set(self.inputs)
        for node in self._forward_walk():
            if node not in input_set:
                node.value = 0.0

        for node in self._forward_walk():
            # At this point all incoming values have been added
            node.activate()

            for edge in node.outgoing:
                edge.nto.value += edge.weight * node.value

    def back_prop(self) -> None:
        for output in self.outputs:
            derivate_error = output.derivative_error()

            # First clear out all path_product variables
            for node in self._back_walk_from(output):
                node.path_product = 1.0

            c1, c2 = 0, 0

            for node in self._back_walk_from(output):
                for edge in node.outgoing:
                    c1 += 1
                    node.path_product *= edge.weight * edge.nto.path_product

            # Now calculate the gradiant on each edge
            for edge in self.edges:
                c2 += 1
                edge.gradiant += (
                    derivate_error * edge.nto.path_product * edge.nfrom.value
                )

                print(c1)
                print(c2)

    def update_gradient(self, gamma: float) -> None:
        for edge in self.edges:
            edge.update_gradient(gamma)


def train_one(graph, input: Vector, output: Vector) -> None:
    graph.put_in_out(input, output)
    graph.forward_prop()
    graph.back_prop()


def train_mini_batch(
    graph: Graph, inputs: List[Vector], outputs: List[Vector], gamma: float
):
    if len(inputs) != len(outputs):
        raise NnException

    for i, o in zip(inputs, outputs):
        train_one(graph, i, o)

    graph.update_gradient(gamma)


def train(
    graph: Graph,
    all_inputs: List[Vector],
    all_outputs: List[Vector],
    gamma: float,
    mini_batch_size: int,
    epochs: int,
) -> None:
    total_mini_batches = 1  # For testing
    for _ in range(epochs):
        print(f"Epoch {_}")
        inds = list(range(len(all_inputs)))
        random.shuffle(inds)
        while len(inds) >= mini_batch_size:
            total_mini_batches -= 1
            print("MINI-BATCH")
            mini_batch_inds = inds[:mini_batch_size]
            inds = inds[mini_batch_size:]
            inputs = [all_inputs[i] for i in mini_batch_inds]
            outputs = [all_outputs[i] for i in mini_batch_inds]
            train_mini_batch(graph, inputs, outputs, gamma)
            if total_mini_batches <= 0:
                return


@attr.s()
class Layer(object):
    node_type: Type[Node] = attr.ib()
    number_nodes: int = attr.ib()


# Makes a dense graph with specified layers
def build_graph(layers: List[Layer]) -> Graph:
    """Makes a dense graph with specified layers
    
    First layer must be inputs, and last layer must be outputs.
    """
    result = Graph()

    # Make layers of nodes
    for i, layer in enumerate(layers):
        nodes: List[Node] = list()
        for j in range(layer.number_nodes):
            new_node = layer.node_type(f"{i}, {j}")
            nodes.append(new_node)
        result.layers.append(nodes)

    # Add edges
    for li in range(len(result.layers) - 1):
        for nfrom in result.layers[li]:
            for nto in result.layers[li + 1]:
                new_edge = Edge(nfrom, nto)
                result.edges.append(new_edge)
                nfrom.outgoing.append(new_edge)
                nto.incoming.append(new_edge)

    return result


# TJ: Magic numbers and strings
inputs, outputs = list(), list()
with open("data/mnist_train.csv", "r") as f:
    while line := f.readline():
        csv = line.split(",")

        output = list()
        target = csv[0]
        for i in range(10):
            output.append(1 if str(i) == target else 0)
        outputs.append(output)

        input = [float(i) for i in csv[1:]]
        inputs.append(input)


graph = build_graph(
    [
        Layer(node_type=InputNode, number_nodes=28 * 28),
        Layer(node_type=BodyNode, number_nodes=80),
        Layer(node_type=OutputNode, number_nodes=10),
    ]
)


print("HELLO")

train(graph, inputs, outputs, 0.05, 100, 1)
