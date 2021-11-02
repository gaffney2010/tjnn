import random
from typing import Any, Iterator, List, Optional, Type

import attr
import numpy as np


Vector = List[float]
Label = int


def equal(*elems):
    """Assert all the elements are equal, and return that element."""
    elist = list(elems)
    assert all(x == elist[0] for x in elist)
    return elist[0]


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

    def _put_in(self, input: Vector) -> None:
        input_sz = equal(len(input), len(self.inputs))

        # Just do the translation node-by-node
        for i in range(input_sz):
            self.inputs[i].set_value(input[i])

    def _put_out(self, output: Vector) -> None:
        output_sz = equal(len(output), len(self.outputs))

        # Just do the translation node-by-node
        for i in range(output_sz):
            self.outputs[i].set_value(output[i])

    def put_in_out(self, input: Vector, output: Vector) -> None:
        self._put_in(input)
        self._put_out(output)

    def _forward_walk(self) -> Iterator[Node]:
        for layer in self.layers:
            for node in layer:
                yield node

    def _back_walk(self) -> Iterator[Node]:
        for layer in self.layers[::-1]:
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

    def infer(self, input: Vector) -> Label:
        self._put_in(input)
        self.forward_prop()

        # Now see what label gets predicted
        best_guess: Optional[Label] = None
        best_guess_score: float = 0.0
        for i, node in enumerate(self.outputs):
            if node.value > best_guess_score:
                best_guess = i
                best_guess_score = node.value
        return best_guess

    def back_prop(self) -> None:
        # First clear out all path_product variables
        for node in self._back_walk():
            node.path_product = 1.0

        # Set the derivate_error for the output nodes
        for node in self.outputs:
            node.path_product = node.derivative_error()

        for node in self._back_walk():
            # Does nothing for output nodes
            for edge in node.outgoing:
                node.path_product *= edge.weight * edge.nto.path_product

        # Now calculate the gradiant on each edge
        for edge in self.edges:
            edge.gradiant += edge.nto.path_product * edge.nfrom.value

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


def score(graph: Graph, all_inputs: List[Vector], all_outputs: List[Vector]) -> float:
    data_sz = equal(len(all_inputs), len(all_outputs))

    n_correct, n_all = 0, 0
    for i in range(data_sz):
        n_all += 1
        n_correct += graph.infer(all_inputs[i]) == all_outputs[i]

    return n_correct / n_all


def train(
    graph: Graph,
    train_inputs: List[Vector],
    train_outputs: List[Vector],
    test_inputs: List[Vector],
    test_outputs: List[Vector],
    gamma: float,
    mini_batch_size: int,
    epochs: int,
    score_every_n_batches=15,
) -> None:
    data_sz = equal(len(train_inputs), len(train_outputs))

    batch_ct = 0
    for _ in range(epochs):
        print(f"Epoch {_}")
        inds = list(range(data_sz))
        random.shuffle(inds)
        while len(inds) >= mini_batch_size:
            print("MINI-BATCH")
            mini_batch_inds = inds[:mini_batch_size]
            inds = inds[mini_batch_size:]
            inputs = [train_inputs[i] for i in mini_batch_inds]
            outputs = [train_outputs[i] for i in mini_batch_inds]
            train_mini_batch(graph, inputs, outputs, gamma)

            # Decide whether to print update
            batch_ct += 1
            if batch_ct % score_every_n_batches == 0:
                print(f"Accuracy after {batch_ct} batches")
                print(f"Train set: {score(graph, train_inputs, train_outputs)}")
                print(f"Test set: {score(graph, test_inputs, test_outputs)}")
                print()


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
def make_dataset(fn):
    inputs, outputs = list(), list()
    with open(fn, "r") as f:
        while line := f.readline():
            csv = line.split(",")

            output = list()
            target = csv[0]
            for i in range(10):
                output.append(1 if str(i) == target else 0)
            outputs.append(output)

            input = [float(i) for i in csv[1:]]
            inputs.append(input)
    return inputs, outputs


train_inputs, train_outputs = make_dataset("data/mnist_train.csv")
test_inputs, test_outputs = make_dataset("data/mnist_test.csv")


graph = build_graph(
    [
        Layer(node_type=InputNode, number_nodes=28 * 28),
        Layer(node_type=BodyNode, number_nodes=80),
        Layer(node_type=OutputNode, number_nodes=10),
    ]
)


print("HELLO")

train(graph, train_inputs, train_outputs, test_inputs, test_outputs, 0.05, 100, 4)
