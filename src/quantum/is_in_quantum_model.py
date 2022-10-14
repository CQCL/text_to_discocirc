import torch
from discopy import Box, Ket
from lambeq import PennyLaneModel
from lambeq.core.types import AtomicType
from tqdm import tqdm


class qDisCoCircIsIn:
    def __init__(self, train_circuits, valid_circuits, ansatz, wire_dim):
        self.train_circuits = train_circuits
        self.valid_circuits = valid_circuits
        self.ansatz = ansatz
        self.wire_dim = wire_dim

        self.text_model = PennyLaneModel.from_diagrams(
            self.train_circuits + self.valid_circuits,
            backend_config={
                "backend": "default.qubit",
                "normalize": False,
                "probabilities": False,
            },
        )
        self.text_model.initialise_weights()

        self.is_in_circ, self.is_in_circ_swapped = self.get_is_in_circs()
        self.is_in_model = PennyLaneModel.from_diagrams(
            [self.is_in_circ, self.is_in_circ_swapped],
            backend_config={
                "backend": "default.qubit",
                "normalize": False,
                "probabilities": False,
            },
        )
        self.is_in_model.initialise_weights()

        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, text_circuit, quesans):
        # get statevector of text diagram, is_in box and is_in_swapped box, swapped is needed, when answer word is located before question word in diagram
        text_state = self.text_model([text_circuit]).squeeze(0)
        is_in_state = self.is_in_model([self.is_in_circ]).squeeze(0).flatten()
        is_in_swapped_state = (
            self.is_in_model([self.is_in_circ_swapped]).squeeze(0).flatten()
        )

        # wires of question word
        ques_ids = [quesans[0] * 2, quesans[0] * 2 + 1]

        # create list of wires that are postselected / discarded
        post_select_indices = list(range(len(text_state.shape)))
        for q in ques_ids:
            post_select_indices.remove(q)

        targets = []
        results = []
        for j in post_select_indices[::2]:
            # save targets
            if j == quesans[1] * 2:
                targets.append(1)
            else:
                targets.append(0)

            # discard on text_state so that only question and answer word wires remain open
            posel_ids = post_select_indices.copy()
            posel_ids.remove(j)
            posel_ids.remove(j + 1)
            sliced_text_state = text_state.clone()
            for i in posel_ids[::-1]:
                sliced_text_state = sliced_text_state.select(i, 0) + sliced_text_state.select(i, 1)
            sliced_text_state = sliced_text_state.flatten()
            sliced_text_state = torch.nn.functional.normalize(sliced_text_state, p=2, dim=0)
            # multiply text_state with dagger is_in_state or is_in_swapped_state depending on whether answer word occurs before or after question word in text diagram
            # only is_in_state.conj() because transpose has no effect on vector
            if j > ques_ids[-1]:
                res = torch.dot(is_in_state.conj(), sliced_text_state)
            elif j < ques_ids[0]:
                res = torch.dot(is_in_swapped_state.conj(), sliced_text_state)
            else:
                raise (NotImplementedError("j in ques_ids should never happen"))
            results.append(torch.abs(res) ** 2)

        targets = torch.tensor(targets, dtype=torch.float64)
        results = torch.stack(results)
        results = self.softmax(results)
        return results, targets

    def get_is_in_circs(self):
        """
        creates circuits of the is_in_box
        """
        if self.wire_dim != 2:
            raise Exception("swapped ciruit not implemented for wire_dim != 2")

        N = AtomicType.NOUN
        is_in_diag = Box("is_in", N**self.wire_dim, N**self.wire_dim)
        is_in_circ = Ket(*[0 for i in range(2 * self.wire_dim)]) >> self.ansatz(
            is_in_diag
        )
        is_in_circ_swapped = is_in_circ.permute(2, 3, 0, 1)

        return is_in_circ, is_in_circ_swapped

    def validation_accuracy(self, valid_labels):
        acc = 0
        for text_circuit, quesans in tqdm(zip(self.valid_circuits, valid_labels)):
            results, targets = self.forward(text_circuit, quesans)

            if torch.argmax(results) == torch.argmax(targets):
                acc += 1

        acc /= len(self.valid_circuits)
        return acc
