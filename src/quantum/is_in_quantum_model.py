import torch
from discopy import Box, Ket
from lambeq import PennyLaneModel
from lambeq.core.types import AtomicType
from tqdm import tqdm
import itertools

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

        self.is_in_circ = self.get_is_in_circs()
        self.is_in_model = PennyLaneModel.from_diagrams(
            [self.is_in_circ],
            backend_config={
                "backend": "default.qubit",
                "normalize": False,
                "probabilities": False,
            },
        )
        self.is_in_model.initialise_weights()

        self.softmax = torch.nn.Softmax(dim=0)



    def forward(self, text_circuits, quesans_list):
        batch_results = []
        batch_targets = []

        for text_circuit, quesans in zip(text_circuits, quesans_list):

            text_state = self.text_model([text_circuit]).squeeze(0)
            is_in_effect = self.is_in_model([self.is_in_circ]).squeeze(0).conj()
            
            # create list of qubit ids of potential answers to the question
            qubits = list(range(len(text_state.shape)))
            it = iter(qubits)
            answer_ids = list(zip(it,it))
            answer_ids.remove((quesans[0] * 2, quesans[0] * 2 + 1))
           
            target = []
            result = []
            for i,j in answer_ids:
                # if answer id is the actual answer --> target = 1, else 0
                if i == quesans[1] * 2:
                    target.append(1)
                else:
                    target.append(0)

                # create lists for tensor contraction of is in effect and textstate
                # ids which is in effect is plugged into 
                is_in_ids = [quesans[0] * 2, quesans[0] * 2 + 1, i, j]
                # create list over all wires of text circuit
                text_ids = list(range(len(text_state.shape)))
                # output ids after tensor contraction should not contain is_in_ids as those wires are psotselected
                out_ids = text_ids.copy()
                for a in is_in_ids:
                    out_ids.remove(a) 
                # tensor contraction
                # is it okay to only conugate, but not transpose is_in_effect? How would you transpose it? Or how would you do this after flattening to a tensor?
                final_state = torch.einsum(is_in_effect, is_in_ids, text_state, text_ids, out_ids)

                # all possible permutations of basis states for the remaining open wires
                n = len(final_state.shape)
                permutations = list(itertools.chain((n*(0,),), (l[0] * (0,) + sum(((1,) + (i-j-1) * (0,) for i, j in zip(l[1:], l[:-1])), ()) + (1,) + (n-l[-1]-1)*(0,) for k in range(1,n+1) for l in itertools.combinations(range(n), k))))
                
                # sum up probabilities for all possible permutations
                prob = 0
                for perm in permutations:
                    state = final_state[perm]
                    prob += torch.abs(state) ** 2
                # print(prob)
                # why is prob sometimes > 1? should not be the case
                if prob > 1:
                    print('prob is bigger 1: ', prob)
                    # raise ValueError('prob can not be > 1')

                
                result.append(prob)

            # print('target: ', target)
            
            target = torch.tensor(target, dtype=torch.float64)
            result = torch.stack(result)
            result = self.softmax(result)
            # print('result: ', result)
            batch_targets.append(target)
            batch_results.append(result)
        
        batch_targets = torch.stack(batch_targets)
        batch_results = torch.stack(batch_results)
        return batch_results, batch_targets




            # print('ques_ids: ', ques_ids)

            # # create list of wires that are postselected / discarded
            # post_select_indices = list(range(len(text_state.shape)))
            # # print('post_select-indices after init: ', post_select_indices)
            # for q in ques_ids:
            #     post_select_indices.remove(q)
            # print('post_select_indices after removing ques_ids: ', post_select_indices)
 
            # target = []
            # result = []
            # # going through post_select_indices ins steps of size 2 because of wire_dim=2
            # # TODO: replace stepsize by wire_dim
            # for j in post_select_indices[::2]:
            #     # save targets
            #     print('j = ', j)
            #     if j == quesans[1] * 2:
            #         target.append(1)
            #         print('target_j = ', j)
            #     else:
            #         target.append(0)
            #     print('not changing? post_select_indices: ', post_select_indices)
            #     print('not changing? text_state: ', text_state.shape)
            #     # discard on text_state so that only question and answer word wires remain open
            #     posel_ids = post_select_indices.copy()
            #     posel_ids.remove(j)
            #     posel_ids.remove(j + 1)
            #     print(f'posel_ids after removing {j}', posel_ids) 
            #     sliced_text_state = text_state.clone()






            #     # going through posel_ids backwards to make sure indices are used correctly. If you discaard at position 2 first and then at position 4, position 4 will be, what was position 5 before. Thus backwards!
            #     for i in posel_ids[::-1]:
            #         # this is same as sliced_text_state[:,...,:,0,:,...,:] + sliced_text_state[:,...,:,1,:,...,:], where the 0 and 1 are at position i of the tensors shape respectively () the dimension that's supposed to be discarded
            #         sliced_text_state = sliced_text_state.select(i, 0) + sliced_text_state.select(i, 1)
            #     sliced_text_state = sliced_text_state.flatten()
            #     sliced_text_state = torch.nn.functional.normalize(sliced_text_state, p=2, dim=0)
                # multiply text_state with dagger is_in_state or is_in_swapped_state depending on whether answer word occurs before or after question word in text diagram
                # only is_in_state.conj() because transpose has no effect on vector
        #         if j > ques_ids[-1]:
        #             res = torch.matmul(is_in_state.conj().transpose(), sliced_text_state)
        #         elif j < ques_ids[0]:
        #             res = torch.matmul(is_in_swapped_state.conj().transpose(), sliced_text_state)
        #         else:
        #             raise (NotImplementedError("j in ques_ids should never happen"))
        #         result.append(torch.abs(res) ** 2)

        #     print('target: ', target)
        #     target = torch.tensor(target, dtype=torch.float64)
        #     result = torch.stack(result)
        #     result = self.softmax(result)
        #     batch_targets.append(target)
        #     batch_results.append(result)
        
        # batch_targets = torch.stack(batch_targets)
        # batch_results = torch.stack(batch_results)
        # return batch_results, batch_targets

    # def state_inner_prod(state_vec, is_in_vec):



    def get_is_in_circs(self):
        """
        creates circuits of the is_in_box
        """
        
        N = AtomicType.NOUN
        is_in_diag = Box("is_in", N**self.wire_dim, N**self.wire_dim)
        is_in_circ = Ket(*[0 for i in range(2 * self.wire_dim)]) >> self.ansatz(
            is_in_diag
        )

        return is_in_circ

    def validation_accuracy(self, valid_labels):
        acc = 0
        for text_circuit, quesans in tqdm(zip(self.valid_circuits, valid_labels)):
            results, targets = self.forward([text_circuit], [quesans])
            results = results.squeeze(0)
            targets = targets.squeeze(0)
            if torch.argmax(results) == torch.argmax(targets):
                acc += 1

        acc /= len(self.valid_circuits)
        return acc