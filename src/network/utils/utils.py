from discopy import PRO
from discopy.monoidal import Functor
from tensorflow import keras

from network.network import Network


def get_nn_functor(nn_boxes, wire_dim):
    def neural_ob(t):
        return PRO(len(t) * wire_dim)
    def neural_ar(box):
        return nn_boxes[box]
    f = Functor(ob=neural_ob, ar=neural_ar, ar_factory=Network)
    return f

def make_lambda_layer(a, b):
    return keras.layers.Lambda(lambda x: x[:, a:b])

def get_fast_nn_functor(nn_boxes, wire_dim):
    def neural_ob(t):
        return PRO(len(t) * wire_dim)

    def neural_ar(box):
        return nn_boxes[box]
    f = Functor(ob=neural_ob, ar=neural_ar, ar_factory=Network)

    def fast_f(diagram):
        # diagram.draw()
        inputs = keras.Input(shape=(len(f(diagram.dom)),))
        outputs = inputs
        # print(f"{inputs=}")
        for fol in diagram.foliation():
            # fol.draw()
            in_idx = 0
            out_idx = 0
            models = []
            layers = fol.layers
            inps = fol.dom
            for i in range(len(fol)):
                left, box, right = layers[i]
                n_wires = len(f(inps[:in_idx+len(left)-out_idx]))
                f_idx = len(f(inps[:in_idx]))
                # f_left = len(f(left))
                if f_idx < n_wires:
                    # print('boo', i, f_idx, f_left, box)
                    model = make_lambda_layer(f_idx,n_wires)(outputs)
                    models.append(model)
                f_dom = len(f(box.dom))
                # print(f"hi {n_wires=} {f_dom=} {box=} {outputs=}")
                model = make_lambda_layer(n_wires, n_wires+f_dom)(outputs)
                models.append(f(box).model(model))
                in_idx = len(left) - out_idx + len(box.dom)
                out_idx = len(left @ box.cod)
            if right:
                model = make_lambda_layer(n_wires+f_dom, None)(outputs)
                models.append(model)
            outputs = keras.layers.Concatenate()(models)
            # print(f"{outputs=}")
        dom, cod = f(diagram.dom), f(diagram.cod)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return Network(dom, cod, model)
    return fast_f


#TODO do not hard-code hidden layers
def initialize_boxes(lexicon, wire_dimension, hidden_layers=[10, 10]):
    """
    Returns a dict of neural networks, and a list of models

    Parameters
    ----------
    lexicon : list
        List of discopy boxes in the lexicon.
    wire_dimension : int
        Dimension of the noun wires.
    """
    nn_boxes = {}
    trainable_models=[]
    for word in lexicon:
        #TODO add names to the model. It does not like [box] and [\box].
        name = word.name
        if '\\' in name:
            name = name.replace('\\', '')
            name = name[1:-1] + '_end'
        elif '[' in name:
            name = name[1:-1] + '_begin'
        nn_boxes[word] = Network.dense_model(
            len(word.dom) * wire_dimension,
            len(word.cod) * wire_dimension,
            hidden_layer_dims=hidden_layers,
            name = name
        )
        trainable_models.append(nn_boxes[word].model)
    return nn_boxes, trainable_models

def get_classification_vocab(lexicon):
    """
    Parameters:
    -----------
    lexicon : list
        list of discopy boxes

    Returns:
    --------
    vocab : list
        list of names of boxes (modulo frames)
    """
    vocab = []
    for box in lexicon:
        name = box.name
        if '[' in name:
            name = name.replace('\\', '')
            name = name[1:-1]
        if name not in vocab:
            vocab.append(name)
    return vocab

def get_params_dict_from_tf_variables(params, split_string, is_state=False):
    params_dict = {}
    for p in params:
        name = p.name[:-2].split(split_string)[0]
        if is_state:
            params_dict[name] = p
        else:
            if name in params_dict:
                params_dict[name].append(p)
            else:
                params_dict[name] = [p]
    return params_dict

def get_box_name(box):
    name = box.name
    if '\\' in name:
        name = name.replace('\\', '')
        name = name[1:-1] + '_end'
    elif '[' in name:
        name = name[1:-1] + '_begin'
    name = name + '_' + str(len(box.dom)) + '_' + str(len(box.cod))
    return name
