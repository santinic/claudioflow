import numpy as np
import layers


class SyntaxOp:
    def __init__(self, input=None):
        self.inputs = [input]

    def __add__(self, other):
        return Sum(self, other)

    def __sub__(self, other):
        raise Exception("Subtraction not implemented")
        # return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __div__(self, other):
        raise Exception("Division not implemented")
        # return Div(self, other)

    def __str__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        input_str = ''
        if len(self.inputs) > 0:
            input_str = ', '.join([str(x) for x in self.inputs])
            input_str = '(%s)' % input_str
        return '%s%s' % (name, input_str)

    def forward_variables(self, input_dict, depth=0, debug=False):
        values = []
        for input_op in self.inputs:
            val = input_op.forward_variables(input_dict, depth + 1, debug)
            values.append(val)
            if debug:
                print('%s%s -> %s' % ('\t' * depth, input_op, val))
        x = values[0] if len(values) == 1 else values
        y = self.layer_forward(np.array(x))
        return y

    def backward_variables(self, deltas, depth=0, debug=False):
        if debug:
            print('%s%s <- deltas=%s' % ('\t' * depth, self, deltas))
        deltas = self.layer_backward(deltas)
        if len(self.inputs) == 1:
            return self.inputs[0].backward_variables(deltas, depth + 1, debug)
        else:
            ret_deltas = []
            for input_op, dJdy in zip(self.inputs, deltas):
                delta = input_op.backward_variables(dJdy, depth + 1, debug)
                ret_deltas.append(delta)
        merged = self.merge_backprop_dicts(ret_deltas)
        return merged

    def update_weights(self, optimizer, depth=0, debug=False):
        if debug:
            print('%s%s' % ('\t' * depth, self))

        self.layer_update_weights(optimizer)
        for input in self.inputs:
            input.update_weights(optimizer, depth + 1, debug)

    @staticmethod
    def merge_backprop_dicts(dicts):
        merged = dicts[0]
        for dict in dicts[1:]:
            for k, v in dict.iteritems():
                if k in merged:
                    merged[k] += v
                else:
                    merged[k] = v
        return merged

    def layer_forward(self, values):
        return self.layer.forward(values)

    def layer_backward(self, dJdy):
        return self.layer.backward(dJdy)

    def layer_update_weights(self, optimizer):
        self.layer.update_weights(optimizer)


class Var(SyntaxOp):
    def __init__(self, variable_name):
        self.variable_name = variable_name
        self.inputs = []

    def forward_variables(self, input_dict, depth, debug):
        val = input_dict[self.variable_name]
        return val

    def backward_variables(self, deltas, depth, debug):
        grad_dict = {}
        grad_dict[self.variable_name] = deltas
        return grad_dict

    def layer_forward(self, values):
        return values

    def layer_backward(self, dJdy):
        return dJdy

    def layer_update_weights(self, optimizer):
        return

    def __str__(self):
        return SyntaxOp.__str__(self, name=self.variable_name)


class Linear(SyntaxOp):
    def __init__(self, in_size, out_size, initialize='random', dtype=None, input=None):
        SyntaxOp.__init__(self, input)
        self.layer = layers.Linear(in_size, out_size, initialize, dtype)


class Tanh(SyntaxOp):
    layer = layers.Tanh()


class Sigmoid(SyntaxOp):
    layer = layers.Sigmoid()


class Sum(SyntaxOp):
    def __init__(self, *args):
        self.layer = layers.Sum()
        self.inputs = args


class Mul(SyntaxOp):
    def __init__(self, *args):
        self.layer = layers.Mul()
        self.inputs = args


class Concat(SyntaxOp):
    def __init__(self, *args):
        self.layer = layers.Concat()
        self.inputs = args


class Const(SyntaxOp):
    def __init__(self, const):
        SyntaxOp.__init__(self)
        self.const = const
        self.inputs = []

    def layer_forward(self, x):
        return np.array([self.const])


class Store(SyntaxOp):
    def __init__(self, in_size, input):
        SyntaxOp.__init__(self, input)
        self.layer = layers.Store(in_size)
        self.read_forward = self.layer.read_forward
