from __future__ import annotations
from copy import deepcopy
from prettytable import PrettyTable

from discocirc.helpers.closed import Func, uncurry_types, Ty


class Expr:
    """
    A class representing an expression in the simply-typed lambda calculus.
    """
    def __init__(self, name, expr_type, typ, head) -> None:
        """
        Initializes an Expr object with the given name, expression type, type, and head.
        """
        self.name = name
        self.expr_type = expr_type
        if expr_type not in ["literal", "lambda", "application", "list"]:
            raise ValueError("Invalid expression type")
        self.typ = typ
        self.head = head
    
    def __repr__(self):
        """
        Returns a string representation of the Expr object.
        """
        return self.to_string()

    def to_string(self, index=False):
        """
        Returns a string representation of the Expr object. 
        """
        if self.expr_type == "literal":
            return get_literal_string(self, index)
        elif self.expr_type == "lambda":
            return get_lambda_string(self, index)
        elif self.expr_type == "application":
            return get_application_string(self, index)
        elif self.expr_type == "list":
            return get_list_string(self, index)
        else:
            raise NotImplementedError(self.expr_type)

    def __call__(self, arg: Expr):
        """
        Applies the self to the given argument and returns self(arg).
        """
        return Expr.apply(self, arg)
    
    def __members(self):
        """
        Returns a tuple of the members of the Expr object.
        """
        if self.expr_type == "literal":
            return (self.expr_type, self.typ, self.name, str(self.head))
        elif self.expr_type == "lambda":
            return (self.expr_type, self.typ, self.name, str(self.head), self.var, self.body)
        elif self.expr_type == "application":
            return (self.expr_type, self.typ, self.name, str(self.head), self.arg, self.fun)
        elif self.expr_type == "list":
            return (self.expr_type, self.typ, self.name, str(self.head), self.expr_list)
        else:
            raise NotImplementedError(self.expr_type)

    def __eq__(self, other) -> bool:
        """
        Returns True if the Expr object is equal to the other object, False otherwise.
        """
        if not isinstance(other, Expr):
            return False
        if self.expr_type != other.expr_type:
            return False
        return self.__members() == other.__members()

    def __hash__(self) -> int:
        """
        Returns the hash value of the Expr object.
        """
        return hash(self.__members())

    @staticmethod
    def literal(name, typ, head=None):
        """
        Returns a new Expr object representing a literal with the given name, type, and head.
        """
        return Expr(name, "literal", typ, head)

    @staticmethod
    def lmbda(var, body, head=None, index=None):
        """
        Returns a new Expr object λvar.body representing a lambda expression with the given variable, body, and head.
        """
        lambda_expr = Expr(body.name, "lambda", Func(var.typ, body.typ, index), head)
        lambda_expr.var = var
        lambda_expr.body = body
        return lambda_expr
    
    @staticmethod
    def application(fun, arg, head=None):
        """
        Returns a new Expr object representing an application of the given function to the given argument.
        """
        if fun.typ.input != arg.typ:
            raise TypeError(f"Type of \n{arg}\n does not"
                            + f"match the input type of \n{fun}")
        app_expr = Expr(f"{fun.name}({arg.name})",
                        "application",
                        fun.typ.output,
                        head)
        app_expr.fun = fun
        app_expr.arg = arg
        return app_expr
    
    @staticmethod
    def lst(expr_list, interchange='auto', head=None):
        """
        Returns a new Expr object representing a list of given Expr objects. 
        """
        if interchange == 'auto':
            interchange = if_interchange_list_type(expr_list)
        flattened_list = get_flattened_expr_list(expr_list)
        expr = Expr(get_expr_list_name(flattened_list),
                    "list",
                    infer_list_type(flattened_list, interchange),
                    head)
        expr.expr_list = tuple(flattened_list)
        expr.interchange = interchange
        return expr

    @staticmethod
    def evl(context, expr):
        """
        performs substitution of context into free variables in expr
        """
        head = expr.head
        if expr.expr_type == "literal":
            if expr in context.keys():
                assert(context[expr].typ == expr.typ)
                return context[expr]
            return expr
        elif expr.expr_type == "lambda":
            new_expr = Expr.lmbda(expr.var, Expr.evl(context, expr.body))
        elif expr.expr_type == "application":
            new_expr = Expr.apply(Expr.evl(context, expr.fun),
                              Expr.evl(context, expr.arg), 
                              context)
        elif expr.expr_type == "list":
            new_expr = Expr.lst([Expr.evl(context, e) for e in expr.expr_list])
        else:
            raise TypeError(f'Unknown type {expr.expr_type} of expression')
        new_expr.head = head
        return new_expr

    @staticmethod
    def apply(fun, arg, context=None, reduce=True, head=None, match_indices=True):
        """
        Apply `fun` to `arg`.
        If the type of `arg` matches only part of the input type of `fun`, then a partial application is returned.
        If `fun` is a lambda expression and `reduce` is True, then the lambda expression is reduced by
        substituting the argument in the body of the lambda expression.
        If `match_indices` is True, then the overlapping indices of `arg` and the input type of `fun` are matched.
        """
        if fun.typ.input != arg.typ:
            new_expr = Expr.partial_apply(fun, arg, context)
        elif fun.expr_type == "lambda" and reduce:
            if context == None:
                context = {}
            if fun.var.expr_type == "list":
                if var_list_matches_arg_list(fun, arg):
                    for var, val in zip(fun.var.expr_list, arg.expr_list):
                        context[var] = val
                    new_expr = Expr.evl(context, fun.body)
                else:
                    new_expr = Expr.application(fun, arg)
            else:
                context[fun.var] = arg
                new_expr = Expr.evl(context, fun.body)
        else:
            new_expr = Expr.application(fun, arg)
        if match_indices:
            index_mapping = create_index_mapping_dict(fun.typ.input, arg.typ)
            new_expr = map_expr_indices(new_expr, index_mapping, reduce)
        new_expr.head = head
        return new_expr

    @staticmethod
    def partial_apply(expr, arg, context=None):
        """
        Return a partial application of `expr` with `arg`.
        """
        num_inputs = 0
        for i in range(len(expr.typ.input) + 1):
            if expr.typ.input[-i:] == arg.typ:
                num_inputs = i
                break
        if num_inputs == 0:
            raise TypeError(f"Type of:\n{arg}\n is not compatible "
                            + f"with the input type of:\n{expr}")

        from discocirc.helpers.discocirc_utils import create_random_variable
        var1 = create_random_variable(expr.typ.input[-i:])
        var2 = create_random_variable(expr.typ.input[:-num_inputs])
        var2_var1 = Expr.lst([var2, var1], interchange=False)
        expr = Expr.lmbda(var1, Expr.lmbda(var2, expr(var2_var1)))
        return Expr.apply(expr, arg, context, reduce=True)


def get_flattened_expr_list(expr_list):
    """
    Returns a list of Expr objects where each Expr objects of type "list" are replaced by their underlying Expr objects.
    """
    new_expr_list = []
    for e in expr_list:
        if e.expr_type == "list":
            new_expr_list.extend(e.expr_list)
        else:
            new_expr_list.append(e)
    return new_expr_list

def get_expr_list_name(expr_list):
    """
    Returns a string representing the name of a list of Expr objects.
    """
    name = ""
    for e in expr_list:
        name += e.name + ", "
    return name

def infer_list_type(expr_list, interchange):
    """
    Infer the type of a list of expressions.
    """
    if interchange:
        final_input = Ty()
        final_output = Ty()
        for e in expr_list:
            f = uncurry_types(e.typ, uncurry_everything=True)
            if isinstance(e.typ, Func):
                final_input = final_input @ f.input
                final_output = final_output @ f.output
            else:
                final_output = final_output @ f
        list_type = final_input >> final_output
    else:
        assert(len(expr_list) > 0)
        list_type = expr_list[0].typ
        for e in expr_list[1:]:
            list_type = list_type @ e.typ
    return list_type

def if_interchange_list_type(expr_list):
    """
    Decides whether to apply the interchange law for the type of a list of expressions.
    """
    for e in expr_list:
        # do not interchange if expr_list has a state
        if not isinstance(e.typ, Func):
            return False
        # do not interchange if expr_list has a higher order map
        for e_inputs in e.typ.input:
            if isinstance(e_inputs, Func):
                return False
    return True

def var_list_matches_arg_list(fun, arg):
    """
    Check if the variable list of `fun` matches the argument list of `arg`.
    """
    if fun.var.expr_type != "list" or arg.expr_type != "list" or \
        len(fun.var.expr_list) != len(arg.expr_list):
        return False
    for var, val in zip(fun.var.expr_list, arg.expr_list):
        if var.typ != val.typ:
            return False
    return True

def create_index_mapping_dict(key_typ, value_typ):
    """
    Create a dictionary mapping the indices of `key_typ` to the indices of `value_typ`.
    """
    mapping = {}
    if isinstance(key_typ, Func):
        mapping |= create_index_mapping_dict(key_typ.input, value_typ.input)
        mapping |= create_index_mapping_dict(key_typ.output, value_typ.output)
    if key_typ.index != None \
        and value_typ.index != None \
        and key_typ.index != value_typ.index:
        for k in key_typ.index:
            mapping[k] = value_typ.index
    return mapping

def map_typ_indices(typ, mapping):
    """
    Map the indices of `typ` according to `mapping`.
    """
    # TODO: remove deepcopy and make sure that typ is not modified
    typ = deepcopy(typ)
    if isinstance(typ, Func):
        input_typ = map_typ_indices(typ.input, mapping)
        output_typ = map_typ_indices(typ.output, mapping)
        typ = Func(input_typ, output_typ, typ.index)
    if typ.index != None:
        new_index = set()
        for idx in typ.index:
            if idx in mapping.keys() and mapping[idx] != None:
                new_index = set.union(new_index, mapping[idx])
            else:
                new_index.add(idx)
        typ.index = new_index
    if len(typ.objects) > 1:
        for obj in typ.objects:
            obj.typ = map_typ_indices(obj, mapping)
    return typ

def map_expr_indices(expr, mapping, reduce=True):
    """
    Map the indices of type of `expr` according to `mapping`.
    """
    if expr.expr_type == "literal":
        new_expr = deepcopy(expr)
        new_expr.typ = map_typ_indices(expr.typ, mapping)
    elif expr.expr_type == "lambda" or expr.expr_type == "list":
        new_expr = expr_type_recursion(expr, map_expr_indices, mapping, reduce)
        if expr.typ.index in mapping.keys():
            new_expr.typ.index = mapping[expr.typ.index]
        else:
            new_expr.typ.index = expr.typ.index
    elif expr.expr_type == "application":
        arg = map_expr_indices(expr.arg, mapping, reduce)
        fun = map_expr_indices(expr.fun, mapping, reduce)
        new_expr = Expr.apply(fun, arg, reduce=reduce, match_indices=False)
    if hasattr(expr, 'head'):
        new_expr.head = expr.head
    return new_expr

def get_literal_string(expr, index):
    """
    Returns a string representing a literal Expr object.
    """
    name = str(expr.name)
    typ = expr.typ.to_string(index)
    length = max(len(name), len(typ))
    string = f'{name:^{length}}' + '\n'
    string += '═' * length + '\n'
    string += f'{typ:^{length}}'
    return string

def get_lambda_string(expr, index):
    """
    Returns a string representing a lambda Expr object.
    """
    tb = PrettyTable(["lambda", "var", "dot", "body"])
    tb.border=False
    tb.preserve_internal_border = False
    tb.header = False
    tb.left_padding_width = 0
    tb.right_padding_width = 0
    tb.align = "l"
    tb.add_row(["λ ", expr.var.to_string(index), " • ", expr.body.to_string(index)])
    tb.valign["lambda"] = "m"
    tb.valign["var"] = "b"
    tb.valign["dot"] = "m"
    tb.valign["body"] = "b"
    string = tb.get_string()
    length = len(string.splitlines()[-1])
    string += '\n' + '─' * length + '\n'
    typ = expr.typ.to_string(index)
    string += f'{typ:^{length}}'
    return string
   

def get_application_string(expr, index):
    """
    Returns a string representing an application Expr object.
    """
    fun = expr.fun.to_string(index)
    arg = expr.arg.to_string(index)
    typ = expr.typ.to_string(index)
    expr_lines = fun.split('\n')
    arg_lines = arg.split('\n')
    empty_arg_lines = [' ' * len(max(arg_lines))] * (len(expr_lines) - len(arg_lines))
    arg_lines = empty_arg_lines + arg_lines
    empty_expr_lines = [' ' * len(max(expr_lines))] * (len(arg_lines) - len(expr_lines))
    expr_lines = empty_expr_lines + expr_lines
    string = ['  '.join([expr_l, arg_l]) for expr_l, arg_l in zip(expr_lines, arg_lines)]
    string.append('─' * len(string[0]))
    string.append(f'{typ:^{len(string[0])}}')
    return '\n'.join(string)

def get_list_string(expr, index):
    """
    Returns a string representing a list Expr object.
    """
    max_lines = max([len(expr.to_string(index).splitlines()) for expr in expr.expr_list])
    tb = PrettyTable()
    tb.border=False
    tb.preserve_internal_border = False
    tb.header = False
    tb.left_padding_width = 0
    tb.right_padding_width = 0
    tb.align = "l"
    tb_list = []
    for i, ex in enumerate(expr.expr_list):
        expr_lines = ex.to_string(index).splitlines()
        if i != len(expr.expr_list) - 1:
            expr_lines[-2] += ' x '
        if max_lines - len(expr_lines) > 0:
            expr_lines = ['\n'*(max_lines - len(expr_lines))] + expr_lines
        expr_lines = '\n'.join(expr_lines)
        tb_list.append(expr_lines)
    tb.add_row(tb_list)
    string = tb.get_string()
    length = len(string.splitlines()[-1])
    string += '\n' + '─' * length + '\n'
    typ = expr.typ.to_string(index)
    string += f'{typ:^{length}}'
    return string

def expr_type_recursion(expr, function, *args, **kwargs):
    """
    Recursively applies the given function to the subexpressions of the given expression.
    This is a useful helper function for implementing new Expr methods.
    """
    if expr.expr_type == "literal":
        new_expr = function(expr, *args, **kwargs)
    elif expr.expr_type == "list":
        new_expr = Expr.lst([function(e, *args, **kwargs)\
                             for e in expr.expr_list])
    elif expr.expr_type == "lambda":
        new_expr = function(expr.body, *args, **kwargs)
        new_var = function(expr.var, *args, **kwargs)
        new_expr = Expr.lmbda(new_var, new_expr)
    elif expr.expr_type == "application":
        arg = function(expr.arg, *args, **kwargs)
        fun = function(expr.fun, *args, **kwargs)
        new_expr = fun(arg)
    else:
        raise TypeError(f'Unknown type {expr.expr_type} of expression')
    if hasattr(expr, 'head'):
        new_expr.head = expr.head
    return new_expr
