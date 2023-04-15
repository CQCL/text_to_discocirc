from __future__ import annotations
from copy import deepcopy
import random
from prettytable import PrettyTable

from discocirc.helpers.closed import Func, uncurry_types, Ty


class Expr:
    def __init__(self, name, expr_type, typ, head) -> None:
        self.name = name
        self.expr_type = expr_type
        if expr_type not in ["literal", "lambda", "application", "list"]:
            raise ValueError("Invalid expression type")
        self.typ = typ
        self.head = head
    
    def __repr__(self):
        if self.expr_type == "literal":
            return get_literal_string(self)
        elif self.expr_type == "lambda":
            return get_lambda_string(self)
        elif self.expr_type == "application":
            return get_application_string(self)
        elif self.expr_type == "list":
            return get_list_string(self)
        else:
            raise NotImplementedError(self.expr_type)

    def __call__(self, arg: Expr):
        return Expr.apply(self, arg)
    
    def __members(self):
        if self.expr_type == "literal":
            return (self.expr_type, self.typ, self.name, self.head)
        elif self.expr_type == "lambda":
            return (self.expr_type, self.typ, self.name, self.head, self.var, self.expr)
        elif self.expr_type == "application":
            return (self.expr_type, self.typ, self.name, self.head, self.arg, self.fun)
        elif self.expr_type == "list":
            return (self.expr_type, self.typ, self.name, self.head, self.expr_list)
        else:
            raise NotImplementedError(self.expr_type)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Expr):
            return False
        if self.expr_type != other.expr_type:
            return False
        return self.__members() == other.__members()

    def __hash__(self) -> int:
        return hash(self.__members())

    @staticmethod
    def literal(name, typ, head=None):
        return Expr(name, "literal", typ, head)

    @staticmethod
    def lmbda(var, expr, head=None):
        lambda_expr = Expr(expr.name, "lambda", var.typ >> expr.typ, head)
        lambda_expr.var = var
        #TODO: rename .expr to something else. possible option: "body"
        lambda_expr.expr = expr
        return lambda_expr
    
    @staticmethod
    def application(fun, arg, head=None):
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
    def lst(expr_list, interchange=True, head = None):
        flattened_list = get_flattened_expr_list(expr_list)
        expr = Expr(get_expr_list_name(flattened_list),
                    "list",
                    infer_list_type(flattened_list, interchange),
                    head)
        expr.expr_list = flattened_list
        return expr

    @staticmethod
    def evl(context, expr):
        """
        performs substitution of context into free variables in expr
        """
        if expr.expr_type == "literal":
            if expr in context.keys():
                return context[expr]
            return expr
        elif expr.expr_type == "lambda":
            return Expr.lmbda(expr.var, Expr.evl(context, expr.expr)) 
        elif expr.expr_type == "application":
            return Expr.apply(Expr.evl(context, expr.fun),
                              Expr.evl(context, expr.arg), 
                              context)
        elif expr.expr_type == "list":
            interchange = all([isinstance(e.typ, Func) for e in expr.expr_list])
            return Expr.lst([Expr.evl(context, e) for e in expr.expr_list],
                            interchange=interchange)
        raise TypeError(f'Unknown type {expr.expr_type} of expression')

    @staticmethod
    def apply(fun, arg, context=None, reduce=True, head=None):
        """
        apply expr to arg
        """
        if fun.typ.input != arg.typ:
            new_expr = Expr.partial_apply(fun, arg, context)
        if fun.expr_type == "lambda" and reduce:
            if context == None:
                context = {}
            if fun.var.expr_type == "list":
                for var, val in zip(fun.var.expr_list, arg.expr_list):
                    context[var] = val
            else:
                context[fun.var] = arg
            new_expr = Expr.evl(context, fun.expr)
        else:
            new_expr = Expr.application(fun, arg)
        new_expr.head = head
        return new_expr

    @staticmethod
    def partial_apply(expr, arg, context=None):
        num_inputs = 0
        for i in range(len(expr.typ.input) + 1):
            if expr.typ.input[-i:] == arg.typ:
                num_inputs = i
                break
        if num_inputs == 0:
            raise TypeError(f"Type of:\n{arg}\n is not compatible "
                            + f"with the input type of:\n{expr}")
        var1 = Expr.literal(f"x_{random.randint(1000,9999)}", expr.typ.input[-i:])
        var2 = Expr.literal(f"x_{random.randint(1000,9999)}", expr.typ.input[:-num_inputs])
        var2_var1 = Expr.lst([var2, var1], interchange=False)
        expr = Expr.lmbda(var1, Expr.lmbda(var2, expr(var2_var1)))
        return Expr.apply(expr, arg, context, reduce=True)


def get_flattened_expr_list(expr_list):
    new_expr_list = []
    for e in expr_list:
        if e.expr_type == "list":
            new_expr_list.extend(e.expr_list)
        else:
            new_expr_list.append(e)
    return new_expr_list

def get_expr_list_name(expr_list):
    name = ""
    for e in expr_list:
        name += e.name + ", "
    return name

def infer_list_type(expr_list, interchange):
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
        list_type = Ty()
        for e in expr_list:
            list_type = list_type @ e.typ
    return list_type


def get_literal_string(expr):
    name = str(expr.name)
    typ = str(expr.typ)
    length = max(len(name), len(typ))
    string = f'{name:^{length}}' + '\n'
    string += '═' * length + '\n'
    string += f'{typ:^{length}}'
    return string

def get_lambda_string(expr):
    var_temp = deepcopy(expr.var)
    if hasattr(var_temp, "name"):
        var_temp.name = 'λ ' + var_temp.name
        var = str(var_temp)
    else:
        var = 'λ ' + str(var_temp)
    expr = str(expr.expr)
    typ = str(expr.typ)
    var_lines = var.split('\n')
    expr_lines = expr.split('\n')
    empty_expr_lines = [' ' * len(max(expr_lines))] * (len(var_lines) - len(expr_lines))
    expr_lines = empty_expr_lines + expr_lines
    empty_var_lines = [' ' * len(max(var_lines))] * (len(expr_lines) - len(var_lines))
    var_lines = empty_var_lines + var_lines
    string = ['  '.join([var_l, expr_l]) for var_l, expr_l in zip(var_lines, expr_lines)]
    string.append('─' * len(string[0]))
    string.append(f'{typ:^{len(string[0])}}')
    output = '\n'.join(string)
    return output

def get_application_string(expr):
    expr = str(expr.fun)
    arg = str(expr.arg)
    typ = str(expr.typ)
    expr_lines = expr.split('\n')
    arg_lines = arg.split('\n')
    empty_arg_lines = [' ' * len(max(arg_lines))] * (len(expr_lines) - len(arg_lines))
    arg_lines = empty_arg_lines + arg_lines
    empty_expr_lines = [' ' * len(max(expr_lines))] * (len(arg_lines) - len(expr_lines))
    expr_lines = empty_expr_lines + expr_lines
    string = ['  '.join([expr_l, arg_l]) for expr_l, arg_l in zip(expr_lines, arg_lines)]
    string.append('─' * len(string[0]))
    string.append(f'{typ:^{len(string[0])}}')
    output = '\n'.join(string)
    return output

def get_list_string(expr):
    max_lines = max([len(str(expr).splitlines()) for expr in expr.expr_list])
    tb = PrettyTable()
    tb.border=False
    tb.preserve_internal_border = False
    tb.header = False
    tb.left_padding_width = 0
    tb.right_padding_width = 0
    tb.align = "l"
    tb_list = []
    for i, expr in enumerate(expr.expr_list):
        expr_lines = str(expr).splitlines()
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
    string += f'{str(expr.typ):^{length}}'
    return string
