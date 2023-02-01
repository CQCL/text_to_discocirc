from __future__ import annotations
from copy import deepcopy


from discocirc.helpers.closed import Func, uncurry_types, Ty


class Expr:
    def __repr__(self):
        if self.expr_type == "literal":
            name = str(self.name)
            typ = str(self.typ)
            length = max(len(name), len(typ))
            string = f'{name:^{length}}' + '\n'
            string += '═' * length + '\n'
            string += f'{typ:^{length}}'
            return string
        elif self.expr_type == "lambda":
            var_temp = deepcopy(self.var)
            var_temp.name = 'λ ' + var_temp.name
            var = str(var_temp)
            expr = str(self.expr)
            typ = str(self.typ)
            var_lines = var.split('\n')
            expr_lines = expr.split('\n')
            empty_expr_lines = [' ' * len(max(expr_lines))] * (len(var_lines) - len(expr_lines))
            expr_lines = empty_expr_lines + expr_lines
            empty_var_lines = [' ' * len(max(var_lines))] * (len(expr_lines) - len(var_lines))
            var_lines = empty_var_lines + var_lines
            string = ['  '.join([var_l, expr_l]) for var_l, expr_l in zip(var_lines, expr_lines)]
            string.append('─' * len(string[0]))
            string.append(f'{typ:^{len(string[0])}}')
            return '\n'.join(string)
        elif self.expr_type == "application":
            expr = str(self.expr)
            arg = str(self.arg)
            typ = str(self.typ)
            expr_lines = expr.split('\n')
            arg_lines = arg.split('\n')
            empty_arg_lines = [' ' * len(max(arg_lines))] * (len(expr_lines) - len(arg_lines))
            arg_lines = empty_arg_lines + arg_lines
            empty_expr_lines = [' ' * len(max(expr_lines))] * (len(arg_lines) - len(expr_lines))
            expr_lines = empty_expr_lines + expr_lines
            string = ['  '.join([expr_l, arg_l]) for expr_l, arg_l in zip(expr_lines, arg_lines)]
            string.append('─' * len(string[0]))
            string.append(f'{typ:^{len(string[0])}}')
            return '\n'.join(string)
        elif self.expr_type == "list":
            names = ', '.join([str(ex.name) for ex in self.expr_list])
            types = str(self.typ)
            length = max(len(names), len(types))
            string = f'{names:^{length}}' + '\n'
            string += '═' * length + '\n'
            string += f'{types:^{length}}'
            return string
        else:
            raise NotImplementedError(self.expr_type)

    def __call__(self, arg: Expr):
        return Expr.apply(self, arg)
    
    def __members(self):
        if self.expr_type == "literal":
            return (self.expr_type,
                    self.typ,
                    self.name)
        elif self.expr_type == "lambda":
            return (self.expr_type,
                    self.typ,
                    self.var,
                    self.expr)
        elif self.expr_type == "application":
            return (self.expr_type,
                    self.typ,
                    self.arg,
                    self.expr)
        elif self.expr_type == "list":
            return (self.expr_type,
                    self.typ,
                    self.expr_list)
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
    def literal(name, typ):
        expr = Expr()
        expr.expr_type = "literal"
        expr.typ = typ
        expr.name = name
        return expr

    @staticmethod
    def lmbda(var, expr):
        lambda_expr = Expr()
        lambda_expr.expr_type = "lambda"
        lambda_expr.var = var
        lambda_expr.expr = expr
        lambda_expr.typ = var.typ >> expr.typ
        return lambda_expr
    
    @staticmethod
    def application(expr, arg):
        if expr.typ.input != arg.typ:
            raise TypeError(f"Type of {arg} does not"
                            + f"match the input type of {expr}")
        app_expr = Expr()
        app_expr.expr_type = "application"
        app_expr.typ = expr.typ.output
        app_expr.expr = expr
        app_expr.arg = arg
        return app_expr
    
    @staticmethod
    def lst(expr_list, interchange=True):
        expr = Expr()
        expr.expr_type = "list"
        new_expr_list = []
        for e in expr_list:
            if e.expr_type == "list":
                new_expr_list.extend(e.expr_list)
            else:
                new_expr_list.append(e)
        expr.expr_list = tuple(new_expr_list)
        expr.typ = Expr.infer_list_type(expr_list, interchange)
        return expr
    
    @staticmethod
    def infer_list_type(expr_list, interchange):
        if interchange:
            final_input = Ty()
            final_output = Ty()
            for e in expr_list:
                f = uncurry_types(e.typ)
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

    @staticmethod
    def evl(context, expr):
        if expr.expr_type == "literal":
            if expr in context.keys():
                return context[expr]
            else:
                return expr
        elif expr.expr_type == "lambda":
            return Expr.lmbda(expr.var, Expr.evl(context, expr.expr))
        elif expr.expr_type == "application":
            return Expr.apply(Expr.evl(context, expr.expr), Expr.evl(context, expr.arg), context)
        elif expr.expr_type == "list":
            return Expr.lst([Expr.evl(context, e) for e in expr.expr_list])
        else:
            raise TypeError(f'Unknown type {expr.expr_type} of expression')

    @staticmethod
    def apply(expr, arg, context=None):
        if expr.typ.input != arg.typ:
            return Expr.partial_apply(expr, arg, context)
        if expr.expr_type == "lambda":
            if context == None:
                context = {}
            if expr.var.expr_type == "list":
                for var, val in zip(expr.var.expr_list, arg.expr_list):
                    context[var] = val
            else:
                context[expr.var] = arg
            return Expr.evl(context, expr.expr)
        else:
            new_expr = Expr.application(expr, arg)
            return new_expr
        
    @staticmethod
    def partial_apply(expr, arg, context=None):
        num_inputs = 0
        for i in range(len(expr.typ.input) + 1):
            if expr.typ.input[-i:] == arg.typ:
                num_inputs = i
                break
        if num_inputs == 0:
            raise TypeError(f"Type of {arg} is not compatible "
                            + f"with the input type of {expr}")
        expr.typ = expr.typ.input[-i:] >> \
                   (expr.typ.input[:-num_inputs] >> expr.typ.output)
        return Expr.apply(expr, arg, context)


    @staticmethod
    def ccg_to_expr(ccg_parse):
        children = [Expr.ccg_to_expr(child) for child in ccg_parse.children]

        result = None
        # Rules with 0 children
        if ccg_parse.rule == CCGRule.LEXICAL:
            closed_type = biclosed_to_closed(ccg_parse.biclosed_type)
            result = Expr.literal(ccg_parse.text, closed_type)

        # Rules with 1 child
        elif ccg_parse.rule == CCGRule.FORWARD_TYPE_RAISING \
                or ccg_parse.rule == CCGRule.BACKWARD_TYPE_RAISING:
            x = Expr.literal(f"x{time.time()}", biclosed_to_closed(ccg_parse.biclosed_type).input)
            result = Expr.lmbda(x, x(children[0]))
        elif ccg_parse.rule == CCGRule.UNARY:
            result = change_expr_typ(children[0], biclosed_to_closed(ccg_parse.biclosed_type))

        # Rules with 2 children
        elif ccg_parse.rule == CCGRule.FORWARD_APPLICATION:
            result = children[0](children[1])
        elif ccg_parse.rule == CCGRule.BACKWARD_APPLICATION:
            result = children[1](children[0])
        elif ccg_parse.rule == CCGRule.FORWARD_COMPOSITION \
                or ccg_parse.rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
            x = Expr.literal(f"temp", biclosed_to_closed(
                ccg_parse.children[1].biclosed_type).input)
            result = Expr.lmbda(x, children[0](children[1](x)))
        elif ccg_parse.rule == CCGRule.BACKWARD_COMPOSITION \
                or ccg_parse.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            x = Expr.literal(f"temp", biclosed_to_closed(
                ccg_parse.children[0].biclosed_type).input)
            result = Expr.lmbda(x, children[1](children[0](x)))
        elif ccg_parse.rule == CCGRule.CONJUNCTION:
            left, right = children[0].typ, children[1].typ
            if CCGAtomicType.conjoinable(left):
                type = right >> biclosed_to_closed(ccg_parse.biclosed_type)
                children[0].typ = type
                result = children[0](children[1])
            elif CCGAtomicType.conjoinable(right):
                type = left >> biclosed_to_closed(ccg_parse.biclosed_type)
                children[1].typ = type
                result = children[1](children[0])
        elif ccg_parse.rule == CCGRule.REMOVE_PUNCTUATION_RIGHT:
            result = children[0]
        elif ccg_parse.rule == CCGRule.REMOVE_PUNCTUATION_LEFT:
            result = children[1]

        if result is None:
            raise NotImplementedError(ccg_parse.rule)

        if ccg_parse.original.cat.var in ccg_parse.original.var_map.keys():
            result.head = ccg_parse.original.variable.fillers
        else:
            result.head = None

        return result

