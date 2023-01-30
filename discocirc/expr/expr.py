from __future__ import annotations

import time

from lambeq import CCGRule, CCGAtomicType

from discocirc.helpers.closed import Func, biclosed_to_closed, uncurry_types, Ty


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
            return f'(λ{self.var}.{self.expr}):{self.typ}'
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
            return f'{self.expr_list}:{self.typ}'
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

    def type_check(self):
        if self.expr_type == "literal":
            return self.typ

        elif self.expr_type == "list":
            expected_type = Ty()
            for expr in self.expr_list:
                element_type = expr.type_check()
                if element_type is None:
                    return None
                expected_type = expected_type @ element_type

            if self.typ == expected_type:
                return expected_type
            else:
                return None

        elif self.expr_type == "application":
            type_arg = self.arg.type_check()
            type_expr = self.expr.type_check()

            if type_arg is None or type_expr is None:
                return None

            if self.typ != type_expr.output:
                return None

            return self.typ
        elif self.expr_type == "lambda":
            type_var = self.var.type_check()
            type_expr = self.expr.type_check()

            if type_var is None or type_expr is None:
                return None

            if self.typ != type_var >> type_expr:
                return None

            return self.typ

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
    def uncurry(expr):
        if expr.expr_type == "literal":
            return Expr.literal(expr.name, uncurry_types(expr.final_type, uncurry_everything=True))
        elif expr.expr_type == "lambda":
            if expr.expr.expr_type == "lambda":
                # a -> b -> c = (a @ b) -> c
                a_b = Expr.lst([Expr.uncurry(expr.var),
                                Expr.uncurry(expr.expr.var)])
                c = Expr.uncurry(expr.expr.expr)
                simple_type = uncurry_types(expr.simple_type,
                                            uncurry_everything=True)
                return Expr.lmbda(a_b, c, simple_type)
            else:
                return Expr.lmbda(Expr.uncurry(expr.var),
                                  Expr.uncurry(expr.expr),
                                  uncurry_types(expr.simple_type,
                                                uncurry_everything=True))
        elif expr.expr_type == "application":
            if expr.expr.expr_type == "application":
                a = Expr.uncurry(expr.arg)
                b = Expr.uncurry(expr.expr.arg)
                c = Expr.uncurry(expr.expr.expr)
                return c(Expr.lst([b, a], interchange=False))
            else:
                arg = Expr.uncurry(expr.arg)
                expr = Expr.uncurry(expr.expr)
                return expr(arg)
        elif expr.expr_type == "list":
            return Expr.lst([Expr.uncurry(e) for e in expr.expr_list], expr.simple_type, interchange=False)
        else:
            raise TypeError(f'Unknown type {expr.expr_type} of expression')

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
            if expr.typ.input[:i] == arg.typ:
                num_inputs = i
                break
        if num_inputs == 0:
            raise TypeError(f"Type of {arg} is not compatible "
                            + f"with the input type of {expr}")
        f_type = expr.typ
        expr.typ = f_type.input[:num_inputs] >> (f_type.input[num_inputs:] >> f_type.output)
        arg.typ = arg.typ
        return Expr.apply(expr, arg, context=None)


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
            x = Expr.literal(f"temp{time.time()}", biclosed_to_closed(ccg_parse.biclosed_type).input)
            result = Expr.lmbda(x, x(children[0]))
        elif ccg_parse.rule == CCGRule.UNARY:
            if children[0].typ != biclosed_to_closed(ccg_parse.biclosed_type):
                raise NotImplementedError("Changing types for UNARY rules")
            result = children[0]

        # Rules with 2 children
        elif ccg_parse.rule == CCGRule.FORWARD_APPLICATION:
            result = children[0](children[1])
        elif ccg_parse.rule == CCGRule.BACKWARD_APPLICATION:
            result = children[1](children[0])
        elif ccg_parse.rule == CCGRule.FORWARD_COMPOSITION \
                or ccg_parse.rule == CCGRule.FORWARD_CROSSED_COMPOSITION:
            x = Expr.literal("temp", biclosed_to_closed(
                ccg_parse.children[1].biclosed_type).input)
            result = Expr.lmbda(x, children[0](children[1](x)))
        elif ccg_parse.rule == CCGRule.BACKWARD_COMPOSITION \
                or ccg_parse.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            x = Expr.literal("temp", biclosed_to_closed(
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
    
