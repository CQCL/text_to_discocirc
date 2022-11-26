from __future__ import annotations

import time

from lambeq import CCGRule, CCGAtomicType

from discocirc.closed import Func, biclosed_to_closed, uncurry_types, Ty


class Expr:
    def __repr__(self):
        if self.expr_type == "literal":
            name = str(self.name)
            final_type = str(self.final_type)
            length = max(len(name), len(final_type))
            string = f'{name:^{length}}' + '\n'
            string += '═' * length + '\n'
            string += f'{final_type:^{length}}'
            return string
        elif self.expr_type == "lambda":
            return f'(λ{self.var}.{self.expr}):{self.final_type}'
        elif self.expr_type == "application":
            expr = str(self.expr)
            arg = str(self.arg)
            final_type = str(self.final_type)
            expr_lines = expr.split('\n')
            arg_lines = arg.split('\n')
            empty_arg_lines = [' ' * len(max(arg_lines))] * (len(expr_lines) - len(arg_lines))
            arg_lines = empty_arg_lines + arg_lines
            empty_expr_lines = [' ' * len(max(expr_lines))] * (len(arg_lines) - len(expr_lines))
            expr_lines = empty_expr_lines + expr_lines
            string = ['  '.join([expr_l, arg_l]) for expr_l, arg_l in zip(expr_lines, arg_lines)]
            string.append('─' * len(string[0]))
            string.append(f'{final_type:^{len(string[0])}}')
            return '\n'.join(string)
        elif self.expr_type == "list":
            return f'{self.expr_list}:{self.final_type}'
        else:
            raise NotImplementedError(self.expr_type)

    def __call__(self, arg: Expr):
        return Expr.apply(self, arg)
    
    def __members(self):
        if self.expr_type == "literal":
            return (self.expr_type,
                    self.simple_type,
                    self.final_type,
                    self.name)
        elif self.expr_type == "lambda":
            return (self.expr_type,
                    self.simple_type,
                    self.final_type,
                    self.var,
                    self.expr)
        elif self.expr_type == "application":
            return (self.expr_type,
                    self.simple_type,
                    self.final_type,
                    self.arg,
                    self.expr)
        elif self.expr_type == "list":
            return (self.expr_type,
                    self.simple_type,
                    self.final_type,
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
    def literal(name, simple_type):
        expr = Expr()
        expr.expr_type = "literal"
        expr.simple_type = simple_type
        expr.final_type = simple_type
        expr.name = name
        return expr

    @staticmethod
    def lmbda(var, expr, simple_type=None):
        lambda_expr = Expr()
        lambda_expr.expr_type = "lambda"
        lambda_expr.var = var
        lambda_expr.expr = expr
        lambda_expr.simple_type = simple_type
        if simple_type == None:
            lambda_expr.simple_type = expr.simple_type
        lambda_expr.final_type = var.final_type >> expr.final_type
        return lambda_expr
    
    @staticmethod
    def application(expr, arg):
        if expr.final_type.input != arg.final_type:
            raise TypeError(f"Type of {arg} does not"
                            + f"match the input type of {expr}")
        app_expr = Expr()
        app_expr.expr_type = "application"
        app_expr.simple_type = expr.simple_type
        app_expr.final_type = expr.final_type.output
        app_expr.expr = expr
        app_expr.arg = arg
        return app_expr
    
    @staticmethod
    def lst(expr_list, simple_type=None, interchange=True):
        expr = Expr()
        expr.expr_type = "list"
        expr.simple_type = simple_type
        new_expr_list = []
        for e in expr_list:
            if e.expr_type == "list":
                new_expr_list.extend(e.expr_list)
            else:
                new_expr_list.append(e)
        expr.expr_list = tuple(new_expr_list)
        expr.final_type = Expr.infer_list_type(expr_list, interchange)
        return expr
    
    @staticmethod
    def infer_list_type(expr_list, interchange):
        if interchange:
            final_input = Ty()
            final_output = Ty()
            for e in expr_list:
                f = uncurry_types(e.final_type)
                if isinstance(e.final_type, Func):
                    final_input = final_input @ f.input
                    final_output = final_output @ f.output
                else:
                    final_output = final_output @ f
            final_type = final_input >> final_output
        else:
            final_type = Ty()
            for e in expr_list:
                final_type = final_type @ e.final_type
        return final_type
    
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
            return Expr.lst([Expr.evl(context, e) for e in expr.expr_list], expr.simple_type)
        else:
            raise TypeError(f'Unknown type {expr.expr_type} of expression')

    @staticmethod
    def apply(expr, arg, context=None):
        if expr.final_type.input != arg.final_type:
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
        for i in range(len(expr.final_type.input)+1):
            if expr.final_type.input[:i] == arg.final_type:
                num_inputs = i
                break
        if num_inputs == 0:
            raise TypeError(f"Type of {arg} is not compatible "
                            + f"with the input type of {expr}")
        f_type = expr.final_type
        expr.final_type = f_type.input[:num_inputs] >> (f_type.input[num_inputs:] >> f_type.output)
        arg.final_type = arg.final_type
        return Expr.apply(expr, arg, context=None)
            

    @staticmethod
    def biclosed_to_expr(diagram):
        terms = []
        for box, offset in zip(diagram.boxes, diagram.offsets):
            if not box.dom:  # is word
                simple_type = biclosed_to_closed(box.cod)
                terms.append(Expr.literal(box.name, simple_type))
            else:
                if len(box.dom) == 2:
                    if box.name.startswith("FA"):
                        term = terms[offset](terms[offset + 1])
                    elif box.name.startswith("BA"):
                        term = terms[offset + 1](terms[offset])
                    elif box.name.startswith("FC"):
                        x = Expr.literal("temp", terms[offset + 1].final_type.input)
                        term = Expr.lmbda(x, terms[offset](terms[offset + 1](x)))
                    elif box.name.startswith("BC") or box.name.startswith(
                            "BX"):
                        x = Expr.literal("temp",
                                         terms[offset].final_type.input)
                        term = Expr.lmbda(x,
                                          terms[offset + 1](terms[offset](x)))
                    else:
                        raise NotImplementedError
                    # term.final_type = biclosed_to_closed(box.cod)
                    terms[offset:offset + 2] = [term]
                elif box.name == "Curry(BA(n >> s))":
                    x = Expr.literal("temp", Ty('n') >> Ty('s'))
                    terms[offset] = Expr.lmbda(x, x(terms[offset]))
                else:
                    raise NotImplementedError
        return terms[0]

    @staticmethod
    def ccg_to_expr(ccg_parse):
        children = [Expr.ccg_to_expr(child) for child in ccg_parse.children]

        result = None
        # Rules with 0 children
        if ccg_parse.rule == CCGRule.LEXICAL:
            closed_type = biclosed_to_closed(ccg_parse.biclosed_type)
            result = Expr.literal(ccg_parse.text, closed_type)

        # Rules with 1 child
        elif ccg_parse.rule == CCGRule.FORWARD_TYPE_RAISING:
            x = Expr.literal(f"temp{time.time()}", biclosed_to_closed(ccg_parse.biclosed_type).input)
            result = Expr.lmbda(x, x(children[0]))
        elif ccg_parse.rule == CCGRule.UNARY:
            if children[0].final_type != biclosed_to_closed(ccg_parse.biclosed_type):
                raise NotImplementedError("Changing types for UNARY rules")
            result = children[0]

        # Rules with 2 children
        elif ccg_parse.rule == CCGRule.FORWARD_APPLICATION:
            result = children[0](children[1])
        elif ccg_parse.rule == CCGRule.BACKWARD_APPLICATION:
            result = children[1](children[0])
        elif ccg_parse.rule == CCGRule.FORWARD_COMPOSITION:
            x = Expr.literal("temp", biclosed_to_closed(ccg_parse.children[1].biclosed_type).input)
            result = Expr.lmbda(x, children[0](children[1](x)))
        elif ccg_parse.rule == CCGRule.BACKWARD_COMPOSITION\
                or ccg_parse.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            x = Expr.literal("temp", biclosed_to_closed(
                ccg_parse.children[0].biclosed_type).input)
            result = Expr.lmbda(x, children[1](children[0](x)))
        elif ccg_parse.rule == CCGRule.CONJUNCTION:
            left, right = children[0].final_type, children[1].final_type
            if CCGAtomicType.conjoinable(left):
                type = right >> biclosed_to_closed(ccg_parse.biclosed_type)
                children[0].simple_type = type
                children[0].final_type = type
                result = children[0](children[1])
            elif CCGAtomicType.conjoinable(right):
                type = left >> biclosed_to_closed(ccg_parse.biclosed_type)
                children[1].simple_type = type
                children[1].final_type = type
                result = children[1](children[0])

        if result is None:
            raise NotImplementedError(ccg_parse.rule)

        if ccg_parse.original.cat.var in ccg_parse.original.var_map.keys():
            result.head = ccg_parse.original.variable.fillers
        else:
            result.head = None

        return result
    
