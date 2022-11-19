from __future__ import annotations

from lambeq import CCGRule

from discocirc import closed
from discocirc.closed import uncurry_types, Ty


class Expr:
    def __repr__(self):
        if self.expr_type == "literal":
            return f'{self.name}:{self.final_type}'
        elif self.expr_type == "lambda":
            return f'(λ{self.var}.{self.expr}):{self.final_type}'
        elif self.expr_type == "application":
            return f'({self.expr} {self.arg}):{self.final_type}'
        elif self.expr_type == "list":
            return f'{self.expr_list}:{self.final_type}'

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
    def lmbda(literal, expr, simple_type=None):
        lambda_expr = Expr()
        lambda_expr.expr_type = "lambda"
        lambda_expr.var = literal
        lambda_expr.expr = expr
        lambda_expr.simple_type = simple_type
        if simple_type == None:
            lambda_expr.simple_type = expr.simple_type
        lambda_expr.final_type = literal.final_type >> expr.final_type
        return lambda_expr
    
    @staticmethod
    def application(expr, arg):
        app_expr = Expr()
        app_expr.expr_type = "application"
        app_expr.simple_type = expr.simple_type
        app_expr.final_type = expr.final_type.output
        app_expr.expr = expr
        app_expr.arg = arg
        return app_expr
    
    @staticmethod
    def lst(expr_list, simple_type=None):
        expr = Expr()
        expr.expr_type = "list"
        expr.simple_type = simple_type
        new_expr_list = []
        final_type = Ty()
        for e in expr_list:
            final_type = final_type @ e.final_type
            if e.expr_type == "list":
                new_expr_list.extend(e.expr_list)
            else:
                new_expr_list.append(e)
        expr.expr_list = tuple(new_expr_list)
        expr.final_type = final_type
        return expr
    
    @staticmethod
    def uncurry(expr):
        if expr.expr_type == "literal":
            return Expr.literal(expr.name, uncurry_types(expr.final_type))
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
                return c(Expr.lst([a, b]))
            else:
                arg = Expr.uncurry(expr.arg)
                expr = Expr.uncurry(expr.expr)
                return expr(arg)
        elif expr.expr_type == "list":
            return Expr.lst([Expr.uncurry(e) for e in expr.expr_list], expr.simple_type)
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
            raise TypeError(f"Type of {arg} does not"
                            + f"match the input type of {expr}")
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
    def biclosed_to_expr(diagram):
        terms = []
        for box, offset in zip(diagram.boxes, diagram.offsets):
            if not box.dom:  # is word
                simple_type = closed.biclosed_to_closed(box.cod)
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
                    # term.final_type = closed.biclosed_to_closed(box.cod)
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
            closed_type = closed.biclosed_to_closed(ccg_parse.biclosed_type)
            result = Expr.literal(ccg_parse.text, closed_type)

        # Rules with 1 child
        # elif ccg_parse.rule == CCGRule.FORWARD_TYPE_RAISING:
        #     result = TR(children[0])
        elif ccg_parse.rule == CCGRule.UNARY:
            result = children[0]

        # Rules with 2 children
        elif ccg_parse.rule == CCGRule.FORWARD_APPLICATION:
            result = children[0](children[1])
        elif ccg_parse.rule == CCGRule.BACKWARD_APPLICATION:
            result = children[1](children[0])
        elif ccg_parse.rule == CCGRule.FORWARD_COMPOSITION:
            x = Expr.literal("temp", closed.biclosed_to_closed(ccg_parse.children[1].biclosed_type).input)
            result = Expr.lmbda(x, children[0](children[1](x)))
        elif ccg_parse.rule == CCGRule.BACKWARD_COMPOSITION\
                or ccg_parse.rule == CCGRule.BACKWARD_CROSSED_COMPOSITION:
            x = Expr.literal("temp", closed.biclosed_to_closed(
                ccg_parse.children[0].biclosed_type).input)
            result = Expr.lmbda(x, children[1](children[0](x)))
        elif ccg_parse.rule == CCGRule.CONJUNCTION:
            assert (len(children) == 2)

            if (ccg_parse.children[0].biclosed_type == Ty('conj')):
                children[0].final_type = children[1].final_type \
                                         >> (children[1].final_type \
                                         >> children[1].final_type)
                children[0].simple_type = children[0].final_type
                result = children[0](children[1])
            elif ccg_parse.children[1].biclosed_type == Ty('conj'):
                children[1].final_type = children[0].final_type \
                                         >> (children[0].final_type \
                                         >> children[0].final_type)
                children[1].simple_type = children[1].final_type
                result = children[1](children[0])
            else:
                raise RuntimeError("This should not happen. Could someone explain conjunctions to me?!?")

            # TODO: does not seem to have a head
            return result


        if result is None:
            raise NotImplementedError(ccg_parse.rule)

        result.head = ccg_parse.original.variable.fillers
        return result