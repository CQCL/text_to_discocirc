from __future__ import annotations

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
        elif self.expr_type == "projection":
            return f'[{self.expr}]_{self.projection_index}:{self.final_type}'
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
    def projection(expr, projection_index):
        proj_expr = Expr()
        proj_expr.expr_type = "projection"
        inputs = []
        output = expr.final_type
        while isinstance(output, Func):
            inputs.append(output.input)
            output = output.output
        final_type = Ty(output[projection_index])
        for i in inputs[::-1]:
            final_type = i >> final_type
        proj_expr.final_type = final_type
        proj_expr.expr = expr
        proj_expr.projection_index = projection_index
        return proj_expr
    
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
        elif expr.expr_type == "projection":
            return Expr.projection(Expr.evl(context, expr.expr), expr.projection_index)
        else:
            raise TypeError(f'Unknown type {expr.expr_type} of expression')

    @staticmethod
    def apply(expr, arg, context=None):
        if expr.final_type.input != arg.final_type:
            raise TypeError(f"Type of {arg} does not match the input type of {expr}")
        if expr.expr_type == "lambda":
            if context == None:
                context = {}
            if expr.var.expr_type == "list":
                for var, val in zip(expr.var.expr_list, arg.expr_list):
                    context[var] = val
            else:
                context[expr.var] = arg
            return Expr.evl(context, expr.expr)
        if expr.expr_type == "list":
            raise NotImplemented(f'Implement list application.')
        elif expr.expr_type == "projection":
            return Expr.projection(Expr.apply(expr.expr, arg, context), expr.projection_index)
        else:
            new_expr = Expr.application(expr, arg)
            return new_expr
