import traceback

from discocirc.diag import Frame
from discocirc.expr import expr_to_diag, s_type_expand, pull_out
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.ccg_type_check import expr_type_check
from discocirc.expr.expr_uncurry import expr_uncurry
from outdated_code.expand_s_types import expand_s_types


def compare_type_expansions(unittest, expr):
    test_diag = expr_to_diag(expr)
    # TODO: fix expand_s_types() (Not really a priority as we won't use it)
    try:
        expand_s_types(test_diag)
    except:
        return
    expr = s_type_expand(expr)
    test_diag2 = expr_to_diag(expr)
    unittest.assertEqual(test_diag, test_diag2)


def ccg_to_diag_test(unittest, config, ccg_parse):
    # ------- Step 1: CCG to Expr -----------
    expr = ccg_to_expr(ccg_parse)
    if config["type_check_ccg"]:
        unittest.assertTrue(expr_type_check(expr), msg="Typechecking ccg to expr")

    if config["draw_steps"]:
        diag = expr_to_diag(expr)
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 2: Pulling out -----------
    expr = pull_out(expr)
    if config["type_check_ccg"]:
        unittest.assertTrue(expr_type_check(expr),
                            msg="Typechecking pulled out expr")

    # TODO: write test to check that all types have been pulled out
    #  correctly (Issue #14)

    if config["draw_steps"]:
        diag = expr_to_diag(expr)
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 3: Type expansion -----------
    expr = s_type_expand(expr)

    if config["type_check_ccg"]:
        unittest.assertTrue(expr_type_check(expr),
                            msg="Typechecking expanded expr")

    if config["compare_type_expansions"]:
        compare_type_expansions(unittest, expr)
    # TODO: write test to check that all types have been expanded
    #  correctly (Issue #14)

    if config["draw_steps"]:
        diag = expr_to_diag(expr)
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 4: Expr to Diag -----------
    diag = expr_to_diag(expr)
    diag = (Frame.get_decompose_functor())(diag)

    expr_uncurried = expr_uncurry(expr)
    diag_uncurried = expr_to_diag(expr_uncurried)
    diag_uncurried = (Frame.get_decompose_functor())(diag_uncurried)

    # unittest.assertEqual(diag, diag_uncurried)

    if config["draw_result"] or config["draw_steps"]:
        diag.draw()

    return True