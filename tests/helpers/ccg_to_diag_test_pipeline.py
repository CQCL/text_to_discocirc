from discocirc.diag import Frame
from discocirc.expr import expr_to_diag, pull_out
from discocirc.expr.ccg_to_expr import ccg_to_expr
from discocirc.expr.normal_form import normal_form
from discocirc.expr.type_check import type_check
from discocirc.expr.coordination_expand import coordination_expand
from discocirc.expr.n_type_expand import n_type_expand
from discocirc.expr.s_type_expand import s_type_expand, p_type_expand
from discocirc.helpers.discocirc_utils import expr_add_indices_to_types
from discocirc.semantics.rewrite import rewrite
from discocirc.expr.uncurry import uncurry
from discocirc.expr.resolve_pronouns import expand_coref

# useful import for debugging
from discocirc.expr.to_discopy_diagram import expr_to_frame_diag


def ccg_to_diag_test(unittest, config, ccg_parse, sentence=None, spacy_model=None):
    """
    Run the pipeline from start to end. Depending on the variables in the
    config, various assertions are made.
    """
    # ------- Step 1: CCG to Expr -----------
    expr = ccg_to_expr(ccg_parse)
    if config["type_check_ccg"]:
        unittest.assertTrue(type_check(expr), msg="Typechecking ccg to expr")

    if config["draw_steps"]:
        diag = expr_to_diag(expr_add_indices_to_types(expr))
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 2: Pulling out -----------
    expr = pull_out(expr)
    if config["type_check_ccg"]:
        unittest.assertTrue(type_check(expr),
                            msg="Typechecking pulled out expr")

    # TODO: write test to check that all types have been pulled out
    #  correctly (Issue #14)

    if config["draw_steps"]:
        diag = expr_to_diag(expr_add_indices_to_types(expr))
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 3: Coordination expansion -----------
    expr = normal_form(expr)
    expr = coordination_expand(expr)

    if config["type_check_ccg"]:
        unittest.assertTrue(type_check(expr),
                            msg="Typechecking coordination expanded expr")

    if config["draw_steps"]:
        diag = expr_to_diag(expr_add_indices_to_types(expr))
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 4: Pulling out again -----------
    expr = pull_out(expr)
    if config["type_check_ccg"]:
        unittest.assertTrue(type_check(expr),
                            msg="Typechecking pulled out expr")

    if config["draw_steps"]:
        diag = expr_to_diag(expr_add_indices_to_types(expr))
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 5: N-type expansion -----------
    expr = n_type_expand(expr)

    if config["type_check_ccg"]:
        unittest.assertTrue(type_check(expr),
                            msg="Typechecking n_type expanded expr")

    if config["draw_steps"]:
        diag = expr_to_diag(expr_add_indices_to_types(expr))
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 6: P-type expansion -----------
    expr = p_type_expand(expr)

    if config["type_check_ccg"]:
        unittest.assertTrue(type_check(expr),
                            msg="Typechecking s_type expanded expr")

    if config["draw_steps"]:
        diag = expr_to_diag(expr_add_indices_to_types(expr))
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 6: S-type expansion -----------
    expr = s_type_expand(expr)

    if config["type_check_ccg"]:
        unittest.assertTrue(type_check(expr),
                            msg="Typechecking s_type expanded expr")

    # TODO: write test to check that all types have been expanded
    #  correctly (Issue #14)

    if config["draw_steps"]:
        diag = expr_to_diag(expr_add_indices_to_types(expr))
        diag = (Frame.get_decompose_functor())(diag)
        diag.draw()

    # ------- Step 7: Co-ref expansion -----------
    if config["coreference_resolution"]:
        if spacy_model and sentence:
            doc = spacy_model(sentence)
            expr = expand_coref(expr, doc)

            print(doc._.coref_chains)

            if config["draw_steps"]:
                diag = expr_to_diag(expr_add_indices_to_types(expr))
                diag = (Frame.get_decompose_functor())(diag)
                diag.draw()
        else:
            print("Warning: Spacy model and sentence not provided. "
                  "Skipping coreference resolution.")

    # ------- Step 8: Expr to Diag -----------
    diag = expr_to_diag(expr_add_indices_to_types(expr))

    # ------- Step 9: Semantic rewrites -----------
    if config["semantic_rewrites"]:
        diag = rewrite(diag, rules='all')
    diag = (Frame.get_decompose_functor())(diag)

    # expr_uncurried = expr_uncurry(expr)
    # diag_uncurried = expr_to_diag(expr_add_indices_to_types(expr_uncurried))
    # diag_uncurried = (Frame.get_decompose_functor())(diag_uncurried)
    #
    # if (diag != diag_uncurried):
    #     diag.draw(figsize=(10, 10))
    #     diag_uncurried.draw(figsize=(10, 10))
    #
    # unittest.assertEqual(diag, diag_uncurried)

    if config["draw_result"] or config["draw_steps"]:
        diag.draw(figsize=(10, 10))

    return True