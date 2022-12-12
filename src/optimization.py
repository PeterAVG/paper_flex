from datetime import timedelta
from time import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.admm import ADMM
from src.base import OBJECTIVE_FUNCTION, SCENARIO_PATH, Case
from src.evaluation import average_opt_results, evaluate_oos
from src.experiment_manager.cache import cache, load_cache
from src.prepare_problem import (
    Problem,
    build_oos_scenarios,
    find_rp_price,
    get_arbitrary_scenarios,
    get_chunk_instance,
    get_scenarios_from_lookback,
    get_variables_and_params,
    sample_lambda_rp,
)
from src.problem import SolverInstance
from src.receding_horizon import solve_receding_horizon_problem


def create_latex_tables_analysis1() -> None:

    ### TABLE 1: compare first 3 cases: SPOT vs mFRR_AND_ENERGY vs mFRR ###
    cache = load_cache()
    res = []
    # get all files in folder
    # r = [results for _params, results in cache.items()]
    for _params, results in cache.items():
        params = eval(_params)
        if "analysis1" == params["analysis"] and (
            (
                Case.SPOT.name in params["case"]
                or Case.mFRR_AND_ENERGY.name in params["case"]
            )
            # or Case.mFRR.name in params["case"]
            and not params.get("gamma", False)
            and not params.get("one_lambda", False)
        ):
            opt_result = results[1]
            if isinstance(opt_result, list) and len(opt_result) > 1:
                opt_result = average_opt_results(opt_result)
            year = params["year"]
            name = params["case"]
            admm = params.get("admm", False)

            name = name.replace(".", " ")
            one_lambda = "One bid" if params["one_lambda"] else ""
            name = (
                name + " " + one_lambda
                if Case.mFRR_AND_ENERGY.name in params["case"]
                else name
            )

            nb = params.get("nb_scenarios_spot", 1)
            print(nb)

            if nb > 0 or nb == -4:
                _nb = "All" if nb == -4 else nb
                res.append(
                    {
                        **{"Name": name},
                        **{"Year": year},
                        **opt_result.__dict__,
                        **{"Scenarios": _nb},
                        **{"ADMM": admm},
                    }
                )
                # print(f"\n{name}: \n{opt_result.__repr__()}")
        elif "analysis3" == params["analysis"] and (
            Case.SPOT.name in params["case"]
            or Case.mFRR_AND_ENERGY.name in params["case"]
            # or Case.mFRR.name in params["case"]
            and not params.get("gamma", False)
            and not params.get("one_lambda", False)
        ):
            opt_result = average_opt_results(results[1])
            name = "Receding horizon"
            one_lambda = "One bid" if params["one_lambda"] else ""
            name = (
                name + " " + one_lambda
                if Case.mFRR_AND_ENERGY.name in params["case"]
                else name
            )
            lookback = params["lookback"]
            year = params["year"]
            admm = params.get("admm", False)

            res.append(
                {
                    **{"Name": name},
                    **{"Year": year},
                    # **{"Lookback": lookback},
                    **opt_result.__dict__,
                    **{"Scenarios": -lookback},
                    **{"ADMM": admm},
                }
            )

    # create dataframe of all results in res. Each element is of type OptimizationResult
    df = pd.DataFrame(res).round(3)
    df.columns = [c.replace("_", " ").capitalize() for c in df.columns]
    df = df.sort_values(["Year", "Name"]).set_index(["Year", "Name"]).T
    df.drop(["Battery capacity"], inplace=True)

    df.loc["% savings"] = (
        (1 - df.loc["Total cost"].astype(float) / df.loc["Base cost today"])
        .astype(float)
        .round(3)
        .values
    ) * 100
    print(df[2021])
    print(df[2022])
    df_2022 = df[2022]
    df_2022.loc["Name", :] = df_2022.columns.tolist()
    table_compare_3 = df_2022.T.query(
        "(Name == 'SPOT') | (Scenarios == -5) | (Scenarios == 50)"
    ).T
    table_compare_3.rename({Case.mFRR_AND_ENERGY.name: "mFRR"}, inplace=True)
    table_compare_3.drop(["Name"], inplace=True)
    print(table_compare_3)
    print(table_compare_3)
    pass
    # # print df to latex
    # align_str = "".join(["c" for _ in range(len(df.columns))])
    # print(df.to_latex(multicolumn_format="c", column_format=f"l{align_str}"))

    # ## TABLE 2: compare next 3 cases: naive vs mFRR_AND_ENERGY vs robust ###
    # res = []
    # for _params, results in cache.items():
    #     params = eval(_params)

    #     if "analysis1" == params["analysis"] and (
    #         Case.NAIVE.name in params["case"]
    #         or Case.ROBUST.name in params["case"]
    #         or Case.mFRR_AND_ENERGY.name in params["case"]
    #     ):
    #         opt_result = results[1]
    #         year = params["year"]
    #         name = params["case"]
    #         name = name.replace(".", " ")
    #         one_lambda = "One bid" if params["one_lambda"] else ""
    #         name = (
    #             name + " " + one_lambda
    #             if (
    #                 Case.mFRR_AND_ENERGY.name in params["case"]
    #                 or Case.ROBUST.name in name
    #             )
    #             else name
    #         )

    #         res.append({**{"Name": name}, **{"Year": year}, **opt_result.__dict__})
    #         print(f"\n{name}: \n{opt_result.__repr__()}")

    # # create dataframe of all results in res. Each element is of type OptimizationResult
    # df = pd.DataFrame(res).round(1)
    # df.columns = [c.replace("_", " ").capitalize() for c in df.columns]
    # df = df.sort_values(["Year", "Name"]).set_index(["Year", "Name"]).T
    # df.drop(["Battery capacity"], inplace=True)

    # df.loc["% savings"] = (
    #     (1 - df.loc["Total cost"] / df.loc["Base cost today"]).round(2).values
    # )
    # print(df)

    # # print df to latex
    # align_str = "".join(["c" for _ in range(len(df.columns))])
    # print(df.to_latex(multicolumn_format="c", column_format=f"l{align_str}"))

    pass


def run_receding_horizon_optimization(params: Dict[str, Any]) -> Any:
    """
    Run receding horizon optimization for 2021-2022
    """
    assert params["analysis"] == "analysis3"
    assert params["year"] == 2022

    lookback = params["lookback"]
    one_lambda = params["one_lambda"]
    case: Case = Case[params["case"]]

    print(f"\n\nUsing {lookback} scenarios as lookback")

    df = pd.read_csv(
        SCENARIO_PATH,
        parse_dates=["HourUTC"],
    )
    dates_to_substract = (  # noqa
        df.groupby(df.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df = df.query("Date != @dates_to_substract").reset_index(drop=True)
    assert df.shape[0] % 24 == 0
    df["SpotPriceDKK"] = df["SpotPriceDKK"] / 1000
    df["mFRR_UpPriceDKK"] = df["mFRR_UpPriceDKK"] / 1000
    df["lambda_rp"] = df.apply(lambda x: find_rp_price(x), axis=1).values / 1000
    df["lambda_rp_diff"] = df["lambda_rp"] - df["SpotPriceDKK"]

    scenarios = get_scenarios_from_lookback(df, lookback)
    instance = get_chunk_instance(scenarios)

    instance.one_lambda = params["one_lambda"]
    instance.elafgift = params["elafgift"]
    instance.moms = params["moms"]
    instance.tariff = np.zeros(24)
    instance.delta_max = params["delta_max"]

    oos_dates = pd.to_datetime(df.query("Date >= '2022-01-01'").Date.unique())

    start = time()
    all_data = []
    print("Preparing data for receding horizon optimization")
    for omega, (is_instance, oos_instance) in enumerate(
        instance.lookback_generator(lookback)
    ):
        print(f"{omega + 1}", end="\r")
        assert is_instance.nb_scenarios == lookback
        assert oos_instance.nb_scenarios == 1
        assert omega + lookback < instance.nb_scenarios
        oos_date = oos_dates[omega]
        assert df.HourUTC.isin([oos_date]).any().any()

        # Sample balancing prices for in-sample instance so that it is more generally representative
        seed = omega
        df_date = df.query(f"HourUTC < '{oos_date}'")
        lambda_rp_diff = sample_lambda_rp(df_date, lookback, seed)
        lambda_rp = is_instance.lambda_rp + lambda_rp_diff
        up_regulation_event = (lambda_rp > is_instance.lambda_spot).astype(int)
        assert is_instance.lambda_spot.shape == lambda_rp.shape
        assert is_instance.lambda_spot.shape == up_regulation_event.shape
        # set is_balancing prices (and therefore also up-regulation events)
        is_instance.lambda_rp = lambda_rp
        is_instance.up_regulation_event = up_regulation_event

        # Assert in-sample and oos does not overlap
        assert all(
            [
                np.isclose(
                    round(df_date.SpotPriceDKK.values[-i], 3),
                    round(is_instance.lambda_spot[-1, -i], 3),
                )
                for i in range(1, 25)
            ]
        )
        assert all(
            [
                np.isclose(
                    round(
                        df.query(
                            f"HourUTC < '{oos_date + timedelta(days=1)}'"
                        ).SpotPriceDKK.values[-i],
                        3,
                    ),
                    round(oos_instance.lambda_spot[0, -i], 3),
                )
                for i in range(1, 25)
            ]
        )
        assert all(
            [
                np.isclose(
                    round(
                        df.query(
                            f"HourUTC < '{oos_date + timedelta(days=1)}'"
                        ).lambda_rp.values[-i],
                        3,
                    ),
                    round(oos_instance.lambda_rp[0, -i], 3),
                )
                for i in range(1, 25)
            ]
        )

        all_data.append(
            (
                SolverInstance.instance_to_dict(is_instance),
                SolverInstance.instance_to_dict(oos_instance),
                OBJECTIVE_FUNCTION[case.name],
                one_lambda,
            )
        )

    assert len(all_data) == len(oos_dates)
    print(f"Time to create all data: {time() - start:.2f} s")

    start = time()
    res = solve_receding_horizon_problem(all_data)
    print(f"Solving receding horizon problem took {time() - start} seconds")

    return res


@cache
def run_optimization(partition: str) -> Any:

    params: Dict[str, Any] = eval(partition)
    generic_solver_instance = SolverInstance()

    if params["analysis"] == "analysis3":
        res = run_receding_horizon_optimization(params)
        return res

    case: Case = Case[params["case"]]
    run_oos: bool = params["run_oos"]
    nb_scenarios_spot: int = params.get("nb_scenarios_spot", 1)

    df_scenarios = pd.read_csv(SCENARIO_PATH, parse_dates=["HourUTC"],).query(
        f"HourUTC.dt.year == {params['year']}"
        # f"HourUTC.dt.year == {params['year']} & HourUTC.dt.month == {1}"
    )
    # build uncertainty set for particular year-month combination
    # scenarios = build_uncertainty_set_v2(df_scenarios, nb=1, run_oos=run_oos)
    # scenarios = build_test_uncertainty_set(df_scenarios)
    # NOTE: in-sample scenarios are sampled arbitrarily from 2021
    # NOTE: out-of-sample scenarios are simply all days in 2022
    scenarios = (
        # build_uncertainty_set_v2(df_scenarios, nb=1, run_oos=run_oos)
        get_arbitrary_scenarios(df_scenarios, nb_scenarios_spot, seed=nb_scenarios_spot)
        if not run_oos
        else build_oos_scenarios(df_scenarios)
    )
    print(f"\n\nUsing {scenarios.lambda_spot.shape[0]} scenarios")

    instance = get_chunk_instance(scenarios)
    instance.one_lambda = params["one_lambda"]

    # Add tariffs and taxes
    instance.elafgift = params["elafgift"]
    instance.moms = params["moms"]
    instance.tariff = np.zeros(24)
    instance.delta_max = params["delta_max"]
    # Divide spot prices by 2 (Coop pays 50% spot price + a fixed price). Fixed price assumed to be 0.7 kr/kWh
    # instance.lambda_spot = scenarios.lambda_spot / 2 + 0.7 # only for tariff B

    if case.name == Case.SPOT.name:
        # old_lambda_rp = instance.lambda_rp.copy()
        # force model to always activate
        instance.lambda_rp = scenarios.lambda_spot + 10
        instance.up_regulation_event = np.ones(shape=instance.lambda_rp.shape)

    if case.name == Case.ROBUST.name and not run_oos:
        instance.convert_to_robust_scenario()

    # if params["analysis"] == "analysis2":
    #     instance.one_lambda = False

    if params.get("admm") and case.name == Case.mFRR_AND_ENERGY.name and not run_oos:
        print("Initializing ADMM")
        assert not params["one_lambda"]
        # Solve with ADMM by decomposing scenarios
        admm = ADMM(
            generic_solver_instance,
            instance,
            OBJECTIVE_FUNCTION[case.name],
            gamma=params.get("gamma", 0.5),
        )
        runs = admm.run_admm()
        res_inst, opt_result = admm.prepare_in_sample_results(runs[-1])

        if params.get("save_admm_iterations") is not None:
            return runs, res_inst, opt_result
    else:

        if not run_oos:
            # Solve normally in-sample
            # Create model instance and set apprioriate constraints
            print("Initializing model instance")
            problem = Problem(generic_solver_instance, instance)
            Problem.customize_constraints(problem.model_instance, instance.one_lambda)
            problem.set_objective(OBJECTIVE_FUNCTION[case.name])

            # if params["analysis"] == "analysis2":
            #     # full reserveation, but allow dynamic bids
            #     fix_p_up_reserve_only(problem, instance, deepcopy(params))

            opt_result = problem.solve()
            res_inst = get_variables_and_params(problem.res_instance)
        else:
            res_inst, opt_result = evaluate_oos(
                generic_solver_instance, instance, case, params
            )

        # if case.name == Case.SPOT.name:
        #     instance.lambda_rp = old_lambda_rp  # needed for plots

    print(opt_result.__repr__())

    return res_inst, opt_result
