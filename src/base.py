from enum import Enum
from typing import Any, Dict, Generator, List

from src.objective import _o_energy_cost, o_rule, o_rule_no_energy

SCENARIO_PATH = "data/scenarios_v2.csv"


class Case(Enum):
    mFRR_AND_ENERGY = "joint_mfrr_and_energy"
    mFRR = "mfrr_only"
    SPOT = "energy_only"
    NAIVE = "naive"  # No scenarios, rule based approach (mFRR_AND_ENERGY)
    ROBUST = "robust"  # single, worst case scenario (mFRR_AND_ENERGY)

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {
            "elafgift": 0.0,
            "moms": 0.0,
            "delta_max": 50,
            "analysis": "analysis1",
        }

    @classmethod
    def valid_cases(cls) -> List["Case"]:
        # return [case for case in self]
        return [cls.mFRR_AND_ENERGY, cls.SPOT]

    @classmethod
    def receding_horizon_cases(cls) -> Generator:
        lookbacks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for case in [cls.mFRR_AND_ENERGY]:
            for lookback in lookbacks:
                params = cls.default_params()
                params["analysis"] = "analysis3"
                params["one_lambda"] = False
                params["lookback"] = lookback
                params["case"] = case.name
                params["year"] = 2022
                yield params

    @classmethod
    def cases(cls) -> Generator:
        year = [2021, 2022]
        run_oos = [False, True]
        _admm = [False, True]
        one_lambda = [False]
        nb_scenarios = [1, 5, 10, 15, 20]
        nb_scenarios = [1, 5, 10, 20, 30, 40, 50, 100, 250]
        # nb_scenarios = [50]
        for case in cls:
            if case not in cls.valid_cases():
                continue
            for _run_oos, _year in zip(run_oos, year):
                if case.name == cls.mFRR_AND_ENERGY.name:
                    for _one_lambda in one_lambda:
                        params = cls.default_params()
                        params["case"] = case.name
                        params["run_oos"] = _run_oos
                        params["year"] = _year
                        params["one_lambda"] = _one_lambda
                        for admm in _admm:
                            params["admm"] = admm
                            if admm and not _one_lambda:
                                for _nb_scenarios in nb_scenarios:
                                    # For OOS, nb_scenarios_spot is used as key to cache
                                    # from IS results
                                    params["nb_scenarios_spot"] = _nb_scenarios
                                    yield params
                            elif admm and _one_lambda:
                                continue
                            else:
                                # NOTE: -4 indicates scenario where we use 20 scenarios as usual
                                # otherwise, we just run the optimization with '_nb_scenarios'
                                # scenarios without using ADMM.
                                for _nb_scenarios in [5, 10]:
                                    # When not using ADMM, nb_scenarios_spot is not used, hence -1
                                    params["nb_scenarios_spot"] = _nb_scenarios
                                    yield params
                else:
                    params = cls.default_params()
                    params["case"] = case.name
                    params["run_oos"] = _run_oos
                    params["year"] = _year
                    params["one_lambda"] = False  # does not matter for SPOT
                    if not _run_oos and case.name == cls.SPOT.name:
                        # NOTE: no policy from IS (2021) is used for OOS (2022) for SPOT
                        # i.e., we always reoptimize since it is "easy"
                        for _nb_scenarios in [1, 5, 10, 20, 30, 40, 50]:
                            params["nb_scenarios_spot"] = _nb_scenarios
                            yield params
                    else:
                        yield params


OBJECTIVE_FUNCTION = {
    Case.mFRR_AND_ENERGY.name: o_rule,
    Case.mFRR.name: o_rule_no_energy,
    Case.SPOT.name: _o_energy_cost,
    Case.NAIVE.name: o_rule,
    Case.ROBUST.name: o_rule,
}
