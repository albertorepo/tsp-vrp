class Problem:

    def _select_neighbor(self, *args, **kwargs):
        raise NotImplementedError("`neighborhood` method has not been implemented.")

    def _evaluate_solution(self, *args, **kwargs):
        raise NotImplementedError("`evaluate_solution` method has not been implemented.")

    def _get_initial_solution(self, *args, **kwargs):
        raise NotImplementedError("`_get_initial_strategy` method has not been implemented.")

    def run(self,*args, **kwargs):
        raise NotImplementedError()


