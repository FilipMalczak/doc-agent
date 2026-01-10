#FIXME this is a wrong module name

import tenacity
from logfire import instrument
from tenacity import stop_after_attempt, sleep


def step(foo):
    return instrument()(foo)

def phase(attempts: int = 3, **kwargs):
    def decorator(foo):
        def _retry():
            x = dict(
                stop=stop_after_attempt(3),
                sleep=sleep(0.5)
            )
            x.update(
                kwargs
            )
            return tenacity.retry(
                **x
            )
        return _retry()(step(foo))
    return decorator



#
# def stage[**P, R](foo: Callable[[P], R]) -> Callable[[P], R]:
#     return default_retry()(
#         instrument()(
#             foo
#         )
#     )
#
# def agent_run[**P, R](foo: Callable[[P], R]) -> Callable[[P], R]:
#     return default_retry()(
#         instrument(
#             stop=stop_after_attempt(2)
#         )(
#             foo
#         )
#     )