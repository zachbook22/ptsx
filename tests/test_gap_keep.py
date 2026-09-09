def test_short_utterance_between_long_turns_is_cut():
    import numpy as np
    from ptsx.turns import assign_turns, TurnParams

    hop = 0.032
    n = 200
    user = np.zeros(n, dtype=bool)
    user[10:40] = True  # ~0.96 s
    user[120:160] = True  # ~1.28 s
    asst = np.zeros(n, dtype=bool)
    asst[70:85] = True  # ~0.48 s one-word between longer turns
    result = assign_turns(user, asst, hop, TurnParams())
    assert not result.keep_asst[72:80].any()
    assert not np.any(result.keep_user & result.keep_asst)
