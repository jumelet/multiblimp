import pandas as pd
import os


def score_tse(model, fn: str):
    tse_df = pd.read_csv(fn, sep='\t')

    tse_df["sen_prob"] = pd.Series(dtype=object).astype(object)
    tse_df["wrong_prob"] = pd.Series(dtype=object).astype(object)

    max_length = None  # ilm_model.model.transformer.config.n_ctx

    for idx, row in tse_df.iterrows():
        sen_prob, wrong_prob = score_pair(model, row.sen, row.wrong_sen, max_length)

        sen_nll = -sen_prob.sum().item()
        wrong_nll = -wrong_prob.sum().item()

        tse_df.at[idx, "sen_prob"] = sen_prob.tolist()
        tse_df.at[idx, "wrong_prob"] = wrong_prob.tolist()

        tse_df.loc[idx, "sen_nll"] = sen_nll
        tse_df.loc[idx, "wrong_nll"] = wrong_nll
        tse_df.loc[idx, "delta"] = wrong_nll - sen_nll

    return tse_df


def score_pair(ilm_model, sen, wrong_sen, max_length):
    sen_len = len(ilm_model.tokenizer.tokenize(sen))
    wrong_sen_len = len(ilm_model.tokenizer.tokenize(wrong_sen))

    if (max_length is not None) and ((sen_len >= max_length) or (wrong_sen_len >= max_length)):
        return 0., 0.

    stimuli = [sen, wrong_sen]

    return ilm_model.sequence_score(stimuli, reduction=lambda x: x)
