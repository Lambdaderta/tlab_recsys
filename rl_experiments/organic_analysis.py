"""Organic (user-chosen) vs pushed (recommender-chosen) listens on Yambda-50M, all users.

uv run python organic_analysis.py   -> out/organic_analysis.json + printed tables
"""
import json

import polars as pl

from common import DAY, DATA, OUT, load_events

ARTIST = DATA.parent.parent / 'artist_item_mapping.parquet'
BURN_IN = 30 * DAY  # "first contact" before day 30 may just be the start of the log (left-censoring)
RETURN_WINDOWS = (7, 30)


def feedback_mix(ev):
    return (ev.group_by('is_organic').agg(
        n=pl.len(),
        skip=(pl.col('fb') == 0).mean(), plus=(pl.col('fb') == 1).mean(),
        like=(pl.col('fb') == 2).mean(), dislike=(pl.col('fb') == 3).mean())
        .sort('is_organic'))


def repeats_and_popularity(ev):
    pop = ev.group_by('item_id').agg(item_pop=pl.len())
    ev = (ev.join(pop, on='item_id')
          .with_columns(first=pl.col('timestamp').rank('ordinal').over('uid', 'item_id') == 1))
    return (ev.group_by('is_organic').agg(
        repeat_share=(~pl.col('first')).mean(),
        median_item_pop=pl.col('item_pop').median(),
        median_item_pop_first=pl.col('item_pop').filter(pl.col('first')).median())
        .sort('is_organic')), ev


def discoveries(ev):
    """First-ever listens of an item after the burn-in: who introduced it, and how did it go."""
    first = ev.filter(pl.col('first') & (pl.col('timestamp') >= BURN_IN))
    return (first.group_by('is_organic').agg(
        n=pl.len(), plus_or_like=pl.col('fb').is_in([1, 2]).mean(), skip=(pl.col('fb') == 0).mean())
        .with_columns(share=pl.col('n') / pl.col('n').sum()).sort('is_organic'))


def artist_returns(ev, t_end):
    """First contact with an artist (after burn-in) -> later ORGANIC listen of the same artist within w days."""
    art = pl.read_parquet(ARTIST).unique('item_id')
    ev = ev.join(art, on='item_id', how='inner').sort('uid', 'timestamp')
    first = (ev.group_by('uid', 'artist_id').agg(
        t0=pl.col('timestamp').first(), org0=pl.col('is_organic').first(), fb0=pl.col('fb').first(),
        item0=pl.col('item_id').first()))
    org_later = ev.filter(pl.col('is_organic') == 1).select('uid', 'artist_id', pl.col('timestamp').alias('t_org'), pl.col('item_id').alias('item_org'))
    # user type: tercile of the user's overall organic share (radio-heavy / mixed / library-heavy)
    utype = (ev.group_by('uid').agg(org_share=pl.col('is_organic').mean())
             .with_columns(utype=pl.col('org_share').qcut(3, labels=['radio', 'mixed', 'library'])))
    out = {}
    for w in RETURN_WINDOWS:
        for gap in (0, 1):  # gap = 1: returns in the same day (session continuation) do not count
            f = first.filter((pl.col('t0') >= BURN_IN) & (pl.col('t0') <= t_end - w * DAY))
            ret = (f.join(org_later, on=['uid', 'artist_id'], how='left')
                   .with_columns(hit=(pl.col('t_org') > pl.col('t0') + gap * DAY) & (pl.col('t_org') <= pl.col('t0') + w * DAY))
                   .group_by('uid', 'artist_id', 'org0', 'fb0').agg(
                       ret=pl.col('hit').any().cast(pl.Float64),
                       # same track (e.g. a liked track replayed from the collection) vs other tracks of the artist
                       ret_same=(pl.col('hit') & (pl.col('item_org') == pl.col('item0'))).any().cast(pl.Float64),
                       ret_other=(pl.col('hit') & (pl.col('item_org') != pl.col('item0'))).any().cast(pl.Float64))
                   .with_columns(user_rate=pl.col('ret').mean().over('uid'))
                   .join(utype.select('uid', 'utype'), on='uid'))
            # excess = cell rate minus the same user's overall rate, averaged over that cell's contacts
            tab = (ret.group_by('org0', 'fb0').agg(n=pl.len(), organic_return=pl.col('ret').mean(),
                                                   same_track=pl.col('ret_same').mean(), other_tracks=pl.col('ret_other').mean(),
                                                   within_user_excess=(pl.col('ret') - pl.col('user_rate')).mean())
                   .sort('org0', 'fb0'))
            out[f'{w}d_gap{gap}d'] = tab
            if w == 7 and gap == 1:
                out['7d_gap1d_pushed_by_user_type'] = (
                    ret.filter(pl.col('org0') == 0).group_by('utype', 'fb0')
                    .agg(n=pl.len(), organic_return=pl.col('ret').mean()).sort('utype', 'fb0'))
    return out


def user_and_time(ev):
    per_user = ev.group_by('uid').agg(org=pl.col('is_organic').mean(), n=pl.len())
    q = per_user.select([pl.col('org').quantile(x).alias(f'q{int(x * 100)}') for x in (0.1, 0.25, 0.5, 0.75, 0.9)])
    weekly = (ev.with_columns(week=(pl.col('timestamp') // (7 * DAY)))
              .group_by('week').agg(org=pl.col('is_organic').mean(), n=pl.len()).sort('week'))
    return q, weekly


if __name__ == '__main__':
    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(20)
    ev = load_events(10_000)
    t_end = int(ev['timestamp'].max())
    res = {}
    fm = feedback_mix(ev); print('feedback mix by is_organic\n', fm); res['feedback_mix'] = fm.to_dicts()
    rp, ev = repeats_and_popularity(ev); print('repeats / popularity\n', rp); res['repeats'] = rp.to_dicts()
    dc = discoveries(ev); print('first-ever item listens after day 30\n', dc); res['discoveries'] = dc.to_dicts()
    q, weekly = user_and_time(ev)
    print('per-user organic share quantiles\n', q); res['user_org_quantiles'] = q.to_dicts()[0]
    print('weekly organic share (first/last 5)\n', weekly.head(5), weekly.tail(5)); res['weekly'] = weekly.to_dicts()
    for w, tab in artist_returns(ev, t_end).items():
        print(f'first artist contact -> organic return, {w} (org0: how contact happened; fb0: reaction)\n', tab)
        res[f'artist_return_{w}'] = tab.to_dicts()
    with open(OUT / 'organic_analysis.json', 'w') as f:
        json.dump(res, f, indent=1)
