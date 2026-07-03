# Calibration Report - 2026-07-03

This report evaluates saved pre-kickoff 1X2 predictions against regular-time results only. It is evaluation-only and does not provide staking, trade execution, wallet, Kelly sizing, or investment advice.

Sample size: `2` usable evaluated prediction snapshot(s).
Usable evaluated predictions: `2`.
Excluded predictions and reasons: Not written to evaluation_dataset.csv; rows without completed 90-minute results or with ambiguous semantics are excluded by construction.

## Raw Model Performance

Raw model: n=2, Brier=0.4498, log loss=0.7929, top-pick accuracy=1.0000, ECE=0.3650.

## Calibrated Model Performance

No production-eligible calibrated probabilities are available in the evaluated dataset.

## Market Performance

Market benchmark sample too small for reliable conclusions.

## Walk-Forward Candidate Methods

Calibration sample too small; using conservative shrinkage.

| model | n | brier_score | log_loss | top_pick_accuracy | expected_calibration_error | brier_delta_vs_raw | log_loss_delta_vs_raw | production_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_model | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | not_promoted_sample_too_small | Need more than 30 completed pre-kickoff predictions for walk-forward scoring. |
| temperature_scaling | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | not_promoted_sample_too_small | Need more than 30 completed pre-kickoff predictions for walk-forward scoring. |
| shrinkage_to_base_rates | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | not_promoted_sample_too_small | Need more than 30 completed pre-kickoff predictions for walk-forward scoring. |
| market_prior_blend | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | not_promoted_sample_too_small | Need more than 30 completed pre-kickoff predictions for walk-forward scoring. |
| behavior_gated_shrinkage | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | not_promoted_sample_too_small | Need more than 30 completed pre-kickoff predictions for walk-forward scoring. |
| confidence_gated_shrinkage | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | not_promoted_sample_too_small | Need more than 30 completed pre-kickoff predictions for walk-forward scoring. |
| combined_conservative_calibrated_model | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | not_promoted_sample_too_small | Need more than 30 completed pre-kickoff predictions for walk-forward scoring. |

## Reliability By Probability Bin

| model | outcome | probability_bin | n | avg_predicted_probability | observed_frequency | calibration_error |
| --- | --- | --- | --- | --- | --- | --- |
| raw_model | home_win | 0.40-0.60 | 2.0 | 0.4525 | 1.0 | 0.5475 |
| raw_model | draw | 0.20-0.40 | 2.0 | 0.2648 | 0.0 | 0.2648 |
| raw_model | away_win | 0.20-0.40 | 2.0 | 0.2827 | 0.0 | 0.2827 |

## Performance By Confidence

| confidence_label | n | top_pick_accuracy | brier_score | multiclass_log_loss |
| --- | --- | --- | --- | --- |
| Low | 2.0 | 1.0 | 0.4498 | 0.7929 |

## Performance By Match Context

| dimension | segment | n | top_pick_accuracy | brier_score | multiclass_log_loss |
| --- | --- | --- | --- | --- | --- |
| favorite_probability_band | 0.40-0.60 | 2.0 | 1.0 | 0.4498 | 0.7929 |
| draw_probability_band | 0.20-0.40 | 2.0 | 1.0 | 0.4498 | 0.7929 |
| behavior_disagreement_band | missing | 2.0 | 1.0 | 0.4498 | 0.7929 |
| market_disagreement_band | missing | 2.0 | 1.0 | 0.4498 | 0.7929 |
| market_join_status | no_event_resolved | 2.0 | 1.0 | 0.4498 | 0.7929 |
| match_context | knockout | 2.0 | 1.0 | 0.4498 | 0.7929 |

## Worst Model Misses

| prediction_id | generated_at_utc | match_id | kickoff_utc | home_team | away_team | competition | round | venue | model_version | home_win_prob_raw | draw_prob_raw | away_win_prob_raw | home_win_prob_calibrated | draw_prob_calibrated | away_win_prob_calibrated | home_win_prob_market | draw_prob_market | away_win_prob_market | behavior_home_prob | behavior_draw_prob | behavior_away_prob | behavior_disagreement_pp | market_disagreement_pp | model_confidence | confidence_score | confidence_label | confidence_reason | confidence_flags | market_join_status | mapping_confidence | data_quality_status | actual_home_goals_90 | actual_away_goals_90 | actual_result_1x2 | advancing_team | result_semantics | top_pick_raw | top_pick_calibrated | probability_assigned_to_actual_raw | probability_assigned_to_actual_calibrated | brier_raw | brier_calibrated | log_loss_raw | log_loss_calibrated | market_probability_assigned_to_actual | brier_market | log_loss_market | raw_model_was_correct | calibrated_model_was_correct | market_was_closer_than_model | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dce6cea073eb3f33 | 2026-06-29T09:51:14+00:00 | wc2026_76 | 2026-06-29T17:00:00+00:00 | Brazil | Japan | FIFA World Cup | Round of 32 | NRG Stadium | baseline_external_calibrated_v1 | 0.4525 | 0.2648 | 0.2827 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | 27.0 | Low | No High confidence without production-eligible calibrated probabilities. Calibration sample too small; using conservative shrinkage. No High confidence when market mapping or usable prices are missing. Knockout matches add draw/advancement interpretation risk. | calibration_not_production_eligible; calibration_sample_too_small; market_not_joined; market_disagreement_unavailable; data_recency_unknown; knockout_match; injuries_news_not_integrated | no_event_resolved | Unavailable | scored_90_minute | 1.0 | 0.0 | home_win | Unavailable | 90-minute regular time | home_win | Unavailable | 0.4525 | Unavailable | 0.4498 | Unavailable | 0.7929 | Unavailable | Unavailable | Unavailable | Unavailable | 1.0 | Unavailable | Unavailable | source=prediction_ledger; calibration_min_sample=30 |
| 0fd4069f946d70a8 | 2026-06-28T19:27:19+00:00 | wc2026_76 | 2026-06-29T17:00:00+00:00 | Brazil | Japan | FIFA World Cup | Round of 32 | NRG Stadium | baseline_external_calibrated_v1 | 0.4525 | 0.2648 | 0.2827 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | 27.0 | Low | No High confidence without production-eligible calibrated probabilities. Calibration sample too small; using conservative shrinkage. No High confidence when market mapping or usable prices are missing. Knockout matches add draw/advancement interpretation risk. | calibration_not_production_eligible; calibration_sample_too_small; market_not_joined; market_disagreement_unavailable; data_recency_unknown; knockout_match; injuries_news_not_integrated | no_event_resolved | Unavailable | scored_90_minute | 1.0 | 0.0 | home_win | Unavailable | 90-minute regular time | home_win | Unavailable | 0.4525 | Unavailable | 0.4498 | Unavailable | 0.7929 | Unavailable | Unavailable | Unavailable | Unavailable | 1.0 | Unavailable | Unavailable | source=prediction_ledger; calibration_min_sample=30 |

## Conclusions

- Calibration improves Brier score: Unavailable; no production-eligible calibrated rows.
- Calibration improves log loss: Unavailable; no production-eligible calibrated rows.
- Model beats market-implied probabilities: Market benchmark sample too small for reliable conclusions.
- Where the model is overconfident: review high favorite-probability bands, high behavior disagreement, and knockout draw rows above.
- Ready to claim calibrated predictive probabilities: No. The current evaluated sample is too small and no candidate is production-eligible.