# Schedule Result Bridge Audit

This audit maps `data/results_ledger.csv` rows to official `data/fixtures.csv` rows before prediction evaluation. It does not overwrite either source ledger.

## Summary

| metric | value |
| --- | --- |
| result_rows | 47 |
| fixture_rows | 70 |
| prediction_rows | 64 |
| result_rows_mapped_to_fixtures | 25 |
| mapped_by_exact_match_id | 13 |
| mapped_by_home_away_date | 11 |
| mapped_by_reversed_team_order_date | 0 |
| mapped_by_normalized_team_date | 1 |
| mapped_by_fuzzy_team_date | 0 |
| ambiguous_result_rows | 0 |
| unmapped_result_rows | 22 |

## Unmapped Result Rows

| result_match_id | result_date_utc | result_home | result_away | join_method | join_reason |
| --- | --- | --- | --- | --- | --- |
| 537327 | 2026-06-11 | Mexico | South Africa | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_11_south_korea_czechia_2_1_fifa_world_cup | 2026-06-11 | South Korea | Czechia | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| 537328 | 2026-06-12 | South Korea | Czechia | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| 537333 | 2026-06-12 | Canada | Bosnia-H. | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_12_canada_bosnia_and_herzegovina_1_1_fifa_world_cup | 2026-06-12 | Canada | Bosnia and Herzegovina | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_12_united_states_paraguay_4_1_fifa_world_cup | 2026-06-12 | United States | Paraguay | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| 537334 | 2026-06-13 | Qatar | Switzerland | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_13_australia_turkey_2_0_fifa_world_cup | 2026-06-13 | Australia | Turkey | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_13_brazil_morocco_1_1_fifa_world_cup | 2026-06-13 | Brazil | Morocco | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_13_haiti_scotland_0_1_fifa_world_cup | 2026-06-13 | Haiti | Scotland | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_14_germany_cura_ao_7_1_fifa_world_cup | 2026-06-14 | Germany | Curaçao | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_14_ivory_coast_ecuador_1_0_fifa_world_cup | 2026-06-14 | Ivory Coast | Ecuador | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_14_netherlands_japan_2_2_fifa_world_cup | 2026-06-14 | Netherlands | Japan | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_14_sweden_tunisia_5_1_fifa_world_cup | 2026-06-14 | Sweden | Tunisia | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_15_belgium_egypt_1_1_fifa_world_cup | 2026-06-15 | Belgium | Egypt | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_15_iran_new_zealand_2_2_fifa_world_cup | 2026-06-15 | Iran | New Zealand | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_15_saudi_arabia_uruguay_1_1_fifa_world_cup | 2026-06-15 | Saudi Arabia | Uruguay | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_15_spain_cape_verde_0_0_fifa_world_cup | 2026-06-15 | Spain | Cape Verde | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_16_argentina_algeria_3_0_fifa_world_cup | 2026-06-16 | Argentina | Algeria | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_16_austria_jordan_3_1_fifa_world_cup | 2026-06-16 | Austria | Jordan | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_16_france_senegal_3_1_fifa_world_cup | 2026-06-16 | France | Senegal | no_fixture_match | No fixture matched this result row by ID, teams, and date. |
| csv_2026_06_16_iraq_norway_1_4_fifa_world_cup | 2026-06-16 | Iraq | Norway | no_fixture_match | No fixture matched this result row by ID, teams, and date. |

## Ambiguous Result Rows

_No rows._

Some result sources use provider IDs while saved prediction snapshots use fixture IDs. The schedule bridge uses date and team names so completed results can be matched to saved predictions without changing the source ledgers.